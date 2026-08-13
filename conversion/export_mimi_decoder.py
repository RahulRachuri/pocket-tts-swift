"""Mimi decoder -> Core AI: functional state-in/state-out rewrite, eager gate, export, engine gate.

One call decodes ONE 12.5 Hz latent frame into 1920 PCM samples at 24 kHz:

    latent [1,512,1] (post-quantizer)  +  12 state tensors
        -> pcm [1,1,1920]              +  12 updated state tensors

Upstream mutates its streaming state in place (`state["previous"][:] = ...`,
`cache[0,:,off:off+k] = k`) which torch.export cannot see through, so every stateful
primitive is re-expressed functionally:

  * StreamingConv1d          -> cat(prev, x) -> conv -> tail slice is the new prev
  * StreamingConvTranspose1d -> convtr -> overlap-add head, tail (minus bias) is new partial
  * KV cache                 -> fixed-capacity SHIFT REGISTER (cat + slice), no scatter,
                                no .item(). Capacity 272 >= context(250) + 16 new positions,
                                which makes the ring mathematically exact for this graph.
                                Ring is zero-initialised, NOT NaN-initialised: upstream slices
                                the NaN tail away, but a masked SDPA still multiplies V by 0
                                and 0 * NaN = NaN.

Gate ladder (zoo style):
  1. eager  : 31-frame replay on the oracle latents vs oracle['mimi/out'] -> must be ~exact
  2. engine : same replay driving the .aimodel, cosine per frame, cpu then gpu
"""

from __future__ import annotations

import argparse, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RING = 272          # >= context 250 + 16 positions written per call
CTX = 250
STEPS = 16          # encoder frames per 12.5 Hz latent (200 Hz / 12.5 Hz)


# --------------------------------------------------------------------- primitives
def elu(x: torch.Tensor) -> torch.Tensor:
    """ELU(alpha=1). aten.elu has no Core AI lowering on coreai-torch 0.4.1, so it is
    written out; expm1 keeps this bit-identical to torch's own kernel."""
    return torch.where(x > 0, x, torch.expm1(x))


def sconv(conv: nn.Conv1d, x: torch.Tensor, prev: torch.Tensor):
    """StreamingConv1d, pad_mode='constant' (the only mode on the decode path)."""
    tp = prev.shape[-1]
    if tp == 0:
        return conv(x), prev
    xc = torch.cat([prev, x], dim=-1)
    return conv(xc), xc[..., -tp:]


def sconvtr(convtr: nn.ConvTranspose1d, x: torch.Tensor, partial: torch.Tensor):
    y = convtr(x)
    pt = partial.shape[-1]
    if pt == 0:
        return y, partial
    y = torch.cat([y[..., :pt] + partial, y[..., pt:]], dim=-1)
    tail = y[..., -pt:]
    if convtr.bias is not None:
        tail = tail - convtr.bias[:, None]
    return y[..., :-pt], tail


def apply_rope(q, k, offset, max_period=10000.0):
    """Verbatim port of modules/rope.py with a tensor offset."""
    B, T, H, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-float(np.log(max_period)) * 2 / D))
    ts = torch.arange(T, device=q.device, dtype=torch.float32) + offset.to(torch.float32)
    ts = ts.view(-1, 1, 1)
    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, H, D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
    qo = torch.stack([(qr * rotr - qi * roti), (qr * roti + qi * rotr)], dim=-1)
    ko = torch.stack([(kr * rotr - ki * roti), (kr * roti + ki * rotr)], dim=-1)
    return qo.view(B, T, H, D).to(q.dtype), ko.view(B, T, H, D).to(k.dtype)


class RingAttention(nn.Module):
    """StreamingMultiheadAttention with a shift-register KV cache."""

    def __init__(self, src, heads: int):
        super().__init__()
        self.in_proj = src.in_proj
        self.out_proj = src.out_proj
        self.h = heads
        self.d = src.dim_per_head

    def forward(self, x, cache, offset):
        b, t, _ = x.shape
        packed = self.in_proj(x).view(b, t, 3, self.h, self.d)
        q, k, v = torch.unbind(packed, dim=2)
        q, k = apply_rope(q, k, offset)
        new = torch.stack([k, v], dim=0)                       # [2,b,t,h,d]
        cache = torch.cat([cache[:, :, t:], new], dim=2)       # shift-append
        k_attn = cache[0].permute(0, 2, 1, 3)
        v_attn = cache[1].permute(0, 2, 1, 3)
        # absolute positions of the ring slots and of the queries
        pos_k = offset.view(1, 1) + t - RING + torch.arange(RING, device=x.device).view(1, -1)
        pos_q = offset.view(1, 1) + torch.arange(t, device=x.device).view(1, -1)
        delta = pos_q[:, :, None] - pos_k[:, None, :]
        mask = (pos_k[:, None, :] >= 0) & (delta >= 0) & (delta < CTX)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k_attn, v_attn, mask[:, None])
        y = y.transpose(1, 2).reshape(b, t, self.h * self.d)
        return self.out_proj(y), cache


class MimiDecoderStep(nn.Module):
    """upsample_mode:
      'convtr' — upstream form, ConvTranspose1d(512,512,k=32,s=16,groups=512).
      'outer'  — equivalent rewrite valid because T==1 and groups==channels: the
                 transposed conv degenerates to a per-channel outer product
                 out[c,:] = x[c,0] * W[c,0,:]. Exists because stride>=8 transposed
                 convs are miscomputed by the Core AI cpu_only delegate (see
                 repro_convtr_cpu.py); this form has no ConvTranspose1d at all.

    fold_front (M2): take the RAW flow-LM latent [1,32] and do the
    `latent * emb_std + emb_mean` rescale plus the quantizer `output_proj`
    (Conv1d 32->512, k=1, bias-free — asserted) in-graph, deleting the last
    host-side tensor math. Verified bit-exact against upstream's
    `mimi.quantizer((lat*std+mean).transpose(-1,-2))` before adoption.

    graph_state (M2): the 12 streaming-state tensors become Core AI in-graph
    state (`state_names=`) instead of round-tripping through the host every
    frame. The forward computes exactly the same new values and then commits
    them with in-place `copy_` on the state inputs (torch.export records these
    as user-input mutations; `remove_functionalization` re-materialises them,
    same mechanics as the flow-LM KV). Chunk-boundary semantics are untouched:
    the host owns the buffers for the whole run and never resets them, which is
    the "Mimi state is NEVER reset across chunks" invariant by construction.
    """

    def __init__(self, mimi, upsample_mode: str = "convtr",
                 fold_front: dict | None = None, graph_state: bool = False):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.graph_state = graph_state
        self.fold = fold_front is not None
        if self.fold:
            self.register_buffer("emb_std", fold_front["emb_std"].detach().clone())
            self.register_buffer("emb_mean", fold_front["emb_mean"].detach().clone())
            self.register_buffer("proj_w", fold_front["proj_w"].detach().clone())
        self.up = mimi.upsample.convtr.convtr
        layers = mimi.decoder_transformer.transformer.layers
        assert mimi.decoder_transformer.input_proj is None
        self.attn = nn.ModuleList([RingAttention(l.self_attn, l.self_attn.num_heads) for l in layers])
        self.layers = layers
        self.m = mimi.decoder.model

    def _tlayer(self, i, x, cache, offset):
        l = self.layers[i]
        upd, cache = self.attn[i](l.norm1(x), cache, offset)
        x = x + l.layer_scale_1(upd)
        x = x + l.layer_scale_2(l.linear2(F.gelu(l.linear1(l.norm2(x)))))
        return x, cache

    def _resnet(self, blk, x, prev):
        v = elu(x)
        v, prev = sconv(blk.block[1].conv, v, prev)
        v = elu(v)
        v = blk.block[3].conv(v)          # kernel 1 -> no history
        return x + v, prev

    def _upsample(self, latent, up_p):
        if self.upsample_mode == "outer":
            # latent [1,512,1] -> [1,512,32] without a ConvTranspose1d
            y = latent * self.up.weight[:, 0, :].unsqueeze(0)
        else:
            y = self.up(latent)
        pt = up_p.shape[-1]
        y = torch.cat([y[..., :pt] + up_p, y[..., pt:]], dim=-1)
        return y[..., :-pt], y[..., -pt:]

    def forward(self, latent, up_p, kv0, kv1, offset,
                c0, c2, c3, c5, c6, c8, c9, c11):
        if self.fold:
            # latent is the RAW [1,32] flow-decoder output
            x32 = latent * self.emb_std + self.emb_mean       # [1,32]
            latent = F.conv1d(x32.unsqueeze(-1), self.proj_w)  # [1,512,1]
        x, up_p_n = self._upsample(latent, up_p)              # [1,512,16]
        h = x.transpose(1, 2)                                 # [1,16,512]
        h, kv0_n = self._tlayer(0, h, kv0, offset)
        h, kv1_n = self._tlayer(1, h, kv1, offset)
        z = h.transpose(1, 2)

        z, c0_n = sconv(self.m[0].conv, z, c0)
        z = elu(z); z, c2_n = sconvtr(self.m[2].convtr, z, c2)
        z, c3_n = self._resnet(self.m[3], z, c3)
        z = elu(z); z, c5_n = sconvtr(self.m[5].convtr, z, c5)
        z, c6_n = self._resnet(self.m[6], z, c6)
        z = elu(z); z, c8_n = sconvtr(self.m[8].convtr, z, c8)
        z, c9_n = self._resnet(self.m[9], z, c9)
        z = elu(z); z, c11_n = sconv(self.m[11].conv, z, c11)
        if self.graph_state:
            # commit every state in place AFTER all reads; nothing is returned
            # but the PCM. `offset` is read only by RoPE/the ring mask above.
            up_p.copy_(up_p_n); kv0.copy_(kv0_n); kv1.copy_(kv1_n)
            c0.copy_(c0_n); c2.copy_(c2_n); c3.copy_(c3_n); c5.copy_(c5_n)
            c6.copy_(c6_n); c8.copy_(c8_n); c9.copy_(c9_n); c11.copy_(c11_n)
            offset.copy_(offset + STEPS)
            return z
        return (z, up_p_n, kv0_n, kv1_n, offset + STEPS,
                c0_n, c2_n, c3_n, c5_n, c6_n, c8_n, c9_n, c11_n)


STATE_NAMES = ("up_p", "kv0", "kv1", "offset", "c0", "c2", "c3", "c5", "c6", "c8", "c9", "c11")
IN_NAMES = ("latent",) + STATE_NAMES
OUT_NAMES = ("pcm",) + tuple(f"{n}_out" for n in STATE_NAMES)


def init_state(dtype):
    z = lambda *s: torch.zeros(*s, dtype=dtype)
    return dict(up_p=z(1, 512, 16), kv0=z(2, 1, RING, 8, 64), kv1=z(2, 1, RING, 8, 64),
                offset=torch.zeros(1, dtype=torch.int32),
                c0=z(1, 512, 6), c2=z(1, 256, 6), c3=z(1, 256, 2), c5=z(1, 128, 5),
                c6=z(1, 128, 2), c8=z(1, 64, 4), c9=z(1, 64, 2), c11=z(1, 64, 2))


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="oracle/orc_a.npz")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--upsample", choices=["convtr", "outer"], default="convtr")
    ap.add_argument("--fold-quantizer", action="store_true",
                    help="graph input is the raw [1,32] latent; rescale+quantizer in-graph")
    ap.add_argument("--graph-state", action="store_true",
                    help="the 12 streaming-state tensors become in-graph Core AI state")
    args = ap.parse_args()

    from pocket_tts import TTSModel
    torch.set_num_threads(1)
    orc = np.load(ROOT / args.oracle)
    gold = torch.from_numpy(orc["mimi/out"])                  # [N,1920]
    n = gold.shape[0]

    model = TTSModel.load_model().eval()
    fold_front = None
    if args.fold_quantizer:
        proj = model.mimi.quantizer.output_proj
        assert proj.bias is None, "fold assumes the bias-free output_proj"
        fold_front = dict(
            emb_std=model.flow_lm.emb_std.to(torch.float32),
            emb_mean=model.flow_lm.emb_mean.to(torch.float32),
            proj_w=proj.weight.to(torch.float32),
        )
        # the raw flow-decoder latents, [n,32] so lat[i:i+1] is the [1,32] graph
        # input; the mimi frames used latents 0..n-1 (the breaking step's latent
        # is discarded upstream, hence 32 steps -> 31 frames)
        lat = torch.from_numpy(orc["step/latent"][:n])
        # sanity: the folded front end must reproduce the captured post-quantizer input
        with torch.no_grad():
            q = F.conv1d((lat * fold_front["emb_std"] + fold_front["emb_mean"])
                         .reshape(n, 32, 1), fold_front["proj_w"])
        dq = (q.reshape(n, 512) - torch.from_numpy(orc["mimi/in"])).abs().max()
        assert dq < 1e-5, f"folded front end diverges from oracle mimi/in: {dq}"
        print(f"[fold] rescale+quantizer front end vs oracle mimi/in: max|Δ| {dq:.2e}")
    else:
        lat = torch.from_numpy(orc["mimi/in"]).unsqueeze(-1)  # [N,512,1]

    step = MimiDecoderStep(model.mimi, args.upsample,
                           fold_front=fold_front, graph_state=args.graph_state).eval()

    def eager_replay():
        st = init_state(torch.float32)
        outs = []
        for i in range(n):
            if args.graph_state:  # states mutate in place, only pcm returns
                outs.append(step(lat[i:i + 1], *[st[k] for k in STATE_NAMES]).reshape(-1))
            else:
                r = step(lat[i:i + 1], *[st[k] for k in STATE_NAMES])
                outs.append(r[0].reshape(-1))
                st = dict(zip(STATE_NAMES, r[1:]))
        return torch.stack(outs)

    # ---- 1. eager gate ---------------------------------------------------
    with torch.no_grad():
        eager = eager_replay()
    cos = F.cosine_similarity(eager, gold, dim=-1)
    print(f"[eager fp32] frames {n}  cos mean {cos.mean():.6f} min {cos.min():.6f}  "
          f"max|Δ| {(eager - gold).abs().max():.3e}")
    if cos.min() < 0.9999:
        print("❌ functional re-author DIVERGES — fix before export"); raise SystemExit(1)
    print("✅ eager functional rewrite reproduces the streaming decoder")
    if args.skip_export:
        return

    # ---- 2. export -------------------------------------------------------
    import asyncio
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    st0 = init_state(dtype)
    lat_example = (torch.zeros(1, 32, dtype=dtype) if args.fold_quantizer
                   else torch.zeros(1, 512, 1, dtype=dtype))
    example = {"latent": lat_example, **st0}
    if args.graph_state:
        in_names, out_names, state_names = ("latent",), ("pcm",), STATE_NAMES
    else:
        in_names, out_names, state_names = IN_NAMES, OUT_NAMES, None
    tag = ("_q" if args.fold_quantizer else "") + ("_gs" if args.graph_state else "")
    print(f"[export] mimi decoder ({args.dtype}, ring {RING}, "
          f"fold={args.fold_quantizer}, graph_state={args.graph_state}) -> Core AI ...",
          flush=True)
    prog = export_to_coreai(step.to(dtype), example, dynamic_shapes=None,
                            input_names=in_names, output_names=out_names,
                            state_names=state_names, externalize_modules=[])
    prog.optimize()
    art = ROOT / args.artifacts; art.mkdir(exist_ok=True)
    path = art / f"mimi_decoder_{args.dtype}_ring{RING}_{args.upsample}{tag}.aimodel"
    shutil.rmtree(path, ignore_errors=True)
    meta = rt.AIModelAssetMetadata(); meta.license = "cc-by-4.0"
    prog.save_asset(path, meta)
    sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {path.name} ({sz:.1f} MB)")

    async def gate(unit):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(
                    getattr(rt.ComputeUnitKind, unit)()))
        m = await rt.AIModel.load(str(path), opts)
        fn = m.load_function("main")
        outs = []
        if args.graph_state:
            state = {k: rt.NDArray(np.ascontiguousarray(v.numpy()))
                     for k, v in init_state(dtype).items()}
            for i in range(n):
                r = await fn(
                    inputs={"latent": rt.NDArray(np.ascontiguousarray(lat[i:i + 1].to(dtype).numpy()))},
                    state=state,
                )
                outs.append(torch.from_numpy(r["pcm"].numpy().astype(np.float32)).reshape(-1))
        else:
            st = {k: v.numpy() for k, v in init_state(dtype).items()}
            for i in range(n):
                feed = {"latent": rt.NDArray(np.ascontiguousarray(lat[i:i + 1].to(dtype).numpy()))}
                feed.update({k: rt.NDArray(v) for k, v in st.items()})
                r = await fn(feed)
                outs.append(torch.from_numpy(r["pcm"].numpy().astype(np.float32)).reshape(-1))
                st = {k: r[f"{k}_out"].numpy() for k in STATE_NAMES}
        eng = torch.stack(outs)
        c = F.cosine_similarity(eng, gold, dim=-1)
        ok = c.mean() > 0.999 and c.min() > 0.99
        print(f"[gate {unit}] frames {n} cos mean {c.mean():.6f} min {c.min():.6f} "
              f"max|Δ| {(eng - gold).abs().max():.4f} -> {'PASS' if ok else 'FAIL'}")
        return ok

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
