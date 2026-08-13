"""Static-shape, in-graph-state rewrites of the pocket-tts flow-LM (graphs b/c) and
the flow decoder (graph d).

Phase A left three export blockers in the flow-LM; all three are answered here:

  * `.item()`-driven KV slice (`modules/transformer.py:14-18`)
        -> a Core AI **state** (`k_cache` / `v_cache`, mutated in graph) written with
           `mutable_slice_update` at a runtime `pos` tensor. No Python int, no
           host round-trip. The whole cache is read back and masked causally, which
           is exactly upstream's `valid = cache[:, :, : offset + T]` semantics
           because a position past `pos + T - 1` is never <= any query position.
  * RNG inside forward (`models/flow_lm.py:131-137`)
        -> noise is an input to the flow decoder graph; the flow-LM step graph does
           not draw at all (it stops at `cond` / `eos_logit`).
  * NaN-as-BOS (`models/tts_model.py:748` -> `models/flow_lm.py:121`)
        -> deleted. An explicit `is_bos` float flag blends `bos_emb` in.

The KV cache is packed layer-first (`[L, 1, H, S, D]`, already in attention layout)
so the whole 6-layer cache is TWO states rather than twelve tensors. FluidAudio's
rank-4 split was a Core ML ANE compiler constraint; the Core AI zoo ships rank-5
packed caches (vibevoice / dots_tts) and they specialize fine, so the split buys
nothing here.

Parity traps honoured (NOTES.md section 7): interleaved-pair RoPE with an absolute
offset, LayerNorm eps 1e-5 in the transformer / 1e-6 in the flow net, the flow net's
non-standard `RMSNorm` (mean-centred, *unbiased* variance, eps outside the rsqrt),
`bos -> voice -> text -> latents` ordering, and the fact that the cache holds
POST-RoPE keys.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from coreai_models.primitives._ops import mutable_slice_update

# --- static graph budget -----------------------------------------------------
# The cache must hold: voice conditioning + the text chunk + EVERY generated frame,
# because the flow-LM has no context window and so nothing ever rolls out of it.
#
#   voice conditioning   76..162 positions — it is NOT 126 for every voice. The eight
#                        shipped English voices measure 126/126/126/126/133/141/162,
#                        and the wider v3 catalogue runs 76..162.
#   text chunk           <= MAX_TOKEN_PER_CHUNK = 50 (upstream's chunker can still
#                        emit more when a long sentence has no comma/colon to split
#                        on — measured at 121 tokens on one Moby Dick sentence).
#   generation           bounded by upstream's own `_estimate_max_gen_len`, which is
#                        ceil((tokens/3 + 2) * 12.5) = 234 frames for a 50-token chunk.
#
# Worst legal case is therefore 162 + 50 + 234 = 446, so **S_MAX = 512**. The Phase B
# value of 256 was sized off the oracle fixture's 32 generated frames and truncates any
# chunk over roughly 35 tokens; see NOTES.md section 16.
S_MAX = int(os.environ.get("POCKET_TTS_S_MAX", "512"))
T_PRE = 16          # prefill chunk width (pad the text chunk up to this)
N_LAYERS = 6
N_HEADS = 16
HEAD_DIM = 64
D_MODEL = 1024
LDIM = 32

STATE_NAMES = ("k_cache", "v_cache")


def build_kv_state(dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Zero-initialised, NOT NaN-initialised — a masked SDPA still multiplies V by a
    zero weight and 0 * NaN = NaN (NOTES.md section 6, blocker iii-b)."""
    shape = (N_LAYERS, 1, N_HEADS, S_MAX, HEAD_DIM)
    return {
        "k_cache": torch.zeros(shape, dtype=dtype),
        "v_cache": torch.zeros(shape, dtype=dtype),
    }


def _write_kv(cache: torch.Tensor, layer: int, pos: torch.Tensor, x: torch.Tensor) -> None:
    """cache[layer, :, :, pos : pos + T] = x, with `pos` a runtime i32 tensor.

    x is [1, H, T, D]. `mutable_slice_update` is the zoo's data-indexed state write;
    it survives `remove_functionalization` and lowers to an in-place state mutation.
    """
    dev = cache.device
    t = x.shape[2]
    li = torch.tensor([layer], dtype=torch.int32, device=dev)
    z = torch.zeros(1, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.int32, device=dev)
    nh = torch.tensor([cache.shape[2]], dtype=torch.int32, device=dev)
    hd = torch.tensor([cache.shape[4]], dtype=torch.int32, device=dev)
    tn = torch.tensor([t], dtype=torch.int32, device=dev)
    p = pos.reshape(1).to(torch.int32)
    begin = torch.cat([li, z, z, p, z])
    end = torch.cat([li + 1, one, nh, p + tn, hd])
    mutable_slice_update(cache, x.unsqueeze(0), begin, end)


def _rope(q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor, max_period: float = 10_000.0):
    """Interleaved-pair (complex) RoPE — `modules/rope.py:36-56`, verbatim except that
    the offset is a runtime tensor instead of a Python int. Rotation math in fp32."""
    b, t, h, d = q.shape
    ds = torch.arange(d // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / d))
    # pos stays rank-1: `coreai.reshape` rejects a rank-0 shape operand, so a
    # `.reshape(())` scalar squeeze does not lower. Broadcasting [T] + [1] is fine.
    ts = (torch.arange(t, device=q.device, dtype=torch.float32) + pos.to(torch.float32)).view(
        -1, 1, 1
    )
    q = q.view(b, t, h, d // 2, 2)
    k = k.view(b, t, h, d // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
    # rotation math in fp32, cast back to the working precision (upstream `rope.py:54`)
    dt = q.dtype
    qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1).to(dt)
    ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1).to(dt)
    return qo.view(b, t, h, d), ko.view(b, t, h, d)


class FlowLMCore(nn.Module):
    """The 6-layer flow-LM backbone with a static KV state.

    forward(x [1,T,1024], pos [1] i32, k_cache, v_cache) -> cond [1,T,1024]

    `x` is already `input_linear(sequence)` for AR steps, or the raw text embeddings
    for prefill — upstream concatenates the two and only ever has one of them
    non-empty per call (`models/flow_lm.py:150`, `models/tts_model.py:356`).
    """

    def __init__(self, flow_lm: nn.Module):
        super().__init__()
        self.layers = flow_lm.transformer.layers
        self.out_norm = flow_lm.out_norm

    def forward(self, x, pos, k_cache, v_cache):
        t = x.shape[1]
        # causal mask over the whole cache: slot j is visible to query row i iff
        # j <= pos + i. Slots above that are either unwritten (zero) or this call's
        # own future rows — exactly upstream's `valid = cache[..., : offset + T]`.
        k_idx = torch.arange(S_MAX, device=x.device, dtype=torch.float32)
        q_pos = torch.arange(t, device=x.device, dtype=torch.float32) + pos.to(torch.float32)
        mask = k_idx.view(1, 1, 1, S_MAX) <= q_pos.view(1, 1, t, 1)

        for i, layer in enumerate(self.layers):
            h = layer.norm1(x)
            projected = layer.self_attn.in_proj(h)
            packed = projected.view(1, t, 3, N_HEADS, HEAD_DIM)
            q, k, v = torch.unbind(packed, dim=2)
            q, k = _rope(q, k, pos)
            q = q.transpose(1, 2)                       # [1,H,T,D]
            _write_kv(k_cache, i, pos, k.transpose(1, 2))
            _write_kv(v_cache, i, pos, v.transpose(1, 2))
            a = F.scaled_dot_product_attention(q, k_cache[i], v_cache[i], mask, dropout_p=0.0)
            a = a.transpose(1, 2).reshape(1, t, D_MODEL)
            x = x + layer.self_attn.out_proj(a)
            # FFN (`modules/mimi_transformer.py:_ff_block`) — no layer scale on the flow-LM
            x = x + layer.linear2(F.gelu(layer.linear1(layer.norm2(x))))
        return self.out_norm(x)


class FlowLMStep(nn.Module):
    """Graph (c) — one AR step.

    latent_in [1,1,32], is_bos [1], pos [1] i32, + KV state
        -> cond [1,1024], eos_logit [1,1]
    """

    def __init__(self, flow_lm: nn.Module):
        super().__init__()
        self.core = FlowLMCore(flow_lm)
        self.input_linear = flow_lm.input_linear
        self.out_eos = flow_lm.out_eos
        self.register_buffer("bos_emb", flow_lm.bos_emb.detach().clone())

    def forward(self, latent_in, is_bos, pos, k_cache, v_cache):
        f = is_bos.reshape(1, 1, 1)
        seq = f * self.bos_emb.view(1, 1, LDIM) + (1.0 - f) * latent_in
        x = self.input_linear(seq)
        cond = self.core(x, pos, k_cache, v_cache)[:, -1]
        return cond, self.out_eos(cond)


class FlowLMPrefill(nn.Module):
    """Graph (b) — one T_PRE-wide text chunk into the same KV state.

    text_emb [1,T_PRE,1024], pos [1] i32, + KV state -> cond [1,1024]

    The output is discarded by the host (upstream's prefill call emits a latent it
    throws away — `models/flow_lm.py:156` with `sequence.shape[1] == 0` keeps the
    whole tensor because `-0 == 0`; see NOTES.md section 7). Short chunks are padded to
    T_PRE: the pad rows write junk into cache slots that no real row can attend to
    (they sit strictly after every real position) and that the first AR steps
    overwrite before ever reading them.
    """

    def __init__(self, flow_lm: nn.Module):
        super().__init__()
        self.core = FlowLMCore(flow_lm)

    def forward(self, text_emb, pos, k_cache, v_cache):
        return self.core(text_emb, pos, k_cache, v_cache)[:, -1]


class _VarFreeRMSNorm(nn.Module):
    """`modules/mlp.py:20-36` without `torch.var`.

    `aten.var.correction` has no Core AI lowering on coreai-torch 0.4.1 (same class of
    gap as the `aten.elu` one Phase A hit in SEANet). The flow net's `RMSNorm` is not
    RMSNorm: it mean-CENTRES, uses the UNBIASED (n-1) estimator, and adds eps OUTSIDE
    the rsqrt. All three are reproduced literally here.
    """

    def __init__(self, src: nn.Module):
        super().__init__()
        self.alpha = nn.Parameter(src.alpha.detach().clone())
        self.eps = src.eps

    def forward(self, x):
        n = x.shape[-1]
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).sum(dim=-1, keepdim=True) / (n - 1)
        return x * (self.alpha * torch.rsqrt(self.eps + var))


class _VarFreeLayerNorm(nn.Module):
    """`modules/mlp.py:39-55` without `torch.var` — biased (n) estimator, eps 1e-6
    INSIDE the sqrt, affine optional (`FinalLayer.norm_final` has none)."""

    def __init__(self, src: nn.Module):
        super().__init__()
        self.eps = src.eps
        self.affine = hasattr(src, "weight")
        if self.affine:
            self.weight = nn.Parameter(src.weight.detach().clone())
            self.bias = nn.Parameter(src.bias.detach().clone())

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias if self.affine else x


def _replace_var_norms(module: nn.Module) -> nn.Module:
    from pocket_tts.modules.mlp import LayerNorm as PtLayerNorm, RMSNorm as PtRMSNorm

    for name, child in module.named_children():
        if isinstance(child, PtRMSNorm):
            setattr(module, name, _VarFreeRMSNorm(child))
        elif isinstance(child, PtLayerNorm):
            setattr(module, name, _VarFreeLayerNorm(child))
        else:
            _replace_var_norms(child)
    return module


class FlowDecoder(nn.Module):
    """Graph (d) — the LSD flow decoder at num_steps=1 (upstream's default,
    `pocket_tts/default_parameters.py:3`).

    cond [1,1024], noise [1,32] -> latent [1,32]

    At num_steps=1 `lsd_decode` is `x0 + v(s=0, t=1, x0)`, so `s`/`t` are baked as
    zeros/ones constants. The in-place `current += flow_dir / num_steps` of
    `models/flow_lm.py:38` is written out-of-place here — that mutation is the trap
    that forced the oracle to clone `step/noise` at capture time.
    """

    def __init__(self, flow_lm: nn.Module):
        super().__init__()
        import copy

        self.flow_net = _replace_var_norms(copy.deepcopy(flow_lm.flow_net))

    def forward(self, cond, noise):
        s = torch.zeros_like(noise[..., :1])
        t = torch.ones_like(noise[..., :1])
        return noise + self.flow_net(cond, s, t, noise)
