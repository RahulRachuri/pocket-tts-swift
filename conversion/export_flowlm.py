"""Flow-LM prefill + AR step -> ONE Core AI multifunction asset, gated on the oracle.

    prefill(text_emb [1,16,1024], pos [1])            -> cond [1,1024]   |  shared
    step   (latent_in [1,1,32], is_bos [1], pos [1])  -> cond, eos_logit |  k/v state

Both entrypoints are staged into one `TorchConverter`, so the 75.5 M transformer
weights are stored once and both functions mutate the SAME `k_cache`/`v_cache`
Core AI states (no host KV round-trip).

Gate ladder:
  1. eager  : seed the state with the shipped voice conditioning, prefill the oracle's
              text embeddings, then teacher-force all 32 AR steps on the oracle's own
              latents. `cond` and `eos_logit` must match `oracle['step/*']`.
  2. engine : the same replay driving the .aimodel, cpu then gpu.

Run (export venv):
    HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
      .venv-export/bin/python conversion/export_flowlm.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from flowlm_graphs import (  # noqa: E402
    LDIM,
    S_MAX,
    STATE_NAMES,
    T_PRE,
    FlowLMPrefill,
    FlowLMStep,
    build_kv_state,
)

VOICE_LEN = 126     # the shipped `alba` conditioning state, as pre-baked in the checkpoint


def seed_voice_state(state: dict, orc, dtype=torch.float32) -> int:
    """Copy the oracle's captured voice KV (upstream layout [2,1,L,H,D]) into the
    packed [layer,1,H,S,D] Core AI state. Returns the resulting absolute position."""
    offs = set()
    for i in range(6):
        c = torch.from_numpy(orc[f"voice/transformer.layers.{i}.self_attn/cache"])
        off = int(orc[f"voice/transformer.layers.{i}.self_attn/offset"].reshape(-1)[0])
        offs.add(off)
        n = c.shape[2]
        state["k_cache"][i, 0, :, :n] = c[0, 0].permute(1, 0, 2).to(dtype)
        state["v_cache"][i, 0, :, :n] = c[1, 0].permute(1, 0, 2).to(dtype)
    assert len(offs) == 1, f"voice offsets disagree across layers: {offs}"
    return offs.pop()


def pad_text(text_emb: torch.Tensor) -> torch.Tensor:
    t = text_emb.shape[1]
    assert t <= T_PRE, f"text chunk {t} exceeds the T_PRE={T_PRE} prefill width"
    return F.pad(text_emb, (0, 0, 0, T_PRE - t))


def replay_eager(prefill, step, orc, dtype=torch.float32):
    """Teacher-forced replay in torch: voice state -> prefill -> 32 AR steps."""
    st = build_kv_state(dtype)
    pos = seed_voice_state(st, orc, dtype)
    text = torch.from_numpy(orc["prefill/text_embeddings"]).to(dtype)
    t_real = text.shape[1]
    with torch.no_grad():
        prefill(pad_text(text), torch.tensor([pos], dtype=torch.int32), st["k_cache"], st["v_cache"])
        pos += t_real
        gold_lat = torch.from_numpy(orc["step/latent"])
        n = gold_lat.shape[0]
        conds, eoss = [], []
        for i in range(n):
            lat = (
                torch.zeros(1, 1, LDIM, dtype=dtype)
                if i == 0
                else gold_lat[i - 1].reshape(1, 1, LDIM).to(dtype)
            )
            c, e = step(
                lat,
                torch.tensor([1.0 if i == 0 else 0.0], dtype=dtype),
                torch.tensor([pos + i], dtype=torch.int32),
                st["k_cache"],
                st["v_cache"],
            )
            conds.append(c.reshape(-1).float())
            eoss.append(e.reshape(-1).float())
    return torch.stack(conds), torch.stack(eoss)


def report(tag, got_cond, got_eos, orc) -> bool:
    gc_ = torch.from_numpy(orc["step/cond"])
    ge = torch.from_numpy(orc["step/eos_logit"])
    cos = F.cosine_similarity(got_cond, gc_, dim=-1)
    d_eos = (got_eos - ge).abs().max()
    # the host applies `> -4.0`; a flipped sign decision is a hard failure
    flips = int(((got_eos > -4.0) != (ge > -4.0)).sum())
    ok = cos.min() > 0.999 and flips == 0
    print(
        f"[{tag}] steps {cos.shape[0]}  cond cos mean {cos.mean():.6f} min {cos.min():.6f}  "
        f"max|Δcond| {(got_cond - gc_).abs().max():.3e}  max|Δeos| {d_eos:.3e}  "
        f"eos-decision flips {flips}  -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="oracle/orc_a.npz")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    orc = np.load(ROOT / args.oracle)

    from pocket_tts import TTSModel

    model = TTSModel.load_model().eval()
    prefill = FlowLMPrefill(model.flow_lm).eval()
    step = FlowLMStep(model.flow_lm).eval()

    # ---- 1. eager gate (always fp32 — this checks the re-author, not the precision)
    c, e = replay_eager(prefill, step, orc, torch.float32)
    if not report("eager fp32", c, e, orc):
        print("❌ static-KV re-author DIVERGES — fix before export")
        raise SystemExit(1)
    print("✅ eager rewrite reproduces the flow-LM AR step")
    if args.skip_export:
        return

    # ---- 2. export: two entrypoints, one weight set, one shared state -------
    import asyncio

    import coreai.runtime as rt
    import coreai_torch
    from coreai_models.export.macos import _EXTERNALIZE_SPECS
    from coreai_models.export.mlir_ops import (
        register_custom_torch_lowering,
        remove_functionalization,
    )

    prefill, step = prefill.to(dtype), step.to(dtype)
    st_p, st_s = build_kv_state(dtype), build_kv_state(dtype)
    ref_p = {
        "text_emb": torch.zeros(1, T_PRE, 1024, dtype=dtype),
        "pos": torch.tensor([VOICE_LEN], dtype=torch.int32),
        **st_p,
    }
    ref_s = {
        "latent_in": torch.zeros(1, 1, LDIM, dtype=dtype),
        "is_bos": torch.zeros(1, dtype=dtype),
        "pos": torch.tensor([VOICE_LEN + T_PRE], dtype=torch.int32),
        **st_s,
    }
    # SDPA stays inlined: the externalized composite carries its own mask/window
    # attrs and we hand it an explicit bool mask over the full cache.
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "scaled_dot_product_attention"]

    def mk(ref):
        def export_fn(m):
            with torch.no_grad():
                ep = torch.export.export(m, args=(), kwargs=ref, dynamic_shapes=None)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            remove_functionalization(ep)
            return ep

        return export_fn

    print(f"[export] flow-LM prefill(T={T_PRE}) + step, S={S_MAX}, {args.dtype} ...", flush=True)
    conv = coreai_torch.TorchConverter()
    conv.add_pytorch_module(
        prefill,
        export_fn=mk(ref_p),
        externalize_modules=specs,
        input_names=("text_emb", "pos"),
        output_names=("cond",),
        state_names=STATE_NAMES,
        entrypoint_name="prefill",
    )
    conv.add_pytorch_module(
        step,
        export_fn=mk(ref_s),
        externalize_modules=specs,
        input_names=("latent_in", "is_bos", "pos"),
        output_names=("cond", "eos_logit"),
        state_names=STATE_NAMES,
        entrypoint_name="step",
    )
    register_custom_torch_lowering(conv)
    prog = conv.to_coreai()
    prog.optimize()

    art = ROOT / args.artifacts
    art.mkdir(exist_ok=True)
    path = art / f"flowlm_{args.dtype}_s{S_MAX}.aimodel"
    shutil.rmtree(path, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "cc-by-4.0"
    prog.save_asset(path, meta)
    sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {path.name} ({sz:.1f} MB)")

    # ---- 3. engine gate ----------------------------------------------------
    np_dtype = np.float16 if dtype is torch.float16 else np.float32

    async def gate(unit):
        opts = (
            rt.SpecializationOptions.cpu_only()
            if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(
                getattr(rt.ComputeUnitKind, unit)()
            )
        )
        m = await rt.AIModel.load(str(path), opts)
        print(f"  functions: {m.function_names}")
        fp, fs = m.load_function("prefill"), m.load_function("step")

        st = build_kv_state(dtype)
        pos = seed_voice_state(st, orc, dtype)
        state = {k: rt.NDArray(np.ascontiguousarray(v.numpy())) for k, v in st.items()}

        text = torch.from_numpy(orc["prefill/text_embeddings"]).to(dtype)
        t_real = text.shape[1]
        await fp(
            inputs={
                "text_emb": rt.NDArray(np.ascontiguousarray(pad_text(text).numpy())),
                "pos": rt.NDArray(np.ascontiguousarray(np.array([pos], np.int32))),
            },
            state=state,
        )
        pos += t_real

        gold_lat = torch.from_numpy(orc["step/latent"])
        conds, eoss = [], []
        for i in range(gold_lat.shape[0]):
            lat = (
                np.zeros((1, 1, LDIM), np_dtype)
                if i == 0
                else gold_lat[i - 1].reshape(1, 1, LDIM).numpy().astype(np_dtype)
            )
            r = await fs(
                inputs={
                    "latent_in": rt.NDArray(np.ascontiguousarray(lat)),
                    "is_bos": rt.NDArray(
                        np.ascontiguousarray(np.array([1.0 if i == 0 else 0.0], np_dtype))
                    ),
                    "pos": rt.NDArray(np.ascontiguousarray(np.array([pos + i], np.int32))),
                },
                state=state,
            )
            conds.append(torch.from_numpy(r["cond"].numpy().astype(np.float32)).reshape(-1))
            eoss.append(torch.from_numpy(r["eos_logit"].numpy().astype(np.float32)).reshape(-1))
        return report(f"gate {unit}", torch.stack(conds), torch.stack(eoss), orc)

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
