"""Flow decoder (graph d) -> Core AI, gated on the exact oracle fixture.

    cond [1,1024] + noise [1,32]  ->  latent [1,32]

Pure function, no state. `s`/`t` are baked at the upstream default
`lsd_decode_steps = 1` (`pocket_tts/default_parameters.py:3`), i.e. one Euler step
from s=0 to t=1. The 8-step variant FluidAudio ships exists and would need `s`/`t`
as inputs (or an unrolled fused graph) — not chased here.

`oracle['step/noise'] -> oracle['step/latent']` at a fixed `oracle['step/cond']` is
an exact fixture: the oracle clones the noise before `lsd_decode` mutates it in place.

Run (export venv):
    HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
      .venv-export/bin/python conversion/export_flow_decoder.py
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

from flowlm_graphs import FlowDecoder  # noqa: E402


def report(tag, got, gold) -> bool:
    cos = F.cosine_similarity(got, gold, dim=-1)
    ok = cos.min() > 0.999
    print(
        f"[{tag}] steps {cos.shape[0]}  cos mean {cos.mean():.6f} min {cos.min():.6f}  "
        f"max|Δ| {(got - gold).abs().max():.3e}  -> {'PASS' if ok else 'FAIL'}"
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
    cond = torch.from_numpy(orc["step/cond"])       # [N,1024]
    noise = torch.from_numpy(orc["step/noise"])     # [N,32]
    gold = torch.from_numpy(orc["step/latent"])     # [N,32]
    n = cond.shape[0]

    from pocket_tts import TTSModel

    model = TTSModel.load_model().eval()
    dec = FlowDecoder(model.flow_lm).eval()

    with torch.no_grad():
        eager = torch.cat([dec(cond[i : i + 1], noise[i : i + 1]) for i in range(n)])
    if not report("eager fp32", eager, gold):
        print("❌ flow-decoder re-author DIVERGES — fix before export")
        raise SystemExit(1)
    print("✅ eager rewrite reproduces lsd_decode(num_steps=1)")
    if args.skip_export:
        return

    import asyncio

    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    dec = dec.to(dtype)
    example = {
        "cond": torch.zeros(1, 1024, dtype=dtype),
        "noise": torch.zeros(1, 32, dtype=dtype),
    }
    print(f"[export] flow decoder ({args.dtype}, 1 LSD step) -> Core AI ...", flush=True)
    prog = export_to_coreai(
        dec,
        example,
        dynamic_shapes=None,
        input_names=("cond", "noise"),
        output_names=("latent",),
        state_names=None,
        externalize_modules=[],
    )
    prog.optimize()
    art = ROOT / args.artifacts
    art.mkdir(exist_ok=True)
    path = art / f"flow_decoder_{args.dtype}_lsd1.aimodel"
    shutil.rmtree(path, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "cc-by-4.0"
    prog.save_asset(path, meta)
    sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {path.name} ({sz:.1f} MB)")

    np_dtype = np.float16 if dtype is torch.float16 else np.float32

    async def gate(unit):
        opts = (
            rt.SpecializationOptions.cpu_only()
            if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(
                getattr(rt.ComputeUnitKind, unit)()
            )
        )
        fn = (await rt.AIModel.load(str(path), opts)).load_function("main")
        outs = []
        for i in range(n):
            r = await fn(
                {
                    "cond": rt.NDArray(np.ascontiguousarray(cond[i : i + 1].numpy().astype(np_dtype))),
                    "noise": rt.NDArray(
                        np.ascontiguousarray(noise[i : i + 1].numpy().astype(np_dtype))
                    ),
                }
            )
            outs.append(torch.from_numpy(r["latent"].numpy().astype(np.float32)).reshape(-1))
        return report(f"gate {unit}", torch.stack(outs), gold)

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
