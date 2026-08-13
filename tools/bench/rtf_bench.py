"""Like-for-like TTS speed benchmark: RTF (generation wall / audio seconds) for the
same text, same voice, same M-series Mac, across every runnable TTS stack we have.

Stacks:
    coreai_fp32     our Core AI port, flow-LM + flow decoder at fp32, GPU compute unit
    coreai_fp16     our Core AI port, flow-LM + flow decoder at fp16, GPU compute unit
                    (Mimi decoder is fp32 in both -- see NOTES.md section 14)
    upstream_torch  upstream Kyutai `pocket_tts` PyTorch package. Tries MPS first,
                    falls back to CPU if MPS is unavailable or fails at runtime, and
                    records which device the timed run actually used.
    mlx             community `pocket-tts-mlx` package (MLX / Metal), if installed.
                    Its chunker is a reimplementation of upstream's, not the literal
                    same function the other three stacks call -- see the caveat this
                    script prints before the MLX rows.

This is a SPEED bench, not a parity bench: each stack draws noise from its own default
RNG, so the audio is not expected to match sample-for-sample across stacks (it does not
even match sample-for-sample across dtypes within a stack -- see NOTES.md section 13).
What's held fixed is the text, the voice, and the chunk-token cap (50, upstream's own
MAX_TOKEN_PER_CHUNK default).

Design notes:
  - Every timed run (warmup or scored) executes in its OWN fresh subprocess, using the
    interpreter from the venv that stack needs. This is deliberate, not incidental: the
    Python `coreai.runtime` bindings leak ~1.9 MB of IOSurface storage per inference
    call and the process dies around ~2,250 calls (NOTES.md section 16.8); one
    generation of this bench's ~150-word paragraph makes on the order of 1,500-2,000
    engine calls, so one generation per process is the safe granularity -- the same
    call-budgeted-worker idea `conversion/sweep_gen.py` uses for the validation sweep,
    just taken to a batch size of one. Running every stack this way (not only the
    Core AI ones) also gives a uniform, trivially-correct way to "exclude model load"
    from the timed number for every stack: each worker measures its own generation
    wall-clock strictly around the generate call and reports that separately from boot.
  - The parent process (this file, run directly) does no ML-framework imports at all --
    only argparse/subprocess/json/statistics -- so it does not need to run inside any
    particular venv. Worker mode (`--_worker ...`) is what imports torch/coreai/mlx,
    and the parent always launches worker subprocesses with the correct venv's
    interpreter explicitly.

Usage:
    # smoke: one short-sentence generation per stack, just to prove each path executes.
    # Numbers from --smoke are NOT the bench -- label them indicative-only.
    python3 tools/bench/rtf_bench.py --smoke

    # the real bench: 1 warmup + 3 timed runs per stack on the fixed paragraph.
    # Takes several minutes; do not run this casually on a machine in use.
    python3 tools/bench/rtf_bench.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

VENV_EXPORT = ROOT / ".venv-export" / "bin" / "python3"   # our Core AI port (coreai.runtime)
VENV_TORCH = ROOT / ".venv" / "bin" / "python3"            # upstream pocket_tts, oracle venv
VENV_MLX = ROOT / ".venv-mlx" / "bin" / "python3"           # community pocket-tts-mlx (may not exist)

VOICE = "alba"

# The bench text: the opening paragraph of Moby Dick chapter 1, the same passage the
# validation sweep's 10-minute long-form run is drawn from (artifacts/longform_passage.txt,
# NOTES.md section 16.5), trimmed to the first sentence-complete ~150 words. Recorded
# verbatim here so the bench is reproducible without the (gitignored) artifacts tree.
BENCH_TEXT = (
    "Call me Ishmael. Some years ago - never mind how long precisely - having little "
    "or no money in my purse, and nothing particular to interest me on shore, I "
    "thought I would sail about a little and see the watery part of the world. It is "
    "a way I have of driving off the spleen and regulating the circulation. Whenever "
    "I find myself growing grim about the mouth; whenever it is a damp, drizzly "
    "November in my soul; whenever I find myself involuntarily pausing before coffin "
    "warehouses, and bringing up the rear of every funeral I meet; and especially "
    "whenever my hypos get such an upper hand of me, that it requires a strong moral "
    "principle to prevent me from deliberately stepping into the street, and "
    "methodically knocking people's hats off - then, I account it high time to get "
    "to sea as soon as I can."
)  # 148 words

SMOKE_TEXT = "This is a short smoke test sentence."

SEED = 1234


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- workers
# Each of these runs ONLY inside its own worker subprocess (see module docstring), under
# the venv that has the package it needs. Each does exactly one generation: model/asset
# load ("boot"), then one call to generate audio, timed separately from boot.


def _worker_coreai(args) -> dict:
    sys.path.insert(0, str(ROOT / "conversion"))
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from pocket_tts import TTSModel

    from e2e_coreai import synth  # reuses the exact gated AR loop -- see NOTES.md section 13

    t_boot = time.monotonic()
    model = TTSModel.load_model().eval()
    art = ROOT / "artifacts"
    assets = {
        "flowlm": art / f"flowlm_{args.dtype}_s512.aimodel",
        "flow": art / f"flow_decoder_{args.dtype}_lsd1.aimodel",
        "mimi": art / "mimi_decoder_float32_ring272_outer.aimodel",  # always fp32
    }
    for k, p in assets.items():
        if not p.exists():
            raise SystemExit(f"missing asset {k}: {p}")
    vs = model.get_state_for_audio_prompt(args.voice)
    voice = {
        f"voice/{mod}/{kk}": vv.detach().cpu().numpy()
        for mod, st in vs.items()
        for kk, vv in st.items()
    }
    boot_s = time.monotonic() - t_boot  # NOT counting asset load -- see below

    text = SMOKE_TEXT if args.smoke else BENCH_TEXT
    import asyncio

    pcm, meta = asyncio.run(
        synth(model, text, voice, assets, args.seed, args.unit, args.dtype)
    )
    # `synth`'s own wall_s starts AFTER its internal AIModel.load calls, so it already
    # excludes asset load; boot_s above (torch model + voice state) undercounts total
    # boot by that asset-load time, which is why we report both separately rather than
    # summing them into one "boot" number we'd have to caveat every time.
    audio_s = pcm.shape[0] / model.sample_rate
    return dict(
        stack=f"coreai_{args.dtype}",
        device=f"gpu ({args.dtype})",
        boot_s=round(boot_s, 3),
        gen_wall_s=meta["wall_s"],
        audio_s=round(audio_s, 3),
        rtf=round(meta["wall_s"] / audio_s, 4) if audio_s else None,
        n_chunks=meta["n_chunks"],
        note="boot_s excludes .aimodel asset load (folded into gen path below it)",
    )


def _worker_upstream_torch(args) -> dict:
    import torch

    torch.set_num_threads(1)
    from pocket_tts import TTSModel

    t_boot = time.monotonic()
    model = TTSModel.load_model().eval()

    device_tried = "cpu"
    mps_note = "mps not available on this torch build"
    if torch.backends.mps.is_available():
        device_tried = "mps"
        model.to("mps")

    voice_state = model.get_state_for_audio_prompt(args.voice)
    boot_s = time.monotonic() - t_boot

    text = SMOKE_TEXT if args.smoke else BENCH_TEXT
    device_used = device_tried
    try:
        t_gen = time.monotonic()
        with torch.no_grad():
            audio = model.generate_audio(voice_state, text, copy_state=True)
        gen_wall_s = time.monotonic() - t_gen
    except Exception as e:  # MPS op unsupported etc. -- fall back to CPU and redo
        if device_tried != "mps":
            raise
        mps_note = f"mps failed at runtime, fell back to cpu: {type(e).__name__}: {e}"[:300]
        device_used = "cpu (mps fallback)"
        model.to("cpu")
        voice_state = model.get_state_for_audio_prompt(args.voice)
        t_gen = time.monotonic()
        with torch.no_grad():
            audio = model.generate_audio(voice_state, text, copy_state=True)
        gen_wall_s = time.monotonic() - t_gen

    audio_np = audio.detach().to("cpu").numpy()
    audio_s = audio_np.shape[-1] / model.sample_rate
    return dict(
        stack="upstream_torch",
        device=device_used,
        boot_s=round(boot_s, 3),
        gen_wall_s=round(gen_wall_s, 3),
        audio_s=round(audio_s, 3),
        rtf=round(gen_wall_s / audio_s, 4) if audio_s else None,
        note=mps_note if device_tried != device_used or device_tried == "cpu" else "",
    )


def _worker_mlx(args) -> dict:
    import numpy as np

    from pocket_tts_mlx import TTSModel

    t_boot = time.monotonic()
    model = TTSModel.load_model()
    state = model.get_state_for_audio_prompt(args.voice)
    boot_s = time.monotonic() - t_boot

    text = SMOKE_TEXT if args.smoke else BENCH_TEXT
    t_gen = time.monotonic()
    audio = model.generate_audio(state, text)  # defaults: max_tokens=50 (matches upstream)
    gen_wall_s = time.monotonic() - t_gen

    arr = np.array(audio)
    audio_s = arr.shape[-1] / model.sample_rate
    return dict(
        stack="mlx",
        device="gpu (mlx/metal, default device)",
        boot_s=round(boot_s, 3),
        gen_wall_s=round(gen_wall_s, 3),
        audio_s=round(audio_s, 3),
        rtf=round(gen_wall_s / audio_s, 4) if audio_s else None,
        note="community pocket-tts-mlx package; own chunker reimplementation, not upstream's",
    )


_WORKERS = {
    "coreai": _worker_coreai,
    "upstream_torch": _worker_upstream_torch,
    "mlx": _worker_mlx,
}


def run_worker_mode(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--_worker", required=True, choices=list(_WORKERS))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--unit", default="gpu", choices=["cpu", "gpu", "neural_engine"])
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        result = _WORKERS[args._worker](args)
        result["smoke"] = args.smoke
        result["error"] = None
    except Exception as e:  # a failed run is a row, not a crashed bench
        result = {"stack": args._worker, "error": f"{type(e).__name__}: {e}"[:500]}
    Path(args.out).write_text(json.dumps(result, indent=2))


# --------------------------------------------------------------------------- driver


class Stack:
    def __init__(self, name: str, label: str, python: Path, worker: str, dtype: str = "float32"):
        self.name = name
        self.label = label
        self.python = python
        self.worker = worker
        self.dtype = dtype

    def available(self) -> tuple[bool, str]:
        if not self.python.exists():
            return False, f"venv interpreter missing: {self.python}"
        return True, ""


def stacks_for_bench() -> list[Stack]:
    stacks = [
        Stack("coreai_fp32", "Core AI port -- fp32 GPU", VENV_EXPORT, "coreai", "float32"),
        Stack("coreai_fp16", "Core AI port -- fp16 GPU", VENV_EXPORT, "coreai", "float16"),
        Stack("upstream_torch", "upstream pocket_tts (PyTorch)", VENV_TORCH, "upstream_torch"),
        Stack("mlx", "community pocket-tts-mlx", VENV_MLX, "mlx"),
    ]
    return stacks


def run_one(stack: Stack, out_dir: Path, smoke: bool, run_idx: int) -> dict:
    out_json = out_dir / f"{stack.name}_run{run_idx}.json"
    cmd = [
        str(stack.python),
        __file__,
        "--_worker",
        stack.worker,
        "--dtype",
        stack.dtype,
        "--voice",
        VOICE,
        "--seed",
        str(SEED),
        "--out",
        str(out_json),
    ]
    if smoke:
        cmd.append("--smoke")
    env = {"HF_HOME": str(ROOT / "weights" / "hf")}
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=full_env, capture_output=True, text=True)
    wall_process = time.monotonic() - t0
    if not out_json.exists():
        return {
            "stack": stack.name,
            "error": f"worker produced no output (exit {proc.returncode}); "
            f"stderr tail: {proc.stderr[-800:]}",
        }
    result = json.loads(out_json.read_text())
    result["wall_process_s"] = round(wall_process, 3)
    return result


def fmt_row(r: dict) -> str:
    if r.get("error"):
        return f"| {r['stack']} | ERROR | - | - | - | {r['error'][:120]} |"
    return (
        f"| {r['stack']} | {r.get('device', '-')} | {r['gen_wall_s']:.3f} | "
        f"{r['audio_s']:.3f} | {r['rtf']:.3f} | {r.get('note', '')} |"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 short-sentence run per stack, no warmup/median -- indicative only",
    )
    ap.add_argument("--runs", type=int, default=3, help="timed runs per stack (default 3)")
    args = ap.parse_args()

    out_dir = ROOT / "tools" / "bench" / "results" / f"_raw_{_now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stacks = stacks_for_bench()
    all_results: dict[str, list[dict]] = {}

    for stack in stacks:
        ok, why = stack.available()
        if not ok:
            print(f"[skip] {stack.name}: {why}")
            all_results[stack.name] = [{"stack": stack.name, "error": why}]
            continue

        if args.smoke:
            print(f"[{stack.name}] smoke run...")
            r = run_one(stack, out_dir, smoke=True, run_idx=0)
            r["label"] = stack.label
            all_results[stack.name] = [r]
            status = "ERROR" if r.get("error") else f"rtf={r.get('rtf')}"
            print(f"  -> {status}")
            continue

        print(f"[{stack.name}] warmup run...")
        warm = run_one(stack, out_dir, smoke=False, run_idx=0)
        if warm.get("error"):
            print(f"  -> warmup ERROR: {warm['error'][:200]}")
            all_results[stack.name] = [warm]
            continue
        print(f"  -> warmup rtf={warm.get('rtf')} (discarded)")

        timed = []
        for i in range(1, args.runs + 1):
            print(f"[{stack.name}] timed run {i}/{args.runs}...")
            r = run_one(stack, out_dir, smoke=False, run_idx=i)
            r["label"] = stack.label
            timed.append(r)
            if r.get("error"):
                print(f"  -> ERROR: {r['error'][:200]}")
            else:
                print(f"  -> wall={r['gen_wall_s']:.3f}s audio={r['audio_s']:.3f}s rtf={r['rtf']:.4f}")
        all_results[stack.name] = timed

    # ---- report
    tag = _now_tag()
    lines = []
    lines.append("# TTS RTF bench" + (" (SMOKE -- indicative only, not a real bench)" if args.smoke else ""))
    lines.append("")
    lines.append(f"- timestamp (UTC): {tag}")
    lines.append(f"- voice: {VOICE}, seed: {SEED} (per-stack default RNG; audio is not expected to match across stacks)")
    lines.append(f"- text: {'SMOKE_TEXT' if args.smoke else f'{len(BENCH_TEXT.split())}-word paragraph'} (recorded in this script)")
    lines.append("")

    if args.smoke:
        lines.append("| stack | device | gen wall (s) | audio (s) | RTF | note |")
        lines.append("|---|---|---:|---:|---:|---|")
        for stack in stacks:
            r = all_results[stack.name][0]
            lines.append(fmt_row(r))
    else:
        lines.append("| stack | device | median gen wall (s) | median audio (s) | median RTF | runs |")
        lines.append("|---|---|---:|---:|---:|---|")
        for stack in stacks:
            runs = [r for r in all_results[stack.name] if not r.get("error")]
            if not runs:
                err = all_results[stack.name][0].get("error", "unknown error")
                lines.append(f"| {stack.name} | ERROR | - | - | - | {err[:150]} |")
                continue
            med_wall = statistics.median(r["gen_wall_s"] for r in runs)
            med_audio = statistics.median(r["audio_s"] for r in runs)
            med_rtf = statistics.median(r["rtf"] for r in runs)
            lines.append(
                f"| {stack.name} | {runs[0]['device']} | {med_wall:.3f} | {med_audio:.3f} | "
                f"{med_rtf:.4f} | {len(runs)}/{args.runs} |"
            )
        lines.append("")
        lines.append("## per-run detail")
        lines.append("")
        lines.append("| stack | run | gen wall (s) | audio (s) | RTF |")
        lines.append("|---|---:|---:|---:|---:|")
        for stack in stacks:
            for i, r in enumerate(all_results[stack.name], start=1):
                if r.get("error"):
                    lines.append(f"| {stack.name} | {i} | ERROR | - | - |")
                else:
                    lines.append(
                        f"| {stack.name} | {i} | {r['gen_wall_s']:.3f} | {r['audio_s']:.3f} | {r['rtf']:.4f} |"
                    )

    lines.append("")
    lines.append(
        "Note: this is a speed bench, not a parity bench -- cross-stack audio need not "
        "match bit-wise. MLX's chunker is a community reimplementation of upstream's, "
        "not the literal same function the other three stacks call."
    )

    report = "\n".join(lines)
    print()
    print(report)

    results_dir = ROOT / "tools" / "bench" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_md = results_dir / f"{tag}.md"
    out_md.write_text(report + "\n")
    print(f"\n[written] {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        # worker mode: dispatched before the driver's own argparse so the two argument
        # sets (driver flags vs worker flags) never collide.
        run_worker_mode(sys.argv[1:])
    else:
        main()
