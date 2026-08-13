"""Gate the long-form run: full-passage WER, WER drift by position, stitch artifacts.

Three questions, because a ten-minute generation can fail in ways a 2.5-second one
cannot:

  intelligibility  ASR round-trip of the whole passage against the Gutenberg source.
  drift            the same WER computed per position decile. A model whose Mimi
                   streaming state accumulates error, or whose voice conditioning
                   decays, degrades monotonically; a flat curve says the state
                   feedback is stable over the production length.
  stitching        chunk audio is concatenated with no crossfade and no inter-sentence
                   silence (upstream's own `torch.cat`), so every boundary is a
                   potential click. Measured as the sample-to-sample step across the
                   join, expressed as a percentile of the same statistic inside the
                   audio, plus the level jump in dB across a 20 ms window either side.

    python conversion/longform_score.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from sweep_score import PARAKEET, PK_ARTIFACTS, to_16k, wer_of  # noqa: E402

SR = 24000


def transcribe(x: np.ndarray, tmp: Path, name: str) -> str:
    import os

    w = tmp / f"{name}.wav"
    scipy.io.wavfile.write(w, SR, x.astype(np.float32))
    w16 = tmp / f"{name}_16k.wav"
    to_16k(w, w16)
    env = dict(os.environ, PARAKEET_ARTIFACTS=str(PK_ARTIFACTS))
    r = subprocess.run(
        [str(PARAKEET), "transcribe", str(w16)], capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        raise SystemExit(f"parakeet failed: {r.stderr[-800:]}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="artifacts/longform_moby_10min.wav")
    ap.add_argument("--work", default="artifacts/longform_work")
    ap.add_argument("--passage", default="artifacts/longform_passage.txt")
    ap.add_argument("--deciles", type=int, default=10)
    a = ap.parse_args()

    work = ROOT / a.work
    chunks = json.loads((work / "chunks.json").read_text())
    summary = json.loads((work / "summary.json").read_text())
    sr, x = scipy.io.wavfile.read(ROOT / a.wav)
    x = x.astype(np.float32)
    passage = (ROOT / a.passage).read_text()
    bounds = summary["boundaries"]
    tmp = Path(tempfile.mkdtemp())

    print(f"[longform] {x.size / sr:.1f}s, {len(chunks)} chunks, {len(bounds)} stitch points")

    # --- 1. whole-passage round trip
    hyp = transcribe(x, tmp, "full")
    w, errs, n = wer_of(passage, hyp)
    print(f"[full] WER {100 * w:.2f}%  ({errs}/{n} words)")

    # --- 2. drift: contiguous groups of chunks, each scored against its own source text
    starts = [0] + list(bounds)
    ends = list(bounds) + [x.size]
    k = a.deciles
    per = max(1, len(chunks) // k)
    print("[drift] WER by position decile")
    dec = []
    for d in range(k):
        i0, i1 = d * per, (d + 1) * per if d < k - 1 else len(chunks)
        if i0 >= len(chunks):
            break
        seg = x[starts[i0] : ends[i1 - 1]]
        ref = " ".join(chunks[i0:i1])
        h = transcribe(seg, tmp, f"d{d}")
        dw, de, dn = wer_of(ref, h)
        dec.append({"decile": d, "chunks": [i0, i1], "seconds": round(seg.size / sr, 1),
                    "wer": round(float(dw), 4), "errs": de, "words": dn})
        print(f"    d{d}  {seg.size / sr:6.1f}s  chunks {i0:3d}-{i1 - 1:3d}  "
              f"WER {100 * dw:6.2f}%  ({de}/{dn})")

    # --- 3. stitch artifacts
    diffs = np.abs(np.diff(x))
    p999 = float(np.percentile(diffs, 99.9))
    pmax = float(diffs.max())
    win = int(0.020 * sr)
    steps, jumps = [], []
    for b in bounds:
        steps.append(float(abs(x[b] - x[b - 1])))
        pre = x[max(0, b - win) : b]
        post = x[b : b + win]
        rp = float(np.sqrt((pre**2).mean()) + 1e-9)
        rq = float(np.sqrt((post**2).mean()) + 1e-9)
        jumps.append(20 * np.log10(rq / rp))
    steps = np.array(steps)
    jumps = np.array(jumps)
    over = int((steps > p999).sum())
    print(
        f"[stitch] {len(bounds)} joins  max step {steps.max():.4f}  mean {steps.mean():.5f}\n"
        f"         in-audio |Δsample| p99.9 {p999:.4f}, max {pmax:.4f}  "
        f"-> {over} join(s) above the in-audio p99.9\n"
        f"         level jump across join: median {np.median(np.abs(jumps)):.1f} dB, "
        f"max {np.abs(jumps).max():.1f} dB"
    )

    out = work / "score.json"
    out.write_text(
        json.dumps(
            {
                "duration_s": round(x.size / sr, 2),
                "full_wer": round(float(w), 4),
                "full_errs": errs,
                "full_words": n,
                "deciles": dec,
                "stitch": {
                    "joins": len(bounds),
                    "max_step": round(float(steps.max()), 5),
                    "mean_step": round(float(steps.mean()), 6),
                    "in_audio_p999_step": round(p999, 5),
                    "in_audio_max_step": round(pmax, 5),
                    "joins_above_p999": over,
                    "level_jump_db_median": round(float(np.median(np.abs(jumps))), 2),
                    "level_jump_db_max": round(float(np.abs(jumps).max()), 2),
                },
                **{k2: summary[k2] for k2 in summary if k2 != "boundaries"},
            },
            indent=2,
        )
    )
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
