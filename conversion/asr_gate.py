"""ASR round-trip acceptance gate for the Core AI TTS pipeline.

Tensor-similarity gates pass while audio is unintelligible (FluidAudio's hard lesson,
NOTES.md section 9), so the end-to-end gate is: synthesize, transcribe with the local
Parakeet host, and compare the transcript to the prompt.

Resamples 24 kHz float32 -> 16 kHz mono PCM16 (what `parakeet-swift transcribe` takes),
runs it, and reports WER.

    .venv-export/bin/python conversion/asr_gate.py artifacts/e2e_orc_gpu.wav \
        --text "The quick brown fox jumps over the lazy dog."
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal

# The ASR half of the gate is an external dependency: a built `parakeet-swift` binary
# and the Core AI artifacts it loads. Override both to match your own checkout.
#   PARAKEET_BIN        path to the parakeet-swift executable (default: found on PATH)
#   PARAKEET_ARTIFACTS  directory holding its Core AI artifacts
PARAKEET = Path(os.environ.get("PARAKEET_BIN", "parakeet-swift"))
ARTIFACTS = Path(os.environ.get("PARAKEET_ARTIFACTS", "artifacts_v2"))


def norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> tuple[float, int, int]:
    r, h = norm(ref), norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(
                d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1])
            )
    return (d[-1, -1] / max(len(r), 1)), int(d[-1, -1]), len(r)


def to_16k_pcm16(path: Path) -> Path:
    sr, x = scipy.io.wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 1.5:  # already integer-scaled
        x = x / 32768.0
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = scipy.signal.resample_poly(x, 16000 // g, sr // g)
    out = Path(tempfile.mkdtemp()) / (path.stem + "_16k.wav")
    scipy.io.wavfile.write(out, 16000, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
    return out


def transcribe(path: Path) -> str:
    env = dict(os.environ, PARAKEET_ARTIFACTS=str(ARTIFACTS))
    r = subprocess.run(
        [str(PARAKEET), "transcribe", str(path)], capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        raise SystemExit(f"parakeet-swift failed ({r.returncode}):\n{r.stderr[-2000:]}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--text", required=True, help="the prompt the wav was synthesized from")
    a = ap.parse_args()
    for w in a.wav:
        p = Path(w)
        t = transcribe(to_16k_pcm16(p))
        rate, errs, n = wer(a.text, t)
        print(f"[asr] {p.name}")
        print(f"      ref: {a.text}")
        print(f"      hyp: {t}")
        print(
            f"      WER {rate * 100:.2f}%  ({errs}/{n} words)  "
            f"-> {'PASS' if rate < 0.10 else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
