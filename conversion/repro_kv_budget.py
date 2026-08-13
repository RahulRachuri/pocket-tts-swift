"""Minimal repro: the static KV capacity silently truncates long sentences.

The flow-LM has **no context window**, so every generated frame occupies a cache slot
for the whole chunk and nothing ever rolls out. The Phase B asset was exported at
S_MAX = 256, a number sized off the oracle fixture (126 voice + 14 text + 32 generated).
Real sentences generate far more than 32 frames: this checkpoint produces roughly 2.2
frames per text token, so the capacity is exhausted at about

    (S_MAX - voice_len) / 3.2   tokens

which is ~40 tokens for a 126-position voice and ~29 for `azelma` (162 positions) —
both well under upstream's own MAX_TOKEN_PER_CHUNK of 50. Past that the utterance is
cut off mid-word with no EOS, and nothing in a per-graph cosine gate can see it: the
step graph is exactly as accurate on step 90 as on step 1.

Runs one sentence through the S=256 and S=512 assets and prints frames / EOS / duration
for each. Both assets pass the identical oracle gate.

    HF_HOME=$PWD/weights/hf DEVELOPER_DIR=... \
      .venv-export/bin/python conversion/repro_kv_budget.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

TEXT = (
    "And they stuck to it till they did gain it; when instantly, a swift tremor "
    "ran through the whole fabric of the ship, and every man aboard knew that the "
    "long chase was over at last."
)
VOICE = "javert"


async def run(s_max: int, text: str, voice: str):
    sys.path.insert(0, str(HERE))
    from sweep_gen import Engine, asset_paths  # noqa: E402  (import AFTER the env var)

    from pocket_tts import TTSModel
    import torch

    torch.set_num_threads(1)
    model = TTSModel.load_model().eval()
    eng = Engine(model, asset_paths(ROOT / "artifacts", "float32"), "gpu", "float32")
    await eng.load()
    pcm, meta = await eng.synth(text, voice, 1234)
    print(f"S_MAX={s_max}  duration {pcm.size / 24000:6.2f}s  chunks {meta['n_chunks']}")
    for c in meta["chunks"]:
        print(
            f"    tokens {c['tokens']:3d}  voice_len {c['voice_len']:3d}  "
            f"steps {c['steps']:3d}  eos {str(c['eos']):>5}  "
            f"kv_high_water {c['kv_high_water']:3d}  capped {c['kv_capped']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-max", type=int, default=None, help="internal: one value per exec")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--voice", default=VOICE)
    a = ap.parse_args()

    if a.s_max is not None:
        asyncio.run(run(a.s_max, a.text, a.voice))
        return
    # S_MAX is read at import time in `flowlm_graphs`, so each value needs its own
    # process. (It also selects which .aimodel is loaded — the name carries S.)
    for s in (256, 512):
        subprocess.run(
            [sys.executable, __file__, "--s-max", str(s), "--text", a.text, "--voice", a.voice],
            env=dict(os.environ, POCKET_TTS_S_MAX=str(s)),
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
