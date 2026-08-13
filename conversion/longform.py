"""Part 2 of the sweep: a ~10-minute long-form generation, the production shape.

The target is long-form narration of roughly ten minutes, so this is the max-length
test rather than a book-length one. A contiguous Moby Dick chapter-1 passage
is run through upstream's OWN long-text handling — `split_into_best_sentences` at
MAX_TOKEN_PER_CHUNK, a KV cache re-seeded from the voice state per chunk, and a Mimi
streaming state that runs continuously across every chunk — and the chunk audio is
**concatenated directly with no inter-sentence silence**, which is exactly what
`TTSModel.generate_audio` does (`torch.cat(audio_chunks, dim=0)`). Inserting silence
would be a nicer-sounding deviation, and it would also hide any boundary artifact this
run is meant to look for.

The IOSurface leak in the Python bindings makes ~2.2k inference calls fatal, and ten
minutes of audio is ~24k calls, so generation is checkpointed: each chunk's PCM is
written to its own .npy and the 12-tensor Mimi state is saved after every chunk, so a
respawned worker resumes bit-exactly where the dead one stopped.

    python conversion/longform.py --prepare        # cut the passage
    python conversion/longform.py                  # driver: generate + stitch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from export_mimi_decoder import STATE_NAMES as MIMI_STATE  # noqa: E402
from sweep_gen import Engine, _c, asset_paths, build_engine_sync, rss_mb  # noqa: E402

# External corpus; same override as conversion/sweep_corpus.py.
MOBY = Path(os.environ.get("CORPUS_ROOT", "corpus")) / "work/moby_dick_gutenberg.txt"


def prepare_passage(target_words: int, out: Path) -> str:
    raw = MOBY.read_text(encoding="utf-8", errors="replace")
    start = raw.find("Call me Ishmael.")
    end = raw.find("CHAPTER 2.")
    body = raw[start : end if end > start else start + 200_000]
    body = re.sub(r"\s+", " ", body)
    body = body.replace("’", "'").replace("‘", "'")
    body = body.replace("“", '"').replace("”", '"')
    body = body.replace("—", " - ").replace("–", "-")
    # cut on a sentence boundary at/after the word target so the reference is a whole
    # number of sentences (the chunker splits on sentence punctuation anyway)
    sents = re.split(r"(?<=[.!?])\s+", body)
    acc, n = [], 0
    for s in sents:
        acc.append(s.strip())
        n += len(s.split())
        if n >= target_words:
            break
    text = " ".join(acc).strip()
    out.write_text(text)
    print(f"[passage] {len(text.split())} words, {len(acc)} sentences -> {out}")
    return text


# --------------------------------------------------------------------------- worker


async def run_worker(args):
    t_boot = time.monotonic()
    text = (ROOT / args.passage).read_text()
    work = ROOT / args.work
    work.mkdir(parents=True, exist_ok=True)
    eng = build_engine_sync(asset_paths(ROOT / args.artifacts, args.dtype), args.unit, args.dtype)
    await eng.load()
    boot = time.monotonic() - t_boot

    from pocket_tts.models.tts_model import MAX_TOKEN_PER_CHUNK, split_into_best_sentences

    chunks = split_into_best_sentences(
        eng.model.flow_lm.conditioner.tokenizer,
        text,
        MAX_TOKEN_PER_CHUNK,
        eng.model.pad_with_spaces_for_short_inputs,
        eng.model.remove_semicolons,
    )
    (work / "chunks.json").write_text(json.dumps(chunks))

    done = sorted(work.glob("c*.npy"))
    i0 = len(done)
    state_p = work / "mimi_state.npz"
    carry = dict(np.load(state_p)) if (i0 and state_p.exists()) else None
    meta_f = (work / "chunks.jsonl").open("a")
    gen_wall = 0.0
    for i in range(i0, len(chunks)):
        if eng.calls > args.call_budget:
            break
        t0 = time.monotonic()
        pcm, meta = await eng.synth_one(chunks[i], args.voice, args.seed, carry)
        gen_wall += time.monotonic() - t0
        carry = eng.last_mimi_state
        np.save(work / f"c{i:04d}.npy", pcm)
        np.savez(state_p, **carry)
        meta_f.write(
            json.dumps({"i": i, "text": chunks[i], "samples": int(pcm.size), **meta}) + "\n"
        )
        meta_f.flush()
    meta_f.close()
    (work / f"worker_{i0:04d}.json").write_text(
        json.dumps(
            {
                "from": i0,
                "to": len(sorted(work.glob("c*.npy"))),
                "boot_s": round(boot, 2),
                "gen_wall_s": round(gen_wall, 2),
                "calls": eng.calls,
                "rss_mb": round(rss_mb(), 1),
            }
        )
    )
    print(
        f"[worker] chunks {i0}..{len(sorted(work.glob('c*.npy')))} of {len(chunks)}, "
        f"{eng.calls} calls, gen {gen_wall:.1f}s, peak RSS {rss_mb():.0f} MB",
        flush=True,
    )


# --------------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    # 2183 words -> 626 s. The checkpoint reads at ~209 wpm, well above the
    # 150-160 wpm of audiobook narration, so ~1600 words only reaches 7.6 minutes.
    ap.add_argument("--target-words", type=int, default=2150)
    ap.add_argument("--passage", default="artifacts/longform_passage.txt")
    ap.add_argument("--work", default="artifacts/longform_work")
    ap.add_argument("--out", default="artifacts/longform_moby_10min.wav")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--unit", default="gpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--call-budget", type=int, default=1500)
    args = ap.parse_args()

    if args.prepare:
        prepare_passage(args.target_words, ROOT / args.passage)
        return
    if args.worker:
        asyncio.run(run_worker(args))
        return

    work = ROOT / args.work
    t0 = time.monotonic()
    n_workers = 0
    while True:
        chunks_p = work / "chunks.json"
        n_total = len(json.loads(chunks_p.read_text())) if chunks_p.exists() else None
        n_done = len(sorted(work.glob("c*.npy"))) if work.exists() else 0
        if n_total is not None and n_done >= n_total:
            break
        n_workers += 1
        print(
            f"[driver] worker {n_workers}: {n_done}/{n_total} chunks, "
            f"{time.monotonic() - t0:.0f}s elapsed",
            flush=True,
        )
        r = subprocess.run(
            [sys.executable, str(HERE / "longform.py"), "--worker"]
            + [f"--{k.replace('_', '-')}={v}" for k, v in vars(args).items()
               if k not in ("worker", "prepare") and not isinstance(v, bool)],
            env=dict(os.environ),
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print(f"[driver]   worker exited {r.returncode} (respawning)", flush=True)
        if n_workers > 200:
            raise SystemExit("[driver] too many respawns — aborting")
    wall = time.monotonic() - t0

    chunks = json.loads((work / "chunks.json").read_text())
    parts = [np.load(work / f"c{i:04d}.npy") for i in range(len(chunks))]
    # direct concat, no inter-sentence silence — upstream's `torch.cat(audio_chunks)`
    pcm = np.concatenate(parts)
    bounds = np.cumsum([p.size for p in parts])[:-1]
    out = ROOT / args.out
    scipy.io.wavfile.write(out, 24000, pcm.astype(np.float32))
    stats = [json.loads(l) for l in (work / "worker_meta.jsonl").read_text().splitlines()] if (
        work / "worker_meta.jsonl"
    ).exists() else []
    wm = [json.loads(p.read_text()) for p in sorted(work.glob("worker_*.json"))]
    summary = {
        "chunks": len(chunks),
        "duration_s": round(pcm.size / 24000, 2),
        "driver_wall_s": round(wall, 1),
        "generation_wall_s": round(sum(w["gen_wall_s"] for w in wm), 1),
        "boot_wall_s": round(sum(w["boot_s"] for w in wm), 1),
        "workers": len(wm),
        "engine_calls": sum(w["calls"] for w in wm),
        "peak_rss_mb": max(w["rss_mb"] for w in wm),
        "boundaries": [int(b) for b in bounds],
        "peak": round(float(np.abs(pcm).max()), 4),
        "rms": round(float(np.sqrt((pcm**2).mean())), 5),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "boundaries"}, indent=2))
    print(f"  -> {out}")
    _ = stats


if __name__ == "__main__":
    main()
