"""Batch generation for the validation sweep, in restartable worker subprocesses.

The Python `coreai.runtime` bindings leak an output IOSurface per inference call and
SIGTRAP somewhere near ~8k calls in a process. This pipeline makes roughly 40 calls per
second of generated audio (one flow-LM step + one flow-decoder + one Mimi frame per
12.5 Hz latent, plus prefill), so a single process cannot generate more than a couple of
minutes of speech. The driver therefore runs generation in **worker subprocesses with an
explicit call budget** and respawns as often as needed; progress is a jsonl the driver
re-reads, so a worker that dies takes at most one clip with it.

Two modes in one file:

    # driver — keeps spawning workers until every corpus row has a row in --meta
    python conversion/sweep_gen.py --corpus artifacts/sweep_corpus.jsonl \
        --out-dir artifacts/sweep_fp32 --meta artifacts/sweep_fp32_gen.jsonl

    # worker — one batch, then exit (the driver's own subprocess call)
    python conversion/sweep_gen.py --worker ...

Everything host-side mirrors `conversion/e2e_coreai.py`, which is the gated Phase B
path: upstream's own chunker, a fresh KV cache per chunk re-seeded from the voice
state, and a Mimi streaming state that is NEVER reset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from export_flowlm import pad_text, seed_voice_state  # noqa: E402
from export_mimi_decoder import STATE_NAMES as MIMI_STATE, init_state as mimi_init_state  # noqa: E402
from flowlm_graphs import LDIM, S_MAX, T_PRE, build_kv_state  # noqa: E402


def _c(x):
    return np.ascontiguousarray(x)


class Engine:
    """Loaded once per worker process: the torch front end (tokenizer, conditioner,
    quantizer, latent rescale) plus the three Core AI assets."""

    def __init__(self, model, assets: dict, unit: str, dtype: str):
        import coreai.runtime as rt

        self.rt = rt
        self.model = model
        self.dtype = dtype
        self.npd = np.float16 if dtype == "float16" else np.float32
        self.td = torch.float16 if dtype == "float16" else torch.float32
        self.calls = 0
        self.voices: dict[str, dict] = {}
        opts = (
            rt.SpecializationOptions.cpu_only()
            if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(
                getattr(rt.ComputeUnitKind, unit)()
            )
        )
        self.opts = opts
        self.emb_std = model.flow_lm.emb_std.to(torch.float32)
        self.emb_mean = model.flow_lm.emb_mean.to(torch.float32)
        self.std = model.temp**0.5
        self.assets = assets

    async def load(self):
        rt = self.rt
        m_lm = await rt.AIModel.load(str(self.assets["flowlm"]), self.opts)
        self.f_pre, self.f_step = m_lm.load_function("prefill"), m_lm.load_function("step")
        self.f_flow = (await rt.AIModel.load(str(self.assets["flow"]), self.opts)).load_function(
            "main"
        )
        self.f_mimi = (await rt.AIModel.load(str(self.assets["mimi"]), self.opts)).load_function(
            "main"
        )

    def voice_state(self, name: str) -> dict:
        if name not in self.voices:
            vs = self.model.get_state_for_audio_prompt(name)
            self.voices[name] = {
                f"voice/{mod}/{k}": v.detach().cpu().numpy()
                for mod, st in vs.items()
                for k, v in st.items()
            }
        return self.voices[name]

    async def _gen_chunk(self, chunk: str, voice: str, seed: int, draw0: int, mimi_state):
        """One <=MAX_TOKEN_PER_CHUNK chunk: fresh KV cache seeded from the voice state,
        windowed prefill, then the AR loop. `mimi_state` is threaded IN and OUT because
        the Mimi streaming state is never reset — not between chunks, and not across a
        worker restart in the long-form run."""
        rt, model = self.rt, self.model
        from pocket_tts.models.tts_model import prepare_text_prompt

        _, frames_guess = prepare_text_prompt(
            chunk, model.pad_with_spaces_for_short_inputs, model.remove_semicolons
        )
        fae = model.model_recommended_frames_after_eos
        if fae is None:
            fae = frames_guess + 2
        prepared = model.flow_lm.conditioner.prepare(chunk)
        with torch.no_grad():
            text_emb = model.flow_lm.conditioner(prepared).to(torch.float32)
        n_text = text_emb.shape[1]
        max_gen_len = model._estimate_max_gen_len(n_text)

        kv = build_kv_state(self.td)
        pos0 = seed_voice_state(kv, self.voice_state(voice), self.td)
        lm_state = {k: rt.NDArray(_c(v.numpy())) for k, v in kv.items()}
        pos = pos0

        for w in range(0, n_text, T_PRE):
            win = text_emb[:, w : w + T_PRE]
            await self.f_pre(
                inputs={
                    "text_emb": rt.NDArray(_c(pad_text(win).numpy().astype(self.npd))),
                    "pos": rt.NDArray(_c(np.array([pos], np.int32))),
                },
                state=lm_state,
            )
            self.calls += 1
            pos += win.shape[1]
        prefill_hw = pos0 + ((n_text + T_PRE - 1) // T_PRE) * T_PRE

        gen = torch.Generator(device="cpu")
        frames = []
        latent = np.zeros((1, 1, LDIM), np.float32)
        is_bos, eos_step, i, draws = 1.0, None, 0, draw0
        for i in range(max_gen_len):
            # HARD STOP at the static cache capacity. Past S_MAX the state write is out
            # of range and the causal mask degenerates, so the run would be meaningless
            # rather than merely truncated; it is recorded as a flag instead.
            if pos + i >= S_MAX:
                break
            r = await self.f_step(
                inputs={
                    "latent_in": rt.NDArray(_c(latent.astype(self.npd))),
                    "is_bos": rt.NDArray(_c(np.array([is_bos], self.npd))),
                    "pos": rt.NDArray(_c(np.array([pos + i], np.int32))),
                },
                state=lm_state,
            )
            self.calls += 1
            draws += 1
            is_bos = 0.0
            cond = r["cond"].numpy()
            if (
                float(r["eos_logit"].numpy().reshape(-1)[0]) > model.eos_threshold
                and eos_step is None
            ):
                eos_step = i
            gen.manual_seed(seed + draws)
            noise = torch.empty(1, LDIM, dtype=torch.float32).normal_(0.0, self.std, generator=gen)
            latent = (
                (
                    await self.f_flow(
                        {
                            "cond": rt.NDArray(_c(cond)),
                            "noise": rt.NDArray(_c(noise.numpy().astype(self.npd))),
                        }
                    )
                )["latent"]
                .numpy()
                .astype(np.float32)
                .reshape(1, 1, LDIM)
            )
            self.calls += 1
            if eos_step is not None and i >= eos_step + fae:
                break  # upstream discards this step's latent
            with torch.no_grad():
                q = model.mimi.quantizer(
                    (torch.from_numpy(latent) * self.emb_std + self.emb_mean).transpose(-1, -2)
                )
            feed = {"latent": rt.NDArray(_c(q.numpy()))}
            feed.update({k: rt.NDArray(v) for k, v in mimi_state.items()})
            rm = await self.f_mimi(feed)
            self.calls += 1
            frames.append(rm["pcm"].numpy().reshape(-1).astype(np.float32))
            mimi_state = {k: _c(rm[f"{k}_out"].numpy()) for k in MIMI_STATE}

        steps = i + 1
        cmeta = {
            "chunk": chunk,
            "voice_len": int(pos0),
            "tokens": int(n_text),
            "max_gen_len": int(max_gen_len),
            "steps": int(steps),
            "eos": None if eos_step is None else int(eos_step),
            "frames": len(frames),
            "kv_high_water": int(max(prefill_hw, pos + steps)),
            "kv_capped": bool(pos + steps >= S_MAX),
            "hit_max_gen_len": bool(eos_step is None and steps >= max_gen_len),
        }
        return frames, cmeta, mimi_state, draws

    def fresh_mimi_state(self):
        return {k: _c(v.numpy()) for k, v in mimi_init_state(torch.float32).items()}

    async def synth(self, text: str, voice: str, seed: int) -> tuple[np.ndarray, dict]:
        """One corpus row: upstream's chunker, then every chunk end to end."""
        from pocket_tts.models.tts_model import MAX_TOKEN_PER_CHUNK, split_into_best_sentences

        chunks = split_into_best_sentences(
            self.model.flow_lm.conditioner.tokenizer,
            text,
            MAX_TOKEN_PER_CHUNK,
            self.model.pad_with_spaces_for_short_inputs,
            self.model.remove_semicolons,
        )
        mimi_state = self.fresh_mimi_state()
        frames, per_chunk, draws = [], [], 0
        calls0 = self.calls
        t0 = time.monotonic()
        for chunk in chunks:
            f, cm, mimi_state, draws = await self._gen_chunk(chunk, voice, seed, draws, mimi_state)
            frames += f
            per_chunk.append(cm)
        self.last_mimi_state = mimi_state
        pcm = np.concatenate(frames) if frames else np.zeros(0, np.float32)
        return pcm, {
            "n_chunks": len(chunks),
            "chunks": per_chunk,
            "wall_s": round(time.monotonic() - t0, 3),
            "calls": self.calls - calls0,
            "no_eos": any(c["eos"] is None for c in per_chunk),
            "kv_capped": any(c["kv_capped"] for c in per_chunk),
            "kv_high_water": max((c["kv_high_water"] for c in per_chunk), default=0),
        }

    async def synth_one(self, chunk: str, voice: str, seed: int, mimi_carry):
        """One already-split chunk, carrying a Mimi state in and out. The noise stream
        is keyed off the chunk text so a resumed worker reproduces the same draws."""
        if mimi_carry is None:
            mimi_carry = self.fresh_mimi_state()
        else:
            mimi_carry = {k: _c(np.asarray(v)) for k, v in mimi_carry.items()}
        calls0 = self.calls
        t0 = time.monotonic()
        base = seed + zlib.crc32(chunk.encode()) % 1_000_000  # str.hash is salted per process
        f, cm, mimi_state, _ = await self._gen_chunk(chunk, voice, base, 0, mimi_carry)
        self.last_mimi_state = mimi_state
        pcm = np.concatenate(f) if f else np.zeros(0, np.float32)
        return pcm, {**cm, "wall_s": round(time.monotonic() - t0, 3), "calls": self.calls - calls0}


def build_engine_sync(assets, unit, dtype):
    from pocket_tts import TTSModel

    torch.set_num_threads(1)
    model = TTSModel.load_model().eval()
    return Engine(model, assets, unit, dtype)


def asset_paths(artifacts: Path, dtype: str) -> dict:
    a = {
        "flowlm": artifacts / f"flowlm_{dtype}_s{S_MAX}.aimodel",
        "flow": artifacts / f"flow_decoder_{dtype}_lsd1.aimodel",
        "mimi": artifacts / "mimi_decoder_float32_ring272_outer.aimodel",
    }
    for k, p in a.items():
        if not p.exists():
            raise SystemExit(f"missing asset {k}: {p}")
    return a


# --------------------------------------------------------------------------- worker


def rss_mb() -> float:
    import resource

    # ru_maxrss is bytes on Darwin, kilobytes on Linux
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


async def run_worker(args):
    t_boot = time.monotonic()
    jobs = [json.loads(l) for l in Path(args.jobs).read_text().splitlines() if l.strip()]
    eng = build_engine_sync(asset_paths(ROOT / args.artifacts, args.dtype), args.unit, args.dtype)
    await eng.load()
    boot = time.monotonic() - t_boot
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_f = (ROOT / args.meta).open("a")
    done = 0
    for job in jobs:
        if eng.calls > args.call_budget:
            break
        try:
            pcm, meta = await eng.synth(job["text"], job["voice"], args.seed)
        except Exception as e:  # a failed clip is a row, not an aborted sweep
            meta_f.write(
                json.dumps({**job, "error": f"{type(e).__name__}: {e}"[:300]}) + "\n"
            )
            meta_f.flush()
            done += 1
            continue
        wav = out_dir / f"{job['id']}.wav"
        scipy.io.wavfile.write(wav, eng.model.sample_rate, pcm)
        peak = float(np.abs(pcm).max()) if pcm.size else 0.0
        meta_f.write(
            json.dumps(
                {
                    **job,
                    "wav": wav.name,
                    "dtype": args.dtype,
                    "unit": args.unit,
                    "samples": int(pcm.size),
                    "duration_s": round(pcm.size / eng.model.sample_rate, 3),
                    "peak": round(peak, 4),
                    "rms": round(float(np.sqrt((pcm**2).mean())) if pcm.size else 0.0, 5),
                    "rss_mb": round(rss_mb(), 1),
                    **meta,
                }
            )
            + "\n"
        )
        meta_f.flush()
        done += 1
    meta_f.close()
    print(
        f"[worker] {done}/{len(jobs)} clips, {eng.calls} engine calls, "
        f"boot {boot:.1f}s, wall {time.monotonic() - t_boot:.1f}s, peak RSS {rss_mb():.0f} MB",
        flush=True,
    )


# --------------------------------------------------------------------------- driver


def run_driver(args):
    corpus = [json.loads(l) for l in (ROOT / args.corpus).read_text().splitlines() if l.strip()]
    meta_p = ROOT / args.meta
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / args.out_dir / "_jobs.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    attempts: dict[str, int] = {}
    t0 = time.monotonic()
    n_workers = 0
    while True:
        done = set()
        if meta_p.exists():
            for line in meta_p.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["id"])
        todo = [j for j in corpus if j["id"] not in done and attempts.get(j["id"], 0) < 2]
        if not todo:
            break
        # Only the FIRST job of a batch is charged an attempt. A worker that stops on
        # its call budget (or dies of the IOSurface leak) leaves the rest of its batch
        # untouched, and those come back round on the next spawn; charging them here
        # would retire them after two innocent rounds.
        attempts[todo[0]["id"]] = attempts.get(todo[0]["id"], 0) + 1
        tmp.write_text("\n".join(json.dumps(j) for j in todo[: args.batch]) + "\n")
        n_workers += 1
        print(
            f"[driver] worker {n_workers}: {len(done)}/{len(corpus)} done, "
            f"{len(todo)} left, {time.monotonic() - t0:.0f}s elapsed",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(HERE / "sweep_gen.py"),
            "--worker",
            "--jobs",
            str(tmp),
            "--out-dir",
            args.out_dir,
            "--meta",
            args.meta,
            "--dtype",
            args.dtype,
            "--unit",
            args.unit,
            "--artifacts",
            args.artifacts,
            "--seed",
            str(args.seed),
            "--call-budget",
            str(args.call_budget),
        ]
        r = subprocess.run(cmd, env=dict(os.environ), cwd=str(ROOT))
        if r.returncode != 0:
            print(f"[driver]   worker exited {r.returncode} (respawning)", flush=True)
    skipped = [j["id"] for j in corpus if attempts.get(j["id"], 0) >= 2 and j["id"] not in done]
    print(
        f"[driver] complete: {len(corpus)} rows, {n_workers} workers, "
        f"{time.monotonic() - t0:.0f}s wall, {len(skipped)} unrecoverable {skipped[:10]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--jobs")
    ap.add_argument("--corpus", default="artifacts/sweep_corpus.jsonl")
    ap.add_argument("--out-dir", default="artifacts/sweep_fp32")
    ap.add_argument("--meta", default="artifacts/sweep_fp32_gen.jsonl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--unit", default="gpu", choices=["cpu", "gpu", "neural_engine"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--call-budget", type=int, default=3000)
    args = ap.parse_args()
    if args.worker:
        asyncio.run(run_worker(args))
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
