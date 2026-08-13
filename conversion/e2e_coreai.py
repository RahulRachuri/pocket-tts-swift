"""End-to-end: text -> wav entirely through Core AI graphs.

    host: sentencepiece + LUT embedding lookup        (graph (a) — a table lookup,
                                                       not worth a graph)
    engine: flowlm.prefill  -> shared k/v state
    engine: flowlm.step     -> cond, eos_logit        } 12.5 Hz autoregressive loop
    engine: flow_decoder    -> latent                 }
    host: latent*emb_std+emb_mean, quantizer k=1 conv (fold into the mimi graph later)
    engine: mimi_decoder    -> 1920 PCM samples/frame

The AR loop mirrors `models/tts_model.py:_autoregressive_generation` exactly, including
the quirk that the latent produced on the breaking step is DISCARDED (32 flow-LM calls
-> 31 mimi frames on the oracle fixture).

Noise uses the oracle's `--rng explicit` protocol: a dedicated Generator re-seeded
`seed + call_index`, where call_index 0 belongs to the text-prefill call whose flow
output upstream throws away. That makes AR step i draw with seed+i+1, so a free-run
here is comparable to the oracle sample-for-sample.

Run (export venv):
    HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
      .venv-export/bin/python conversion/e2e_coreai.py --out artifacts/e2e_orc.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from export_mimi_decoder import STATE_NAMES as MIMI_STATE, init_state as mimi_init_state  # noqa: E402
from export_flowlm import VOICE_LEN, pad_text, seed_voice_state  # noqa: E402
from flowlm_graphs import LDIM, STATE_NAMES, T_PRE, build_kv_state  # noqa: E402

PROMPT = "The quick brown fox jumps over the lazy dog."
VOICE = "alba"


def _c(x):
    return np.ascontiguousarray(x)


async def synth(
    model, text: str, voice_state: dict, assets: dict, seed: int, unit: str, dtype: str
):
    """Text -> float32 PCM at 24 kHz, plus timings. `dtype` selects the flow-LM /
    flow-decoder precision; the Mimi decoder asset is fp32 either way (its 12-tensor
    streaming state feeds back at 12.5 Hz — the exact shape FluidAudio measured fp16
    error compounding into audible beeping on ANE, NOTES.md section 9)."""
    npd = np.float16 if dtype == "float16" else np.float32
    td = torch.float16 if dtype == "float16" else torch.float32
    import coreai.runtime as rt

    from pocket_tts.models.tts_model import (
        MAX_TOKEN_PER_CHUNK,
        prepare_text_prompt,
        split_into_best_sentences,
    )

    opts = (
        rt.SpecializationOptions.cpu_only()
        if unit == "cpu"
        else rt.SpecializationOptions.from_preferred_compute_unit_kind(
            getattr(rt.ComputeUnitKind, unit)()
        )
    )
    m_lm = await rt.AIModel.load(str(assets["flowlm"]), opts)
    f_pre, f_step = m_lm.load_function("prefill"), m_lm.load_function("step")
    f_flow = (await rt.AIModel.load(str(assets["flow"]), opts)).load_function("main")
    f_mimi = (await rt.AIModel.load(str(assets["mimi"]), opts)).load_function("main")

    # ---- host front end: upstream's own chunker, so token ids match the oracle.
    # `_generate_audio_stream_short_text` tokenizes the CHUNK, not the prepared text —
    # prepare_text_prompt is consulted only for the frames-after-EOS guess.
    chunks = split_into_best_sentences(
        model.flow_lm.conditioner.tokenizer,
        text,
        MAX_TOKEN_PER_CHUNK,
        model.pad_with_spaces_for_short_inputs,
        model.remove_semicolons,
    )

    # The Mimi streaming state is NEVER reset — it runs continuously across chunks.
    # The flow-LM KV cache IS reset to the voice state per chunk (upstream deep-copies).
    mimi_state = {k: _c(v.numpy()) for k, v in mimi_init_state(torch.float32).items()}
    gen = torch.Generator(device="cpu")
    std = model.temp**0.5
    emb_std = model.flow_lm.emb_std.to(torch.float32)
    emb_mean = model.flow_lm.emb_mean.to(torch.float32)
    frames, calls, per_chunk = [], 0, []

    t0 = time.monotonic()
    for chunk in chunks:
        _, frames_guess = prepare_text_prompt(
            chunk, model.pad_with_spaces_for_short_inputs, model.remove_semicolons
        )
        frames_after_eos = model.model_recommended_frames_after_eos
        if frames_after_eos is None:
            frames_after_eos = frames_guess + 2
        prepared = model.flow_lm.conditioner.prepare(chunk)
        with torch.no_grad():
            text_emb = model.flow_lm.conditioner(prepared).to(torch.float32)
        n_text = text_emb.shape[1]
        max_gen_len = model._estimate_max_gen_len(n_text)

        kv = build_kv_state(td)
        pos = seed_voice_state(kv, voice_state, td)
        lm_state = {k: rt.NDArray(_c(v.numpy())) for k, v in kv.items()}

        # prefill, windowed over the static T_PRE graph, carrying pos across windows
        for w in range(0, n_text, T_PRE):
            win = text_emb[:, w : w + T_PRE]
            await f_pre(
                inputs={
                    "text_emb": rt.NDArray(_c(pad_text(win).numpy().astype(npd))),
                    "pos": rt.NDArray(_c(np.array([pos], np.int32))),
                },
                state=lm_state,
            )
            pos += win.shape[1]

        latent = np.zeros((1, 1, LDIM), np.float32)
        is_bos, eos_step, i = 1.0, None, 0
        for i in range(max_gen_len):
            r = await f_step(
                inputs={
                    "latent_in": rt.NDArray(_c(latent.astype(npd))),
                    "is_bos": rt.NDArray(_c(np.array([is_bos], npd))),
                    "pos": rt.NDArray(_c(np.array([pos + i], np.int32))),
                },
                state=lm_state,
            )
            is_bos = 0.0
            cond = r["cond"].numpy()
            if (
                float(r["eos_logit"].numpy().reshape(-1)[0]) > model.eos_threshold
                and eos_step is None
            ):
                eos_step = i
            calls += 1
            gen.manual_seed(seed + calls)  # call 0 is the discarded prefill draw
            noise = torch.empty(1, LDIM, dtype=torch.float32).normal_(0.0, std, generator=gen)
            latent = (
                (
                    await f_flow(
                        {
                            "cond": rt.NDArray(_c(cond)),
                            "noise": rt.NDArray(_c(noise.numpy().astype(npd))),
                        }
                    )
                )["latent"]
                .numpy()
                .astype(np.float32)
                .reshape(1, 1, LDIM)
            )
            if eos_step is not None and i >= eos_step + frames_after_eos:
                break  # upstream discards this step's latent
            # host: rescale + the k=1 quantizer conv (both belong inside graph (e) later)
            with torch.no_grad():
                q = model.mimi.quantizer(
                    (torch.from_numpy(latent) * emb_std + emb_mean).transpose(-1, -2)
                )
            feed = {"latent": rt.NDArray(_c(q.numpy()))}
            feed.update({k: rt.NDArray(v) for k, v in mimi_state.items()})
            rm = await f_mimi(feed)
            frames.append(rm["pcm"].numpy().reshape(-1).astype(np.float32))
            mimi_state = {k: _c(rm[f"{k}_out"].numpy()) for k in MIMI_STATE}
        if eos_step is None:
            print(f"  ⚠ max_gen_len reached without EOS on chunk {chunk!r}")
        per_chunk.append(dict(tokens=n_text, steps=i + 1, eos=eos_step, fae=frames_after_eos))
    wall = time.monotonic() - t0
    return np.concatenate(frames), dict(
        wall_s=round(wall, 3), n_chunks=len(chunks), n_frames=len(frames), chunks=per_chunk
    )


def compare(got: np.ndarray, gold: np.ndarray):
    n = min(got.shape[0], gold.shape[0])
    a, b = torch.from_numpy(got[:n]), torch.from_numpy(gold[:n])
    print(
        f"[wav vs oracle] samples {got.shape[0]} vs {gold.shape[0]}  "
        f"cos {F.cosine_similarity(a, b, dim=0):.6f}  max|Δ| {(a - b).abs().max():.4f}  "
        f"rms got {a.pow(2).mean().sqrt():.4f} / oracle {b.pow(2).mean().sqrt():.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=PROMPT)
    ap.add_argument("--voice", default=VOICE)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--unit", default="gpu", choices=["cpu", "gpu", "neural_engine"])
    ap.add_argument("--oracle", default="oracle/orc_a.npz")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts/e2e.wav")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = ap.parse_args()

    torch.set_num_threads(1)
    from pocket_tts import TTSModel

    model = TTSModel.load_model().eval()
    art = ROOT / args.artifacts
    assets = {
        "flowlm": art / f"flowlm_{args.dtype}_s256.aimodel",
        "flow": art / f"flow_decoder_{args.dtype}_lsd1.aimodel",
        "mimi": art / "mimi_decoder_float32_ring272_outer.aimodel",
    }
    for k, p in assets.items():
        if not p.exists():
            raise SystemExit(f"missing asset {k}: {p}")

    # the shipped voice state, in the oracle's own capture layout
    vs = model.get_state_for_audio_prompt(args.voice)
    voice = {}
    for mod, st in vs.items():
        for kk, vv in st.items():
            voice[f"voice/{mod}/{kk}"] = vv.detach().cpu().numpy()

    pcm, meta = asyncio.run(
        synth(model, args.text, voice, assets, args.seed, args.unit, args.dtype)
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(out, model.sample_rate, pcm)
    dur = pcm.shape[0] / model.sample_rate
    print(
        f"[e2e {args.unit} {args.dtype}] {meta}  duration {dur:.3f}s  "
        f"peak {np.abs(pcm).max():.3f} rms {np.sqrt((pcm**2).mean()):.4f}  -> {out}"
    )
    print(f"  wall/audio = {meta['wall_s'] / dur:.2f}x (lower is faster than realtime)")

    if args.text == PROMPT:
        orc = np.load(ROOT / args.oracle)
        compare(pcm, orc["wav"])
        assert VOICE_LEN == 126


if __name__ == "__main__":
    main()
