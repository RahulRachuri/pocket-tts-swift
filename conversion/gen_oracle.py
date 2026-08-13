"""PyTorch reference ("oracle") run for the pocket-tts Core AI port.

Runs upstream inference on a fixed prompt/voice and captures every tensor a later
Core AI graph must reproduce. Two RNG modes:

  --rng explicit  (default)  noise drawn from a dedicated torch.Generator, one draw
                             per flow-LM call, seeded seed+call_index. Deterministic
                             by construction and independent of thread scheduling.
  --rng stock                upstream path (global RNG via torch.nn.init.normal_),
                             seeded with torch.manual_seed. Used to test whether the
                             stock path reproduces at all.

Writes: oracle/<tag>.npz, oracle/<tag>.wav, oracle/<tag>.json
"""

import argparse, hashlib, json, platform, sys, time
from functools import partial
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

from pocket_tts import TTSModel
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.models.flow_lm import FlowLMModel, lsd_decode
from pocket_tts.models.mimi import MimiModel

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
ORACLE = ROOT / "oracle"

PROMPT = "The quick brown fox jumps over the lazy dog."
VOICE = "alba"


class Capture:
    def __init__(self, seed: int, mode: str):
        self.seed = seed
        self.mode = mode
        self.calls = 0            # flow-LM forward calls (prefill counts as call 0)
        self.rec = {}             # name -> list of arrays
        self.gen = torch.Generator(device="cpu")

    def add(self, key, t):
        self.rec.setdefault(key, []).append(
            t.detach().to(torch.float32).cpu().numpy().copy()
        )


def instrument(model: TTSModel, cap: Capture):
    """Replace FlowLMModel.forward with an exact copy that records + injects noise."""

    def flow_forward(self, sequence, text_embeddings, model_state, lsd_decode_steps,
                     temp, noise_clamp, eos_threshold):
        idx = cap.calls
        cap.calls += 1
        is_prefill = sequence.shape[1] == 0

        cap.add(f"in/seq_len_text_{idx}", torch.tensor([text_embeddings.shape[1]]))
        if is_prefill:
            cap.add("prefill/text_embeddings", text_embeddings)

        # --- upstream body, verbatim (models/flow_lm.py forward) ---
        sequence = torch.where(torch.isnan(sequence), self.bos_emb, sequence)
        input_ = self.input_linear(sequence)
        transformer_out = self.backbone(input_, text_embeddings, sequence, model_state=model_state)
        transformer_out = transformer_out.to(torch.float32)
        assert lsd_decode_steps > 0
        transformer_out = transformer_out[:, -1]
        eos_logit = self.out_eos(transformer_out)
        out_eos = eos_logit > eos_threshold

        noise_shape = transformer_out.shape[:-1] + (self.ldim,)
        std = temp ** 0.5
        noise = torch.empty(noise_shape, dtype=transformer_out.dtype, device=transformer_out.device)
        if cap.mode == "explicit":
            cap.gen.manual_seed(cap.seed + idx)
            if noise_clamp is None:
                noise.normal_(mean=0.0, std=std, generator=cap.gen)
            else:
                torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-noise_clamp, b=noise_clamp)
        else:
            if noise_clamp is None:
                torch.nn.init.normal_(noise, mean=0.0, std=std)
            else:
                torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-noise_clamp, b=noise_clamp)

        if not is_prefill:
            cap.add("step/cond", transformer_out)       # [1,1024] flow-LM hidden, post out_norm
            cap.add("step/eos_logit", eos_logit)        # [1,1]
            cap.add("step/noise", noise)                # [1,32] BEFORE lsd_decode mutates it

        conditioned_flow = partial(self.flow_net, transformer_out)
        latent = lsd_decode(conditioned_flow, noise, lsd_decode_steps)
        if not is_prefill:
            cap.add("step/latent", latent)              # [1,32]
        return latent, out_eos

    FlowLMModel.forward = flow_forward

    orig_decode = MimiModel.decode_from_latent

    def mimi_decode(self, latent, mimi_state):
        cap.add("mimi/in", latent)                      # [1,512,1] post-quantizer
        out = orig_decode(self, latent, mimi_state)
        cap.add("mimi/out", out)                        # [1,1,1920]
        return out

    MimiModel.decode_from_latent = mimi_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rng", choices=["explicit", "stock"], default="explicit")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--text", default=PROMPT)
    ap.add_argument("--voice", default=VOICE)
    args = ap.parse_args()
    tag = args.tag or f"oracle_{args.rng}_s{args.seed}"
    ORACLE.mkdir(exist_ok=True)

    model = TTSModel.load_model()
    model.eval()
    cap = Capture(args.seed, args.rng)
    instrument(model, cap)

    voice_state = model.get_state_for_audio_prompt(args.voice)
    # voice conditioning state, as shipped (the no-cloning checkpoint ships it pre-baked)
    for mod_name, st in voice_state.items():
        for k, v in st.items():
            cap.add(f"voice/{mod_name}/{k}", v)

    tok = model.flow_lm.conditioner.prepare(args.text)
    cap.add("prefill/text_tokens", tok.tokens)

    torch.manual_seed(args.seed)
    t0 = time.monotonic()
    audio = model.generate_audio(voice_state, args.text, copy_state=True)
    wall = time.monotonic() - t0

    wav = audio.numpy().astype(np.float32)
    scipy.io.wavfile.write(ORACLE / f"{tag}.wav", model.sample_rate, wav)

    out = {}
    for key, lst in cap.rec.items():
        if key.startswith(("step/", "mimi/")):
            out[key] = np.concatenate([a.reshape(1, -1) for a in lst], axis=0)
        else:
            out[key] = lst[0] if len(lst) == 1 else np.stack(lst)
    out["wav"] = wav
    np.savez(ORACLE / f"{tag}.npz", **out)

    digest = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
              for k, v in sorted(out.items())}
    meta = dict(
        tag=tag, rng=args.rng, seed=args.seed, text=args.text, voice=args.voice,
        n_flowlm_calls=cap.calls, n_steps=out["step/latent"].shape[0],
        n_mimi_frames=out["mimi/out"].shape[0],
        wav_samples=int(wav.shape[0]), sample_rate=int(model.sample_rate),
        duration_s=round(wav.shape[0] / model.sample_rate, 4), wall_s=round(wall, 3),
        temp=model.temp, lsd_decode_steps=model.lsd_decode_steps,
        noise_clamp=model.noise_clamp, eos_threshold=model.eos_threshold,
        frames_after_eos=model.model_recommended_frames_after_eos,
        shapes={k: list(v.shape) for k, v in sorted(out.items())},
        sha256_16=digest,
        versions=dict(
            python=sys.version.split()[0], torch=torch.__version__,
            numpy=np.__version__, platform=platform.platform(),
            pocket_tts=__import__("importlib.metadata", fromlist=["version"]).version("pocket-tts"),
        ),
    )
    (ORACLE / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in
                      ("tag","rng","seed","n_flowlm_calls","n_steps","n_mimi_frames",
                       "duration_s","wall_s")}, indent=2))
    print("shapes:")
    for k, v in sorted(meta["shapes"].items()):
        print(f"  {k:34s} {v}")


if __name__ == "__main__":
    main()
