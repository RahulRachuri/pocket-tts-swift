# pocket-tts → Core AI — working notes

Technical state of the port, written so a cold session can resume without re-deriving
anything. Phase A = license/variant verification, PyTorch oracle, decomposition map,
export-blocker inventory, and a first Core AI graph (the Mimi decoder). Phase B
(§11 onward) = the state-mechanism verdict and the remaining graphs, to audible audio.

Last updated **2026-08-12** (Phase B).

---

## 1. License & variant verdict — GREEN

Use **`kyutai/pocket-tts-without-voice-cloning`**, revision `e041936c75475d350b405bc870bcf7c22da4e9e6`.

| repo | license | gated | evidence |
|---|---|---|---|
| `kyutai/pocket-tts-without-voice-cloning` | `cc-by-4.0` | **`gated: false`, `private: false`** | <https://huggingface.co/api/models/kyutai/pocket-tts-without-voice-cloning> (JSON), card at <https://huggingface.co/kyutai/pocket-tts-without-voice-cloning> |
| `kyutai/pocket-tts` | `cc-by-4.0` | **gated** — acceptance of usage terms required | <https://huggingface.co/kyutai/pocket-tts> |
| `FluidInference/pocket-tts-coreml` | `cc-by-4.0` (inherited, attribution to Kyutai required) | no | <https://huggingface.co/FluidInference/pocket-tts-coreml> |

The ungated repo is a complete model, not a stub: the upstream loader tries the gated
checkpoint first and silently falls back (`pocket_tts/models/tts_model.py:203-207`,
`config/english.yaml` `weights_path_without_voice_cloning`). With no HF token the
fallback fires and `has_voice_cloning` becomes `False` — verified, and generation works
normally, because the shipped voices are **pre-baked conditioning states**, not audio:
`get_state_for_audio_prompt("alba")` loads `embeddings_v3/alba.safetensors` straight into
the KV cache (`tts_model.py:853-869`, `_import_model_state` at `:1055`). Only *live*
cloning from a wav needs the gated weights (it needs the Mimi **encoder** + a speaker
projection). **No gated weights were downloaded.**

Correction to the brief: the checkpoint is not "6-layer English"-only — it is 6-layer
*for English*; the non-English packs ship a 24-layer variant (`*_24l`). Ours is English,
6 transformer layers, **109.50 M parameters total** (measured).

---

## 2. Environment

Two venvs in this repo (both gitignored); nothing outside the repo was touched.

| venv | purpose | pins |
|---|---|---|
| `.venv` | oracle / reference inference | python 3.13.14, `pocket-tts` 2.1.0, torch **2.13.0**, numpy 2.5.2 |
| `.venv-export` | Core AI export + gates | python 3.13.14, torch **2.11.0**, `coreai-torch` 0.4.1, `coreai-core` 1.0.0b2, `coreai-opt` 0.2.1, `transformers` 5.15.0, `pocket-tts` 2.1.0 |

The `apple/coreai-models` overlay is wired into `.venv-export` with a one-line `.pth`
(`site-packages/coreai_models_overlay.pth` → the overlay's `python/src`), **not**
`pip install -e` — that would downgrade `coreai-torch` to the dead 0.4.0.

Export-venv build order that works: install `torch==2.11.0` + the three `coreai-*` wheels
first, then `pocket-tts` (it accepts torch ≥ 2.5, so the pin survives), then
`transformers` (only `coreai_models.export.pipeline` needs it). `coreai-opt` and
`transformers` are both imported transitively by `coreai_models.export.macos` — installing
them up front saves two failed runs. `coreai-opt` pins `safetensors<=0.7.0` and
`pocket-tts` wants 0.8.0; the resulting pip warning is cosmetic, nothing at runtime
touches the conflict.

Weights live under `weights/` via `HF_HOME=$PWD/weights/hf` — every command below sets it.

---

## 3. Model facts (measured, `conversion/inspect_state.py`)

Config: `pocket_tts/config/english.yaml` (`english` == `english_2026-04`).

```
sample rate 24000     frame rate 12.5 Hz     frame size 1920 samples
latent dim (ldim) 32  model dim 1024         mimi encoder frame rate 200 Hz
mimi hop_length 120   steps per latent 16    tokenizer sentencepiece, 4000 bins
```

| block | params | shape |
|---|---|---|
| `flow_lm.transformer` | 75.52 M | 6 layers, d_model 1024, 16 heads × 64, FFN 4096, RoPE max_period 1e4, **no context window (full causal)** |
| `flow_lm.flow_net` (`SimpleMLPAdaLN`) | 9.76 M | 6 `ResBlock`s, width 512, in/out 32, cond 1024, **2 time conditions (s, t)** |
| `flow_lm.conditioner` | 4.10 M | `nn.Embedding(4001, 1024)` |
| `mimi.decoder_transformer` | 6.30 M | 2 layers, d_model 512, 8 heads × 64, FFN 2048, layer_scale 0.01, **context 250 (sliding)** |
| `mimi.decoder` (SEANet) | 3.97 M | ratios [6,5,4], n_filters 64, `pad_mode: constant` |
| `mimi.upsample` / `mimi.quantizer` | 0.02 M each | ConvTranspose1d(512,512,k32,s16,g512,bias=False) / Conv1d(32→512,k1,bias=False) |

Generation defaults (`pocket_tts/default_parameters.py`): `temp 0.7`,
**`lsd_decode_steps 1`**, `noise_clamp None`, `eos_threshold -4.0`,
`MAX_TOKEN_PER_CHUNK 50`.

### On the LSD step count — the brief was right for this checkpoint

A parallel research pass flagged "upstream default is 1-step" as unsupported. It is
supported, directly: `pocket_tts/default_parameters.py:3` reads
`DEFAULT_LSD_DECODE_STEPS = 1`, and it is the default of both `TTSModel.load_model`
(`models/tts_model.py:238`) and the CLI (`main.py:262`). FluidAudio passes
`lsd_decode_steps=8` explicitly and bakes `s=i/8, t=(i+1)/8` into their fused decoder —
that is a deliberate 8× quality/speed trade against upstream's own knob, not upstream's
default. Both run here (1 → 2.560 s / rms 0.1243, 8 → 2.560 s / rms 0.1179 on the same
seed and text), so the choice is ours to make and should be A/B'd on audio quality, not
inherited. "LSD" itself is Lagrangian Self-Distillation (`models/flow_lm.py:19-40`, cites
arXiv 2505.18825) — the training scheme that makes a low step count viable, not a mode.

---

## 4. The oracle — the parity authority

`conversion/gen_oracle.py`. Fixed prompt `"The quick brown fox jumps over the lazy dog."`,
voice `alba`, seed 1234, English config. Outputs `oracle/<tag>.{npz,wav,json}` (gitignored).

```
HF_HOME=$PWD/weights/hf .venv/bin/python conversion/gen_oracle.py --tag orc_a
.venv/bin/python conversion/compare_oracle.py oracle/orc_a.npz oracle/orc_b.npz
```

**Determinism: fully reproducible, both RNG modes, bit-identical on all 54–55 keys.**

Upstream draws the flow-matching noise *inside* `FlowLMModel.forward`
(`models/flow_lm.py:133-137`) from the global CPU RNG, on a background thread. It still
reproduces under `torch.manual_seed` because that thread is the only RNG consumer (the
Mimi decode thread draws nothing), so draw order is fixed. That is fragile, so the script
also offers `--rng explicit`: a dedicated `torch.Generator` re-seeded `seed + call_index`
per call. The explicit mode is what later gates should use — it makes each step's noise a
*value*, reproducible without reference to any thread history, and it is exactly the shape
the Core AI graph needs (noise as a graph input).

Captured (fp32, per the run above: 14 text tokens, 32 AR steps, 31 Mimi frames, 2.48 s):

| key | shape | what |
|---|---|---|
| `prefill/text_tokens` | [1,14] | SentencePiece ids |
| `prefill/text_embeddings` | [1,14,1024] | text conditioning output — graph (a)'s output |
| `voice/transformer.layers.{0..5}.self_attn/{cache,offset}` | [2,1,126,16,64], [1] | the shipped voice state, i.e. prefill already done for 126 positions |
| `step/cond` | [32,1024] | flow-LM hidden after `out_norm`, last position — graph (b)'s output |
| `step/eos_logit` | [32,1] | raw `out_eos` logit (threshold applied host-side) |
| `step/noise` | [32,32] | the noise fed to the flow decoder, captured **before** `lsd_decode` mutates it |
| `step/latent` | [32,32] | flow decoder output — graph (c)'s output |
| `mimi/in` | [31,512] | post-quantizer latent — graph (d)'s input |
| `mimi/out` | [31,1920] | PCM per frame — graph (d)'s output |
| `wav` | [59520] | final audio, peak 0.564, rms 0.122 |

Two traps the capture had to dodge, both re-usable: `lsd_decode` does
`current += flow_dir / num_steps` **in place on the noise tensor**, so noise must be cloned
at capture time; and the text-prefill call also runs the flow head and consumes one RNG
draw even though its output is discarded (`tts_model.py:722-725`) — step indices must
account for it.

---

## 5. Decomposition map

Five graphs. Shapes are what the *port* should use, `[batch=1, …]`, fp32 for gating.

| # | graph | inputs | outputs | state carried |
|---|---|---|---|---|
| a | **text conditioning** | `tokens [1,T] i32` | `text_emb [1,T,1024]` | none — a pure embedding lookup (`conditioners/text.py:74-76`). Cheap enough to keep host-side in Swift as a 4001×1024 table; not worth a graph. |
| b | **flowlm prefill** | `text_emb [1,T,1024]`, 6× `kv [2,1,S,16,64]`, `offset [1]` | 6× `kv'`, `offset'` | 6-layer KV. Voice conditioning is *already in* the shipped state (126 positions), so prefill only ever appends text. T ≤ 50 tokens by `MAX_TOKEN_PER_CHUNK`. |
| c | **flowlm AR step** | `latent_prev [1,1,32]` (or a BOS flag), 6× `kv`, `offset` | `cond [1,1024]`, `eos_logit [1,1]`, 6× `kv'`, `offset'` | same 6-layer KV, one position per call |
| d | **flow decoder** | `cond [1,1024]`, `noise [1,32]`, (`s`,`t` if not baked) | `latent [1,32]` | none — pure function. N Euler steps, N=1 upstream. |
| e | **mimi decoder** | `latent [1,512,1]`, **12 state tensors** | `pcm [1,1,1920]`, 12 state tensors | see below |

Host (Swift) owns: sentence chunking (≤50 tokens), the SentencePiece tokenizer, the AR
loop, the EOS threshold compare, the noise draw, the `latent*emb_std+emb_mean` rescale and
the 32→512 quantizer conv (a k=1 conv — fold it into graph (e)), and state marshaling.

### The Mimi streaming state — enumerated

`init_states(model.mimi, ...)` returns **28 modules / 52 tensors**, but that walks the
encoder too. The **decode path** (`upsample` → `decoder_transformer` → `decoder`) is
**14 modules / 24 tensors**, of which **13 are live**:

| module | tensor | shape |
|---|---|---|
| `upsample.convtr` | `partial` | [1,512,16] |
| `decoder_transformer.transformer.layers.{0,1}.self_attn` | `cache`, `offset` | [2,1,L,8,64], [1] i64 |
| `decoder.model.0` | `previous` | [1,512,6] |
| `decoder.model.2` | `partial` | [1,256,6] |
| `decoder.model.3.block.1` | `previous` | [1,256,2] |
| `decoder.model.5` | `partial` | [1,128,5] |
| `decoder.model.6.block.1` | `previous` | [1,128,2] |
| `decoder.model.8` | `partial` | [1,64,4] |
| `decoder.model.9.block.1` | `previous` | [1,64,2] |
| `decoder.model.11` | `previous` | [1,64,2] |

The other 11 are dead weight and should not be plumbed: three `…block.3.previous` are
**zero-width** `[1,C,0]` (the k=1 convs in each resnet block have no history), and every
`first` bool flag is only read by `pad_mode="replicate"`, which no decode-path conv uses
(`config/english.yaml` sets `pad_mode: constant`; only the encoder-side downsample is
replicate). FluidAudio's "23 tensors" is the same set counted slightly differently.
Zero-width tensors are worth deleting on principle — FluidAudio documents them crashing
the Core ML CPU/GPU Espresso backend.

**Finding the brief did not anticipate: the Mimi decoder is not KV-free.** Its
`decoder_transformer` is a 2-layer streaming transformer with `context: 250`, and upstream
allocates its cache at `max_gen_len × 16` and never rolls it — the cache grows with
utterance length even though attention only ever looks back 250. A static graph therefore
needs a bounded cache. **272 is exact**: a query at local index j ∈ [0,16) needs keys back
to `pos+j-249`, so the live span is 250+15 = 265 positions; a 272-slot shift register
covers it with no approximation. This is used in the shipped graph below and gates
bit-exact.

---

## 6. Export blockers — located, with fixes

All line numbers are in the installed `pocket_tts` 2.1.0 package.

**(i) `.item()`-based KV offset.** `modules/transformer.py:14`
```python
offset_value = int(offset.view(-1)[0].item())
cache[0, :, offset_value : offset_value + k.shape[1]] = k     # :16
cache[1, :, offset_value : offset_value + v.shape[1]] = v     # :17
valid = cache[:, :, : offset_value + k.shape[1]]              # :18
```
A data-dependent Python int drives three slices — untraceable, and the in-place writes are
invisible to `torch.export` regardless. Same pattern at `models/tts_model.py:427`
(`_flow_lm_current_end`) and `:761` (`is_eos.item()`, host-side, fine to keep in Swift).
*Fix used here:* a fixed-capacity **shift register** — `cat(cache[:, :, T:], new)` — plus an
absolute-position mask built from `arange`. No scatter, no `.item()`, no dynamic slice.
Correct for the Mimi decoder because its attention has a finite context; the flow-LM has
**no** context window, so graph (c) needs the one-hot-multiply-add write instead (or Core
AI's in-graph state primitive, see §9).

**(ii) RNG inside forward.** `models/flow_lm.py:131-137` allocates `noise` and fills it
with `torch.nn.init.normal_` / `trunc_normal_` inside the traced region.
*Fix:* lift it to a graph input. Already done on the oracle side (`step/noise` is captured
per step), matching FluidAudio, who generate noise host-side with a seeded xoshiro256** and
pass `latent_init` in.

**(iii) NaN-as-BOS sentinel.** Produced at `models/tts_model.py:748-753`
(`torch.full((1,1,32), float("NaN"))`), consumed at `models/flow_lm.py:121`
(`torch.where(torch.isnan(sequence), self.bos_emb, sequence)`).
*Fix: delete the protocol, do not port it.* Feed `bos_emb` directly on the first step, or
pass an explicit `is_bos` flag. FluidAudio hit exactly this — their rank-4 ANE graphs drop
the NaN protocol because "the ANE mangles NaN inputs before `isnan` evaluates" — and their
cross-model playbook states it as a general rule. An explicit flag removes the risk class
entirely at zero cost.

**(iii-b) NaN also poisons the KV cache.** `modules/transformer.py:52-56` and
`models/tts_model.py:411-418` both initialise/extend the cache with `float("NaN")`.
Upstream gets away with it by slicing the unwritten tail off before attention. A
fixed-capacity graph cannot slice, and a *masked* SDPA still multiplies V by a zero
weight — `0 * NaN = NaN`. **Fixed-capacity caches must be zero-initialised.** (Independently
confirmed by FluidAudio, who scrub `where(isnan(keys), 0, keys)` in their rank-5 trace and
require zero-filled caches in the rank-4 one.)

**(iv) New — `aten.elu` has no Core AI lowering** on `coreai-torch` 0.4.1:
`ValueError: The exported program contains unsupported ATen ops: aten.elu.default`. SEANet
uses ELU everywhere. Writing it out as `torch.where(x > 0, x, torch.expm1(x))` exports fine
and is bit-identical to torch's kernel (verified: max|Δ| 0.0 across 31 frames).

---

## 7. Parity traps (verified in source)

**LayerNorm eps really is a 1e-5 / 1e-6 mix, split by subsystem:**

| site | file:line | eps | variance convention |
|---|---|---|---|
| transformer `norm1`/`norm2` (both flow-LM **and** Mimi) | `modules/mimi_transformer.py:26,27` | **1e-5** | `nn.LayerNorm` (biased) |
| flow-LM `out_norm` | `models/flow_lm.py:89` | **1e-5** | `nn.LayerNorm` |
| flow-net `ResBlock.in_ln` | `modules/mlp.py:96` | **1e-6** | custom, explicit `unbiased=False` |
| flow-net `FinalLayer.norm_final` | `modules/mlp.py:121` | **1e-6** | custom, `elementwise_affine=False` |
| `TimestepEmbedder` `RMSNorm` | `modules/mlp.py:29` | 1e-5 | **see below** |

Rule of thumb: **the transformer stack is 1e-5, the flow net is 1e-6.** The flow net's
`LayerNorm` is a hand-written reimplementation ("because the default one doesn't support
jvp") — same math as `nn.LayerNorm`, but a Swift/Metal port must not assume the eps.

**`RMSNorm` is not RMSNorm** (`modules/mlp.py:20-36`):
```python
var = eps + x.var(dim=-1, keepdim=True)          # torch default correction=1 → UNBIASED
y = x * (alpha * torch.rsqrt(var))
```
It mean-*centres* (via `var`), uses the **unbiased** estimator (n−1), adds eps **before**
`rsqrt` rather than inside a `sqrt`, and does not re-add a mean. Reimplementing it as
`x * rsqrt(mean(x²) + eps)` will be wrong three ways. Only used inside `TimestepEmbedder`,
so it lands in graph (d).

**RoPE is the interleaved-pair (complex) convention, not rotate-half**
(`modules/rope.py:36-56`): `q.view(B,T,H,D//2,2)`, real = `[...,0]`, imag = `[...,1]`,
`freqs = exp(ds * (-log(1e4) * 2 / D))`, `ts = arange(T) + offset`. Rotation math is
promoted to fp32 and cast back. A GPT-NeoX-style `rotate_half` port silently produces
plausible-but-wrong audio. The `offset` is **absolute** and grows unbounded — it feeds
`cos/sin` directly, so it must be carried as a real value, not a ring index.

**Conditioning concat order** (load-bearing, and a historical bug in FluidAudio's port):
* `models/flow_lm.py:150` — `cat([text_embeddings, input_], dim=1)`: conditioning is
  **prepended** to the latent, and the output is sliced back to the latent's length at
  `:156` (`transformer_out[:, -sequence.shape[1]:]`).
* `models/tts_model.py:356` — `cat([text_embeddings, audio_conditioning], dim=1)`; in
  practice exactly one of the two is non-empty per call.
* `models/tts_model.py:894` — `cat([bos_before_voice, prompt], dim=1)`.
* Net sequence order: **`bos_before_voice` → voice conditioning → text tokens → latents.**

**`transformer_out[:, -sequence.shape[1]:]` with `shape[1] == 0`** keeps the *whole*
tensor (`-0` is `0` in Python), not an empty one — which is why the text-prefill call also
emits a latent and burns an RNG draw. Do not "fix" this when reimplementing.

**Sliding-window mask** (`modules/transformer.py:22-29`): `mask = (pos_k >= 0) &
(delta >= 0) & (delta < context)`, `context = 250` for Mimi, `None` for the flow-LM.

---

## 8. Mimi decoder on Core AI — exported and gated ✅

`conversion/export_mimi_decoder.py`. One call = one 12.5 Hz latent → 1920 PCM samples.

```
latent [1,512,1] + 12 state tensors  ->  pcm [1,1,1920] + 12 state tensors
```
(12 = the 13 live tensors minus one, because both transformer layers share a single
`offset` input; the graph returns `offset + 16`.)

Every stateful primitive was re-expressed functionally, since upstream mutates in place:
`StreamingConv1d` → `cat(prev, x)` → conv → tail slice is the new `prev`;
`StreamingConvTranspose1d` → convtr → overlap-add the head, tail minus bias is the new
`partial`; KV cache → the 272-slot shift register from §5.

**Gate ladder and results** (fp32, 31 frames, vs `oracle/orc_a.npz` `mimi/in` → `mimi/out`):

| gate | cos mean | cos min | max abs Δ | |
|---|---|---|---|---|
| eager functional rewrite vs upstream streaming | 1.000000 | 0.999999 | **0.0** | ✅ bit-identical |
| Core AI engine, `SpecializationOptions.gpu` | 1.000000 | 0.999999 | 0.0000 | ✅ PASS |
| Core AI engine, `SpecializationOptions.cpu_only` | **−0.010196** | −0.209464 | 0.8740 | ❌ **FAIL** |
| …after the `--upsample outer` rewrite, `cpu_only` | 1.000000 | 1.000000 | 0.0000 | ✅ PASS |
| …after the rewrite, `gpu` | 1.000000 | 0.999999 | 0.0000 | ✅ PASS |

Artifact: `mimi_decoder_float32_ring272_outer.aimodel`, **41.2 MB**. Nothing was timed —
`cpu_only` is a parity option, never a benchmark.

### The CPU failure was a runtime bug, not our graph

Per-output diffing of frame 0 (`conversion/diag_cpu.py`) put the divergence on the **first**
op: `up_p_out` was already wrong while `offset_out` was exact. Minimal repro
(`conversion/repro_convtr_cpu.py`) isolates it to `ConvTranspose1d` on the `cpu_only`
delegate, with a clean threshold at **stride ≥ 8 / kernel ≥ 16** and no dependence on
channel count or grouping:

| op | gpu | cpu_only |
|---|---|---|
| `ConvTranspose1d(64,64,k=4,s=2)` … `(k=12,s=6)` | OK | **OK** |
| `ConvTranspose1d(64,64,k=16,s=8)` … `(k=32,s=16)` | OK | **WRONG** (cos 0.03–0.08) |
| `ConvTranspose1d(512,512,k=32,s=16,groups=512)` | OK | WRONG (cos 0.036) |
| `ConvTranspose1d(4,4,k=32,s=16)` | OK | WRONG (cos −0.047) |
| `Conv1d(512,512,k=7)` — control | OK | OK |

So exactly one op in the whole decoder is affected: `mimi.upsample.convtr`
(k=32, s=16, groups=512). SEANet's own transposed convs (k=12/s=6, 10/5, 8/4) sit under
the threshold. Because that op runs on a `T == 1` input with `groups == channels`, it
degenerates to a per-channel outer product, `out[c,:] = x[c,0] * W[c,0,:]` — a broadcast
multiply with no `ConvTranspose1d` at all. `--upsample outer` uses that form, and it takes
`cpu_only` from cos −0.01 to **bit-exact**, which both confirms the diagnosis and is the
production fix.

Environment for the repro: `coreai-torch` 0.4.1, `coreai-core` 1.0.0b2, `coreai-opt` 0.2.1,
torch 2.11.0, macOS 27 beta, Xcode 27 beta. Worth an Apple Feedback (user-gated), same as
the Python-bindings IOSurface leak from the companion port.

Trap-list items honoured: `SpecializationOptions.cpu_only()` for parity, no
`expectFrequentReshapes` anywhere, gated on CPU before GPU.

---

## 9. What FluidAudio does, and where we would diverge

FluidAudio's port is Apache-2.0 (Swift runtime `FluidInference/FluidAudio`, Python
conversion lab `FluidInference/mobius`). Their decomposition matches ours graph-for-graph —
`cond_prefill`, `flowlm_step`, `flow_decoder`, `mimi_decoder`, plus an optional
`mimi_encoder` for cloning — which is the main confirmation we wanted. Deltas worth
carrying:

* **In-graph state, from the start.** Their default `.gpu` placement round-trips the KV
  cache through host `MLMultiArray`s every call; their later `.aneState` placement uses a
  real in-graph `MLState` behind a **multifunction package exposing `prefill` and
  `generate` over one shared state**, and measured **−14.9 % end-to-end**. Their CosyVoice3
  port ships in-graph state as its *default*. Two independent convergences. **Phase B
  should treat in-graph/stateful KV as the reference design, not an optimisation** — and
  the first thing to check is whether Core AI exposes a multi-entry-point-over-shared-state
  construct, which would collapse graphs (b) and (c) into two functions of one asset.
* **Rank-5 → rank-4.** Their rank-5 `[2,1,512,16,64]` combined K/V cache was rejected by
  the Core ML ANE compiler; splitting it into separate rank-4 K and V tensors took
  `flowlm_step` from 0 % to 100 % ANE. Core AI's constraints are its own, but the
  *principle* (a rejected shape is usually a construct problem) transfers, and the
  rank-4 split costs nothing to adopt pre-emptively.
* **Euler-step fusion is not free.** It worked for their pocket-tts flow decoder and was
  *declined* for another port whose per-step fp32 IO casts were doing real
  error-containment. With upstream defaulting to 1 step here, the question may not even
  arise — decide it on measured audio, not on their number.
* **Mimi off the fast accelerator.** They pin `mimi_decoder` to `.cpuOnly` with three
  documented ANE failure modes (zero-length-tensor crash, fp16 state-feedback beeping at
  ~1e-3/frame compounding, 64-byte stride segfault). Our Core AI result is *inverted*:
  the GPU delegate is bit-exact and the CPU delegate is the broken one (§8). Their
  conclusion is a Core ML/ANE finding and must not be copied as a Core AI default —
  measure all three units.
* **Acceptance gate.** Their hard lesson is that tensor-similarity gates pass while audio
  is unintelligible; they require an ASR round-trip (WER < 10 %, PyTorch-vs-port
  transcription diff < 2 %). Cosine is fine as a per-graph smoke check (and is what §8
  uses), but **the Phase B end-to-end gate must be an ASR round-trip** — there is a fast
  local Parakeet ASR available for exactly this.
* Where we already diverge: we take upstream's `lsd_decode_steps=1` rather than their 8;
  we drop the NaN-BOS protocol outright instead of keeping it for one placement; and the
  ungated checkpoint gives us the voice states pre-baked, so graph (b) never has to run
  the Mimi *encoder*.

---

## 10. What Phase B needs

1. **Graph (c), the flow-LM AR step, with in-graph KV.** 6 layers × `[1,S,16,64]` split
   K/V (rank-4), zero-initialised, S = 512 (126 voice + ≤50 text + generation headroom, the
   same budget that sets `MAX_TOKEN_PER_CHUNK = 50`). No context window means no shift
   register: needs either Core AI's state primitive or a one-hot multiply-add write.
   Gate against `oracle/orc_a.npz` `step/cond` and `step/eos_logit` for all 32 steps,
   driven by the captured `voice/*` state.
2. **Answer the state question first** — does Core AI have an `MLState`/multifunction
   equivalent? It decides whether (b) and (c) are two graphs or two functions, and it is
   worth ~15 % by FluidAudio's measurement.
3. **Graph (d), the flow decoder.** Pure function, no state, `cond` + `noise` in,
   `latent` out; `s`/`t` baked at N=1. Smallest remaining graph, and `step/noise` →
   `step/latent` is already captured as an exact fixture. Watch `RMSNorm`'s unbiased
   variance and the 1e-6 eps.
4. **Graph (b), prefill.** Fixed `T_max` with masking; the ungated checkpoint means only
   text is ever prefilled.
5. **Fold the quantizer** (`Conv1d(32→512, k=1)`) and the `latent*emb_std+emb_mean`
   rescale into graph (e) so the host passes the raw `[1,32]` latent.
6. **Swift host**: SentencePiece tokenizer, ≤50-token chunker, AR loop with the −4.0 EOS
   compare and `frames_after_eos`, seeded Gaussian noise, state marshaling, and a fresh KV
   cache per chunk while the Mimi state is **never** reset.
7. **fp16 pass** — everything above is fp32. Expect the flow decoder's tiny AdaLN MLP to
   be the sensitive one, and re-check the Mimi state feedback (23-ish tensors at 12.5 Hz)
   for the compounding artifacts FluidAudio heard on ANE.
8. **ASR round-trip harness** as the acceptance gate (§9).

---

## 11. State mechanism — **YES on both counts** (Phase B step 0)

Core AI has both of the things FluidAudio needed Core ML's `MLState` + multifunction
packages for. This is not inferred, it is in the shipped API surface and in working
zoo ports.

| question | verdict | evidence |
|---|---|---|
| in-graph mutable state? | **yes** | `coreai/runtime/__init__.pyi:151` — `_InferenceFunction.__call__(inputs, state)`; declared at export via `TorchConverter.add_pytorch_module(..., state_names=)` (`coreai_torch/converter.py:266`). A "state" is any input torch.export reports as mutated: `_classify_stateful_inputs` (`coreai_torch/_utils.py:1683`) picks up mutable buffers and `user_inputs_to_mutate`. |
| several functions over one weight set + one state? | **yes** | `entrypoint_name=` per staged program; `AIModel.function_names` / `load_function(name)` at runtime. `coreai_models.export.macos.export_to_coreai_multifunction` documents "Constants are deduplicated across entrypoints". Shipped in the zoo: `conversion/qwen3_asr/export_unified.py:127-131` and `conversion/unlimited_ocr/export_decoder.py:79-83` both stage `prefill` + `decode` with identical `state_names`. |

Confirmed empirically here: our two-entrypoint asset is **302.4 MB** for a 75.52 M-param
fp32 backbone (= 302.1 MB of weights). One copy, two functions.

Mechanics that matter: the in-place write must survive `torch.export`, so the export
path is `run_decompositions(get_decomp_table())` → **`remove_functionalization(ep)`** →
convert. Without that call the mutation is silently dropped (the zoo's own doctor has a
lint for exactly this: `cli/coreai_doctor.py:513`). The write op is
`coreai_models.primitives._ops.mutable_slice_update(cache, update, begin, end)` with
`begin`/`end` as **runtime i32 tensors**.

### The June MPSGraph KV-write bug does not reproduce here

`coreai-model-zoo/knowledge/coreai-beta-mpsgraph-kvwrite-bug.md` reports that a
`slice_update` at a *runtime-tensor* index SIGTRAPs on the Mac GPU on this beta, and
prescribes a host-supplied one-hot `write_mask` blend as the escape. **We wrote the
graph the "crashing" way on purpose (data-indexed `mutable_slice_update` at a runtime
`pos`) and it loads, specializes and runs clean on both `cpu_only` and `gpu`** — same as
the zoo's own later vibevoice/dots_tts ports, which use the same helper. Either the
runtime moved or the trigger is narrower than documented. The mask-blend remains the
fallback if a later toolchain regresses; it costs a full-cache read-modify-write per
layer per step, which is why it is not the default.

**Decision taken:** in-graph state, packed **rank-5** `k_cache`/`v_cache`
`[6, 1, 16, 256, 64]` — layer-first, already in attention layout, so the whole 6-layer
cache is TWO states rather than twelve. FluidAudio's rank-4 K/V split was a *Core ML ANE
compiler* workaround; Core AI specializes the packed rank-5 form fine and the zoo ships
it. The NaN protocols are deleted (explicit `is_bos` float flag; zero-initialised cache).

**S = 256**, justified: 126 voice positions (shipped pre-baked) + a ≤50-token text chunk
(`MAX_TOKEN_PER_CHUNK`) + generation headroom = 208 worst case; 256 is the next power of
two. The oracle run uses 126 + 14 + 32 = 172.

---

## 12. Graphs (b), (c), (d) — exported and gated ✅

`conversion/flowlm_graphs.py` (the static-shape re-authoring) +
`conversion/export_flowlm.py` / `conversion/export_flow_decoder.py` (export + gates).

```
prefill(text_emb [1,16,1024], pos [1] i32)             -> cond [1,1024]     |  shared
step   (latent_in [1,1,32], is_bos [1], pos [1] i32)   -> cond, eos_logit   |  k/v state
flow   (cond [1,1024], noise [1,32])                   -> latent [1,32]     |  stateless
```

Gate method: seed the state with the oracle's captured `voice/*` KV, prefill the
oracle's `prefill/text_embeddings`, then **teacher-force all 32 AR steps on the oracle's
own latents** so each step is judged independently rather than through a compounding
free-run.

| gate | cond cos mean / min | max&#124;Δcond&#124; | max&#124;Δeos&#124; | eos-decision flips | |
|---|---|---|---|---|---|
| flow-LM eager fp32 | 1.000000 / 1.000000 | 1.25e−6 | 2.86e−6 | 0 | ✅ |
| flow-LM engine `cpu_only` | 1.000000 / 1.000000 | 2.82e−6 | 1.43e−5 | 0 | ✅ PASS |
| flow-LM engine `gpu` | 1.000000 / 1.000000 | 3.94e−6 | 1.24e−5 | 0 | ✅ PASS |
| flow decoder eager fp32 | 1.000000 / 1.000000 | 9.54e−7 | — | — | ✅ |
| flow decoder engine `cpu_only` | 1.000000 / 1.000000 | 3.34e−6 | — | — | ✅ PASS |
| flow decoder engine `gpu` | 1.000000 / 1.000000 | 1.43e−6 | — | — | ✅ PASS |

Artifacts: `flowlm_float32_s256.aimodel` **302.4 MB** (two functions),
`flow_decoder_float32_lsd1.aimodel` **39.1 MB**. The eos gate is on the *decision*
(`logit > −4.0`), not just the value — a flipped comparison truncates the utterance.

### Blockers cleared, and one new one

* `.item()` KV offset (`modules/transformer.py:14`) → `mutable_slice_update` at a runtime
  `pos` input into a Core AI state. The causal mask `k_idx <= pos + i` over the *whole*
  256-slot cache is exactly upstream's `valid = cache[:, :, : offset + T]`, because an
  unwritten slot is always past every query position.
* RNG in forward (`models/flow_lm.py:131-137`) → `noise` is a graph input; the step graph
  stops at `cond`/`eos_logit` and never draws.
* NaN-BOS (`models/tts_model.py:748`) → deleted, replaced by an `is_bos` float that blends
  `bos_emb` in. Cache is zero-initialised.
* **New — `aten.var.correction` has no Core AI lowering** on coreai-torch 0.4.1 (the same
  class of gap as Phase A's `aten.elu`). It is hit by *both* flow-net norms
  (`modules/mlp.py:22` unbiased, `:52` `unbiased=False`). Fix: two-pass mean/variance
  written out (`_VarFreeRMSNorm`, `_VarFreeLayerNorm`), reproducing all three of that
  "RMSNorm"'s oddities — mean-centring, the (n−1) estimator, and eps *outside* the rsqrt.
  Eager max&#124;Δ&#124; 9.5e−7 against the oracle, so the rewrite is faithful.
* **New — `coreai.reshape` rejects a rank-0 shape operand.** `pos.reshape(())` to squeeze
  a `[1]` position into a scalar fails at `prog.optimize()`; keep `pos` rank-1 and let
  `[T] + [1]` broadcast.

### Prefill: chosen shape and the production note

Prefill is a **second function over the shared state**, static width `T_PRE = 16`, short
chunks zero-padded. Pad rows write junk into cache slots that sit strictly after every
real position, so no real row can attend to them (causal mask), and the first AR steps
overwrite them before ever reading them — verified by the gate, which prefills 14 real
tokens into a 16-wide graph. For production a ≤50-token chunk wants either `T_PRE = 64`
or a windowed loop over the 16-wide graph carrying `pos` across windows (FluidAudio's
`cond_prefill` does the latter); the multifunction shape is already right either way.

---

## 13. End to end — text → audible wav, entirely on Core AI ✅

`conversion/e2e_coreai.py`. Host owns only the sentencepiece tokenizer, the 4001×1024
LUT lookup (graph (a) — not worth a graph), the `latent*emb_std+emb_mean` rescale, the
k=1 quantizer conv, the EOS compare and the seeded noise draw. Everything else is engine.

Free-running (no teacher forcing), noise from the oracle's `--rng explicit` protocol so
the run is comparable sample-for-sample. Chunking follows upstream exactly: a fresh KV
cache per chunk (re-seeded from the voice state), and the **Mimi streaming state is never
reset** across chunks.

| run | framing | wav vs oracle | wall (audio 2.480 s) |
|---|---|---|---|
| fp32, gpu | 14 tokens → 32 steps, EOS at 28, 31 frames — **identical to the oracle** | cos **1.000005**, max&#124;Δ&#124; **1.0e−4**, rms 0.1224 vs 0.1224 | **0.528 s** = 4.7× realtime |
| fp16, gpu | identical framing | cos **0.999467**, max&#124;Δ&#124; 0.0870, rms 0.1224 | 0.480 s = 5.2× realtime |

Bit-exactness end to end was never expected — the AR loop feeds its own output back, so
any fp divergence compounds. What matters is that the EOS step and frame count land in
the same place and the audio is intelligible.

### Acceptance gate — ASR round-trip (`conversion/asr_gate.py`)

Resamples 24 kHz float32 → 16 kHz mono PCM16 and transcribes with the local
`parakeet-swift` host (`PARAKEET_ARTIFACTS` = the zoo's `artifacts_v2`).

| wav | WER |
|---|---|
| Core AI fp32, oracle prompt | **0.00 %** (0/9 words) |
| Core AI fp16, oracle prompt | **0.00 %** |
| PyTorch oracle, same prompt (control) | **0.00 %** |
| Core AI fp32, **fresh out-of-oracle sentence** (17 words, 27 tokens, 5.200 s) | **0.00 %** (0/17) |

The fresh sentence matters twice: it proves the pipeline generalizes past the fixture,
and at 27 tokens it exercises the **windowed prefill** (two passes through the T_PRE=16
graph carrying `pos`), which the oracle's 14-token prompt does not.

## 14. fp16 pass — green, no chase needed

Re-exported the flow-LM and the flow decoder at fp16; Mimi deliberately stays fp32
(12 state tensors feeding back at 12.5 Hz is exactly the shape FluidAudio measured fp16
error compounding into audible artifacts, §9).

| graph | unit | cos mean / min | max&#124;Δ&#124; | size |
|---|---|---|---|---|
| flow-LM fp16 | `cpu_only` | 0.999837 / 0.999760 | 6.4e−2 | 151.3 MB (was 302.4) |
| flow-LM fp16 | `gpu` | 0.999999 / 0.999999 | 2.9e−3 | |
| flow decoder fp16 | `cpu_only` | 0.999987 / 0.999923 | 5.5e−2 | 19.6 MB (was 39.1) |
| flow decoder fp16 | `gpu` | 1.000000 / 1.000000 | 2.4e−3 | |

Zero eos-decision flips at fp16, and the e2e framing is unchanged. Note the inversion vs
Phase A's Mimi result: here `cpu_only` is the *less* accurate delegate on both graphs,
which is consistent with it being a reference-precision path only in the sense of being
un-fused, not in the sense of being more accurate.

**One fp16 export trap:** RoPE's fp32 rotation math must be cast back to the working
dtype (`modules/rope.py:54` does; our first draft dropped it) or `torch.export` fails
with `query.dtype: float, key.dtype: c10::Half` at SDPA. In fp32 the missing cast is
invisible.

## 15. What Phase C (Swift host + iPhone) needs

1. **Swift host**: sentencepiece tokenizer + the ≤50-token chunker, the LUT lookup, the
   AR loop with the −4.0 EOS compare and `frames_after_eos`, seeded Gaussian noise
   (FluidAudio use xoshiro256\*\* + Box-Muller), and `InferenceFunction.MutableViews`
   for the two KV states. Everything the Python harness does host-side is ~200 lines.
2. **Fold the quantizer + rescale into the Mimi graph** so the host passes a raw `[1,32]`
   latent — the only remaining host-side tensor math.
3. **Widen `T_PRE` to 64** (or keep the window loop; both work — the window loop is
   already gated). A 50-token chunk currently costs 4 prefill calls.
4. **Give the Mimi decoder in-graph state too.** It is currently the only graph still
   round-tripping 12 tensors through the host every frame, because Phase A exported it
   before the state verdict was in. Same `state_names=` mechanics.
5. **Re-run every gate on the device**, and expect the compute-unit answer to move: our
   `gpu`-vs-`cpu_only` results are Mac-side, and the zoo's own note is that ANE
   eligibility is a separate question that must be measured, not planned.
6. **Watch the IOSurface leak** (carried over from the sibling ASR port's notes): the Python bindings leak an
   output buffer per call and SIGTRAP at ~8k calls. This spike makes ~130 calls per 2.5 s
   clip (32 step + 32 flow + 31 mimi + prefill), so **~60 clips per process** — fine for
   gates, fatal for a book. The open question that matters for Phase C is whether the
   **native Swift path leaks the same way**; the AR loop makes tens of calls per second.
7. **Time it properly.** The 0.5 s number here is one untuned fp32 Mac-GPU run with
   per-call Python async overhead in the loop; it is an existence proof, not a benchmark.

---

## Repo layout

```
conversion/
  inspect_state.py        shapes + streaming-state enumeration
  gen_oracle.py           the oracle run (--rng explicit|stock, --seed, --text, --voice)
  compare_oracle.py       bit-compare two oracle npz captures
  wav_stats.py            sanity stats on a capture
  export_mimi_decoder.py  functional rewrite + eager gate + Core AI export + cpu/gpu gates
  diag_cpu.py             per-output localisation of a compute-unit divergence
  repro_convtr_cpu.py     minimal repro of the ConvTranspose1d cpu_only bug
  flowlm_graphs.py        static-shape re-authoring: flow-LM core/step/prefill + flow decoder
  export_flowlm.py        graphs (b)+(c) as ONE multifunction asset + eager/cpu/gpu gates
  export_flow_decoder.py  graph (d) + eager/cpu/gpu gates
  e2e_coreai.py           text -> wav through the four graphs, vs the oracle wav
  asr_gate.py             the acceptance gate: 16 kHz resample -> parakeet-swift -> WER
oracle/    weights/    artifacts/    .venv/    .venv-export/     (all gitignored)
```

Reproduce everything:
```
HF_HOME=$PWD/weights/hf .venv/bin/python conversion/gen_oracle.py --tag orc_a
HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  .venv-export/bin/python conversion/export_mimi_decoder.py --upsample outer \
  --fold-quantizer --graph-state    # the M2 `_q_gs` production asset (sec 19); drop the
                                    # last two flags for the M1-era round-trip variant
HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  .venv-export/bin/python conversion/export_flowlm.py        # add --dtype float16
HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  .venv-export/bin/python conversion/export_flow_decoder.py
HF_HOME=$PWD/weights/hf DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  .venv-export/bin/python conversion/e2e_coreai.py --out artifacts/e2e_orc_gpu.wav
.venv-export/bin/python conversion/asr_gate.py artifacts/e2e_orc_gpu.wav \
  --text "The quick brown fox jumps over the lazy dog."
```

---

## 16. Validation sweep — 302 sentences + a 10-minute long-form run

Phase B's "2 sentences at 0.00 % WER" is a smoke test. The sibling ASR port passed
chapter-level gates and then failed on full books, so this sweep is the full-book
analogue: a broad corpus, every shipped voice, and a production-length generation, with
the explicit goal of finding the failure tail before anything goes public.

Harness: `conversion/sweep_corpus.py` (corpus), `sweep_gen.py` (driver + call-budgeted
worker subprocesses), `sweep_score.py` (ASR round trip, WER, sanity flags),
`sweep_oracle.py` (the upstream PyTorch control), `longform.py` + `longform_score.py`,
`repro_kv_budget.py`. All artifacts are gitignored.

### 16.1 The finding that mattered: S_MAX = 256 truncates real sentences

**The flow-LM has no context window**, so every generated frame holds a KV slot for the
whole chunk and nothing ever rolls out. Phase B sized the cache at 256 from the oracle
fixture (126 voice + 14 text + **32 generated frames**). Measured over 349 real chunks,
this checkpoint generates **2.47 frames per text token on average** (p90 3.2), so the
capacity is exhausted at roughly `(S_MAX - voice_len) / 3.2` tokens — about **40 tokens
for a 126-position voice**, well inside upstream's own `MAX_TOKEN_PER_CHUNK = 50`.

Past that the utterance is cut off mid-word with no EOS, and **no per-graph gate can see
it**: the step graph is exactly as accurate on step 90 as on step 1, and §12's cosine
table stays at 1.000000. It is only visible end to end.

`conversion/repro_kv_budget.py`, one sentence, voice `javert`, same asset family:

```
S_MAX=256   35-token chunk  -> 96 steps, eos None, high-water 257, CAPPED
S_MAX=512   35-token chunk  -> 107 steps, eos 103, high-water 268, ok
```

A 79-word Moby Dick sentence was worse: 0.72 s of audio for 79 words at S=256, because
upstream's chunker had emitted a single 121-token chunk (see §16.5) and 126 + 121 = 247
left ten slots to generate into.

**Two corrections to §11's budget note.** (a) The voice conditioning is *not* 126
positions for every voice — the eight shipped English voices measure 126/126/126/126/
133/141/162 and the wider v3 catalogue runs 76..162. (b) The generation term is bounded
by upstream's own `_estimate_max_gen_len`, i.e. `ceil((tokens/3 + 2) * 12.5)` = **234
frames** for a 50-token chunk. Worst legal case is therefore **162 + 50 + 234 = 446**, so
**S_MAX = 512**, and the value is now derived in `flowlm_graphs.py` rather than guessed.

Re-exported at S=512, both precisions, and both re-pass the §12 oracle gate unchanged:

| asset | size | cpu_only | gpu |
|---|---|---|---|
| `flowlm_float32_s512.aimodel` | 302.4 MB | cos 1.000000, max&#124;Δ&#124; 3.0e−6, 0 eos flips | cos 1.000000, 3.9e−6, 0 flips |
| `flowlm_float16_s512.aimodel` | 151.3 MB | cos 0.999837, 6.4e−2, 0 flips | cos 0.999999, 3.3e−3, 0 flips |

Every number below is from the S=512 assets. Across all 302 sweep clips: **0 capped
chunks, 0 chunks that failed to fire EOS**.

### 16.2 The sentence sweep

302 sentences: 150 LibriSpeech test transcripts (test-clean + test-other, lower-cased),
100 Moby Dick sentences, 52 hand-built hard cases (numbers, years, money,
abbreviations, acronyms, invented proper nouns, 1-3 word utterances, 40+ word
sentences). The eight shipped voices are cycled over the shuffled corpus, so each gets
~38 sentences spread evenly across sources and length buckets. fp32, GPU. Total 1756 s
of audio, 388 s of generation wall, 40 worker processes.

**Corpus WER 4.27 %** (300 ASR-usable clips of 302; 4.97 % over the 271 unflagged).

| voice | n | WER | conditioning positions | median RMS |
|---|---:|---:|---:|---:|
| alba | 38 | 4.41 % | 126 | 0.124 |
| azelma | 38 | 2.50 % | 162 | 0.101 |
| cosette | 38 | 5.65 % | 126 | **0.030** |
| eponine | 37 | 4.46 % | 141 | 0.094 |
| fantine | 38 | 3.53 % | 133 | 0.095 |
| javert | 38 | 4.01 % | 126 | **0.045** |
| jean | 37 | 4.89 % | 126 | 0.114 |
| marius | 36 | 5.14 % | 126 | **0.042** |

No voice is broken; the 2.5–5.7 % spread is inside the corpus-composition noise. But the
**output level spread is real and large: cosette sits 12.3 dB below alba** at the same
settings. That is inherited from the conditioning clips, not a bug, and it means a
player must apply per-voice gain normalisation or the voice picker will read as a volume
control.

By source: librispeech 3.47 %, moby 3.69 %, hard cases 9.69 %.
By length: **1-3 words 28.3 %, 4-6 words 23.8 %, 7-15 words 7.8 %, 16-30 words 3.2 %,
31+ words 1.4 %.** Long-form prose — the actual use case — is the *best* regime.

### 16.3 Systematic failure modes found

**(1) Short utterances are a weak, high-variance regime — and it is upstream's, not the
port's.** WER falls monotonically with length (above). To separate a port defect from a
property of the checkpoint or of the ASR gate, the same 63 short rows were generated by
upstream `TTSModel.generate_audio` (pure PyTorch, `sweep_oracle.py`) and scored by the
identical scorer, at three seeds each:

| seed | Core AI fp32 | upstream PyTorch |
|---|---:|---:|
| 1234 | 23.5 % | 15.0 % |
| 777 | 15.6 % | 15.0 % |
| 20260813 | 17.9 % | **30.1 %** |
| mean | **19.0 %** | **20.0 %** |

The seed-to-seed spread swamps the path-to-path difference and the port sits inside
upstream's own range, so **the short-utterance weakness is not port-attributable**. On
the matched >6-word rows the two paths are 2.22 % (port) vs 1.89 % (upstream) — equal.
Note also `pad_with_spaces_for_short_inputs` is **False** for this config, so upstream's
own short-input mitigation is not even switched on; turning it on is worth an A/B if
short utterances ever matter for the product.

**(2) Upstream's chunker overshoots its own limit.** 26 of 302 rows (26 of 349 chunks)
contained a chunk above `MAX_TOKEN_PER_CHUNK = 50` — median 69 tokens, **max 121** —
because `split_into_best_sentences` only sub-splits on `,;:` and gives up when a long
sentence has none. Upstream logs a warning and carries on. Quality did not suffer
(those rows score 1.61 % WER), but the KV high-water mark of the whole sweep was **494
of 512**, from exactly that 121-token chunk. So S=512 is only safe *given* a bounded
chunk: the Swift host must enforce a hard token cap (whitespace-split the remainder)
rather than trusting the chunker.

**(3) No runaway, no clipping, no dead voice.** 0 clips hit `max_gen_len` without EOS,
0 clips clipped (max peak 0.940 over the sweep), 0 empty clips, 1 clip flagged
`short_audio`. The one genuine per-clip failure in the whole sweep is `s0002`, cosette,
"So it has always been" — 5 words in 0.96 s at RMS 0.030, i.e. a rushed, mumbled
delivery. Everything else above threshold is a scoring artifact (see §16.6).

**(4) The ASR gate itself is framing-sensitive — the sibling port's failure, on the gate
side.** `parakeet-swift` returns an **empty** transcript for an isolated sub-2-second
utterance no matter how much silence is padded around it, and on one 7.1 s clip whose
audio is a clean, correct ten-word count it ran away to **1445 words** of `"ten, ten,
ten…"` under 0.25 s padding while transcribing perfectly under 1.0 s padding. The scorer
now detects both (empty, length-implausible, and repeated-n-gram loops) and retries under
a different framing; 12 of 302 clips needed a retry and 2 remain unscorable. Left
undetected these alone put the headline WER at 30 %.

### 16.4 fp16

48-clip stratified subset (6 per voice), same corpus rows, matched against fp32:

| path | n | WER |
|---|---:|---:|
| Core AI fp32 | 48 | 5.31 % |
| Core AI fp16 | 48 | 5.96 % |

0.65 pp on 48 clips is inside the noise, and the framing (EOS step, chunk count) is
unchanged. **fp16 is a green light for the flow-LM and flow decoder**; Mimi stays fp32
per §14.

### 16.5 The 10-minute long-form run

2183 contiguous words of Moby Dick chapter 1, one voice (`alba`), fp32, GPU. Chunked by
upstream's own `split_into_best_sentences` into 97 chunks; KV cache re-seeded from the
voice state per chunk; **Mimi streaming state never reset**, and carried across worker
restarts through an npz checkpoint so a respawn resumes bit-exactly. Chunk audio is
**concatenated directly, no crossfade and no inserted silence** — that is what
`TTSModel.generate_audio` does (`torch.cat(audio_chunks, dim=0)`), and inserting silence
would both deviate from upstream and hide the boundary artifacts this run exists to look
for.

```
duration        626.5 s (10.4 min), 97 chunks, 96 stitch points
generation      125.5 s wall  ->  RTF 0.20  (5.0x realtime)
+ worker boot    18.1 s over 15 processes; 157.1 s driver wall end to end
engine calls    23 963      peak RSS 3.0 GB per worker
full-passage WER 2.53 %  (55 / 2174 words)
```

The RTF is indicative, not a benchmark: one untuned fp32 Mac-GPU run, with per-call
Python async overhead inside the AR loop and an editor plus a browser resident. The
speaking rate is **209 wpm**, noticeably faster than the 150-160 wpm of typical audiobook
narration — 1600 words came out at 7.6 minutes, which is why the passage had to be
extended to 2183 words to reach ten.

**Drift: none.** WER by position decile — 1.00, 2.31, 1.48, 4.71, 0.00, 2.38, 2.80, 0.43,
3.77, 2.42 % — is flat and non-monotonic; the linear trend is +0.09 pp per decile,
+0.8 pp over the whole ten minutes, against a decile-to-decile spread of 4.7 pp. The
Mimi state feeding back at 12.5 Hz for 7 800 consecutive frames does not accumulate
audible error at fp32.

**Stitching: clean.** The largest sample-to-sample step across any of the 96 joins is
**0.0246**, against an in-audio 99.9th percentile of 0.1878 and a maximum of 0.4800 —
i.e. every join is an order of magnitude gentler than ordinary signal motion, and **zero
joins exceed the in-audio p99.9**. The reason is structural: `frames_after_eos` leaves a
tail of near-silence, so 92 of 96 joins have silence on at least one side (median join
window RMS 0.0010 before / 0.0046 after, against a speech-frame median of 0.1294). The
four speech-on-speech joins jump ≤ 13.3 dB, which is ordinary sentence-onset dynamics.
**No crossfade is needed.**

Artifact: `artifacts/longform_moby_10min.wav` (gitignored).

### 16.6 Every clip over 10 % WER or flagged, verbatim

Read this table with the failure taxonomy in mind — the great majority are **scoring
artifacts, not model failures**: the TTS correctly expands `Dr.`→"Doctor", `St.`→"Saint",
`Capt.`→"Captain" while the reference keeps the abbreviation (the whole `abbr` bucket);
LibriSpeech references carry their own defects (`mouthwhat`, `to morrow`, `'mong`); the
invented `fantasy` names are audibly right but unspellable by an ASR; and
British/American spelling (`honor`/`honour`) and compounding (`schoolboy`/`school boy`)
each cost a substitution. The 26 `chunk_over_max` rows are excluded here — they are a
chunker property, listed in §16.3, and score 1.61 % WER.

| WER | voice | bucket | dur | flags | reference / transcript |
|---:|---|---|---:|---|---|
| 320% | cosette | s | 1.0s | - | `So it has always been`<br>→ `There has always been a lot of people who are not going to be able to do it.` |
| 100% | cosette | s | 1.0s | - | `of stock fish.`<br>→ `The stockfish` |
| 100% | marius | s | 1.4s | low_level,asr_unreliable | `Indeed ah`<br>→ `*(empty)*` |
| 100% | eponine | s | 1.2s | asr_unreliable | `Aw shucks`<br>→ `*(empty)*` |
| 75% | fantine | s | 1.8s | - | `Don't begin said annie`<br>→ `That begins that Annie.` |
| 73% | fantine | abbr | 4.4s | - | `Dr. Watson met Mr. Holmes at 221B Baker St. on Tuesday.`<br>→ `Doctor Watson met Mr. Holmes at two hundred and twenty one B Baker Saint on Tuesday.` |
| 67% | jean | s | 1.1s | - | `Luff, luff a point!`<br>→ `Love for point, love for point, love a point.` |
| 50% | eponine | fantasy | 3.6s | - | `Xiuhtecuhtli Anaxagorou spoke the ninth syllable.`<br>→ `Shirtikutli and Exegoru spoke the ninth syllable.` |
| 50% | javert | fantasy | 4.6s | - | `The Thaumaturge Ysolde Vennarion bound the Skarn to her will.`<br>→ `The Thaumaturge, wise old Evanarian, bound the scar into her will.` |
| 50% | jean | short | 1.0s | - | `Not yet.`<br>→ `My yet not yet` |
| 50% | alba | s | 2.2s | - | `S'pose he opened his mouthwhat then`<br>→ `Suppose he opened his mouth, what then?` |
| 44% | azelma | abbr | 5.3s | - | `Capt. Ahab vs. the whale, Vol. II, ch. 4.`<br>→ `Captain Ahab versus the Whale, Volume 2, Ch four.` |
| 40% | cosette | s | 1.7s | - | `At another time harald asked`<br>→ `But at another time, Harold asked.` |
| 40% | marius | s | 1.7s | - | `The horizon seems extremely distant`<br>→ `Coaxin seems extremely distant.` |
| 40% | javert | fantasy | 4.4s | - | `Kaelthorn Vraskyr rode from Ilmenwyth to the gates of Zharudan.`<br>→ `Caelthorne Vraskir rode from Ilmenwith to the gates of Jarudan.` |
| 40% | alba | s | 1.1s | - | `Well they all do jim`<br>→ `Will they all do gym?` |
| 38% | alba | fantasy | 3.0s | - | `Orithane and Belzuvath argued over the Sunder Stone.`<br>→ `Orathane and Belzuvath argued over the Thunderstone.` |
| 33% | cosette | s | 1.2s | - | `Fathom six feet`<br>→ `Atom six feet.` |
| 33% | alba | short | 1.0s | - | `Who is there?`<br>→ `How is there?` |
| 33% | jean | acro | 3.8s | - | `IEEE 802.11 is a Wi-Fi standard.`<br>→ `IEE 802 11 is a Wi-Fi standard.` |
| 33% | jean | short | 1.0s | - | `Call me Ishmael.`<br>→ `Call me Israel.` |
| 33% | eponine | s | 2.5s | - | `Hello stephanos here comes the dedalus`<br>→ `Hello, Stefanos, here comes the Daedalus.` |
| 33% | azelma | s | 3.0s | - | `God goes 'mong the worlds blackberrying.`<br>→ `God goes among the world's blackberrying.` |
| 33% | jean | s | 1.8s | - | `That hazard shall not be thine.`<br>→ `The hacker shall not be thine.` |
| 33% | alba | s | 1.2s | - | `What does it mean cried one`<br>→ `What does it mean for anyone?` |
| 33% | javert | s | 2.1s | - | `Gobey's crab stew`<br>→ `Goby's crab stew.` |
| 30% | marius | m | 2.6s | - | `Yes, now that I think of it, here's his bedfellow!`<br>→ `Yes, now that I think of it, here's this bed fellow.` |
| 29% | fantine | m | 2.0s | - | `I will come to morrow said margaret`<br>→ `I will come tomorrow, said Margaret.` |
| 29% | alba | abbr | 5.0s | - | `Prof. Adams, Ph.D., wrote the foreword, etc.`<br>→ `Prof, Adams, Ph., D, wrote the foreword, etc.` |
| 27% | javert | m | 6.0s | - | `Start her, now; give 'em the long and strong stroke, Tashtego.`<br>→ `Start her now. Give him the long and strong stroke, Tash Tago.` |
| 27% | marius | m | 2.6s | - | `It spiralizes in ye; forks out at the serpent-snapping eye.`<br>→ `Spiralizes and he forks out at the serpent-snapping eye.` |
| 25% | azelma | s | 2.4s | - | `Anders face grew red`<br>→ `Ander's face grew red.` |
| 25% | alba | s | 1.4s | - | `God bless ye, men.`<br>→ `God bless you, men.` |
| 25% | marius | s | 1.7s | - | `The whale, the whale!`<br>→ `Whale, the whale.` |
| 25% | eponine | s | 1.8s | - | `Ah, poor Hay-Seed!`<br>→ `Ah, poor hay feed.` |
| 25% | eponine | fantasy | 3.7s | - | `Beyond Quel'Doreth lies the drowned city of Nyxhavel.`<br>→ `Beyond Queldoreth lies the drowned city of Nixavel.` |
| 25% | marius | fantasy | 3.6s | - | `The Aeltherim call it Vaskirion; the Drenn call it the Long Dark.`<br>→ `The Elfrim call it Baskirian, the Dren call it the long dark.` |
| 22% | marius | m | 1.9s | - | `About nine thirty a m i was shot down`<br>→ `About 9:30 and I was shot down.` |
| 22% | marius | num | 1.8s | - | `Room 101 is on the 4th floor.`<br>→ `Room 01 is on the fourth floor.` |
| 21% | javert | l | 7.4s | - | `At a street corner a black eyed school boy was parting from a rosy faced school girl whose music roll he was reluctantly surrendering`<br>→ `At a street corner a black-eyed schoolboy was parting from a rosy-faced schoolgirl whose music role he was reluctantly surrendering.` |
| 20% | jean | m | 4.8s | - | `The whole calamity, with the falling form of Macey, was plainly descried from the ship.`<br>→ `The whole calamity with the fallen form of Macy was plainly described from the ship.` |
| 20% | eponine | s | 2.4s | - | `I say, Quohog, blast ye!`<br>→ `I say Quahog blast ye.` |
| 20% | eponine | s | 2.2s | - | `That is well father fauvent`<br>→ `That is well, Father Favant.` |
| 18% | fantine | mixed | 5.0s | - | `The file (version 2.1, dated Jan. 3rd) is attached.`<br>→ `The file, version two point one, dated January, is attached.` |
| 18% | marius | m | 2.9s | - | `He knew the silver fleece his and zora's must be ruined`<br>→ `We knew the silver fleece is, and Zora's must be ruined.` |
| 18% | eponine | mixed | 4.2s | - | `Re-entering, he re-read the co-operative's pre-arranged agreement.`<br>→ `Re-entering, he re-read the co-operative's prearranged agreement.` |
| 18% | alba | abbr | 3.8s | - | `St. Mary's Hospital is on Elm Ave. near the Jr. college.`<br>→ `Saint Mary's Hospital is on Elm Ave, near the Junior College.` |
| 18% | jean | m | 2.6s | - | `That is the way i get to the roths answered polly`<br>→ `That is the way I get to the Roths Answer Poly.` |
| 18% | eponine | l | 5.2s | - | `Thus it is that the honor of three is saved our country's our master's and our own`<br>→ `Thus it is that the honour of three is saved our countries our masters and our own.` |
| 17% | javert | s | 2.3s | - | `Hullo he said who are you`<br>→ `Hello, he said. Who are you?` |
| 17% | jean | s | 2.1s | - | `Whelped somewhere by the sharkish sea.`<br>→ `Welcome somewhere by the Sharkish Sea` |
| 17% | eponine | l | 5.5s | - | `When he get home he try an try to brush that soot off but it done get into the skin an it stay there`<br>→ `When he gets home, he try and try to brush that soot off, but it doesn't get into the skin and it stay there.` |
| 17% | fantine | num | 4.3s | - | `He paid 99 cents for the first one and $19.99 for the second.`<br>→ `He paid ninety-nine cents for the first one and nineteen dollars ninety-nine for the second.` |
| 17% | fantine | s | 3.6s | - | `Beware of the horrible plague!" "Gabriel!`<br>→ `Beware of the horrible plague. Gabrielle` |
| 17% | alba | s | 2.1s | - | `Suppose i try said mister hale`<br>→ `Suppose I try said Mr. Hale.` |
| 15% | cosette | m | 4.2s | - | `Missus thornton the only mother he has i believe said mister hale quietly`<br>→ `Mrs. Thornton, the only mother he has, I believe, said Mr. Hale quietly.` |
| 15% | marius | m | 2.2s | short_audio | `I saw your sign and i know a boy who needs the job`<br>→ `It's all your sign, and I know a boy who needs the job.` |
| 14% | javert | m | 4.5s | - | `Hussey." And so it turned out; Mr.`<br>→ `Hossie. And so it turned out. Mr.` |
| 14% | javert | m | 2.9s | - | `That is going too far replied hermon`<br>→ `That is going too far, replied Herman.` |
| 14% | fantine | fantasy | 3.0s | - | `Grimwald Thistlebottom of Underhollow refused the summons.`<br>→ `Grimwald Thistlebottom of Underholler refused the summons.` |
| 14% | javert | m | 3.2s | - | `Do you son was the smiling rejoinder`<br>→ `To you, son, was the smiling rejoinder.` |
| 14% | alba | acro | 2.7s | - | `WER is measured by ASR after TTS.`<br>→ `Wr is measured by ASR after TTS.` |
| 14% | jean | m | 1.9s | - | `The middle ribs were the most arched.`<br>→ `The middle rims were the most arched.` |
| 14% | marius | m | 3.5s | - | `Once fairly a wing however he wheeled and made back hurriedly for his perch`<br>→ `Once fairly awing, however, he wheeled and made back hurriedly for his perch.` |
| 14% | azelma | mixed | 3.0s | - | `It's the crew's boat, not the captain's.`<br>→ `It's the cruise boat, not the captain's.` |
| 14% | fantine | m | 2.0s | - | `Cost em more but they'd be respectable`<br>→ `Cost them more, but they'd be respectable.` |
| 14% | eponine | m | 3.2s | - | `The landowner chuckled under his white mustaches`<br>→ `The landowner chuckled under his white moustaches.` |
| 14% | alba | l | 5.4s | - | `Now the Cinque Ports are partially or somehow under the jurisdiction of a sort of policeman or beadle, called a Lord Warden.`<br>→ `Now the Sinky ports are partially or somehow under the jurisdiction of a sort of policeman or beetle called the Lord Warden.` |
| 13% | javert | l | 15.6s | - | `Did you fixedly gaze, too, upon that ribbed and dented brow; there also, you would see still stranger foot-prints - the foot-prints of his one unsleeping, ever-pacing thought.`<br>→ `Did you fixedly gaze, too, upon that ribbed and dented brow? There also you would see still stranger footprints, the footprints of his one unsleeping, ever-pacing thought.` |
| 13% | marius | l | 11.7s | - | `Do not charge a fee for access to, viewing, displaying, performing, copying or distributing any Project Gutenberg works unless you comply with paragraph 1.E.8 or 1.E.9.`<br>→ `Do not charge a fee for access to viewing, displaying, performing, copying, or distributing any Project Gutenberg works unless you comply with paragraph 1. E 8 or 1 E 9` |
| 12% | azelma | m | 2.6s | - | `I can't you don't want him do you`<br>→ `I can you don't want him, do you?` |
| 12% | javert | num | 3.1s | - | `Chapter 12 begins on page 308.`<br>→ `Chapter twelve begins on page three oh eight` |
| 12% | jean | l | 9.0s | - | `But what's that he says now - hist!" "I look, you look, he looks; we look, ye look, they look." "Why, he's getting it by heart - hist!`<br>→ `But what's that? he says now, hissed. I look, you look, he looks, we look, you look, they look why, he's getting it by heart, hissed.` |
| 11% | alba | m | 3.8s | - | `But, to this, Bishop Jebb's anticipative answer is ready.`<br>→ `But to this, Bishop Jeb's anticipative answer is ready.` |
| 11% | cosette | abbr | 2.8s | - | `The Rev. Mapple climbed the pulpit at 6 a.m.`<br>→ `The Reverend Mapple climbed the pulpit at six A.M.` |
| 0% | eponine | short | 0.7s | low_level | `Wait.`<br>→ `Wait. Wait. Wait.` |
| 0% | marius | m | 2.2s | low_level | `It would be a good thing to have two men for it`<br>→ `It would be a good thing to have two men for it.` |

### 16.7 Go / no-go

**Go, with two host-side conditions.** The port is faithful: on everything longer than a
short phrase it tracks upstream within measurement noise (2.22 % vs 1.89 % on matched
rows), and the production shape — ten minutes of continuous book prose — round-trips at
**2.53 % WER with no quality drift and no audible stitch artifacts**, at 5x realtime on
an untuned Mac GPU path. fp16 costs nothing measurable. No voice is broken, nothing
clips, nothing runs away. The one class of failure that would have been fatal in the
field was invisible to every per-graph gate and only fell out of a corpus sweep: a static
KV budget sized from a fixture rather than from the model's own generation bound, which
silently truncated any sentence past ~40 tokens. That is now derived, not guessed, and
re-gated. The two conditions before this is public: (a) the Swift host must enforce a
hard chunk-token cap of its own, because upstream's chunker overshoots to 121 tokens and
that took the cache to 494 of 512; (b) the eight shipped voices differ by 12 dB in output
level and need per-voice gain normalisation. The remaining soft spot — 1-6 word
utterances at ~19 % WER — is upstream's, reproduced at the same rate by pure PyTorch, and
irrelevant to a ten-minute recap; it would matter only if the product ever synthesises
one-word interjections, and `pad_with_spaces_for_short_inputs` is the untested lever
there. Two caveats on the numbers themselves: WER is an intelligibility floor, not an
audio-quality measure, so a human listen to `artifacts/longform_moby_10min.wav` is still
the gate for prosody and naturalness; and every number here is Mac-side, so §15's
"re-run every gate on device" is unchanged.

### 16.8 Operational note — the IOSurface leak is worse than 8k calls here

§15 carried "~8k calls per process" from the ASR port. With this pipeline's state sizes
the ceiling is much lower: workers reliably died of
`Failed to allocate storage for NDArray … storageKind: ioSurface` at **~2 250 calls**,
with RSS climbing about 1.9 MB per call to ~3 GB. Every harness here therefore runs
generation in subprocesses with a **1 500-call budget** and a driver that respawns
(40 workers for the sentence sweep, 15 for the ten-minute run). Boot is cheap — 1.2 s
including all three assets, JIT-cached after the first load — so the restart tax was
18 s out of 157 s on the long-form run. This is a Python-bindings problem; whether the
native Swift path leaks the same way is still the open Phase C question, and it matters
more now that the measured ceiling is 2 k rather than 8 k calls.

## 17. Quiet-machine RTF baseline (2026-08-13, M4 Pro, everything closed)

Same 148-word paragraph, voice alba, load excluded, 1 warmup + 3 timed runs, sequential
stacks, fresh subprocess per run (`tools/bench/rtf_bench.py`; raw tables in
`tools/bench/results/`, gitignored).

| stack | device | median RTF | ×realtime |
|---|---|---:|---:|
| coreai fp32 | gpu | 0.1663 | 6.0× |
| coreai fp16 | gpu | 0.1562 | 6.4× |
| upstream PyTorch | mps | 0.2027 | 4.9× |
| pocket-tts-mlx 0.2.1 | metal | 0.0953 | 10.5× |

Reading: the port beats upstream PyTorch by ~1.25×, MLX beats the port by ~1.7×. Spread
within each stack < 4%. Ambient-vs-quiet delta was modest here (0.20 → 0.166), unlike the
ASR port's 2×. Caveats: MLX's chunker is a community reimplementation (not the same
splitter); speed bench only, no cross-stack parity claim. Honest card framing: faster than
upstream on the same Mac, behind Metal-native MLX; the port's case is the phone + the
Swift host, not Mac headline speed.

## 18. M1 — the production Swift host (2026-08-13)

`Package.swift` + `Sources/`: a macOS SwiftPM CLI (`swift run -c release pocket-tts-cli
--text ... --voice alba --out out.wav`) that runs the whole pipeline natively — **no
fixture-supplied inputs anywhere**. The Core AI plumbing (multifunction flow-LM over one
shared rank-5 KV state via `MutableViews`, the AR loop, EOS, the 12-tensor Mimi
round-trip) is ported from the device bench harness; the genuinely new pieces and their
verification:

| piece | implementation | verified how |
|---|---|---|
| sentencepiece | native unigram Viterbi + byte-fallback over `tokenizer.model` (hand-rolled protobuf; the model's normalizer is **identity** + dummy-prefix, asserted at load) | 26/26 encode strings vs Python `sentencepiece` (punctuation, numbers, apostrophes, unicode, emoji/CJK byte fallback); 52/52 decode incl. mid-span segments |
| chunker | `prepare_text_prompt` + `split_into_best_sentences` ported statement-for-statement, then **hard enforcement** of `voice_pos + n_text + max_gen_len(n_text) <= 512` (whitespace hard-split; upstream only warns) + a runtime overflow `precondition` in the AR loop | chunk-for-chunk identical to upstream on 4 corpus texts; the one divergence is a 51-token upstream chunk (over its own cap) that we split |
| noise | torch MT19937 + `normal_fill` Box–Muller, seeded `seed + call_index` per the oracle protocol | float-exact (9 sig digits) vs `torch.Generator.normal_` across 3 seeds × 4 call indices |
| voice conditioning | `embeddings/<voice>.safetensors` loaded + permuted natively; `voice_pos` from the file's own offset (126–162, never hardcoded) | gate (a) framing identical to oracle |
| host constants | LUT / `emb_std` / `emb_mean` / quantizer read from `model.safetensors` (BF16 widened, exact) | gate (a) |
| gain | per-voice normalisation to RMS 0.10 from the §16 sweep table, peak-clamped at 0.99, `--no-gain` to disable | cosette ×3.33, alba ×0.81; parity runs use `--no-gain` |

**Gate (a) — oracle parity (fp32, gpu, oracle seed/noise): PASS.**
Framing identical (14 tokens → 32 steps, EOS at 28, 31 frames, 59 520 samples);
wav vs `oracle/orc_a.wav` **cos 1.000000, max|Δ| 1e-4** (gate ≥ 0.9999).

**Gate (b) — ASR round trip (parakeet-swift): PASS.**
Oracle prompt **0.00 %** (0/9); fresh 20-word sentence **0.00 %** (0/20); the 148-word
bench paragraph, 7 chunks, **1.38 %** (2/145). (A first fresh sentence scored 9.52 %
purely from an ASR compound-join — "lighthouse keeper" → "lighthousekeeper".)
fp16 (`--dtype float16`, not the gated config): identical framing, wav cos 0.998,
ASR 0.00 %.

**RTF, same 148-word paragraph as §17, alba, gpu, 1 warmup + 3 timed** (ambient
machine, not the §17 quiet protocol): fp32 median **0.1384 (7.2× RT)** vs the Python
harness's 0.1663 (6.0×); fp16 **0.1259 (7.9×)** vs 0.1562 (6.4×). The Swift host is
~1.2× faster than the Python harness on the same assets — per-call async/bindings
overhead, not graph time. First-call specialization costs ~4.5 s across the four
functions; `--warmup` keeps it out of timed runs.

**The §16.8 leak question is answered for the native path: no leak.** One process ran
4 773 engine calls (2.1× the Python bindings' ~2 250-call IOSurface death ceiling) with
peak footprint flat at **300 MB** (Python workers climbed ~1.9 MB/call to 3 GB). The
worker-respawn machinery is a Python-bindings workaround only; the Swift host needs none.

**M2 remains:** in-graph Mimi state (`state_names=`, drop the 12-tensor round trip),
`T_PRE = 64` (a 50-token chunk costs 4 prefill calls today), folding the quantizer +
`emb_std/emb_mean` rescale into the Mimi graph (the last host-side tensor math),
device (iPhone) bring-up of this host, and the perf levers (§17's MLX gap).

## 19. M2 — graph consolidation (Mac-side)

### 19.1 Profile first: where the wall clock actually goes

The Swift host now separates engine-run time from every class of host glue (LUT
lookup, input NDArray marshaling, the rescale+quantizer matvec, output flattening;
the remainder is loop/async overhead). §17/§18 protocol: the 148-word paragraph,
alba, gpu, `--warmup`, 3 runs (medians below; spread < 1.5% on every row). fp32
framing: 7 chunks, 18 prefill windows, 531 steps, 524 Mimi frames, 1604 engine calls.

| component | fp32 ms | share | per call | fp16 ms | share |
|---|---:|---:|---:|---:|---:|
| flow-LM step (531 calls) | 3395 | 59.2 % | 6.4 ms | 3074 | 56.8 % |
| Mimi decode (524 calls) | 1565 | 27.3 % | 3.0 ms | 1606 | 29.7 % |
| flow decoder (531 calls) | 476 | 8.3 % | 0.9 ms | 446 | 8.3 % |
| flow-LM prefill (18 calls) | 123 | 2.1 % | 6.8 ms | 117 | 2.2 % |
| host: input marshaling | 130 | 2.3 % | — | 130 | 2.4 % |
| host: LUT + quantize + flatten | 10 | 0.2 % | — | 10 | 0.2 % |
| loop/async other | 34 | 0.6 % | — | 29 | 0.5 % |
| **wall** | **5731** | | | **5408** | |

(fp16 keeps the fp32 Mimi asset per §14, which is why its Mimi share is higher.)

### 19.2 What the profile does to the planned levers

* **T_PRE = 64 — SKIPPED.** Prefill is 2.1 % of the wall in total. Collapsing 18
  windowed calls to 7 (one per chunk) saves at most ~65 ms ≈ 1.1 % even assuming the
  64-wide call costs no more than the 16-wide one. Under the 2 % bar; not worth a
  flow-LM re-export. The windowed-prefill path stays (it is gated, and RNG/KV
  semantics are untouched).
* **Quantizer + rescale fold — not a perf lever (0.05 %), folded in anyway.** The
  matvec is 2.2 ms end to end. It rides along in the Mimi re-export that the state
  lever forces regardless, because it deletes the last host-side tensor math at zero
  marginal export cost.
* **Mimi in-graph state — the one material lever.** 27–30 % of the wall at 3.0
  ms/call, of which the 12-tensor round trip (≈ 4.5 MB of state IO per call, two
  1.1 MB KV rings each way) is pure protocol overhead. Proceeding.
* **Host marshaling (2.3 %) — left alone.** Shrinking it means reusing input
  NDArrays across async engine calls; ≤ 1–2 % upside against a real aliasing-bug
  risk class. Same skip logic as T_PRE.

### 19.3 The two Mimi levers — shipped, gated, numbers

Two staged export variants of the one graph, each gated before the next landed;
the M1 assets are untouched on disk and the host cut over per commit.

**`_q` — rescale + quantizer folded in-graph** (`--fold-quantizer`). The graph
takes the raw `[1,32]` flow-decoder latent; `latent*emb_std+emb_mean` and the
bias-free k=1 `output_proj` (asserted) run in-graph, verified against the
oracle's captured `mimi/in` (max|Δ| 2.86e−6) before export. The host's last
tensor math is gone. Perf-neutral as predicted; taken for the device story.

**`_q_gs` — the 12 streaming-state tensors become in-graph Core AI state**
(`--graph-state`, `state_names=` + `remove_functionalization`, same mechanics as
the flow-LM KV). The forward computes identical new values and commits them with
in-place `copy_` on the state inputs — torch.export records them as user-input
mutations, and plain `copy_` survives the pipeline (no `mutable_slice_update`
needed for full-tensor writes). The host owns the 12 buffers for the whole run
and hands `MutableViews` per frame; nothing round-trips, and the never-reset-
across-chunks invariant is now structural (the buffers simply persist).

| gate (fp32 unless stated) | `_q` | `_q_gs` |
|---|---|---|
| eager rewrite vs upstream streaming, 31 frames | cos 1.000000 / min 0.999999, max&#124;Δ&#124; 0.0 | same |
| engine `cpu_only` | cos 1.000000 / 0.999999, max&#124;Δ&#124; 0.0 | cos 1.000000 / 0.999999, max&#124;Δ&#124; 0.0 |
| engine `gpu` | cos 1.000000 / 1.000000, max&#124;Δ&#124; 0.0 | cos 1.000000 / 1.000000, max&#124;Δ&#124; 0.0 |
| e2e oracle prompt vs `orc_a.wav` | cos 1.000000, max&#124;Δ&#124; 1e−4, framing identical (14 tok → 32 steps, EOS 28, 31 frames, 59 520 samples) | same |
| ASR round trip, oracle prompt | 0.00 % (0/9) | 0.00 % (0/9) |
| ASR, 148-word paragraph | 1.38 % (2/145) | 1.38 % (2/145) |
| **multi-chunk state persistence**, 7 chunks / 524 frames | — | vs the `_q` round-trip wav: **cos 1.000000, max&#124;Δ&#124; 0.0000** (bit-identical); vs an independent Python-harness e2e on the M1 assets: max&#124;Δ&#124; 3e−6 over 1 006 080 samples |

The state-persistence row is the gate §16's lesson demanded: it is only visible
end to end, and it proves the in-graph state advances across chunk boundaries
exactly as the round-tripped state did.

fp16 (flow-LM + flow decoder fp16, Mimi fp32 as always): framing identical,
wav vs oracle cos 0.998 (M1's own fp16 number), ASR 0.00 % — unchanged.

### 19.4 Re-bench — §17/§18 protocol (148-word paragraph, alba, gpu, 1 warmup + 3 timed)

Ambient machine, same as §18's rows (not the §17 quiet protocol). Medians of 3;
spread < 1.3 %.

| config | M1 (§18) | M2 | Δ wall | ×realtime |
|---|---:|---:|---:|---:|
| fp32 | 0.1384 (7.2×) | **0.1281 (7.8×)** | −7.4 % | 7.8× |
| fp16 | 0.1259 (7.9×) | **0.1171 (8.5×)** | −7.0 % | 8.5× |

(This session's re-measured M1 baselines were 0.1368 / 0.1271 — §18's numbers
were slightly ambient-pessimistic; the Δ against those is −6.4 % / −7.9 %.)

Where it came from: Mimi engine time 1565 → 1189 ms (2.99 → 2.27 ms/frame,
−24 % per frame) — entirely the in-graph state; the quantizer fold was
perf-neutral as profiled. Against MLX's 10.5× (§17, RTF 0.0953): the fp16 gap
narrows from 1.32× to **1.23×**, with the remaining wall now 58 % flow-LM step —
i.e. the gap left is per-step transformer time, not protocol overhead, and the
next lever class (if ever needed) is inside the step graph, not around it.

**M2 Mac-side is done.** Still open for device bring-up (§15): run every gate on
the iPhone (needs the phone), re-measure compute units there, and the ANE
question stays parked behind the confirmed specialization-abort bug.

## 20. M3 — the production host on the iPhone (2026-08-13)

The device harness (`pocket-tts-ios-bench`) is now a thin shell around **PocketTTSKit**,
consumed as a local SwiftPM package (relative path, `../pocket-tts-swift`); the
fixture-driven Phase C harness is deleted. The phone runs the production pipeline
verbatim — native tokenizer → chunker → AR loop → in-graph-state Mimi → wav, no
fixtures in the loop — against the M2 `_q_gs` s512 assets.

Device: iPhone 17 Pro Max (`iPhone18,2`, A19 Pro), iOS 27.0, Release build, JIT
`.aimodel`. Power state during every run below: **charging (100 %)**, thermal
**nominal before and after** every run. Candidate-4 discipline: every fp32 load states an explicit `gpu` or
`cpuOnly` preference — since M2 *both* main graphs carry in-graph state, and fp32 +
{`.default`, `.neuralEngine`} is an uncatchable specialization abort on this device.
The harness refuses those pairs in code rather than by convention.

### 20.1 Gate stack (fp32 = parity config), all PASS

| gate | result |
|---|---|
| per-graph exact transfer, `cpuOnly`, device vs M4 Pro (`GraphProbe`, identical kit code both sides) | all 8 dumps **bit-identical** (max&#124;Δ&#124; = 0): prefill cond, step cond + eos, flow latent, 3-frame Mimi pcm, post-run `kv0`/`up_p`/`offset` state |
| e2e oracle prompt, gpu, vs `oracle/orc_a.wav` | framing identical (14 tok → 32 steps, EOS 28, 31 frames, 59 520 samples); **cos 1.000000, max&#124;Δ&#124; 4.3e−5** (the Mac's own gate-a run is 1e−4) |
| ASR round trip (Mac-side parakeet-swift) | oracle prompt **0.00 %** (0/9); 148-word §17 paragraph **1.38 %** (2/145) — the Mac's exact number |
| fp16 (not the gated config) | framing identical, ASR 0.00 % / 1.38 %; wav vs oracle cos **0.998974** (Mac fp16: 0.998). fp16 does **not** transfer bit-wise across A19 Pro/M4 Pro — expected, documented, not chased |

One honest note from the long run: gpu fp32 is *not* bit-transferable across GPU
architectures either — the 46.8 s free run lands at cos 0.9997 / max|Δ| 0.38 vs the
Mac's gpu wav, with framing chunk-for-chunk identical and identical WER. The transfer
property belongs to `cpuOnly` (bit-exact, gate above); gpu correctness is carried by
the oracle-prompt cos + framing + ASR gates, same as on the Mac.

### 20.2 s512 truncation confirm-run — CLOSED

The exact Phase C text that truncated at s256 (the 163-word "long" fixture text, 47 s,
whose chunks overflowed at rows 82/86 past S_MAX=256): regenerated on device, fp32 gpu,
through the enforcing chunker + s512 assets. **Zero cache overflows** (the AR-loop
`precondition` is armed and never fired; min headroom 338 of 512), **EOS fired in all
8 chunks** (steps 58–99, none hit `max_gen_len`), framing chunk-for-chunk identical to
the Mac run of the same text, 46.800 s audio. WER **2.45 % (4/163) — exactly the Mac's**
on the same ungained config (a gained Mac wav scores 1.84 %; the delta is ASR
number-normalization noise, not audio). The truncation issue is closed.

### 20.3 Device perf (zoo-card rows) — Release, 148-word paragraph, alba, 1 warmup + 3 timed

Power **charging (100 %)**, thermal **nominal → nominal** on every row, model load
excluded (reported separately). Peak RSS is the task ledger peak.

| config | RTF median (runs) | ×RT | peak RSS | load | vs M4 Pro (M2) |
|---|---|---:|---:|---:|---|
| fp32 gpu | **0.1646** (.1647/.1646/.1638) | **6.1×** | 202 MB | 0.7 s | 0.78× of 0.1281 |
| fp16 gpu | **0.1281** (.1281/.1276/.1355) | **7.8×** | 169 MB | 0.4 s | 0.91× of 0.1171 |
| fp32 cpuOnly (spot check) | 0.5324 (1 run) | 1.9× | 225 MB | 0.02 s | — |

Compute-unit spot check: gpu is **3.2×** faster than cpuOnly on device (measured, not
assumed from the Mac). Warmup (first-call specialization) is ~0.3 s on device across
the four functions vs ~4.5 s on the Mac. The phone at fp16 matches the M4 Pro's M1-era
throughput and holds 7.8× real time on a 42 s paragraph with a 169 MB peak footprint —
the device story the port exists for. ANE remains parked behind candidate 4.

**M3 is done.** The Swift host, the assets, and every gate now hold on the target
hardware. Remaining before the zoo PR: the writeup/card itself (numbers above), and the
candidate-4 minimal repro + Mac check before filing that report.

## 21. Same-device comparable: FluidAudio Core ML route (2026-08-13)

FluidInference `pocket-tts-coreml` through their own FluidAudio SDK 0.15.5, same iPhone
17 Pro Max (iOS 27.0, charging, thermal nominal), default configuration (their real
shipping path: fp16, gpu placement for cond/flowlm/flow_decoder, Mimi pinned cpuOnly by
their loader), ~150-word Moby Dick ch.1 passage via their whole-utterance API, 1 warmup +
3 timed.

| route | RTF median | ×realtime |
|---|---:|---:|
| this port, fp16 gpu (§20) | 0.128 | 7.8× |
| this port, fp32 gpu (§20) | 0.165 | 6.1× |
| FluidAudio pocket-tts-coreml, defaults | 0.399 | 2.51× |

Our fp16 path is ~3.1× faster on the same hardware. Fairness notes: passage is
length-matched but not byte-identical to the §17 paragraph (the harness holding it was
frozen during this run); their internal ≤50-token chunking happens inside the single
timed call, no external chunker imposed. Their per-run spread: 2.53×/2.51×/2.43×.

MLX on iOS: no MLX-Swift port of pocket-tts exists (`pocket-tts-mlx` is Python, Mac-only;
`mlx-audio`/`mlx-swift-audio` carry no pocket-tts/Mimi support). Core ML and Core AI are
the only phone routes for this model today.
