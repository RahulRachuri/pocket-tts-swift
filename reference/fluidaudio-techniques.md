# FluidAudio / mobius TTS conversion techniques — reference for pocket-tts → Core AI port

Research date: 2026-08-12. Repos cloned read-only into this scratchpad:
- `FluidAudio/` = https://github.com/FluidInference/FluidAudio @ 667181a (Swift runtime, ships to users)
- `mobius/` = https://github.com/FluidInference/mobius @ d2398af (Python conversion lab, PyTorch→CoreML, trial logs)

Convention: **[VERIFIED]** = read directly in cloned source/docs, file:line cited. **[CLAIMED]** = stated in
FluidAudio/mobius docs but not independently cross-checked against kyutai upstream or a running build.
**[CORRECTED]** = contradicts a premise in the task brief; the correction is itself [VERIFIED] against these repos.

---

## 1. PocketTTS port — does it exist, and precisely what is it

**[VERIFIED]** Yes. Three-repo split, matching mobius's own documented convention
(`mobius/Documentation/ModelConversion.md:7-21`):
- **Conversion (Python, PyTorch→CoreML):** `mobius/models/tts/pocket_tts/coreml/` — traceable wrappers,
  convert scripts, trial logs.
- **Model artifacts (HF):** `FluidInference/pocket-tts-coreml` (v2, v2.1, `.mlmodelc`/`.mlpackage`, `constants_bin/`).
- **Runtime (Swift, ships to users):** `FluidAudio/Sources/FluidAudio/TTS/PocketTTS/` — 18 files, ~2500 lines.

### Graph decomposition — three parallel placement strategies, not one

The most important structural finding: FluidAudio did **not** converge on a single graph split. They shipped
**three interchangeable placement configurations** of the same weights, selected by
`PocketTtsModelPlacement` **[VERIFIED]** `FluidAudio/Sources/FluidAudio/TTS/PocketTTS/Pipeline/PocketTtsModelStore.swift:36,48-64`:

| Placement | KV-cache representation | Where it lives | Status |
|---|---|---|---|
| `.gpu` (default) | Combined rank-5 `[2,1,512,16,64]` (K+V stacked), **host-side** `MLMultiArray`s, round-tripped in/out of every `predict()` call | `cond_prefill`/`flowlm_step` forced to GPU because the ANE compiler rejects the rank-5 scatter | Shipping default |
| `.ane` | Split rank-4 `[1,512,16,64]` K and V as **separate** host-side arrays (`k_cache{i}`/`v_cache{i}`) | `flowlm_step` reaches **100% ANE**; `cond_prefill_ane` stays `.all`→GPU (single fat call is faster there) | Shipping, opt-in |
| `.aneState` | True **in-graph CoreML `MLState`**, via a multifunction `pocket_state.mlmodelc` package exposing `prefill` and `generate` functions that share one state | One shared KV state, no host round-trip at all | Shipping, opt-in, requires macOS 15/iOS 18+ (`MLState` + multifunction) |

**[VERIFIED]** citations: rank-5 vs rank-4 cache struct — `PocketTtsSynthesizer+KVCache.swift:11-34`; placement
dispatch and measured per-model compute units — `PocketTtsModelStore.swift:98-165`; `.aneState` multifunction
loading (`functionName: "prefill"|"generate"`, `.cpuAndNeuralEngine`) — `PocketTtsModelStore.swift:217-251`;
`.aneState` session routing — `PocketTtsSynthesizer+State.swift:1-91`.

**[VERIFIED — mobius]** `.aneState` is called **"Trial 23"** in the conversion lab and its win is quantified:
"MLState pipeline Swift-wired 2026-06-10 (`.aneState`, FluidAudio feat/pocket-tts-mlstate): **−14.9% e2e
measured**, −1.66 ms/frame ... WER 0, bit-identical seeded WAVs, MLState×multifunction clean on macOS" —
`mobius/Documentation/ANE_Candidates.md:132`. The rank-4 split-KV win for `.ane` is dated to the same
campaign: "flowlm 0%→100% ANE, cond_prefill 0%→92%, bit-identical" (Trials 19/20) —
`mobius/Documentation/ANE_Candidates.md:19`.

**Gap noted:** `mobius/models/tts/pocket_tts/coreml/TRIALS.md` only documents Trials 1–15 (ends at the
"split-Mimi shelved" verdict). Trials 16–23 (rank-4 split-KV, MLState multifunction) are referenced by name
from `ANE_Candidates.md` and from Swift comments but their narrative log was not present in the cloned
commit — likely landed after this mobius snapshot. Treat the *existence and headline numbers* of Trials 19–23
as **[VERIFIED]** (present in `ANE_Candidates.md` and in shipped Swift code/comments) but the blow-by-blow
methodology as **[CLAIMED]**.

### Where KV cache "lives" — answer is placement-dependent

Not a single answer. `.gpu`/`.ane`: host-owns rank-5/rank-4 `MLMultiArray`s, explicitly copied in
(`addCacheInputs`) and read back out (`extractCacheOutputs`) every `predict()` call —
`PocketTtsSynthesizer+KVCache.swift:102-154`. `.aneState`: in-graph `MLState`, no host copy per step — this is
the closest existing analog to whatever ComputeStream/state primitive Core AI exposes.

### What exactly the ANE compiler rejected

**[VERIFIED]** Two independently-documented failure descriptions, consistent with each other:
- mobius trial log (early, `.gpu`-era): "MPSGraph internal assertion when ANE partitioner tries to fold prefill
  into an ANE block ... KV cache shape `(2, 1, 512, 16, 64)` — rank 5, leading `2` is k/v stack. ANE's compiler
  hits a rank-planner edge case." — `mobius/models/tts/pocket_tts/coreml/TRIALS.md:34-43` (this is `IOS_COREML_ISSUES.md:15,39` too, "tripped MPSGraph rank-5 / zero-shape assert").
- FluidAudio doc (later, v2.1-era), same root cause stated at the compiler-contract level: "their rank-5
  KV-cache `scatter` is rejected by the ANE compiler at *any* precision" —
  `FluidAudio/Documentation/TTS/PocketTTS.md:36-37`.

**The workaround that actually shipped is not "give up on ANE," it's "change the tensor rank."** Splitting the
combined `[2,1,L,16,64]` cache into two separate `[1,L,16,64]` K/V tensors (rank-4) made the *same* scatter
ANE-eligible (Trial 19/20, 0%→100%). This is the single most load-bearing technique in the whole corpus — see
§6.

**[VERIFIED — general rule, not model-specific]** mobius's cross-model playbook states this as a first
principle: "ANECCompile rejections are usually construct problems, not model problems. Rank-5 tensors →
split/reshape to rank-4. `scatter` → one-hot multiply-add (exact circular-buffer semantics). `masked_fill(-inf)`
→ additive mask. Runtime trig on big tensors → precompute or hoist." — `mobius/Documentation/ANE_Candidates.md:15-19`.

### Per-graph precision and compute units

**[VERIFIED]** `PocketTtsModelStore.swift:98-127` (comment block) + `PRECISION.md`:

| Graph | Precision | Compute units (measured fastest, M-series/macOS 26) |
|---|---|---|
| `cond_prefill` | fp16 | `.all` (compiles to GPU; rank-5 blocks ANE; 4.7ms vs 7.5ms `.cpuAndGPU`) |
| `flowlm_step` (default) | fp16 | `.all` → GPU, 3.4ms (rank-5 scatter blocks ANE) |
| `flowlm_stepv2` (opt-in) | **selective int8**: `attn{i}_in_proj`, `attn{i}_out_proj`, `linear{i}_1/2` only; everything else (LayerNorms, `input_linear`, `out_eos`) stays fp16 | same routing as `flowlm_step` |
| `flow_decoder_fused` | fp16 | `.all` → **100% ANE**, 1.09ms — the only model that reaches ANE in `.gpu` placement |
| `mimi_decoder` | fp16 (state) | **`.cpuOnly`**, 6.0ms — deliberately off ANE and off GPU |
| `.ane` placement: `flowlm_step_ane` | fp16 only (int8 not offered) | `.cpuAndNeuralEngine` → 100% ANE, 3.68ms (slower than GPU's 3.04ms — "the trade buys a GPU-free decode loop") |

Quantization rationale (`PRECISION.md:35-174`, **[VERIFIED]**): matches upstream's own selection
(`kyutai-labs/pocket-tts#147`, "Add int8 quantization for FlowLM transformer linears" — **[external, not
fetched]**), but the *mechanism* differs: upstream uses true dynamic int8 (int8×int8 GEMM via FBGEMM/torchao);
FluidAudio uses CoreML weight-only PTQ (`coremltools.optimize.torch.quantization.PostTrainingQuantizer`,
per-channel symmetric, `global_config=None` load-bearing) — weights stored int8, dequantized to fp16 at
inference, activations always fp16. Win is DRAM→SRAM bandwidth, not GEMM throughput, because "the ANE has no
exposed int8 GEMM kernel, its MAC array is fp16-native" (`PRECISION.md:122-128`).

### Euler steps — 8, hardcoded, fused

**[VERIFIED]** `PocketTtsSynthesizer+Flow.swift:6-18` (doc comment) and `PocketTtsModelStore.swift` confirm:
flow decoder runs **exactly 8 Euler steps**, fused into a single CoreML graph call in v2.1
(`flow_decoder_fused.mlpackage`) rather than the v2 approach of 8 separate `predict()` calls per frame. The
step count is baked in at conversion time and must match the Python conversion flag: "the `s`/`t` time
endpoints are baked in at conversion for the chosen step count, so `numSteps` here MUST match the value passed
to `convert_flow_decoder_fused.py --num-steps`" (`PocketTtsSynthesizer+Flow.swift:15-18`). Conversion-time
detail in `mobius/models/tts/pocket_tts/coreml/CONVERSION.md:217-220`: `s=i/8, t=(i+1)/8` per step,
`latent += velocity * (1/8)`.

**[CORRECTED]** The task brief's premise — "hard-code 8 Euler steps ... where upstream's default is 1-step
distilled (LSD) decode" — is **not supported by anything in these repos**, and appears to conflate two
different things. "LSD" in every occurrence found (`CONVERSION.md:126`,
`traceable_flow_decoder.py:1,10`) stands for **"Lagrangian Self-Distillation"**, a training-time distillation
scheme that lets a flow-matching decoder use *fewer* steps than a non-distilled model — it is not itself "the
1-step mode." The upstream kyutai API that these conversion scripts call is
`TTSModel.load_model(lsd_decode_steps=8)` (used consistently across all traceable/convert scripts, e.g.
`traceable_flow_decoder.py:62`, `convert_flowlm_step.py:96`) — i.e. **8 is the value FluidAudio explicitly
requested from upstream's own configurable parameter**, not a value they invented against upstream's default.
Nothing in mobius or FluidAudio mentions a "1-step" mode anywhere (grepped both repos, zero hits for
"1-step"/"one-step"). **Recommend verifying the actual kyutai `pocket-tts` upstream default directly** (not
fetched here — out of scope for a FluidAudio-only clone) before treating "upstream defaults to 1 step" as true
for the Core AI port; on the evidence here, 8 looks like a deliberate quality/speed tradeoff FluidAudio picked
using the model's own knob, and there's a documented history of a **correctness bug** in the two-time-input
`(s,t)` scheme (Trial 7, see §4) that would apply at any step count.

### Mimi decoder pinned to CPU

**[VERIFIED]**, extensively. Not "we didn't try" — **[VERIFIED]** three independently-diagnosed ANE failure
modes for `mimi_decoder`, all documented with root cause:
1. Zero-length tensor Espresso crash (`res*_conv1_prev: [1,C,0]` state tensors from `StreamingConv1d(kernel_size=1)`) — CPU/GPU Espresso backend crashes on zero-element blobs; ANE backend on macOS tolerated it. Fixed by MIL-level stripping + Swift zero-length skip.
2. **fp16 precision compounding → audible beeping** — the claim in the task brief, confirmed exactly:
   "23 streaming state tensors feed back every 80ms ... ANE's fp16 quantization perturbations (~1e-3 per frame)
   compound across 75 frames/sec into audible artifacts." Diagnosis ladder ruled out the model itself (bit-identical
   MIL-stripped vs original in Python) and isolated it to ANE dispatch specifically (Python CPU_AND_GPU: clean;
   Swift `.all` with ANE active: beeping). — `mobius/.../IOS_COREML_ISSUES.md:76-96,206-230`.
3. 64-byte stride misalignment segfault — ANE requires last-axis tile strides aligned to 64 bytes; Mimi's
   residual channel dims (32/64/128/256/512) sit exactly on that boundary and some intermediate transposed-conv
   outputs drop below it.
Also **[VERIFIED]** a *fourth*, separate attempt to split the transformer-half of Mimi onto ANE and leave SEANet
on GPU (Trial 15/16) was built, benchmarked with careful interleaved A/B methodology, and **regressed 13%
e2e** purely from cross-model dispatch/state-plumbing overhead, not precision — shelved, not shipped
(`TRIALS.md:382-517`).

### Noise/RNG handling

**[VERIFIED]** Host-side, not in-graph. `SeededRNG` (xoshiro256**, seeded via SplitMix64) generates Gaussian
noise via Box-Muller on the CPU in Swift; `flowDecode()` builds `latent_init = randn(32) * sqrt(temperature)`
and passes it as a graph **input** — `PocketTtsSynthesizer+Flow.swift:19-33,83-127`. The graph itself is
deterministic given `latent_init`; reproducibility is achieved entirely by seeding the host RNG
(`PocketTtsSynthesizer.swift:441`, `SeededRNG(seed:)`).

### NaN-as-BOS sentinel

**[VERIFIED]**, but placement-dependent — and this interacts directly with the ANE-rejection story. For the
rank-5 (`.gpu`, default) graph: first-step `sequence` input is filled with `Float.nan` across all 32 dims
(`createNaNSequence()`), and inside the traced PyTorch wrapper the model does
`sequence = torch.where(torch.isnan(sequence), bos_emb, sequence)` — `PocketTtsSynthesizer.swift:1249-1277`,
`traceable_flowlm_step.py:237-238`. For the rank-4 `.ane` graph, **the NaN-BOS protocol is dropped entirely**
and the BOS latent embedding is passed directly as the first-step `sequence`, because
**"the ANE mangles NaN inputs before `isnan` evaluates"** —
`PocketTtsSynthesizer.swift:1251-1254,545-548`. This is stated as a **general rule**, not a pocket-tts quirk,
in the cross-model playbook: "NaN inputs are mangled by the ANE before `isnan` evaluates — never use NaN
protocols in ANE-bound graphs." — `mobius/Documentation/ANE_Candidates.md:31-32`.

Inside attention itself, a *second* NaN-handling layer exists for the KV cache: unwritten cache slots must be
explicitly zero-filled (never NaN) because attention does `keys = torch.where(isnan(keys), 0, keys)` before
masking — `traceable_flowlm_step.py:180-183`; the rank-4 ANE allocator enforces zero-fill for exactly this
reason (`"The rank-4 _ane models REQUIRE zero-filled caches: their traces drop the rank-5 packs' NaN scrub"` —
`PocketTtsSynthesizer+KVCache.swift:38-40`).

### Text conditioning / sentence chunking / streaming state

**[VERIFIED]** SentencePiece tokenizer → `constants.textEmbedTable` lookup (numpy-style flat array indexed by
`id*dim`) → **voice-conditioning tokens first, then text tokens**, fed one-by-one (or in one shot via
`cond_prefill`) into the KV cache. Order is load-bearing and was itself a historical bug (Trial 11, §4).
Chunking: `chunkTextWithMetadata` splits at sentence boundaries first, then clause (`,;:`), then word, capped
at ≤50 tokens (512 KV slots − ~125 voice − ~25 overhead) — `PocketTtsSynthesizer.swift:856-1100`. Each chunk
gets a **fresh KV cache** (voice+text re-prefilled); Mimi's 23-tensor streaming state is **never reset**,
carried continuously across chunks and even across whole utterances in session mode — `PocketTtsSynthesizer+Mimi.swift:6-11`,
`PocketTTS.md:140-150`. A one-shot `cond_prefill` fast path exists (v2.1) that windows arbitrarily-long
conditioning blocks through a fixed `T_max` in a loop, carrying KV position across windows —
`PocketTtsSynthesizer+KVCache.swift:181-240`.

### Performance numbers **[VERIFIED — self-reported, not independently reproduced]**

- v2→v2.1 (rank-5/.gpu placement): **~905ms → ~452-520ms per utterance (~1.8× RTFx)**, "M-series/macOS 26" —
  `PocketTTS.md:20-47`. Note this is a **per-utterance wall-time ratio between two of FluidAudio's own builds**,
  not an absolute real-time factor (RTFx) against audio duration in the normal sense; treat the "~1.8x
  realtime" framing in the task brief as directionally right but imprecise — mobius's own vocabulary calls it
  "RTFx" loosely too (see cosyvoice3 section below for a graph where RTFx *is* audio-duration-normalized:
  0.33×, i.e. **slower** than real-time).
- `.aneState` over `.gpu`: **−14.9% e2e**, −1.66 ms/frame, WER 0, bit-identical seeded WAVs
  (`ANE_Candidates.md:132`).
- Cross-engine pipelining experiment (mimi CPU overlapped with flowlm GPU + flow ANE): **measured no win**
  on M5 Pro — serial `.ane` 1.108s vs pipelined 1.124s; serial `.gpu` ~1.33s vs pipelined ~1.33s. Shipped
  disabled (`useCrossEnginePipeline = false`) — `PocketTtsSynthesizer.swift:695-705`.
- Original ANE dispatch summary (pre-v2.1): only ~10% of total pipeline wall-time actually executes on ANE;
  ~90% on CPU/GPU, with `mimi_decoder` setting the architectural floor —
  `mobius/.../IOS_COREML_ISSUES.md:21-28`.

### Voice cloning

**[VERIFIED]** Not blanket-gated. A separate `mimi_encoder.mlmodelc` (language-agnostic, downloaded on first
`cloneVoice()` call) converts a 1-30s audio sample into conditioning embeddings; these feed the same
`cond_step`/`cond_prefill` path as shipped voices. **[VERIFIED — partial gate]** Live cloning is explicitly
**unsupported for 6-layer (non-English) packs**: "the upstream pocket-tts flow LM early-EOSes on the encoded
voice conditioning (confirmed against the PyTorch reference, #793)" — only 24-layer packs and English clone
reliably — `PocketTtsModelStore.swift:463-481`. So "voice cloning weights are gated" in the task brief is
**[CORRECTED]**: the *encoder itself* is open and shipped; what's actually restricted is architecture-specific
(6L packs don't clone well) plus a per-language "speaker projection" asset that's missing for some
non-English packs (§793).

---

## 2. Other TTS ports

### Kokoro (`KokoroAneManager`) — non-autoregressive, 7-stage graph split

**[VERIFIED]** `FluidAudio/Documentation/TTS/KokoroAne.md`. Not AR+vocoder shaped (Kokoro is duration-predictor
+ alignment + vocoder, no token-by-token loop) so less directly transferable to pocket-tts's shape, but still
useful as a **graph-decomposition and compute-unit-pinning precedent**: Albert/PostAlbert/Alignment/Vocoder on
`cpuAndNeuralEngine`, Prosody/Noise/Tail on `.all` — 4 ANE-resident stages, 3 GPU+CPU. Two OS-level ANE bugs
documented as non-model, non-version-gated regressions (§4). G2P is a **separate CoreML BART seq2seq model**
for English (text→IPA) and a **from-scratch rule-based Swift pipeline** for Mandarin (dict lookup + tone sandhi
+ Bopomofo, ~10MB of packaged phrase tables, zero network, modeled on `misaki`'s `zh_frontend.py`) —
`KokoroAne.md:165-197`. Relevant to pocket-tts comparison: pocket-tts explicitly has **no phoneme stage at
all** — raw SentencePiece text tokens go straight into the transformer, pronunciation lives entirely in
weights (`PocketTTS.md:249-286`) — so none of Kokoro's G2P infrastructure transfers to a pocket-tts-shaped port.

### CosyVoice3 (Mandarin) — **[VERIFIED] closest architectural analog to pocket-tts**

Not mentioned in the task brief but the single most relevant cross-port comparison found: Qwen2-0.5B AR LLM →
flow-matching (CFM) decoder → HiFT (iSTFT) vocoder. Same three-stage AR+flow+vocoder shape as pocket-tts.
`mobius/models/tts/cosyvoice3/coreml/REPORT.md`, `TRIALS_AND_ERRORS.md` (1157 lines).

- **Shipping KV cache is in-graph `MLState`, not host-side arrays** — the opposite default choice from
  pocket-tts's `.gpu` placement, and confirms `.aneState`/MLState is a *production-shipped* pattern here, not
  just an experimental branch: "`LLM-Decode-M768-fp16-stateful` ... Single-step AR decode against 768-slot KV
  cache held in `MLState` (48 per-layer buffers)", min OS macOS 15/iOS 18 — `REPORT.md:8-20`.
- **Static-shape KV cache**: `[24,1,2,768,64]` fp16 = `(layers, batch, num_kv_heads, max_ctx, head_dim)`, GQA
  with 2 KV heads/14 Q heads. Prefill (`T_pre=256` fixed) and decode are **separate mlpackages** deliberately,
  because "prefill is compute-bound over long context, decode is memory-bound over single-step — different
  optimal compute-unit assignment on ANE" — `TRIALS_AND_ERRORS.md:173-188`. This directly validates a
  cond_prefill/flowlm_step split shape.
- **fp16-safe mask sentinel**: `torch.tensor(float('-inf'))` overflows to NaN in fp16 softmax; fixed by using
  `-1e4` instead — `TRIALS_AND_ERRORS.md:189-195`.
- **fp16 RMSNorm overflow** on Qwen2 activation outliers; fixed with selective fp32 pinning of
  `{pow, reduce_mean, rsqrt, softmax}` via `ct.transform.FP16ComputePrecision(op_selector=...)` —
  `TRIALS_AND_ERRORS.md:196-218`.
- **Flow DiT (22 blocks, CFM/flow-matching, exactly the pocket-tts-decoder's genre) fp16 = catastrophic NaN**,
  not mild drift. Root cause: a **fused `layer_norm` MIL op** cannot be reached by op-type-based fp32 pinning
  because its internal scalar ops aren't decomposed in the graph; adding `layer_norm`/`gelu` to the fp32-pin
  set still didn't fix it in one attempt, then later did in another (root cause of the discrepancy left
  unresolved in the log) — `TRIALS_AND_ERRORS.md:224-308`.
- **Flow-graph ANE port attempted and reverted**: a BC1S-layout rewrite of the Flow DiT compiled clean, ran
  ~3× faster, passed a "no NaN" gate, but **collapsed the mel dynamic range** (audio unintelligible to ASR)
  despite no NaN — a silent precision failure distinct from the loud NaN failures elsewhere. Hypothesis:
  precision loss in AdaLN `(1+scale)*norm` or manual-SDPA softmax compounds across 22 blocks × 10 Euler steps
  × CFG batch=2 as progressive magnitude attenuation — `REPORT.md:87-120`, `TRIALS_AND_ERRORS.md:1117-1143`.
  **This is the concrete evidence for "BC1S-style layout" in this codebase** — it exists, but as a reverted
  experiment, not a shipped pattern.
- **`MLMultiArray` stride padding bug** (Swift-side, general — not CosyVoice3-specific): CoreML pads array
  dimensions for alignment (e.g. `[1,80,500]` fp32 physically strided `[40960,512,1]` — time dim padded
  500→512, "likely 64-byte/SIMD alignment"). Raw `dataPointer` linear reads silently read padding bytes.
  **Fix: stride-aware accessors everywhere an MLMultiArray crosses the Swift boundary.** —
  `TRIALS_AND_ERRORS.md:407-424`. This is a strong candidate to check against Core AI's tensor/array bridging
  API — if it uses a similar dense-buffer-with-padding representation, the same bug class will recur.
- Performance: RTFx ~0.33× (i.e. slower than real-time) on the shipping fixture, Flow stage dominates at ~65%
  of total synth time — `REPORT.md:35-49`. Useful contrast to pocket-tts's ~1.8×-ish claim: architecture and
  step count (22 DiT blocks × 2 CFG × N Euler steps vs pocket-tts's tiny 32-dim MLP-AdaLN × 8 steps) matter far
  more than any ANE trick.

### Other TTS entries in mobius (not deep-dived, noted for completeness)

`models/tts/`: `kittentts`, `magpie`, `qwen3`, `styletts2` (also shipped in FluidAudio as `StyleTTS2Manager`),
`supertonic-3` (shipped as `Supertonic3`), `voxcpm-1.5`. None of these were read in depth for this memo;
`supertonic-3/coreml/trials.md` independently documents a fused `VectorEstimator` loop (same
"fuse-the-Euler-loop" pattern as pocket-tts's `flow_decoder_fused`) that was **declined** — 99.5% ANE but only
2-3ms/chunk saved, and parity regressed 30× outside tolerance because "the host loop's per-step fp32 IO casts
act as error-containment barriers for the precision-sensitive LSD denoiser" (`ANE_Candidates.md:116-125`) —
i.e. **the exact inverse lesson from pocket-tts's flow decoder, where fusing 8 Euler steps into one graph call
worked cleanly.** The playbook's own gloss: "loop fusion is NOT free when the looped graph is
precision-sensitive... (Contrast: PocketTTS flow decoder fused cleanly because its Euler updates tolerate
fp16.)" (`ANE_Candidates.md:123-125`). **This is a direct, load-bearing warning for the Core AI flow-decoder
graph**: whether fusing your Euler loop is safe is not generic to "flow decoders," it depends on that
specific decoder's numerical sensitivity — test both fused and per-step, don't assume.

---

## 3. Cross-port shared techniques (mobius `ANE_Candidates.md` playbook, `ModelConversion.md`)

**[VERIFIED]** These are stated explicitly as a *transferable playbook*, distilled across all ports:

1. **Rank-5 → rank-4 is the standard fix for ANE scatter/gather rejections**, proven on pocket-tts's KV cache
   (Trials 19/20). General form: "`scatter` → one-hot multiply-add (exact circular-buffer semantics)."
2. **`masked_fill(-inf)` → additive mask.** Both pocket-tts's manual attention (`traceable_flowlm_step.py:204`
   uses `masked_fill(~attn_mask, float("-inf"))`, fp32 internally though — a fp16-safe additive mask would be
   the ANE-ready form) and CosyVoice3's `-1e4` sentinel fix are instances of this.
3. **Attention/SDPA for ANE eligibility: manual QKV + explicit matmul/softmax, not `F.scaled_dot_product_attention`.**
   `traceable_flowlm_step.py:141-213` — explicit comment "avoids `scaled_dot_product_attention` op for iOS 17
   compat." `ModelConversion.md`'s incompatibility table lists this as a general rule:
   `scaled_dot_product_attention` → "Manual attention implementation (for iOS 17 compat)" (`ModelConversion.md:144`).
4. **RoPE**: computed via interleaved-pair complex rotation (`q_complex = q.view(B,T,H,half_d,2)`), internal
   math promoted to fp32 (`.float()`) then cast back to input dtype at the end — `traceable_flowlm_step.py:94-139`.
   Frequencies precomputed from `torch.exp(ds * (-log(period)*2/D))`, not a runtime `torch.pow`.
5. **Layout convention: NOT BC1S by default.** pocket-tts's transformer KV cache and hidden states use
   standard `[batch, seq, heads, head_dim]` / `[K/V, batch, seq, heads, head_dim]` layouts throughout — no
   evidence of channel-first BC1S convention in the shipped transformer graphs. BC1S only appears in the
   **reverted** CosyVoice3 Flow-DiT ANE experiment. Read as: BC1S is a lever mobius reaches for on conv-heavy
   or ANE-first rewrites, not a default transformer convention.
6. **LayerNorm/eps handling**: `nn.LayerNorm(1024, eps=1e-5)` used as-is in pocket-tts's traceable step model
   (`traceable_flowlm_step.py:45-46`); no special eps massaging needed there. CosyVoice3's RMSNorm (Qwen2) is
   the one that needed selective fp32 pinning (`pow/reduce_mean/rsqrt`) — the failure mode is architecture-specific
   (RMSNorm's `pow`+`rsqrt` reduction, not LayerNorm's own `eps`), so don't assume pocket-tts's LayerNorm
   experience predicts what a Core AI RMSNorm-based backbone will need.
7. **Parity/tolerance gating during conversion**: no single universal tolerance found; each port defines its
   own bar. pocket-tts: end-to-end **ASR round-trip (Whisper) is mandatory**, not just tensor MAE — "Traditional
   verification methods (spectral similarity, mel-spectrogram comparison) can pass even when generated audio
   contains incorrect words" (`ModelConversion.md:188-220`), require WER<10% and PyTorch-vs-CoreML transcription
   diff <2%. CosyVoice3 target: MAE 7e-6, max|Δ| 3e-5 (= int16 quantization floor), SNR 78dB on final audio
   (`TRIALS_AND_ERRORS.md:438-439`), but got there only after finding two unrelated non-numerical bugs (stride
   padding, stale fixture) masquerading as parity failures.
8. **Tokenizer in Swift**: pocket-tts ships a **hand-rolled SentencePiece decoder in Swift**
   (`FluidAudio/Sources/FluidAudio/TTS/PocketTTS/Tokenizer/SentencePieceProto.swift`,
   `SentencePieceTokenizer.swift`), not a C library binding — worth reading directly if the Core AI port needs
   the same tokenizer and wants a native-Swift reference. Kokoro's G2P is the outlier (CoreML BART model for
   English, from-scratch rule engine for Mandarin) — not applicable to pocket-tts's raw-token frontend.
9. **Warmup/first-load specialization cost**: **[VERIFIED]** documented for both Kokoro ("Cold load ... ≈20s on
   M1; warm load ≈0.3s" — `KokoroAne.md:211-212`) and pocket-tts ("first launch ... takes significantly longer
   as CoreML compiles `.mlpackage` files to `.mlmodelc`" — `IOS_COREML_ISSUES.md:155-166`, mitigated by caching
   compiled `.mlmodelc` on disk after first compile). No quantitative pocket-tts cold-load number found in
   these repos.
10. **Placement ≠ speed; always A/B with interleaving.** "`.all` often prefers GPU on M-series; ANE residency
    is a host policy choice. Interleaved A/B ... before shipping, always." Also: `MLComputePlan`'s reported
    "preferred" device is a **static intent dump**, not a runtime trace — a model can show "100% CPU" in the
    plan while actually dispatching at GPU speed (`ANE_Candidates.md:33-38`, Pyannote example). **Any Core AI
    equivalent of a static compute-plan inspector should be treated the same way — verify with wall-clock
    interleaved A/B, not the planner's own report.**

---

## 4. Documented failure modes (fp16 artifacts, ANE op rejections, stateful pitfalls, memory/thermal)

All **[VERIFIED]** from the repos, consolidated (see full detail above for citations):

- **fp16 state-feedback compounding on ANE** → audible beeping (pocket-tts Mimi, 23 tensors/80ms; root cause
  ANE fp16 quant error ~1e-3/frame compounding over 75 frames/sec).
- **fp16 catastrophic NaN in flow-matching/CFM decoders** (CosyVoice3 Flow DiT) vs **fp16 fine in pocket-tts's
  flow decoder** — same *class* of model (flow-matching), opposite outcome; sensitivity is graph-size/depth
  dependent (22 DiT blocks vs a tiny MLP-AdaLN), not a property of "flow matching" as a category.
- **ANE mangles NaN before `isnan` evaluates** → any NaN-as-sentinel protocol (BOS markers, etc.) must be
  dropped for ANE-targeted graphs; use a real input value instead.
- **Rank-5 tensors and dynamic-shape `scatter`** are rejected by the ANE compiler; fix by reshaping to rank-4,
  not by abandoning ANE.
- **64-byte stride alignment on ANE** — tensor channel dims that don't align to 64 bytes (32 fp16 elements)
  cause hard segfaults on some intermediate tensors, not graceful fallback.
- **Zero-length tensor dimensions crash Espresso (CPU/GPU) on-device**, even though the same model loads fine
  on macOS/ANE in some configs — always check for zero-length state tensors in stateful streaming models.
- **`MLMultiArray` stride padding**: CoreML pads output array dims (observed: time dim 500→512) for
  alignment; naive linear `dataPointer` reads silently read garbage padding. Fix requires stride-aware
  accessors at every Swift/CoreML boundary crossing.
- **iOS Simulator produces plausible-looking but silent (all-zero) audio** for pocket-tts — the simulator's
  CPU-only Espresso backend loads/runs without crashing but doesn't compute correctly; **must test on real
  device**, macOS CLI is an acceptable dev-time substitute.
- **KV-cache circular-buffer wraparound bugs are easy to get wrong and silent, not crashing**: pocket-tts
  Trial 12 — Mimi's streaming attention cache had no modulo wrap, silently corrupted audio after ~16 frames
  (~1.28s) with no error, only audible degradation on long-form text; short verification clips never caught it.
- **OS-version-gated Apple runtime bugs, not app bugs**: Kokoro's `KokoroAneManager` hit both a BNNS CPU
  segfault (iOS/macOS 26.4-26.5.x, fixed in 26.6) and a GPU RNN JIT assertion failure (`GPURNNOps.mm:
  'JIT not supported'`, still live on 26.6) — neither avoidable by app-level compute-unit choice alone; the
  JIT one requires routing RNN-bearing stages off GPU entirely (`KokoroAne.md:232-249`).
- **Cross-engine pipelining (GPU/ANE/CPU overlap) does not automatically overlap in practice** — pocket-tts
  built and measured a producer/consumer pipeline expecting CPU-bound Mimi to overlap with GPU/ANE flowlm+flow;
  measured **zero win** on-device, shipped disabled. `CoreML predict()` calls may not actually run concurrently
  the way the scheduling model assumes.
- **Per-model conversion attempts documented as failing entirely, worth knowing before starting fresh**: full
  monolithic model trace (dynamic control flow, autoregressive loop) is not traceable by `torch.jit.trace` at
  all — CoreML requires static graphs, this is why pocket-tts (and every other port here) is split into
  step-level submodels from the start, not something to attempt as "convert everything as one graph first."
- No memory/jetsam-specific failure mode was found documented in either repo for pocket-tts specifically (not
  confirmed either way — absence of evidence, not evidence of absence).

---

## 5. Per-technique transfer assessment for the Core AI pocket-tts port

| # | Technique | Assessment |
|---|---|---|
| 1 | Rank-5→rank-4 KV cache split to satisfy compiler | **Transfers with adaptation.** Core AI's coreai_torch export + JIT specialization may have entirely different rank/shape constraints than the Core ML ANE compiler. But the *principle* — a rejected op/shape is usually a construct problem, try reshaping before giving up on the fast compute unit — is the transferable part, not the specific "avoid rank 5" rule. Budget time to rediscover Core AI's actual shape constraints empirically rather than assuming CoreML's carry over. |
| 2 | Manual attention (no SDPA op) for compute-unit eligibility | **Verify first, don't assume.** CoreML's SDPA gap was an iOS17-era limitation; Core AI is new enough it may have native fused-SDPA support for ComputeStream from day one. Worth an explicit go/no-go test before committing to a manual QKV+softmax rewrite — it added real complexity for CoreML and may be unnecessary overhead for Core AI. |
| 3 | Precompute RoPE freqs, promote to fp32 internally, cast back | **Transfers as-is.** This is dtype/numerics hygiene independent of the runtime — precomputing static frequency tables and doing rotation math in fp32 before casting to the working precision is good practice regardless of Core AI vs Core ML. |
| 4 | NaN-as-BOS sentinel is ANE-hostile; drop it for ANE/accelerator-bound graphs | **Transfers as a general warning, mechanism may differ.** If Core AI's chosen accelerator path has any float16-native fast path, re-verify whether NaN survives `isnan`-equivalent checks there before reusing this sentinel pattern. Cheap to just avoid NaN sentinels entirely and use an explicit `is_bos: Bool` flag input instead — removes the whole risk class. |
| 5 | In-graph state (CoreML `MLState`/multifunction) beats host round-trip KV cache | **High-value signal, transfers as a design goal.** Two independent teams (pocket-tts Trial 23, CosyVoice3 shipping default) converged on in-graph state outperforming host-side round-trip by a measurable margin (pocket-tts: −14.9% e2e). If Core AI's ComputeStream/state primitive supports something equivalent to MLState, prefer it over host-side array marshaling for the flowlm-step KV cache from the start rather than treating it as a later optimization — this directly validates prioritizing whatever Core AI's in-graph-state story is, early. |
| 6 | Selective fp16/fp32 op pinning by op-type (not blanket precision) | **Transfers as methodology, not values.** The pattern "pin specific op types to fp32, leave the rest fp16" is reusable; but CosyVoice3's hard lesson — a *fused* `layer_norm` MIL op can't be reached by op-type pinning because its internals aren't decomposed — means the actual mechanism must be re-verified against however Core AI's precision/op-pinning API works. Don't assume op-type-level control exists or behaves the same. |
| 7 | Fusing an N-step Euler/ODE loop into one graph call | **Case-by-case, not a default.** Worked cleanly for pocket-tts's tiny flow decoder (32-dim MLP-AdaLN, fp16-tolerant), was declined for Supertonic's VectorEstimator (fp32 IO casts were doing real error-containment work that fusion removed). Test both fused and per-step for whatever flow decoder ships in the Core AI port; do not assume fusion is free just because it worked for pocket-tts once. |
| 8 | Mimi decoder off the fast accelerator (CPU/GPU only), never ANE | **Transfers as a strong prior, re-verify the mechanism.** The failure is specifically "fp16-native accelerator + long-running streaming state feedback = compounding quantization error," which is a property of Mimi's *architecture* (23-tensor streaming state at 12.5Hz), not of CoreML/ANE per se. If Core AI's flow/vocoder/codec-decoder compute path has an equivalent low-precision-forced fast unit, expect the same class of artifact and plan to keep the Mimi-equivalent stage on a full-precision unit by default, then measure before trying to move it. |
| 9 | Static/fixed-shape everything, no dynamic dims | **Transfers as-is, likely non-negotiable.** Both CoreML and (per the task brief) Core AI's JIT specialization model appear to require concrete shapes; the whole "cond_prefill windows a variable-length block through a fixed T_max" pattern (pocket-tts) and CosyVoice3's fixed `T_pre=256`/`LLM_MAX_LEN=768` are the standard answer and should carry over directly. |
| 10 | MLMultiArray stride-padding gotcha | **Core ML-specific mechanism, but the *class* of bug transfers.** Whatever tensor/array type crosses the Core AI Swift boundary should be checked for the same possibility (padded dims, non-contiguous strides) before writing naive raw-pointer readers — this bit CosyVoice3 silently (wrong audio, no crash, no error) for a while before being found. |
| 11 | Mandatory ASR round-trip (WER) parity gate for TTS, not just tensor MAE | **Transfers as-is, methodology-only.** Nothing CoreML-specific about verifying synthesized speech via ASR transcription rather than trusting mel/tensor similarity — directly reusable for the Core AI pocket-tts port's validation harness. |
| 12 | Cross-engine (CPU/GPU/ANE-equivalent) pipelining for overlap | **Do not assume it works; measure.** pocket-tts's own attempt at this (mimi CPU overlapping flowlm GPU + flow ANE) measured zero win on-device despite a plausible theoretical case. If Core AI's ComputeStream submission model has different concurrency guarantees than CoreML's `predict()`, this could go either way — treat as an experiment to run late, not an assumption to design around early. |
| 13 | Interleaved A/B benchmarking methodology (mono/split/mono/split…) for placement decisions | **Transfers as-is.** Pure measurement methodology (cancels thermal/drift bias vs. sequential-batch A/B) — directly reusable regardless of the underlying runtime. |
| 14 | `.aneState` requires macOS15/iOS18 floor; CosyVoice3's MLState similarly needs macOS15/iOS18 | **Check Core AI's OS floor requirement for state/stream primitives explicitly** — if Core AI is WWDC-2026-era, its equivalent primitive may have its own new-enough minimum-OS gate that needs to be decided (and reconciled with any downstream consumer's minimum-OS floor) before committing to state-based KV cache as the default. |

---

## Does this change the planned four/five-graph split (cond/prefill, flowlm step w/ in-graph KV, flow decoder, mimi decoder)?

**No fundamental change, one strong reinforcement, one open question, one thing to de-risk early:**

- **Reinforcement**: the plan's choice of **in-graph KV state for the flowlm step** is exactly what both
  pocket-tts's own `.aneState` (Trial 23, −14.9% e2e) and CosyVoice3's *shipping default* (not an experiment)
  converged on independently. Don't treat host-side KV marshaling (pocket-tts's `.gpu` default) as the
  reference design to beat — treat the in-graph-state design as the reference design, and validate that Core
  AI's equivalent state primitive is available and performant from the start rather than deferring it as an
  optimization pass.
- **Open question**: whether a **separate cond_prefill graph is still needed as a distinct model**, or whether
  Core AI's state/stream model makes it cleaner to fold prefill and per-frame generation into two *functions*
  of one package (pocket-tts's `.aneState` does exactly this — one `pocket_state.mlmodelc` exposing `prefill`
  and `generate` as separate functions sharing state, rather than two separate mlpackages). If Core AI supports
  a similar multi-entry-point-over-shared-state construct, that's a cleaner mapping than 4-5 fully separate
  graphs, worth checking against Core AI's actual API before finalizing graph count.
- **De-risk early**: the flow decoder's fuse-vs-don't-fuse decision (Euler loop) and the Mimi-equivalent
  codec decoder's compute-unit pinning are both **empirical, model-specific findings that must be re-measured
  on the actual pocket-tts weights on Core AI**, not assumptions to carry over from these numbers. The evidence
  here shows both directions happen (fusion helped pocket-tts's flow decoder, hurt Supertonic's; ANE helped
  pocket-tts's flowlm, hurt CosyVoice3's Flow DiT) — so the four/five-graph split's *shape* looks sound, but
  each graph's precision/fusion/compute-unit choice needs its own on-device A/B, exactly as this corpus did it.
