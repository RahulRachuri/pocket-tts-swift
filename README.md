# pocket-tts-swift

Port of Kyutai's pocket-tts to Apple's Core AI framework, with a native Swift host.

Text in, wav out, entirely through Core AI graphs from a native Swift CLI. No Python and
no fixtures in the loop, on Mac and on iPhone.

On an iPhone 17 Pro Max (A19 Pro, iOS 27.0), a 148-word paragraph synthesizes at **7.8×
realtime** in fp16 with a 169 MB peak footprint, or 6.1× in fp32. Correctness is gated
against the PyTorch fp32 oracle rather than judged by ear: cos 1.000000 end to end,
bit-identical per-graph transfer between device and Mac under `cpuOnly`, and an ASR
round trip at 0.00 % WER on the oracle prompt and 1.38 % on the paragraph. NOTES.md §20
has the full tables.

Converted bundles: **[rahulrachuri/pocket-tts-coreai](https://huggingface.co/rahulrachuri/pocket-tts-coreai)**
on the Hugging Face Hub.

Companion project: [parakeet-swift](https://github.com/RahulRachuri/parakeet-swift)
(same toolchain, same gate discipline; also the ASR used by the acceptance gate).

## Build & run (Xcode 27 toolchain, macOS 27 SDK)

```
swift build -c release
swift run -c release pocket-tts-cli --text "Hello there." --voice alba --out out.wav
```

Run from the repo root so the default `weights/` and `artifacts/` paths resolve
(`--root` / `--assets` override them). Flags: `--seed`, `--dtype float32|float16`,
`--unit gpu|cpu|cpuOnly`, `--no-gain`, `--ref-wav` (cosine vs a reference wav),
`--pcm16`, `--warmup`. Debug subcommands `tokenize` / `detok` / `chunk` / `noise`
print machine-diffable output for parity checks against the Python reference.

## Use as a package

`PocketTTSKit` is a library product, so an app can depend on it and let it resolve its
own assets — no checkout, no `weights/` tree, no environment variables.

```swift
.package(url: "https://github.com/RahulRachuri/pocket-tts-swift", from: "0.1.0")
// target dependency: .product(name: "PocketTTSKit", package: "pocket-tts-swift")
```


```swift
import PocketTTSKit

let tts = try await TTSPipeline.fromHub()
let out = try await tts.synthesize(text: "Hello there.", voice: try tts.voice("alba"),
                                   seed: 0, applyGain: true)
```

`fromHub` resolves two repositories, both pinned to an immutable revision: the converted
bundles published here, and Kyutai's checkpoint, which owns the weights, the sentencepiece
model and the voice embeddings. Only what the run needs is fetched — at the default
`float16` with one voice that is about 415 MB, against 9.8 GB for Kyutai's repository
whole. Files land in `~/Library/Caches/pocket-tts-swift/hub/` (the app's own caches
directory on iOS) and are reused, so a warm start pays a cache check rather than the
download.

Every LFS-stored payload is checked against the SHA-256 the Hub reports, and moved into
place only once verified — a truncated or interrupted download cannot be mistaken for a
complete one. The revision pin is still what SECURITY.md says it is: the digest proves you
received the bytes the Hub holds, not that those bytes are the ones that were gated.

Options: `voices: nil` for the whole English catalogue (about 165 MB more), `dtype:` to
pick the fp32 bundles, `progress:` for a closure to drive a loading UI, and
`cacheDirectory:` to put the cache somewhere else. The progress closure is `@Sendable` and
is called from URLSession's queue, so hop to the main actor before touching UI.

## Layout

- `NOTES.md` — working notes: decomposition map, export blockers, gate results.
- `Package.swift`, `Sources/` — the Swift host: `PocketTTSKit` (tokenizer, chunker,
  RNG, weights, pipeline, Hub resolution) + the `pocket-tts-cli` executable.
- `conversion/` — Python export + gate scripts (own venv, see NOTES).
- `reference/` — notes on techniques studied from other ports.

Some conversion scripts depend on things outside this repo and read their locations from
the environment: `PARAKEET_BIN` and `PARAKEET_ARTIFACTS` for the ASR gate, and
`CORPUS_ROOT` for the validation sweep's LibriSpeech and Project Gutenberg sources.

## Licence and attribution

The code in this repository — conversion scripts, gates, and the Swift host — is
licensed under the Apache Licence, Version 2.0. See [LICENSE](LICENSE).

The model is not. It is Kyutai's
[`pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning),
released under CC-BY-4.0; any weights or converted bundles you run through this host stay
under that licence and carry its attribution requirement. No weights are distributed in
this repository. Voice embeddings come from Kyutai's catalogue — see
[kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) for per-voice licensing.

The conversion route traces Kyutai's [`pocket-tts`](https://pypi.org/project/pocket-tts/)
Python package (MIT) as an installed dependency; no source from it is vendored here.
`reference/fluidaudio-techniques.md` summarizes techniques studied from
[FluidAudio](https://github.com/FluidInference/FluidAudio) (Apache-2.0); no source from it
is vendored or imported either.
