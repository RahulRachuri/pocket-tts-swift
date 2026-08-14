# Security

This repository is a host: Swift code plus the Python scripts that produced the Core AI
bundles it runs. It distributes no weights and no bundles. The interesting question is
therefore less "is there a bug in this code" and more "can you tell whether the bundles
you loaded are the ones that were gated." This page answers that, including where the
answer is no.

## Reporting

Open a [security advisory](../../security/advisories/new) for anything you would rather
not say in public, such as a bundle that behaves unlike its source model, or a repository
that looks tampered with. Ordinary bugs belong in [issues](../../issues). This is a
personal project, so expect a reply in days rather than hours.

## What the integrity story actually is

**A pinned revision, not a signature.** The bundles this host runs are published at
[rahulrachuri/pocket-tts-coreai](https://huggingface.co/rahulrachuri/pocket-tts-coreai).
Pin the revision, currently `ad989309a578`, rather than tracking the branch. A later push
to that repository cannot change what a pinned consumer receives.

**What is not done.** The bundles are not code-signed, and there is no checksum manifest
beyond what Hugging Face itself stores. Conversion is not byte-deterministic, so a
checksum of your own rebuild will not match the published one even when the rebuild is
correct. Integrity rests on the revision pin and on Hugging Face's storage, not on a
signature you can verify offline.

**A `.aimodel` bundle is data that a runtime executes.** Treat one from any source the way
you would treat a binary dependency. The scripts in `conversion/` are the provenance:
they show exactly what was built, from which checkpoint, and what it was checked against.

## Checking the bundles yourself

Nothing here asks you to trust this repository. Every exporter in `conversion/` gates its
graph against a PyTorch oracle capture before writing a bundle, in eager, `cpu_only` and
`gpu`, and refuses to write on failure. To reproduce that end to end:

1. Generate the oracle capture with `conversion/gen_oracle.py --tag orc_a`. This needs
   the `pocket-tts` pip package and Kyutai's checkpoint.
2. Re-run the exporters per the recipe. Each prints its own gate result.
3. Run the end-to-end check with `conversion/e2e_coreai.py`, and the intelligibility gate
   with `conversion/asr_gate.py`, which transcribes generated audio and reports WER.

The ASR gate exists because tensor similarity can pass while audio is unintelligible.
Cosine alone is not sufficient evidence that a TTS bundle is correct.

## Model weights are upstream's

The model is Kyutai's
[`pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning),
CC-BY-4.0, read directly from Kyutai's own repository at revision
`e041936c75475d350b405bc870bcf7c22da4e9e6`. It is not mirrored here. What the model says,
what it was trained on, and its licence are upstream's concern.

## Scope

In scope: this host, the conversion and gate scripts, and the published bundles and their
provenance. Out of scope: the upstream model's content, Hugging Face's own storage, and
Apple's Core AI runtime. If you are shipping something you have to support, mirror the
bundles you depend on into storage you control rather than fetching a personal Hugging
Face namespace at runtime, and re-verify after any OS or toolchain bump. The
`coreai-core` wheel is OS-coupled, and a beta bump has already invalidated previously
exported bundles once.
