//
// TTSPipeline+Hub.swift — the package-style entry point.
//
// `TTSPipeline(assetsDir:layout:dtype:unit:)` takes directories and asks no questions
// about where they came from; that is the right seam for a CLI run from a checkout.
// This adds the other way in, for an app that has no checkout: name a voice, get a
// pipeline. Everything else is unchanged.
//

import Foundation

extension TTSPipeline {

    /// Resolve both pinned repositories and build a pipeline from them.
    ///
    ///     let tts = try await TTSPipeline.fromHub()
    ///
    /// Only what the run needs is fetched. At the default `float16` with one voice that
    /// is roughly 415 MB — the three bundles for that dtype, Kyutai's checkpoint and
    /// sentencepiece model, and that voice's embedding. Passing `voices: nil` takes the
    /// whole English voice catalogue instead, which adds about 165 MB.
    ///
    /// Files land in the caches directory and are reused, so the cost is paid once.
    public static func fromHub(
        dtype: String = "float16",
        voices: Set<String>? = ["alba"],
        unit: ComputeUnit = .gpu,
        cacheDirectory: URL? = nil,
        progress: (@Sendable (HubProgress) -> Void)? = nil
    ) async throws -> TTSPipeline {
        let assetsDir = try await HubStore.ensure(
            .bundles,
            where: { bundleFiles(dtype: dtype).contains(where: $0.hasPrefix) || $0 == "config.json" },
            cacheDirectory: cacheDirectory,
            progress: progress)

        let english = "languages/english/"
        let weightsDir = try await HubStore.ensure(
            .weights,
            where: { path in
                if path == english + "model.safetensors" || path == english + "tokenizer.model" {
                    return true
                }
                let embeddings = english + "embeddings/"
                guard path.hasPrefix(embeddings) else { return false }
                guard let voices else { return true }
                let name = String(path.dropFirst(embeddings.count))
                    .replacingOccurrences(of: ".safetensors", with: "")
                return voices.contains(name)
            },
            cacheDirectory: cacheDirectory,
            progress: progress)

        let layout = WeightsLayout(
            modelURL: weightsDir.appendingPathComponent(english + "model.safetensors"),
            tokenizerURL: weightsDir.appendingPathComponent(english + "tokenizer.model"),
            embeddingsDir: weightsDir.appendingPathComponent(english + "embeddings"))

        return try await TTSPipeline(assetsDir: assetsDir, layout: layout, dtype: dtype, unit: unit)
    }

    /// The bundle directories `TTSPipeline.init` will look for at this dtype. Mimi is
    /// fp32 at every setting — see the note on `TTSPipeline`.
    static func bundleFiles(dtype: String) -> [String] {
        ["flowlm_\(dtype)_s\(Model.sMax).aimodel/",
         "flow_decoder_\(dtype)_lsd1.aimodel/",
         "mimi_decoder_float32_ring272_outer_q_gs.aimodel/"]
    }
}

extension TTSPipeline {
    /// Load one of the shipped voices from wherever this pipeline's weights came from.
    ///
    ///     let tts = try await TTSPipeline.fromHub()
    ///     let out = try await tts.synthesize(text: "Hello there.", voice: try tts.voice("alba"),
    ///                                        seed: 0, applyGain: true)
    ///
    /// After `fromHub(voices:)` only the voices that were requested are present.
    public func voice(_ name: String) throws -> VoiceState {
        try VoiceState(name: name, embeddingsDir: layout.embeddingsDir)
    }
}
