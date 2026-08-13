import CoreAI
import Foundation

/// Per-graph exact-transfer probe (NOTES.md §20).
///
/// The port's established property is that `cpuOnly` is a *reference-precision* mode:
/// the same graph fed the same bits produces the same bits on every Apple-silicon
/// machine. This probe is how that property is checked across machines — one
/// deterministic call per graph function, raw float32 output dumps, and the comparison
/// is `max|Δ| == 0` file-for-file between a Mac run and a device run.
///
/// Everything here is deterministic by construction: inputs come from the kit's own
/// torch-protocol RNG (bit-exact on both platforms by §18's gate), states start at
/// zero, and the call sequence is fixed. The probe deliberately runs through the SAME
/// `TTSPipeline` plumbing the production path uses (`gd`, `MutableViews`, `mimiFrame`)
/// so a transfer failure implicates arithmetic, not marshaling.
///
/// Call sequence (one process, one load):
///   1. `prefill`  — RNG text embeddings [1,16,1024], pos 0, zero KV   → `prefill_cond`
///   2. `step`     — RNG latent, is_bos 1, pos 16, the KV from (1)     → `step_cond`, `step_eos`
///   3. `flow`     — the cond from (2) verbatim + RNG noise            → `flow_latent`
///   4. `mimi` ×3  — the latent from (3), then two RNG latents, fresh
///                   in-graph state advancing across the three frames  → `mimi_pcm`,
///                   plus the post-run state buffers `mimi_kv0`, `mimi_upp`, `mimi_offset`
///
/// Step (2) exercises the KV state written by (1); the three Mimi frames exercise the
/// in-graph streaming state feeding back — so the dumps cover the mutable-state paths,
/// not just the pure functions.
public enum GraphProbe {
    public static let seed: UInt64 = 20_260_813

    /// Runs the probe and writes `<name>.f32` dumps + `manifest.txt` into `outDir`.
    public static func run(pipeline p: TTSPipeline, outDir: URL,
                           log: (String) -> Void = { _ in }) async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: outDir, withIntermediateDirectories: true)
        var manifest: [String] = ["graph-probe seed \(seed) dtype \(p.half ? "float16" : "float32")"]
        var junkA: UInt64 = 0, junkB: UInt64 = 0   // timing sinks; the probe measures nothing

        func dump(_ v: [Float], _ name: String) throws {
            let data = v.withUnsafeBufferPointer { Data(buffer: $0) }   // little-endian on arm64
            try data.write(to: outDir.appendingPathComponent("\(name).f32"))
            var s = 0.0
            for x in v { s += Double(x) }
            let head = v.prefix(4).map { String(format: "%.9g", $0) }.joined(separator: " ")
            let line = String(format: "%@  n %d  sum %.9g  head [%@]", name, v.count, s, head)
            manifest.append(line)
            log("probe: \(line)")
        }

        var ns = NoiseSource(seed: seed, temp: 1.0)

        // 1. prefill — one full T_PRE window into a zero cache at pos 0.
        var kCache = makeState([Float](repeating: 0, count: Model.nLayers * Model.nHeads * Model.sMax * Model.headDim),
                               shape: [Model.nLayers, 1, Model.nHeads, Model.sMax, Model.headDim], half: p.half)
        var vCache = makeState([Float](repeating: 0, count: Model.nLayers * Model.nHeads * Model.sMax * Model.headDim),
                               shape: [Model.nLayers, 1, Model.nHeads, Model.sMax, Model.headDim], half: p.half)
        let textEmb = ns.nextNoise(count: Model.tPre * Model.dModel)
        do {
            var states = InferenceFunction.MutableViews()
            states.insert(&kCache, for: "k_cache")
            states.insert(&vCache, for: "v_cache")
            var out = try await p.fPrefill.run(
                inputs: ["text_emb": p.gd(textEmb, [1, Model.tPre, Model.dModel]),
                         "pos": nd([Int32(0)], [1])],
                states: states)
            try dump(try take(&out, "cond"), "prefill_cond")
        }

        // 2. step — reads the KV rows prefill just wrote.
        let stepLatent = ns.nextNoise(count: Model.ldim)
        var stepCondND: NDArray
        do {
            var states = InferenceFunction.MutableViews()
            states.insert(&kCache, for: "k_cache")
            states.insert(&vCache, for: "v_cache")
            var out = try await p.fStep.run(
                inputs: ["latent_in": p.gd(stepLatent, [1, 1, Model.ldim]),
                         "is_bos": p.gd([1.0], [1]),
                         "pos": nd([Int32(Model.tPre)], [1])],
                states: states)
            guard let cond = out.remove("cond")?.ndArray else {
                throw TTSError.message("probe: step returned no 'cond'")
            }
            stepCondND = cond
            try dump(flat(cond), "step_cond")
            try dump(try take(&out, "eos_logit"), "step_eos")
        }

        // 3. flow decoder — the step's cond verbatim (same dtype, no round trip).
        var flowLatent: [Float]
        do {
            var out = try await p.fFlow.run(
                inputs: ["cond": stepCondND,
                         "noise": p.gd(ns.nextNoise(count: Model.ldim), [1, Model.ldim])])
            flowLatent = try take(&out, "latent")
            try dump(flowLatent, "flow_latent")
        }

        // 4. Mimi — three frames through fresh in-graph state, so frames 2 and 3 see
        // state written by the frames before them. Always fp32 (the asset is).
        var mimiState = TTSPipeline.MimiState()
        var pcm: [Float] = []
        for frame in 0..<3 {
            let latent = frame == 0 ? flowLatent : ns.nextNoise(count: Model.ldim)
            pcm += try await p.mimiFrame(latent: nd(latent, [1, Model.ldim]), state: &mimiState,
                                         engineNanos: &junkA, flattenNanos: &junkB)
        }
        try dump(pcm, "mimi_pcm")
        try dump(flat(mimiState.kv0), "mimi_kv0")
        try dump(flat(mimiState.upP), "mimi_upp")
        var offsetVal: Int32 = 0
        mimiState.offset.view(as: Int32.self).withUnsafePointer { ptr, _, _ in offsetVal = ptr[0] }
        try dump([Float(offsetVal)], "mimi_offset")

        try manifest.joined(separator: "\n").appending("\n")
            .write(to: outDir.appendingPathComponent("manifest.txt"), atomically: true, encoding: .utf8)
    }
}
