import CoreAI
import Foundation

/// Per-chunk telemetry from a synthesis run.
public struct ChunkStat {
    public var text = ""
    public var nText = 0
    public var nWindows = 0
    public var posStart = 0        // voicePos + nText: the row the first AR step writes
    public var headroom = 0        // AR steps available before the KV write hits S_MAX
    public var maxGenLen = 0
    public var steps = 0
    public var frames = 0
    public var eosStep: Int? = nil
    public var hitMaxGenLen = false
    public var framesAfterEOS = 0
    public var durationSeconds = 0.0
}

public struct SynthResult {
    public var samples: [Float] = []
    public var chunks: [ChunkStat] = []
    public var wallSeconds = 0.0          // generation only, excludes model load
    public var prefillMillis = 0.0        // engine run only — input marshaling excluded
    public var stepMillis = 0.0
    public var flowMillis = 0.0
    public var mimiMillis = 0.0
    /// Host-side glue, measured separately so a profile can attribute the wall clock:
    /// LUT lookup, float32→NDArray input marshaling (incl. fp16 narrowing), and
    /// NDArray→[Float] output flattening. Whatever is left of `wallSeconds` after
    /// engine + these is loop/async overhead. (The rescale+quantizer matvec was host
    /// work until M2 folded it into the Mimi graph.)
    public var lutMillis = 0.0
    public var marshalMillis = 0.0
    public var flattenMillis = 0.0
    public var engineCalls = 0
    public var gainApplied = 1.0

    public var hostMillis: Double { lutMillis + marshalMillis + flattenMillis }
    public var engineMillis: Double { prefillMillis + stepMillis + flowMillis + mimiMillis }
    public var otherMillis: Double { wallSeconds * 1000 - engineMillis - hostMillis }

    public var durationSeconds: Double { Double(samples.count) / Double(Model.sampleRate) }
    public var rtf: Double { wallSeconds / max(durationSeconds, 1e-9) }
}

/// The production Swift host: everything the pipeline does that is *not* a graph.
///
/// Three assets, four graphs:
///   * `flowlm_<dtype>_s512.aimodel` — multifunction: `prefill` + `step` over one weight
///     set and one shared in-graph KV state (`k_cache`/`v_cache`, packed rank-5
///     `[6,1,16,512,64]`). The host owns the buffers and hands the runtime mutable views;
///     the cache never round-trips.
///   * `flow_decoder_<dtype>_lsd1.aimodel` — pure function, `cond` + `noise` → `latent`.
///   * `mimi_decoder_float32_ring272_outer_q_gs.aimodel` — fp32 at every dtype setting
///     (12 state tensors feeding back at 12.5 Hz is exactly where fp16 error compounds
///     audibly, NOTES.md §9/§14). Takes the raw [1,32] latent; the rescale + quantizer
///     are in-graph, and since M2 the 12 streaming-state tensors are in-graph Core AI
///     state too — the host owns the buffers for the whole run (never reset across
///     chunks, upstream's invariant) and hands the runtime mutable views per frame; no
///     state tensor ever round-trips.
///
/// The host owns: sentencepiece tokenization, the budget-enforcing chunker, the LUT
/// embedding lookup, the windowed T_PRE=16 prefill carrying `pos`, the AR loop with the
/// −4.0 EOS compare and `frames_after_eos`, the seeded torch-protocol noise (the
/// rescale + quantizer moved in-graph with the `_q` Mimi asset), Mimi state hand-off, chunk
/// concatenation (direct concat — upstream's `torch.cat`; the sweep measured every join
/// an order of magnitude below in-audio signal motion, §16.5), and per-voice gain.
public final class TTSPipeline {
    public let lmAsset: Asset
    public let flowAsset: Asset
    public let mimiAsset: Asset
    let fPrefill: InferenceFunction
    let fStep: InferenceFunction
    let fFlow: InferenceFunction
    let fMimi: InferenceFunction
    public let half: Bool
    public let loadSeconds: Double

    public let weights: HostWeights
    public let sp: SentencePieceModel
    public let chunker: Chunker
    /// Where this pipeline's weights came from. Retained so a caller that did not build
    /// the layout itself — `fromHub`, or an app that was handed a pipeline — can still
    /// load a voice from it.
    public let layout: WeightsLayout

    public init(assetsDir: URL, layout: WeightsLayout, dtype: String, unit: ComputeUnit) async throws {
        self.half = (dtype == "float16")
        let lmURL = assetsDir.appendingPathComponent("flowlm_\(dtype)_s\(Model.sMax).aimodel")
        let flowURL = assetsDir.appendingPathComponent("flow_decoder_\(dtype)_lsd1.aimodel")
        let mimiURL = assetsDir.appendingPathComponent("mimi_decoder_float32_ring272_outer_q_gs.aimodel")
        for u in [lmURL, flowURL, mimiURL] where !FileManager.default.fileExists(atPath: u.path) {
            throw TTSError.message("missing asset \(u.lastPathComponent) in \(assetsDir.path)")
        }
        let t0 = nowNanos()
        lmAsset = try await Asset(url: lmURL, unit: unit)
        flowAsset = try await Asset(url: flowURL, unit: unit)
        mimiAsset = try await Asset(url: mimiURL, unit: unit)
        fPrefill = try lmAsset.function("prefill")
        fStep = try lmAsset.function("step")
        fFlow = try flowAsset.function("main")
        fMimi = try mimiAsset.function("main")
        weights = try HostWeights(modelURL: layout.modelURL)
        sp = try SentencePieceModel(url: layout.tokenizerURL)
        chunker = Chunker(sp: sp)
        self.layout = layout
        loadSeconds = Double(nowNanos() - t0) / 1e9
    }

    /// Graph-dtype NDArray from float32 host values.
    @inline(__always) func gd(_ v: [Float], _ shape: [Int]) -> NDArray {
        half ? ndHalf(v, shape) : nd(v, shape)
    }

    /// The 12 Mimi streaming-state buffers, host-owned for the lifetime of one
    /// `synthesize` run. The runtime mutates them in place through `MutableViews`
    /// (in-graph state, same mechanics as the flow-LM KV) — nothing round-trips.
    /// A `var` struct so `&state.field` satisfies `MutableViews.insert`'s inout.
    struct MimiState {
        var upP = zeros([1, 512, 16])
        var kv0 = zeros([2, 1, Model.mimiRing, 8, 64])
        var kv1 = zeros([2, 1, Model.mimiRing, 8, 64])
        var offset: NDArray = {
            var a = NDArray(shape: [1], scalarType: .int32)
            var mv = a.mutableView(as: Int32.self)
            mv.withUnsafeMutablePointer { p, _, _ in p[0] = 0 }
            return a
        }()
        var c0 = zeros([1, 512, 6]); var c2 = zeros([1, 256, 6]); var c3 = zeros([1, 256, 2])
        var c5 = zeros([1, 128, 5]); var c6 = zeros([1, 128, 2])
        var c8 = zeros([1, 64, 4]); var c9 = zeros([1, 64, 2]); var c11 = zeros([1, 64, 2])

        /// Zero-initialised, never NaN (NOTES.md §6, blocker iii-b). Fresh state is
        /// allocated once per *run*, never per chunk — upstream never resets the Mimi
        /// state, and resetting it would click at every chunk boundary.
        static func zeros(_ shape: [Int]) -> NDArray {
            var a = NDArray(shape: shape, scalarType: .float32)
            var mv = a.mutableView(as: Float.self)
            mv.withUnsafeMutablePointer { p, shp, _ in
                let n = shp.indices.reduce(1) { $0 * shp[$1] }
                for i in 0..<n { p[i] = 0 }
            }
            return a
        }
    }

    /// One Mimi frame: raw flow-decoder latent `[1,32]` in, 1920 PCM samples out. The
    /// state advances in place inside the graph.
    /// `engineNanos` accrues the run call only; `flattenNanos` the NDArray→[Float] copy —
    /// separated so the profile can attribute the wall clock.
    func mimiFrame(latent: NDArray, state: inout MimiState,
                   engineNanos: inout UInt64, flattenNanos: inout UInt64) async throws -> [Float] {
        var states = InferenceFunction.MutableViews()
        states.insert(&state.upP, for: "up_p")
        states.insert(&state.kv0, for: "kv0")
        states.insert(&state.kv1, for: "kv1")
        states.insert(&state.offset, for: "offset")
        states.insert(&state.c0, for: "c0"); states.insert(&state.c2, for: "c2")
        states.insert(&state.c3, for: "c3"); states.insert(&state.c5, for: "c5")
        states.insert(&state.c6, for: "c6"); states.insert(&state.c8, for: "c8")
        states.insert(&state.c9, for: "c9"); states.insert(&state.c11, for: "c11")
        let te = nowNanos()
        var out = try await fMimi.run(inputs: ["latent": latent], states: states)
        engineNanos &+= nowNanos() &- te
        let tf = nowNanos()
        let pcm = try take(&out, "pcm")
        flattenNanos &+= nowNanos() &- tf
        return pcm
    }

    // MARK: - synthesis

    /// Text → 24 kHz float32 PCM through the four graphs, free-running.
    ///
    /// Lifetime note that shapes this function: `InferenceFunction.MutableViews` is
    /// `~Escapable` and borrows its arrays only up to `function.run`, so what each state
    /// buffer needs is storage outliving every call it is inserted into — not one enclosing
    /// scope. Here the KV cache is per-chunk and the Mimi state per-run, so both are locals
    /// and prefill and the AR loop end up sharing a scope. A graph whose state lives as long
    /// as its owner can hold it in stored properties instead and split the calls up.
    public func synthesize(
        text: String, voice: VoiceState, seed: UInt64, applyGain: Bool,
        log: (String) -> Void = { _ in }
    ) async throws -> SynthResult {
        var r = SynthResult()
        var noise = NoiseSource(seed: seed, temp: Model.temp)
        var mimiState = MimiState()        // NEVER reset across chunks
        var stepN: UInt64 = 0, flowN: UInt64 = 0, mimiN: UInt64 = 0, preN: UInt64 = 0
        var lutN: UInt64 = 0, marshN: UInt64 = 0, flatN: UInt64 = 0

        let chunkTexts = try chunker.chunk(text, voicePos: voice.positions)
        let t0 = nowNanos()

        for chunkText in chunkTexts {
            var cs = ChunkStat()
            cs.text = chunkText

            // frames_after_eos: model_recommended is nil for this config, so it is the
            // prepare_text_prompt guess (3 for <=4 words else 1) + 2.
            cs.framesAfterEOS = try Chunker.prepareTextPrompt(chunkText).framesGuess + 2

            // `_generate_audio_stream_short_text` tokenizes the CHUNK, not the prepared
            // text — prepare_text_prompt is consulted only for the frames guess.
            let tokens = sp.encode(chunkText)
            let nText = tokens.count
            cs.nText = nText
            cs.maxGenLen = Model.maxGenLen(tokens: nText)
            precondition(voice.positions + nText + cs.maxGenLen <= Model.sMax,
                         "chunker invariant violated: \(voice.positions)+\(nText)+\(cs.maxGenLen) > \(Model.sMax)")
            let tl = nowNanos()
            let textEmb = weights.embed(tokens)
            lutN &+= nowNanos() &- tl

            // Fresh KV cache per chunk, re-seeded from the voice state — upstream
            // deep-copies it per chunk, so a chunk never sees the previous chunk's text.
            var kCache = makeState(voice.kSeed, shape: [6, 1, 16, Model.sMax, 64], half: half)
            var vCache = makeState(voice.vSeed, shape: [6, 1, 16, Model.sMax, 64], half: half)
            var pos = Int32(voice.positions)

            // Windowed prefill over the static T_PRE=16 graph, carrying `pos` across
            // windows. Pad rows write junk into cache slots strictly after every real
            // position; the causal mask makes them unreachable and the first AR steps
            // overwrite them before ever reading them.
            var w = 0
            while w < nText {
                let width = min(Model.tPre, nText - w)
                let tm = nowNanos()
                var win = [Float](repeating: 0, count: Model.tPre * Model.dModel)
                for j in 0..<(width * Model.dModel) { win[j] = textEmb[w * Model.dModel + j] }
                let winND = gd(win, [1, Model.tPre, Model.dModel])
                let posND = nd([pos], [1])
                marshN &+= nowNanos() &- tm
                var states = InferenceFunction.MutableViews()
                states.insert(&kCache, for: "k_cache")
                states.insert(&vCache, for: "v_cache")
                let ta = nowNanos()
                var out = try await fPrefill.run(
                    inputs: ["text_emb": winND, "pos": posND],
                    states: states)
                preN &+= nowNanos() &- ta
                r.engineCalls += 1
                _ = out.remove("cond")   // upstream discards the prefill latent too
                pos += Int32(width)
                w += width
                cs.nWindows += 1
            }
            cs.posStart = Int(pos)
            cs.headroom = Model.sMax - Int(pos)

            // The AR loop, mirroring `_autoregressive_generation`: EOS is a threshold
            // compare on the host; the latent produced on the breaking step is DISCARDED
            // (32 flow-LM calls -> 31 Mimi frames on the oracle fixture).
            let chunkSampleStart = r.samples.count
            var latent = [Float](repeating: 0, count: Model.ldim)
            var isBOS: Float = 1.0
            var eosStep: Int? = nil
            var i = 0
            while i < cs.maxGenLen {
                // The overflow assert (NOTES.md §16.1): past S_MAX the write corrupts
                // context silently and EOS fires late. The chunker's budget makes this
                // unreachable; it stays as a hard stop, not a warning.
                let writeRow = Int(pos) + i
                precondition(writeRow < Model.sMax,
                             "AR write position \(writeRow) reached S_MAX=\(Model.sMax) — cache overflow")

                let tm = nowNanos()
                let latND = gd(latent, [1, 1, Model.ldim])
                let bosND = gd([isBOS], [1])
                let posND = nd([pos + Int32(i)], [1])
                marshN &+= nowNanos() &- tm
                var states = InferenceFunction.MutableViews()
                states.insert(&kCache, for: "k_cache")
                states.insert(&vCache, for: "v_cache")
                let ta = nowNanos()
                var out = try await fStep.run(
                    inputs: ["latent_in": latND, "is_bos": bosND, "pos": posND],
                    states: states)
                stepN &+= nowNanos() &- ta
                r.engineCalls += 1
                isBOS = 0
                guard let condND = out.remove("cond")?.ndArray else {
                    throw TTSError.message("step: missing 'cond'")
                }
                var tf = nowNanos()
                let eos = try take(&out, "eos_logit")[0]
                flatN &+= nowNanos() &- tf
                if eos > Model.eosThreshold, eosStep == nil { eosStep = i }

                // `cond` goes straight from the step graph's output into the flow
                // decoder's input — same dtype, no host round-trip.
                let tn = nowNanos()
                let ns = noise.nextNoise(count: Model.ldim)
                let nsND = gd(ns, [1, Model.ldim])
                marshN &+= nowNanos() &- tn
                let tb = nowNanos()
                var fout = try await fFlow.run(
                    inputs: ["cond": condND, "noise": nsND])
                flowN &+= nowNanos() &- tb
                r.engineCalls += 1
                tf = nowNanos()
                latent = try take(&fout, "latent")
                flatN &+= nowNanos() &- tf

                if let e = eosStep, i >= e + cs.framesAfterEOS { i += 1; break }

                // Raw [1,32] latent straight into the Mimi graph — the rescale +
                // quantizer live in-graph since the `_q` asset. Always fp32: the
                // Mimi asset is fp32 at every --dtype setting (§14).
                let tw = nowNanos()
                let latMimiND = nd(latent, [1, Model.ldim])
                marshN &+= nowNanos() &- tw
                let pcm = try await mimiFrame(latent: latMimiND, state: &mimiState,
                                              engineNanos: &mimiN, flattenNanos: &flatN)
                r.engineCalls += 1
                r.samples.append(contentsOf: pcm)
                i += 1
            }
            cs.steps = i
            cs.eosStep = eosStep
            cs.hitMaxGenLen = (eosStep == nil)
            cs.frames = (r.samples.count - chunkSampleStart) / Model.frameSamples
            cs.durationSeconds = Double(r.samples.count - chunkSampleStart) / Double(Model.sampleRate)
            if cs.hitMaxGenLen { log("WARNING: chunk hit max_gen_len without EOS: \"\(chunkText.prefix(50))\"") }
            log(String(format: "  chunk %d tokens  windows %d  pos %d  headroom %d  steps %d  frames %d  eos@%@  %.2f s",
                       cs.nText, cs.nWindows, cs.posStart, cs.headroom, cs.steps, cs.frames,
                       eosStep.map(String.init) ?? "-", cs.durationSeconds))
            r.chunks.append(cs)
        }

        r.wallSeconds = Double(nowNanos() - t0) / 1e9
        r.prefillMillis = Double(preN) / 1e6
        r.stepMillis = Double(stepN) / 1e6
        r.flowMillis = Double(flowN) / 1e6
        r.mimiMillis = Double(mimiN) / 1e6
        r.lutMillis = Double(lutN) / 1e6
        r.marshalMillis = Double(marshN) / 1e6
        r.flattenMillis = Double(flatN) / 1e6
        if applyGain {
            r.gainApplied = VoiceGain.apply(&r.samples, voice: voice.name)
        }
        return r
    }
}
