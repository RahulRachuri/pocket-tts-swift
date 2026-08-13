import Foundation
import PocketTTSKit

/// pocket-tts-cli — text in, wav out, entirely through the Core AI graphs.
///
///     swift run -c release pocket-tts-cli --text "..." --voice alba --out out.wav
///
/// Debug subcommands (parity tooling; each prints machine-diffable output):
///     pocket-tts-cli tokenize --text "..."        sentencepiece ids, space-separated
///     pocket-tts-cli chunk --text "..." [--voice] chunk plan with token counts
///     pocket-tts-cli noise --seed N --calls K     the K-th call's 32 noise floats
///
/// Defaults assume the repo layout (artifacts/, weights/hf/...) relative to --root,
/// which defaults to the current directory.
struct Args {
    var mode = "synth"
    var text = "The quick brown fox jumps over the lazy dog."
    var voice = "alba"
    var out = "out.wav"
    var seed: UInt64 = 1234
    var dtype = "float32"
    var unit = ComputeUnit.gpu
    var root = FileManager.default.currentDirectoryPath
    var assets: String? = nil
    var noGain = false
    var refWav: String? = nil
    var pcm16 = false
    var calls = 1
    var quiet = false
    var warmup = false

    static func parse() -> Args {
        var a = Args()
        var argv = Array(CommandLine.arguments.dropFirst())
        if let first = argv.first, !first.hasPrefix("--") {
            a.mode = first
            argv.removeFirst()
        }
        var i = 0
        func next(_ flag: String) -> String {
            guard i + 1 < argv.count else { fail("\(flag) needs a value") }
            i += 1
            return argv[i]
        }
        while i < argv.count {
            switch argv[i] {
            case "--text": a.text = next("--text")
            case "--voice": a.voice = next("--voice")
            case "--out": a.out = next("--out")
            case "--seed": a.seed = UInt64(next("--seed")) ?? 1234
            case "--dtype":
                a.dtype = next("--dtype")
                guard ["float32", "float16"].contains(a.dtype) else { fail("--dtype must be float32|float16") }
            case "--unit":
                guard let u = ComputeUnit(rawValue: next("--unit")) else { fail("--unit must be gpu|cpu|ane|cpuOnly|def") }
                a.unit = u
            case "--root": a.root = next("--root")
            case "--assets": a.assets = next("--assets")
            case "--no-gain": a.noGain = true
            case "--ref-wav": a.refWav = next("--ref-wav")
            case "--pcm16": a.pcm16 = true
            case "--calls": a.calls = Int(next("--calls")) ?? 1
            case "--quiet": a.quiet = true
            case "--warmup": a.warmup = true
            case "--help", "-h":
                print("usage: pocket-tts-cli [tokenize|chunk|noise] --text ... --voice alba --out out.wav")
                print("flags: --seed N --dtype float32|float16 --unit gpu|cpu|cpuOnly --root DIR --assets DIR")
                print("       --no-gain --ref-wav WAV --pcm16 --quiet")
                exit(0)
            default: fail("unknown argument \(argv[i])")
            }
            i += 1
        }
        return a
    }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

let args = Args.parse()
let rootURL = URL(fileURLWithPath: args.root)

do {
    let layout = try WeightsLayout.discover(root: rootURL)

    switch args.mode {
    case "tokenize":
        let sp = try SentencePieceModel(url: layout.tokenizerURL)
        print(sp.encode(args.text).map(String.init).joined(separator: " "))

    case "detok":
        // --text carries space-separated token ids; prints the decoded text.
        let sp = try SentencePieceModel(url: layout.tokenizerURL)
        let ids = args.text.split(separator: " ").compactMap { Int($0) }
        print(sp.decode(ids))

    case "chunk":
        let sp = try SentencePieceModel(url: layout.tokenizerURL)
        let chunker = Chunker(sp: sp)
        let voice = try VoiceState(name: args.voice, embeddingsDir: layout.embeddingsDir)
        let budget = Chunker.tokenBudget(voicePos: voice.positions)
        print("voice \(voice.name): \(voice.positions) positions, token budget \(budget)")
        for (idx, c) in try chunker.chunk(args.text, voicePos: voice.positions).enumerated() {
            let n = sp.encode(c).count
            print("[\(idx)] \(n) tokens, max_gen_len \(Model.maxGenLen(tokens: n)): \(c)")
        }

    case "noise":
        var src = NoiseSource(seed: args.seed, temp: Model.temp)
        var draw: [Float] = []
        for _ in 0..<args.calls { draw = src.nextNoise(count: Model.ldim) }
        print(draw.map { String(format: "%.9g", $0) }.joined(separator: " "))

    case "probe":
        // Per-graph exact-transfer dumps (GraphProbe): the Mac side of the device gate.
        //     pocket-tts-cli probe --unit cpuOnly --out artifacts/probe_mac_f32_cpuOnly
        let assetsDir = args.assets.map { URL(fileURLWithPath: $0) }
            ?? rootURL.appendingPathComponent("artifacts")
        let outDir = URL(fileURLWithPath: args.out, relativeTo: rootURL)
        let pipeline = try await TTSPipeline(
            assetsDir: assetsDir, layout: layout, dtype: args.dtype, unit: args.unit)
        print("probe: loaded \(args.dtype)/\(args.unit.rawValue) in \(String(format: "%.2f", pipeline.loadSeconds)) s")
        try await GraphProbe.run(pipeline: pipeline, outDir: outDir) { print($0) }
        print("probe: dumps in \(outDir.path)")

    case "synth":
        let assetsDir = args.assets.map { URL(fileURLWithPath: $0) }
            ?? rootURL.appendingPathComponent("artifacts")
        let log: (String) -> Void = args.quiet ? { _ in } : { print($0) }

        let voice = try VoiceState(name: args.voice, embeddingsDir: layout.embeddingsDir)
        log("voice \(voice.name): \(voice.positions) conditioning positions")

        let pipeline = try await TTSPipeline(
            assetsDir: assetsDir, layout: layout, dtype: args.dtype, unit: args.unit)
        log(String(format: "loaded %@/%@ in %.2f s (flow-LM functions: %@)",
                   args.dtype, args.unit.rawValue, pipeline.loadSeconds,
                   pipeline.lmAsset.functionNames.joined(separator: ",")))

        if args.warmup {
            // One throwaway sentence so first-call graph specialization (~4–5 s across the
            // four functions) stays out of the timed run. A warmup synth shares nothing
            // with the timed one — Mimi state, KV state, and the noise counter are all
            // per-`synthesize` — so the audio is identical with or without it.
            _ = try await pipeline.synthesize(
                text: "Warm up.", voice: voice, seed: args.seed, applyGain: false)
            log("warmup done")
        }

        let r = try await pipeline.synthesize(
            text: args.text, voice: voice, seed: args.seed, applyGain: !args.noGain, log: log)

        let outURL = URL(fileURLWithPath: args.out, relativeTo: rootURL)
        if args.pcm16 {
            try WAV.writePCM16(r.samples, sampleRate: Model.sampleRate, to: outURL)
        } else {
            try WAV.writeFloat32(r.samples, sampleRate: Model.sampleRate, to: outURL)
        }

        let peak = r.samples.reduce(Float(0)) { max($0, abs($1)) }
        log(String(format: "%d chunk(s), %d engine calls  peak %.3f  rms %.4f  gain %.3f",
                   r.chunks.count, r.engineCalls, peak, rms(r.samples), r.gainApplied))
        log(String(format: "engine: prefill %.0f ms  step %.0f ms  flow %.0f ms  mimi %.0f ms",
                   r.prefillMillis, r.stepMillis, r.flowMillis, r.mimiMillis))
        log(String(format: "host:   lut %.1f ms  marshal %.1f ms  flatten %.1f ms  other %.0f ms",
                   r.lutMillis, r.marshalMillis, r.flattenMillis, r.otherMillis))
        print(String(format: "%.3f s audio in %.3f s  RTF %.4f (%.1fx realtime)  -> %@",
                     r.durationSeconds, r.wallSeconds, r.rtf, 1.0 / r.rtf, outURL.path))

        if let ref = args.refWav {
            let (gold, sr) = try WAV.readMono(URL(fileURLWithPath: ref, relativeTo: rootURL))
            guard sr == Model.sampleRate else { fail("--ref-wav sample rate \(sr) != \(Model.sampleRate)") }
            let n = min(gold.count, r.samples.count)
            let c = cosine(r.samples[0..<n], gold[0..<n])
            let d = maxAbsDiff(r.samples[0..<n], gold[0..<n])
            print(String(format: "wav vs ref: samples %d vs %d  cos %.6f  max|d| %.4f  rms %.4f vs %.4f",
                         r.samples.count, gold.count, c, d, rms(r.samples), rms(gold)))
        }

    default:
        fail("unknown mode '\(args.mode)' (expected synth|tokenize|chunk|noise|probe)")
    }
} catch {
    fail("\(error)")
}
