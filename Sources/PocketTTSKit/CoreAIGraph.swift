import CoreAI
import Foundation

/// Which compute unit a bundle is specialized for.
///
/// The porting rule (NOTES.md §8): parity runs may use `cpuOnly`, but anything timed
/// must state a real preference — `cpu_only` is a reference-precision mode, never a
/// benchmark configuration. On this Mac the GPU delegate is the bit-exact one.
public enum ComputeUnit: String, Sendable, CaseIterable {
    case gpu, cpu, ane, cpuOnly, def

    var options: SpecializationOptions {
        switch self {
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .cpu: return SpecializationOptions(preferredComputeUnitKind: .cpu)
        case .ane: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .cpuOnly: return SpecializationOptions.cpuOnly
        case .def: return SpecializationOptions.default
        }
    }
}

public enum TTSError: Error, CustomStringConvertible {
    case message(String)
    public var description: String { switch self { case .message(let m): return m } }
}

@inline(__always) public func nowNanos() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

/// One loaded `.aimodel` bundle. Keeps the `AIModel` around and exposes `function(_:)`
/// because the flow-LM asset is a **multifunction** package: `prefill` and `step` are two
/// entry points over one weight set. What loading it once buys is the weights — the two
/// functions share their KV cache through host-owned `NDArray`s handed over as
/// `MutableViews`, so it is the buffers that carry the state across calls, not the
/// `AIModel` identity.
public final class Asset {
    public let url: URL
    public let unit: ComputeUnit
    public let model: AIModel
    public let loadSeconds: Double

    public init(url: URL, unit: ComputeUnit) async throws {
        self.url = url
        self.unit = unit
        let t0 = nowNanos()
        self.model = try await AIModel(contentsOf: url, options: unit.options)
        self.loadSeconds = Double(nowNanos() - t0) / 1e9
    }

    public var functionNames: [String] { model.functionNames }

    public func function(_ name: String) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: name) else {
            throw TTSError.message("\(url.lastPathComponent): no function '\(name)'")
        }
        return fn
    }
}

// MARK: - NDArray helpers

/// Build a float32 NDArray from a Swift array.
public func nd(_ values: [Float], _ shape: [Int]) -> NDArray {
    NDArray(scalars: values, shape: shape)
}

public func nd(_ values: [Int32], _ shape: [Int]) -> NDArray {
    NDArray(scalars: values, shape: shape)
}

/// Build a float16 NDArray by narrowing float32 values. Used when the flow-LM / flow
/// decoder assets are the fp16 export — the graph's declared input dtype must be matched
/// exactly or the runtime rejects the call.
public func ndHalf(_ values: [Float], _ shape: [Int]) -> NDArray {
    NDArray(scalars: values.map { Float16($0) }, shape: shape)
}

/// Allocate an NDArray of the given scalar type and fill it from float32 source data.
///
/// This is how the KV state gets created: a *state* is a buffer the runtime mutates in
/// place across calls; the host owns it, hands the runtime a mutable view each call, and
/// never reads it back. Zero-initialised, not NaN — a masked SDPA still multiplies V by a
/// zero weight and `0 * NaN` is `NaN` (NOTES.md §6, blocker iii-b).
public func makeState(_ values: [Float], shape: [Int], half: Bool) -> NDArray {
    var a = NDArray(shape: shape, scalarType: half ? .float16 : .float32)
    if half {
        var mv = a.mutableView(as: Float16.self)
        mv.withUnsafeMutablePointer { p, _, _ in
            for i in 0..<values.count { p[i] = Float16(values[i]) }
        }
    } else {
        var mv = a.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { p, _, _ in
            for i in 0..<values.count { p[i] = values[i] }
        }
    }
    return a
}

/// Flatten any float output to `[Float]`, row-major, widening fp16.
public func flat(_ array: NDArray) -> [Float] {
    switch array.scalarType {
    case .float16: return flatten(array, as: Float16.self)
    case .float32: return flatten(array, as: Float.self)
    default: preconditionFailure("unsupported output scalar type \(array.scalarType)")
    }
}

private func flatten<T: BinaryFloatingPoint & BitwiseCopyable>(_ a: NDArray, as _: T.Type) -> [Float] {
    let total = a.shape.reduce(1, *)
    var out = [Float](repeating: 0, count: total)
    a.view(as: T.self).withUnsafePointer { p, shp, strides in
        var expected = 1
        var contiguous = true
        for d in stride(from: shp.count - 1, through: 0, by: -1) {
            if strides[d] != expected { contiguous = false; break }
            expected *= shp[d]
        }
        if contiguous {
            for i in 0..<total { out[i] = Float(p[i]) }
            return
        }
        var idx = [Int](repeating: 0, count: shp.count)
        for i in 0..<total {
            var off = 0
            for d in 0..<shp.count { off += idx[d] * strides[d] }
            out[i] = Float(p[off])
            var d = shp.count - 1
            while d >= 0 {
                idx[d] += 1
                if idx[d] < shp[d] { break }
                idx[d] = 0
                d -= 1
            }
        }
    }
    return out
}

/// Pull one named output as `[Float]`, consuming it out of the `Outputs` bag.
public func take(_ outputs: inout InferenceFunction.Outputs, _ name: String) throws -> [Float] {
    guard let v = outputs.remove(name)?.ndArray else {
        throw TTSError.message("missing output '\(name)'")
    }
    return flat(v)
}

// MARK: - metrics

/// Cosine similarity in double precision, so the metric itself never diverges.
public func cosine(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for (x, y) in zip(a, b) {
        dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y)
    }
    if na == 0 || nb == 0 { return na == nb ? 1.0 : 0.0 }
    return dot / (na.squareRoot() * nb.squareRoot())
}

public func maxAbsDiff(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Double {
    var m = 0.0
    for (x, y) in zip(a, b) { m = max(m, abs(Double(x) - Double(y))) }
    return m
}

public func rms(_ a: [Float]) -> Double {
    var s = 0.0
    for x in a { s += Double(x) * Double(x) }
    return (s / Double(max(1, a.count))).squareRoot()
}
