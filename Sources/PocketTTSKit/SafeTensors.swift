import Foundation

/// Minimal safetensors reader — exactly what the host needs, nothing more.
///
/// Format: 8-byte little-endian header length, a JSON header mapping tensor name to
/// `{dtype, shape, data_offsets}` (offsets relative to the end of the header), then the
/// raw buffer. Supported dtypes: F32, BF16 (widened to F32 — the checkpoint stores its
/// constants in bfloat16 and upstream loads them into fp32 modules, so widening here is
/// bit-faithful to what PyTorch computes with), and I64 (the voice-state offsets).
public struct SafeTensors {
    public struct Entry {
        public let dtype: String
        public let shape: [Int]
        public let byteRange: Range<Int>
    }

    private let data: Data
    private let dataStart: Int
    public let entries: [String: Entry]

    public init(url: URL) throws {
        let raw = try Data(contentsOf: url, options: .mappedIfSafe)
        guard raw.count >= 8 else { throw TTSError.message("\(url.lastPathComponent): truncated") }
        var headerLen: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLen) { raw.copyBytes(to: $0, from: 0..<8) }
        let hEnd = 8 + Int(headerLen)
        guard hEnd <= raw.count else { throw TTSError.message("\(url.lastPathComponent): bad header length") }
        let header = try JSONSerialization.jsonObject(with: raw.subdata(in: 8..<hEnd))
        guard let dict = header as? [String: Any] else {
            throw TTSError.message("\(url.lastPathComponent): header is not a JSON object")
        }
        var out: [String: Entry] = [:]
        for (name, v) in dict where name != "__metadata__" {
            guard let t = v as? [String: Any],
                  let dtype = t["dtype"] as? String,
                  let shape = t["shape"] as? [Int],
                  let offs = t["data_offsets"] as? [Int], offs.count == 2 else {
                throw TTSError.message("\(url.lastPathComponent): malformed entry '\(name)'")
            }
            out[name] = Entry(dtype: dtype, shape: shape, byteRange: offs[0]..<offs[1])
        }
        self.entries = out
        self.dataStart = hEnd
        self.data = raw
    }

    public func shape(_ name: String) throws -> [Int] {
        guard let e = entries[name] else { throw TTSError.message("missing tensor '\(name)'") }
        return e.shape
    }

    /// Tensor as float32, widening BF16 (a bf16 value is exactly the top 16 bits of the
    /// equal-valued float32, so the widening is a bit shift, not a rounding).
    public func floats(_ name: String) throws -> [Float] {
        guard let e = entries[name] else { throw TTSError.message("missing tensor '\(name)'") }
        let lo = dataStart + e.byteRange.lowerBound
        let hi = dataStart + e.byteRange.upperBound
        let raw = data.subdata(in: lo..<hi)
        switch e.dtype {
        case "F32":
            return raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        case "BF16":
            return raw.withUnsafeBytes { buf in
                let u = buf.bindMemory(to: UInt16.self)
                var out = [Float](repeating: 0, count: u.count)
                for i in 0..<u.count { out[i] = Float(bitPattern: UInt32(u[i]) << 16) }
                return out
            }
        default:
            throw TTSError.message("tensor '\(name)' has unsupported dtype \(e.dtype)")
        }
    }

    public func int64s(_ name: String) throws -> [Int64] {
        guard let e = entries[name] else { throw TTSError.message("missing tensor '\(name)'") }
        guard e.dtype == "I64" else { throw TTSError.message("tensor '\(name)' is \(e.dtype), expected I64") }
        let lo = dataStart + e.byteRange.lowerBound
        let hi = dataStart + e.byteRange.upperBound
        return data.subdata(in: lo..<hi).withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
    }
}
