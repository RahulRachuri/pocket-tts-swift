import Foundation

/// WAV I/O. The host writes IEEE-float32 WAV by default — the oracle wavs are float32
/// (scipy writes the float array as-is), so the parity comparison never runs through a
/// 16-bit quantisation. `writeWAV16` exists for players that want PCM16.
public enum WAV {
    public static func writeFloat32(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        let dataBytes = samples.count * 4
        var out = Data()
        func le32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func le16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        out.append(contentsOf: Array("RIFF".utf8))
        le32(UInt32(36 + dataBytes))
        out.append(contentsOf: Array("WAVEfmt ".utf8))
        le32(16)
        le16(3)                          // IEEE float
        le16(1)                          // mono
        le32(UInt32(sampleRate))
        le32(UInt32(sampleRate * 4))     // byte rate
        le16(4)                          // block align
        le16(32)                         // bits per sample
        out.append(contentsOf: Array("data".utf8))
        le32(UInt32(dataBytes))
        samples.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        try out.write(to: url)
    }

    public static func writePCM16(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        var pcm = [Int16](repeating: 0, count: samples.count)
        for i in 0..<samples.count {
            let v = max(-1.0, min(1.0, samples[i])) * 32767.0
            pcm[i] = Int16(v.rounded())
        }
        let dataBytes = pcm.count * 2
        var out = Data()
        func le32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        func le16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        out.append(contentsOf: Array("RIFF".utf8))
        le32(UInt32(36 + dataBytes))
        out.append(contentsOf: Array("WAVEfmt ".utf8))
        le32(16)
        le16(1)                          // PCM
        le16(1)
        le32(UInt32(sampleRate))
        le32(UInt32(sampleRate * 2))
        le16(2)
        le16(16)
        out.append(contentsOf: Array("data".utf8))
        le32(UInt32(dataBytes))
        pcm.withUnsafeBufferPointer { out.append(Data(buffer: $0)) }
        try out.write(to: url)
    }

    /// Read a mono WAV as float32 samples (PCM16 or IEEE-float32). Used by `--ref-wav`
    /// to compare a run against the oracle wav.
    public static func readMono(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: url)
        guard data.count > 44, data.subdata(in: 0..<4) == Data("RIFF".utf8) else {
            throw TTSError.message("\(url.lastPathComponent): not a RIFF/WAV file")
        }
        var i = 12
        var fmt: (format: Int, channels: Int, rate: Int, bits: Int)? = nil
        while i + 8 <= data.count {
            let cid = data.subdata(in: i..<(i + 4))
            var sz: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &sz) { data.copyBytes(to: $0, from: (i + 4)..<(i + 8)) }
            let body = i + 8
            if cid == Data("fmt ".utf8) {
                func u16(_ o: Int) -> Int {
                    var v: UInt16 = 0
                    _ = withUnsafeMutableBytes(of: &v) { data.copyBytes(to: $0, from: o..<(o + 2)) }
                    return Int(v)
                }
                func u32(_ o: Int) -> Int {
                    var v: UInt32 = 0
                    _ = withUnsafeMutableBytes(of: &v) { data.copyBytes(to: $0, from: o..<(o + 4)) }
                    return Int(v)
                }
                fmt = (u16(body), u16(body + 2), u32(body + 4), u16(body + 14))
            } else if cid == Data("data".utf8) {
                guard let f = fmt else { throw TTSError.message("WAV: data chunk before fmt") }
                let end = min(body + Int(sz), data.count)
                let raw = data.subdata(in: body..<end)
                var samples: [Float]
                switch (f.format, f.bits) {
                case (1, 16):
                    samples = raw.withUnsafeBytes { buf in
                        buf.bindMemory(to: Int16.self).map { Float($0) / 32768.0 }
                    }
                case (3, 32):
                    samples = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                default:
                    throw TTSError.message("WAV: unsupported format \(f.format)/\(f.bits)-bit")
                }
                if f.channels > 1 {   // average down to mono
                    let ch = f.channels
                    var mono = [Float](repeating: 0, count: samples.count / ch)
                    for j in 0..<mono.count {
                        var acc: Float = 0
                        for c in 0..<ch { acc += samples[j * ch + c] }
                        mono[j] = acc / Float(ch)
                    }
                    samples = mono
                }
                return (samples, f.rate)
            }
            i = body + Int(sz) + (Int(sz) & 1)
        }
        throw TTSError.message("\(url.lastPathComponent): no data chunk")
    }
}
