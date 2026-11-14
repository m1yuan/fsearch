import Foundation

enum EmbedError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self {
        case .message(let text): return text
        }
    }
}

func l2Normalize(_ values: inout [Float]) {
    var sum: Float = 0
    for value in values { sum += value * value }
    let norm = sum.squareRoot()
    guard norm > 0 else { return }
    for index in values.indices { values[index] /= norm }
}

/// Matryoshka: keep the leading `dim` values and L2-normalize again.
func truncateNormalize(_ values: [Float], dim: Int) -> [Float] {
    var out = Array(values.prefix(dim))
    if out.count < dim {
        return out
    }
    l2Normalize(&out)
    return out
}

func bf16ToFloat(_ bits: UInt16) -> Float {
    Float(bitPattern: UInt32(bits) << 16)
}

struct EmbedTable {
    let rows: Int
    let hidden: Int
    let scale: Float
    private let data: Data

    init(url: URL, rows: Int, hidden: Int, scale: Float) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let need = rows * hidden * 2
        guard data.count >= need else {
            throw EmbedError.message("embed table \(url.path) is \(data.count) bytes, expected \(need)")
        }
        self.data = data
        self.rows = rows
        self.hidden = hidden
        self.scale = scale
    }

    /// One token. The file is raw bfloat16, unscaled; Gemma's forward multiplies by sqrt(hidden).
    func lookup(_ id: Int) -> [Float] {
        guard id >= 0, id < rows else { return [Float](repeating: 0, count: hidden) }
        var out = [Float](repeating: 0, count: hidden)
        let start = id * hidden
        data.withUnsafeBytes { raw in
            let words = raw.bindMemory(to: UInt16.self)
            for index in 0..<hidden {
                out[index] = bf16ToFloat(words[start + index]) * scale
            }
        }
        return out
    }
}

struct Manifest: Decodable {
    var model: String?
    var nativeDim: Int?
    var hidden: Int?
    var embedScale: Double?
    var embedTable: String?
    var embedRows: Int?
    var buckets: [Int]?
    var inputs: Inputs?
    var output: String?

    struct Inputs: Decodable {
        var embeds: String?
        var mask: String?
    }

    enum CodingKeys: String, CodingKey {
        case model
        case nativeDim = "native_dim"
        case hidden
        case embedScale = "embed_scale"
        case embedTable = "embed_table"
        case embedRows = "embed_rows"
        case buckets
        case inputs
        case output
    }
}

func loadManifest(in directory: URL) -> Manifest {
    let url = directory.appendingPathComponent("manifest.json")
    guard let data = try? Data(contentsOf: url), let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else {
        return Manifest(model: "embeddinggemma-2", nativeDim: 768, hidden: 512, embedScale: 22.627417, embedTable: "embed_tokens.bf16", embedRows: 262_144, buckets: [64, 128, 256, 512], inputs: nil, output: "embedding")
    }
    return manifest
}
