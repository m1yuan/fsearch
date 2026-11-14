import CoreML
import Foundation
import Tokenizers

#if canImport(CoreAI)
import CoreAI
#endif

struct Health: Codable {
    var ok: Bool
    var backend: String
    var model: String
    var nativeDim: Int
    var buckets: [Int]
    var compute: String
    var placement: String
    var error: String?

    enum CodingKeys: String, CodingKey {
        case ok, backend, model, buckets, compute, placement, error
        case nativeDim = "native_dim"
    }
}

/// Packs Gemma tokens on the host and runs a fixed-shape bucket.
actor Sidecar {
    let directory: URL
    let manifest: Manifest
    let nativeDim: Int
    let hidden: Int
    let buckets: [Int]
    let table: EmbedTable
    let tokenizer: any Tokenizer
    let embedName: String
    let maskName: String
    let outputName: String
    private var coreml: [Int: MLModel] = [:]
    private var placement = "not loaded"
    private var backend = "none"
    private var loadError: String?

    #if canImport(CoreAI)
    private var coreai: [Int: InferenceFunction] = [:]
    #endif

    init(directory: URL) async throws {
        self.directory = directory
        let manifest = loadManifest(in: directory)
        self.manifest = manifest
        nativeDim = manifest.nativeDim ?? 768
        hidden = manifest.hidden ?? 512
        buckets = (manifest.buckets ?? [64, 128, 256, 512]).sorted()
        let rows = manifest.embedRows ?? 262_144
        let scale = Float(manifest.embedScale ?? Double(hidden).squareRoot())
        let tableName = manifest.embedTable ?? "embed_tokens.bf16"
        table = try EmbedTable(url: directory.appendingPathComponent(tableName), rows: rows, hidden: hidden, scale: scale)
        tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        embedName = manifest.inputs?.embeds ?? "inputs_embeds"
        maskName = manifest.inputs?.mask ?? "attention_mask"
        outputName = manifest.output ?? "embedding"
        try await load()
    }

    func health() -> Health {
        Health(
            ok: loadError == nil && backend != "none",
            backend: backend,
            model: manifest.model ?? "embeddinggemma-2",
            nativeDim: nativeDim,
            buckets: buckets,
            compute: backend == "coreai" ? "neuralEngine-preferred" : "cpuAndNeuralEngine",
            placement: placement,
            error: loadError
        )
    }

    func embed(texts: [String], dimensions: Int?) async throws -> [[Float]] {
        if let loadError { throw EmbedError.message(loadError) }
        let dim = dimensions ?? nativeDim
        guard dim > 0, dim <= nativeDim else {
            throw EmbedError.message("dimensions \(dim) is outside 1...\(nativeDim)")
        }
        var out: [[Float]] = []
        for text in texts {
            var vector = try await embedOne(text)
            if vector.count > nativeDim, vector.count % nativeDim == 0 {
                vector = meanRows(vector, width: nativeDim)
            } else if vector.count > hidden, vector.count != nativeDim, vector.count % hidden == 0 {
                vector = meanRows(vector, width: hidden)
            }
            if vector.count > dim {
                vector = truncateNormalize(vector, dim: dim)
            } else {
                l2Normalize(&vector)
            }
            out.append(vector)
        }
        return out
    }

    private func embedOne(_ text: String) async throws -> [Float] {
        let limit = buckets.max() ?? 512
        var ids = tokenizer.encode(text: text, addSpecialTokens: true)
        if ids.count <= limit {
            return try await run(ids: ids)
        }
        // Longer than the widest bucket: non-overlapping windows, then a mean.
        // fsearch chunks below this, so this is the safety net.
        var acc = [Float](repeating: 0, count: nativeDim)
        var parts = 0
        var start = 0
        let window = max(8, limit - 2)
        while start < ids.count {
            let end = min(ids.count, start + window)
            var slice = Array(ids[start..<end])
            if let eos = tokenizer.eosTokenId, slice.last != eos { slice.append(eos) }
            let vector = try await run(ids: slice)
            let used = vector.count >= nativeDim ? Array(vector.prefix(nativeDim)) : vector
            if acc.count != used.count { acc = [Float](repeating: 0, count: used.count) }
            for index in used.indices { acc[index] += used[index] }
            parts += 1
            start = end
        }
        if parts > 1 {
            for index in acc.indices { acc[index] /= Float(parts) }
        }
        l2Normalize(&acc)
        return acc
    }

    private func run(ids: [Int]) async throws -> [Float] {
        let length = buckets.first { $0 >= max(ids.count, 1) } ?? (buckets.last ?? ids.count)
        let packed = pack(ids: ids, length: length)
        #if canImport(CoreAI)
        if let function = coreai[length] ?? coreai[buckets.last() ?? length] {
            return try await runCoreAI(function, embeds: packed.embeds, mask: packed.mask, length: length)
        }
        #endif
        if let model = coreml[length] ?? coreml[buckets.last() ?? length] {
            return try runCoreML(model, embeds: packed.embeds, mask: packed.mask, length: length)
        }
        throw EmbedError.message("no bucket for \(ids.count) tokens (have \(buckets))")
    }

    private func pack(ids: [Int], length: Int) -> (embeds: [Float], mask: [Float]) {
        var embeds = [Float](repeating: 0, count: length * hidden)
        var mask = [Float](repeating: 0, count: length)
        for (row, id) in ids.prefix(length).enumerated() {
            let values = table.lookup(id)
            for col in 0..<hidden { embeds[row * hidden + col] = values[col] }
            mask[row] = 1
        }
        return (embeds, mask)
    }

    private func meanRows(_ values: [Float], width: Int) -> [Float] {
        let rows = values.count / width
        guard rows > 0 else { return values }
        var acc = [Float](repeating: 0, count: width)
        for row in 0..<rows {
            for col in 0..<width { acc[col] += values[row * width + col] }
        }
        let scale = Float(rows)
        for col in acc.indices { acc[col] /= scale }
        return acc
    }

    private func load() async throws {
        let force = ProcessInfo.processInfo.environment["FSEARCH_EMBED_BACKEND"]
        if force != "coreml" {
            #if canImport(CoreAI)
            do {
                if try await loadCoreAI() { return }
            } catch {
                fputs("ane-embed: Core AI load failed (\(error)); trying Core ML\n", stderr)
                loadError = "\(error)"
            }
            #endif
        }
        if force == "coreai" {
            throw EmbedError.message(loadError ?? "Core AI backend was requested and did not load")
        }
        try loadCoreML()
    }

    private func loadCoreML() throws {
        let dir = directory.appendingPathComponent("coreml")
        let root = FileManager.default.fileExists(atPath: dir.path) ? dir : directory
        var found = 0
        for length in buckets {
            let url = root.appendingPathComponent("embed_\(length).mlpackage")
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let config = MLModelConfiguration()
            // CPU + Neural Engine. `.all` allows the GPU; do not use it.
            config.computeUnits = .cpuAndNeuralEngine
            coreml[length] = try MLModel(contentsOf: url, configuration: config)
            found += 1
        }
        guard found > 0 else {
            let message = "no Core ML packages under \(root.path) (expected embed_{64,128,256,512}.mlpackage)"
            loadError = message
            throw EmbedError.message(message)
        }
        backend = "coreml"
        let sample = root.appendingPathComponent("embed_\(buckets.first ?? 0).mlpackage")
        placement = describePlan(sample)
        loadError = nil
        fputs("ane-embed: Core ML fallback, \(found) buckets, computeUnits=cpuAndNeuralEngine\n", stderr)
    }

    private func describePlan(_ package: URL) -> String {
        // compile + MLComputePlan is async and slow. The sync load path reports
        // the configuration; validate_ane.sh prints powermetrics, which is the
        // measurement that actually shows ANE vs GPU.
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return "configured cpuAndNeuralEngine for \(package.lastPathComponent). Run scripts/validate_ane.sh to sample ane_power and gpu_power; GPU should stay near 0 mW."
    }

    #if canImport(CoreAI)
    private func loadCoreAI() async throws -> Bool {
        let dir = directory.appendingPathComponent("coreai")
        guard FileManager.default.fileExists(atPath: dir.path) else { return false }
        // A preference, not an allow-list: Core AI's public SpecializationOptions
        // has no "CPU + Neural Engine, never GPU" switch. `.default` minimizes
        // latency and picks the GPU for a dynamic or loosely-shaped graph, which
        // is Apple's macOS recipe. Fixed-shape packages plus this preference are
        // the supported way to stay on the ANE. powermetrics is the check.
        let options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        var found = 0
        for length in buckets {
            let url = dir.appendingPathComponent("embed_\(length).aimodel")
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let model = try await AIModel(contentsOf: url, options: options)
            let function = try model.loadFunction(named: "embed_\(length)") ?? model.loadFunction(named: "main")
            guard let function else {
                throw EmbedError.message("\(url.lastPathComponent) has no embed_\(length) or main function")
            }
            coreai[length] = function
            found += 1
        }
        guard found > 0 else { return false }
        backend = "coreai"
        placement = "SpecializationOptions(preferredComputeUnitKind: .neuralEngine) on \(found) fixed-shape buckets. This does not forbid a per-op GPU fallback; confirm with powermetrics (ANE up, GPU ~0)."
        loadError = nil
        fputs("ane-embed: Core AI, \(found) buckets, preferred compute unit Neural Engine\n", stderr)
        return true
    }

    private func runCoreAI(_ function: InferenceFunction, embeds: [Float], mask: [Float], length: Int) async throws -> [Float] {
        var embedsArray = NDArray(shape: [1, length, hidden], scalarType: .float16)
        var maskArray = NDArray(shape: [1, length], scalarType: .float16)
        write(embeds, into: &embedsArray)
        write(mask, into: &maskArray)
        var outputs = try await function.run(inputs: [embedName: embedsArray, maskName: maskArray])
        guard let value = outputs.remove(outputName), let array = value.ndArray else {
            throw EmbedError.message("Core AI output \(outputName) missing")
        }
        return read(array)
    }

    private func write(_ values: [Float], into array: inout NDArray) {
        var view = array.mutableView(as: Float16.self)
        guard var elements = view.contiguousElements else { return }
        let count = min(values.count, elements.count)
        for index in 0..<count { elements[index] = Float16(values[index]) }
    }

    private func read(_ array: NDArray) -> [Float] {
        if array.scalarType == .float32 {
            var out: [Float] = []
            array.view(as: Float.self).withUnsafePointer { pointer, shape, _ in
                let count = spanCount(shape)
                out.reserveCapacity(count)
                for index in 0..<count { out.append(pointer[index]) }
            }
            return out
        }
        var out: [Float] = []
        array.view(as: Float16.self).withUnsafePointer { pointer, shape, _ in
            let count = spanCount(shape)
            out.reserveCapacity(count)
            for index in 0..<count { out.append(Float(pointer[index])) }
        }
        return out
    }

    private func spanCount(_ shape: Span<Int>) -> Int {
        var count = 1
        var index = 0
        while index < shape.count {
            count *= shape[index]
            index += 1
        }
        return max(count, 0)
    }
    #endif

    private func runCoreML(_ model: MLModel, embeds: [Float], mask: [Float], length: Int) throws -> [Float] {
        let embedsArray = try multiArray(embeds, shape: [1, length, hidden])
        let maskArray = try multiArray(mask, shape: [1, length])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            embedName: MLFeatureValue(multiArray: embedsArray),
            maskName: MLFeatureValue(multiArray: maskArray),
        ])
        let output = try model.prediction(from: provider)
        guard let multi = output.featureValue(for: outputName)?.multiArrayValue else {
            throw EmbedError.message("Core ML output \(outputName) missing")
        }
        let count = multi.count
        var values = [Float](repeating: 0, count: count)
        if multi.dataType == .float16 {
            let pointer = multi.dataPointer.bindMemory(to: Float16.self, capacity: count)
            for index in 0..<count { values[index] = Float(pointer[index]) }
        } else {
            let pointer = multi.dataPointer.bindMemory(to: Float.self, capacity: count)
            for index in 0..<count { values[index] = pointer[index] }
        }
        return values
    }

    private func multiArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let numbers = shape.map { NSNumber(value: $0) }
        let array = try MLMultiArray(shape: numbers, dataType: .float16)
        let pointer = array.dataPointer.bindMemory(to: Float16.self, capacity: values.count)
        for index in values.indices { pointer[index] = Float16(values[index]) }
        return array
    }
}
