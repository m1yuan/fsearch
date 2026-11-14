import Foundation

@main
struct AneEmbed {
    static func main() async {
        do {
            try await run()
        } catch {
            fputs("ane-embed: \(error)\n", stderr)
            exit(1)
        }
    }

    static func run() async throws {
        var port: UInt16 = 8765
        var directory = defaultDirectory()
        var args = Array(CommandLine.arguments.dropFirst())
        var index = 0
        while index < args.count {
            switch args[index] {
            case "--port":
                index += 1
                port = UInt16(args[index]) ?? port
            case "--model":
                index += 1
                directory = URL(fileURLWithPath: args[index])
            case "-h", "--help":
                fputs("usage: ane-embed [--port 8765] [--model DIR]\n", stderr)
                fputs("serves POST /v1/embeddings and GET /health on 127.0.0.1\n", stderr)
                return
            default:
                fputs("ane-embed: unknown argument \(args[index])\n", stderr)
                exit(2)
            }
            index += 1
        }
        if let env = ProcessInfo.processInfo.environment["FSEARCH_EMBED_PORT"], let value = UInt16(env) {
            port = value
        }
        if let env = ProcessInfo.processInfo.environment["FSEARCH_EMBED_DIR"], !env.is_empty {
            directory = URL(fileURLWithPath: env)
        }

        let sidecar: Sidecar
        do {
            fputs("ane-embed: loading \(directory.path)\n", stderr)
            fputs("ane-embed: first specialization can take a few minutes and is cached afterwards\n", stderr)
            sidecar = try await Sidecar(directory: directory)
        } catch {
            // Still serve /health so a failed conversion is visible without guessing.
            let message = "\(error)"
            fputs("ane-embed: model not loaded: \(message)\n", stderr)
            Http.serve(port: port) { method, path, _ in
                if method == "GET", path == "/health" {
                    let body = "{\"ok\":false,\"backend\":\"none\",\"error\":\(jsonString(message))}\n"
                    return (200, body)
                }
                return (503, "{\"error\":{\"message\":\(jsonString(message)),\"type\":\"model_not_loaded\"}}\n")
            }
        }

        let health = await sidecar.health()
        fputs("ane-embed: \(health.backend) native_dim=\(health.nativeDim) \(health.placement)\n", stderr)
        Http.serve(port: port) { method, path, body in
            if method == "GET", path == "/health" {
                let value = blocking { await sidecar.health() }
                return (200, (try? encode(value)) ?? "{\"ok\":false}\n")
            }
            guard method == "POST", path == "/v1/embeddings" else {
                return (404, "{\"error\":{\"message\":\"not found\",\"type\":\"not_found\"}}\n")
            }
            do {
                let request = try JSONSerialization.jsonObject(with: body) as? [String: Any] ?? [:]
                let texts = inputTexts(request["input"])
                if texts.isEmpty { throw EmbedError.message("input must be a string or a list of strings") }
                let dim = (request["dimensions"] as? NSNumber)?.intValue ?? (request["dimensions"] as? Int)
                let vectors = try blocking { try await sidecar.embed(texts: texts, dimensions: dim) }
                var data: [[String: Any]] = []
                var tokens = 0
                for (index, vector) in vectors.enumerated() {
                    tokens += vector.count
                    data.append(["object": "embedding", "index": index, "embedding": vector.map { Double($0) }])
                }
                let payload: [String: Any] = [
                    "object": "list",
                    "data": data,
                    "model": health.model,
                    "usage": ["prompt_tokens": tokens, "total_tokens": tokens],
                ]
                let raw = try JSONSerialization.data(withJSONObject: payload)
                return (200, String(decoding: raw, as: UTF8.self) + "\n")
            } catch {
                return (400, "{\"error\":{\"message\":\(jsonString(String(describing: error))),\"type\":\"invalid_request_error\"}}\n")
            }
        }
    }

    static func defaultDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FSearch/models/embeddinggemma-2")
    }
}

private func inputTexts(_ value: Any?) -> [String] {
    switch value {
    case let text as String: return [text]
    case let list as [String]: return list
    case let list as [Any]: return list.compactMap { $0 as? String }
    default: return []
    }
}

private func jsonString(_ text: String) -> String {
    let data = try? JSONSerialization.data(withJSONObject: [text])
    let wrapped = data.flatMap { String(data: $0, encoding: .utf8) } ?? "[\"error\"]"
    return String(wrapped.dropFirst().dropLast())
}

private func encode<T: Encodable>(_ value: T) throws -> String {
    let data = try JSONEncoder().encode(value)
    return String(decoding: data, as: UTF8.self) + "\n"
}

/// The HTTP loop is synchronous. Model calls are async (Core AI specialization
/// and `InferenceFunction.run`). One request is in flight at a time.
private func blocking<T>(_ body: @escaping () async throws -> T) throws -> T {
    let box = ResultBox<T>()
    let sem = DispatchSemaphore(value: 0)
    Task {
        do { box.result = .success(try await body()) }
        catch { box.result = .failure(error) }
        sem.signal()
    }
    sem.wait()
    guard let result = box.result else { throw EmbedError.message("embed task did not finish") }
    return try result.get()
}

private final class ResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}
