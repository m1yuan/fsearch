import Darwin
import Foundation

/// One HTTP/1.1 request at a time, bound to 127.0.0.1. The ANE call is the
/// slow part; overlapping it does not help and the model is not re-entrant here.
enum Http {
    static func serve(port: UInt16, handler: (String, String, Data) -> (Int, String)) -> Never {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        if fd < 0 { fail("socket") }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout.size(ofValue: yes)))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        if bound != 0 { fail("bind 127.0.0.1:\(port)") }
        if listen(fd, 16) != 0 { fail("listen") }
        fputs("ane-embed listening on 127.0.0.1:\(port)\n", stderr)
        while true {
            let client = accept(fd, nil, nil)
            if client < 0 { continue }
            var buffer = Data()
            var tmp = [UInt8](repeating: 0, count: 64 * 1024)
            while true {
                let n = read(client, &tmp, tmp.count)
                if n <= 0 { break }
                buffer.append(tmp, count: n)
                if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                    let head = String(data: buffer.subdata(in: 0..<headerEnd.lowerBound), encoding: .utf8) ?? ""
                    let length = contentLength(head)
                    let bodyStart = headerEnd.upperBound
                    if buffer.count >= bodyStart + length {
                        let body = buffer.subdata(in: bodyStart..<(bodyStart + length))
                        let (status, payload) = handler(method(head), path(head), body)
                        let bytes = Data(payload.utf8)
                        var response = "HTTP/1.1 \(status) \r\nContent-Type: application/json\r\nContent-Length: \(bytes.count)\r\nConnection: close\r\n\r\n"
                        response += payload
                        _ = response.withCString { write(client, $0, strlen($0)) }
                        break
                    }
                }
                if buffer.count > 8 << 20 { break }
            }
            close(client)
        }
    }

    private static func method(_ head: String) -> String {
        head.split(separator: " ").first.map(String.init) ?? "GET"
    }

    private static func path(_ head: String) -> String {
        let parts = head.split(separator: " ")
        return parts.count > 1 ? String(parts[1]) : "/"
    }

    private static func contentLength(_ head: String) -> Int {
        for line in head.split(separator: "\r\n") {
            let pair = line.split(separator: ":", maxSplits: 1)
            if pair.count == 2, pair[0].trimmingCharacters(in: .whitespaces).caseInsensitiveCompare("content-length") == .orderedSame {
                return Int(pair[1].trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }
        return 0
    }

    private static func fail(_ what: String) -> Never {
        fputs("ane-embed: \(what): \(String(cString: strerror(errno)))\n", stderr)
        exit(1)
    }
}
