//! OpenAI-compatible embeddings client, and the prompt strings EmbeddingGemma
//! expects. Prefixes are applied here so the sidecar embeds text as given
//! and another model is a config change, not a code change.

use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Client {
    pub endpoint: String,
    pub model: String,
    pub timeout: Duration,
}

impl Client {
    /// One batch. Vectors come back L2-normalized and exactly `dim` long
    /// (a longer server vector is truncated Matryoshka-style and renormalized).
    pub fn embed(&self, inputs: &[String], dim: usize) -> Result<Vec<Vec<f32>>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if dim == 0 || dim > 8192 {
            return Err(format!("bad embedding dim {dim}"));
        }
        let body = json!({
            "input": inputs,
            "model": self.model,
            "dimensions": dim,
            "encoding_format": "float",
        });
        let raw = post_json(&self.endpoint, &body.to_string(), self.timeout)?;
        let v: Value = serde_json::from_str(&raw).map_err(|e| format!("embeddings response: {e}"))?;
        if let Some(err) = v.get("error") {
            let msg = err.get("message").and_then(Value::as_str).unwrap_or("embeddings error");
            return Err(msg.to_string());
        }
        let data = v.get("data").and_then(Value::as_array).ok_or("embeddings response has no data")?;
        let mut rows: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for item in data {
            let index = item.get("index").and_then(Value::as_u64).unwrap_or(rows.len() as u64) as usize;
            let emb = item.get("embedding").and_then(Value::as_array).ok_or("embedding is not an array")?;
            let mut vec = Vec::with_capacity(emb.len());
            for n in emb {
                let x = n.as_f64().ok_or("embedding value is not a number")? as f32;
                if !x.is_finite() {
                    return Err("embedding contained a non-finite value".into());
                }
                vec.push(x);
            }
            rows.push((index, vec));
        }
        if rows.len() != inputs.len() {
            return Err(format!("embeddings returned {} vectors for {} inputs", rows.len(), inputs.len()));
        }
        rows.sort_by_key(|(i, _)| *i);
        rows.into_iter().map(|(_, v)| fit(v, dim)).collect()
    }
}

fn fit(mut v: Vec<f32>, dim: usize) -> Result<Vec<f32>, String> {
    if v.len() < dim {
        return Err(format!("endpoint returned {} dims, config wants {dim}", v.len()));
    }
    v.truncate(dim);
    l2_normalize(&mut v);
    Ok(v)
}

pub fn l2_normalize(v: &mut [f32]) {
    let mut s = 0f32;
    for x in v.iter() {
        s += x * x;
    }
    let n = s.sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// `title: {title} | text: {text}` for EmbeddingGemma. `{title}` is the file
/// name, or `none`. A template with no `{text}` still gets the chunk appended,
/// so a typo in the config cannot embed the title alone.
pub fn format_document(template: &str, title: &str, text: &str) -> String {
    let title = if title.is_empty() { "none" } else { title };
    let mut s = template.replace("{title}", title).replace("{text}", text);
    if !template.contains("{text}") {
        if !s.is_empty() && !s.ends_with([' ', '\n']) {
            s.push(' ');
        }
        s.push_str(text);
    }
    s
}

pub fn format_query(template: &str, query: &str) -> String {
    if template.is_empty() {
        return query.to_string();
    }
    let s = template.replace("{query}", query).replace("{q}", query);
    if template.contains("{query}") || template.contains("{q}") { s } else { format!("{s}{query}") }
}

pub fn file_title(path: &[u8]) -> String {
    let name = path.rsplit(|b| *b == b'/').next().unwrap_or(path);
    let s = String::from_utf8_lossy(name);
    if s.is_empty() { "none".into() } else { s.into_owned() }
}

/// Bag of character n-grams. Deterministic stand-in for a model: shared
/// phrases land near each other, which is all the integration test needs.
#[cfg_attr(not(test), allow(dead_code))]
pub fn hashed_embedding(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut v = vec![0f32; dim];
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        v[0] = 1.0;
        return v;
    }
    // Trigrams and longer, not unigrams: a shared prompt ("title:", "query:")
    // is mostly common letters, and those were drowning the actual phrase.
    for n in 2..=4 {
        if chars.len() < n {
            continue;
        }
        let weight = n as f32;
        for gram in chars.windows(n) {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for c in gram {
                h ^= *c as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
            let i = (h as usize) % dim;
            let sign = if h & 0x100 == 0 { weight } else { -weight };
            v[i] += sign;
        }
    }
    l2_normalize(&mut v);
    v
}

fn post_json(url: &str, body: &str, timeout: Duration) -> Result<String, String> {
    let (host, port, path) = parse_url(url)?;
    let mut stream = TcpStream::connect((host.as_str(), port)).map_err(|e| format!("embeddings {url}: {e}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).map_err(|e| format!("embeddings write: {e}"))?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| format!("embeddings read: {e}"))?;
    parse_http(&buf)
}

fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("embeddings endpoint must be http://, got {url}"))?;
    let (hostport, path) = match rest.split_once('/') {
        Some((h, p)) => (h, format!("/{p}")),
        None => (rest, "/".into()),
    };
    if hostport.is_empty() {
        return Err(format!("embeddings endpoint has no host: {url}"));
    }
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) if !h.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => {
            (h.to_string(), p.parse::<u16>().map_err(|_| format!("bad port in {url}"))?)
        }
        _ => (hostport.to_string(), 80),
    };
    Ok((host, port, path))
}

fn parse_http(buf: &[u8]) -> Result<String, String> {
    let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").ok_or("embeddings response had no header")?;
    let head = std::str::from_utf8(&buf[..head_end]).map_err(|_| "embeddings response header is not utf-8")?;
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap_or("");
    let code: u16 = status.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut len = None;
    for line in lines {
        if let Some(v) = line.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).map(|(_, v)| v.trim()) {
            len = v.parse::<usize>().ok();
        }
    }
    let body = &buf[head_end + 4..];
    let body = match len {
        Some(n) if body.len() >= n => &body[..n],
        Some(n) => return Err(format!("embeddings response truncated ({}/{} bytes)", body.len(), n)),
        None => body,
    };
    let text = String::from_utf8_lossy(body).into_owned();
    if !(200..300).contains(&code) {
        return Err(format!("embeddings HTTP {code}: {text}"));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn templates_match_embeddinggemma() {
        assert_eq!(format_document("title: {title} | text: {text}", "Notes.md", "hello"), "title: Notes.md | text: hello");
        assert_eq!(format_document("title: {title} | text: {text}", "", "hello"), "title: none | text: hello");
        assert_eq!(format_query("task: search result | query: {query}", "lights"), "task: search result | query: lights");
        assert_eq!(format_document("{text}", "a.rs", "fn main"), "fn main");
    }

    #[test]
    fn hashed_vectors_are_unit_and_stable() {
        let a = hashed_embedding("charged particles from the sun", 32);
        let b = hashed_embedding("charged particles from the sun", 32);
        let c = hashed_embedding("a recipe for bread", 32);
        assert_eq!(a, b);
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        assert!((dot(&a, &a) - 1.0).abs() < 1e-5);
        assert!(dot(&a, &c) < dot(&a, &hashed_embedding("particles from the sun hit the atmosphere", 32)));
    }

    #[test]
    fn prefixed_chinese_is_closer_than_a_gitignore() {
        let q = format_query(crate::semantic::QUERY_TEMPLATE, "太阳带电粒子进入大气层");
        let sky = format_document(crate::semantic::DOC_TEMPLATE, "sky.md", "北极光是太阳带电粒子进入大气层产生的现象。\n");
        let git = format_document(crate::semantic::DOC_TEMPLATE, ".gitignore", "secret/\n");
        let (qv, sv, gv) = (hashed_embedding(&q, 32), hashed_embedding(&sky, 32), hashed_embedding(&git, 32));
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(dot(&qv, &sv) > dot(&qv, &gv), "sky {} git {}", dot(&qv, &sv), dot(&qv, &gv));
    }

    #[test]
    fn client_sorts_truncates_and_normalizes() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let port = spawn_mock(seen.clone(), Mock::ExtraDims);
        let client = Client { endpoint: format!("http://127.0.0.1:{port}/v1/embeddings"), model: "test".into(), timeout: Duration::from_secs(5) };
        let got = client.embed(&["second".into(), "first".into()], 8).unwrap();
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|v| v.len() == 8));
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((norm(&got[0]) - 1.0).abs() < 1e-4);
        let inputs = seen.lock().unwrap().clone();
        assert_eq!(inputs, vec![vec!["second".to_string(), "first".to_string()]]);
    }

    enum Mock {
        ExtraDims,
    }

    fn spawn_mock(seen: Arc<Mutex<Vec<Vec<String>>>>, _mode: Mock) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten().take(8) {
                let _ = answer(conn, &seen);
            }
        });
        port
    }

    fn answer(mut conn: TcpStream, seen: &Mutex<Vec<Vec<String>>>) -> std::io::Result<()> {
        conn.set_read_timeout(Some(Duration::from_secs(2))).ok();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = conn.read(&mut tmp)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = std::str::from_utf8(&buf[..i]).unwrap_or("");
                let len: usize = head
                    .split("\r\n")
                    .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).map(|(_, v)| v.trim().parse().ok()))
                    .flatten()
                    .unwrap_or(0);
                let have = buf.len() - (i + 4);
                if have >= len {
                    let body = &buf[i + 4..i + 4 + len];
                    let v: Value = serde_json::from_slice(body).unwrap_or(json!({}));
                    let inputs: Vec<String> = match &v["input"] {
                        Value::String(s) => vec![s.clone()],
                        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                        _ => Vec::new(),
                    };
                    seen.lock().unwrap().push(inputs.clone());
                    let dim = v["dimensions"].as_u64().unwrap_or(8) as usize;
                    let mut data = Vec::new();
                    // Reverse the order: the client must sort by index.
                    for (i, text) in inputs.iter().enumerate().rev() {
                        let mut emb = hashed_embedding(text, dim + 4);
                        emb.extend([0.0, 0.0]);
                        data.push(json!({"object": "embedding", "index": i, "embedding": emb}));
                    }
                    let resp = json!({"object": "list", "data": data, "model": "test"}).to_string();
                    let msg = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp}",
                        resp.len()
                    );
                    conn.write_all(msg.as_bytes())?;
                    break;
                }
            }
        }
        Ok(())
    }
}
