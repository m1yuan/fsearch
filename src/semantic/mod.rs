//! Local semantic search. Optional (`--features semantic`) and, even then,
//! idle until `semantic.json` sets `"enabled"` or a semantic request arrives.
//!
//! The daemon's lexical index is untouched. This module reads the same files
//! `content::eligible` would put in the trigram index (size caps, gitignore,
//! skip list, generated-file names), chunks them, and asks a local
//! OpenAI-compatible server for vectors. Vectors live in SQLite next to the
//! other index files. A change notification only enqueues a directory; a
//! background thread embeds, slowly, and a failed batch stays queued.
//!
//! Hybrid search asks the engine for a filename search and a content grep,
//! then fuses those lists with the vector list by reciprocal rank fusion.
//! The fusion itself is [`rrf`]; the engine call sits in the daemon.

mod chunk;
mod embed;
mod rrf;
mod store;

use chunk::ChunkOptions;
use embed::Client;
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub use rrf::{Fused, K as RRF_K, Ranked, fuse};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8765/v1/embeddings";
const DOC_TEMPLATE: &str = "title: {title} | text: {text}";
const QUERY_TEMPLATE: &str = "task: search result | query: {query}";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
}

impl DType {
    fn parse(s: &str) -> Result<DType, String> {
        match s {
            "f16" | "fp16" => Ok(DType::F16),
            "f32" | "fp32" => Ok(DType::F32),
            other => Err(format!("dtype must be f16 or f32, got {other}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub enabled: bool,
    pub endpoint: String,
    pub model: String,
    pub dim: usize,
    pub dtype: DType,
    pub batch: usize,
    pub document_template: String,
    pub query_template: String,
    pub target_tokens: usize,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub max_chunks: usize,
    pub batch_pause_ms: u64,
    pub timeout_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: DEFAULT_ENDPOINT.into(),
            model: "embeddinggemma-2".into(),
            dim: 256,
            dtype: DType::F16,
            batch: 8,
            document_template: DOC_TEMPLATE.into(),
            query_template: QUERY_TEMPLATE.into(),
            target_tokens: 384,
            max_tokens: 512,
            overlap_tokens: 64,
            max_chunks: 64,
            batch_pause_ms: 25,
            timeout_ms: 30_000,
        }
    }
}

impl Config {
    pub fn chunk_options(&self) -> ChunkOptions {
        ChunkOptions {
            target_tokens: self.target_tokens,
            max_tokens: self.max_tokens,
            overlap_tokens: self.overlap_tokens,
            max_chunks: self.max_chunks,
        }
    }

    /// `dir/semantic.json`, then `FSEARCH_SEMANTIC`, `FSEARCH_EMBED_URL`,
    /// `FSEARCH_SEMANTIC_DIM` and `FSEARCH_SEMANTIC_DTYPE`. A missing file is
    /// the default (disabled). A broken file is an error so a typo is not
    /// silent; the daemon logs it and keeps serving lexical search.
    pub fn load(dir: &Path) -> Result<Config, String> {
        let path = dir.join("semantic.json");
        let mut cfg = match std::fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Config::default(),
            Err(e) => return Err(format!("{}: {e}", path.display())),
            Ok(b) => parse_config(&b).map_err(|e| format!("{}: {e}", path.display()))?,
        };
        if let Ok(v) = std::env::var("FSEARCH_SEMANTIC") {
            cfg.enabled = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Ok(v) = std::env::var("FSEARCH_EMBED_URL")
            && !v.is_empty()
        {
            cfg.endpoint = v;
        }
        if let Ok(v) = std::env::var("FSEARCH_SEMANTIC_DIM")
            && let Ok(n) = v.parse::<usize>()
        {
            cfg.dim = n;
        }
        if let Ok(v) = std::env::var("FSEARCH_SEMANTIC_DTYPE") {
            cfg.dtype = DType::parse(&v)?;
        }
        cfg.check()?;
        Ok(cfg)
    }

    fn check(&self) -> Result<(), String> {
        if self.dim == 0 || self.dim > 8192 {
            return Err(format!("dim {} is out of range", self.dim));
        }
        if self.batch == 0 {
            return Err("batch must be at least 1".into());
        }
        if self.max_tokens < self.target_tokens {
            return Err("max_tokens must be >= target_tokens".into());
        }
        if self.overlap_tokens >= self.target_tokens {
            return Err("overlap_tokens must be < target_tokens".into());
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct RawConfig {
    #[serde(default)]
    enabled: bool,
    endpoint: Option<String>,
    model: Option<String>,
    dim: Option<usize>,
    dtype: Option<String>,
    batch: Option<usize>,
    document_template: Option<String>,
    query_template: Option<String>,
    target_tokens: Option<usize>,
    max_tokens: Option<usize>,
    overlap_tokens: Option<usize>,
    max_chunks: Option<usize>,
    batch_pause_ms: Option<u64>,
    timeout_ms: Option<u64>,
}

fn parse_config(bytes: &[u8]) -> Result<Config, String> {
    let raw: RawConfig = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let mut cfg = Config { enabled: raw.enabled, ..Config::default() };
    if let Some(v) = raw.endpoint {
        cfg.endpoint = v;
    }
    if let Some(v) = raw.model {
        cfg.model = v;
    }
    if let Some(v) = raw.dim {
        cfg.dim = v;
    }
    if let Some(v) = raw.dtype {
        cfg.dtype = DType::parse(&v)?;
    }
    if let Some(v) = raw.batch {
        cfg.batch = v;
    }
    if let Some(v) = raw.document_template {
        cfg.document_template = v;
    }
    if let Some(v) = raw.query_template {
        cfg.query_template = v;
    }
    if let Some(v) = raw.target_tokens {
        cfg.target_tokens = v;
    }
    if let Some(v) = raw.max_tokens {
        cfg.max_tokens = v;
    }
    if let Some(v) = raw.overlap_tokens {
        cfg.overlap_tokens = v;
    }
    if let Some(v) = raw.max_chunks {
        cfg.max_chunks = v;
    }
    if let Some(v) = raw.batch_pause_ms {
        cfg.batch_pause_ms = v;
    }
    if let Some(v) = raw.timeout_ms {
        cfg.timeout_ms = v;
    }
    cfg.check()?;
    Ok(cfg)
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SyncStats {
    pub seen: usize,
    pub embedded: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub chunks: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Status {
    pub enabled: bool,
    pub files: usize,
    pub chunks: usize,
    pub pending: usize,
    pub dim: usize,
    pub dtype: &'static str,
    pub endpoint: String,
    pub model: String,
    pub document_template: String,
    pub query_template: String,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub path: String,
    pub snippet: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub lexical_score: Option<f32>,
}

fn clip(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        return s.to_string();
    }
    let end = s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len());
    format!("{}…", &s[..end])
}

/// The index the daemon and the tests share. Writes go through one SQLite
/// connection; the search matrix is rebuilt when a write lands.
pub struct Index {
    home: Vec<u8>,
    cfg: Config,
    store: store::Store,
    client: Client,
    cache: Mutex<Option<Arc<store::Matrix>>>,
}

impl Index {
    pub fn open(dir: &Path, home: &str) -> Result<Index, String> {
        let cfg = Config::load(dir)?;
        let home = home.trim_end_matches('/').as_bytes().to_vec();
        let store = store::Store::open(&dir.join("semantic.sqlite"))?;
        let client = Client { endpoint: cfg.endpoint.clone(), model: cfg.model.clone(), timeout: Duration::from_millis(cfg.timeout_ms) };
        Ok(Index { home, cfg, store, client, cache: Mutex::new(None) })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn reset(&self) -> Result<(), String> {
        self.store.reset()?;
        *self.cache.lock().unwrap() = None;
        Ok(())
    }

    pub fn status(&self) -> Result<Status, String> {
        let (files, chunks, pending) = self.store.counts()?;
        Ok(Status {
            enabled: self.cfg.enabled,
            files,
            chunks,
            pending,
            dim: self.cfg.dim,
            dtype: self.cfg.dtype.as_str(),
            endpoint: self.cfg.endpoint.clone(),
            model: self.cfg.model.clone(),
            document_template: self.cfg.document_template.clone(),
            query_template: self.cfg.query_template.clone(),
        })
    }

    pub fn pending(&self) -> Result<usize, String> {
        Ok(self.store.counts()?.2)
    }

    pub fn enqueue(&self, path: &[u8], recursive: bool) -> Result<(), String> {
        self.store.enqueue(path, recursive)
    }

    /// Run one queued directory. `Ok(false)` means the queue had nothing due.
    /// An embeddings outage defers the job and returns the error; the row
    /// stays, so a restart continues.
    pub fn drain_one(&self) -> Result<bool, String> {
        let Some(job) = self.store.next_job(now())? else { return Ok(false) };
        match self.sync_path(&job.path, job.recursive) {
            Ok(_) => {
                self.store.finish_job(job.id)?;
                Ok(true)
            }
            Err(e) => {
                self.store.defer_job(job.id, now().saturating_add(2))?;
                Err(e)
            }
        }
    }

    pub fn sync_path(&self, path: &[u8], recursive: bool) -> Result<SyncStats, String> {
        self.ensure_layout()?;
        if !path_exists(path) {
            let gone = self.store.paths_under(path, true)?;
            let removed = self.store.delete_paths(&gone)?;
            self.invalidate();
            return Ok(SyncStats { removed, ..SyncStats::default() });
        }
        if path_is_file(path) {
            return self.sync_one(path);
        }
        self.walk(path, recursive)
    }

    /// Vector search. `under` keeps hits inside that folder. Scores are cosine.
    pub fn search(&self, query: &str, limit: usize, under: Option<&[u8]>) -> Result<Vec<Hit>, String> {
        let query = query.trim();
        if query.is_empty() {
            return Err("empty semantic query".into());
        }
        self.ensure_layout()?;
        let text = embed::format_query(&self.cfg.query_template, query);
        let vecs = self.client.embed(&[text], self.cfg.dim)?;
        let matrix = self.matrix()?;
        let q = &vecs[0];
        let allow = |row: &store::Row| under.is_none_or(|root| inside(&row.path, root));
        let top = store::nearest(&matrix, q, limit, allow);
        let texts = self.store.texts(&top.iter().map(|(i, _)| matrix.rows[*i].id).collect::<Vec<_>>())?;
        let mut by_id: std::collections::HashMap<i64, String> = texts.into_iter().collect();
        Ok(top
            .into_iter()
            .map(|(i, score)| {
                let row = &matrix.rows[i];
                Hit {
                    path: String::from_utf8_lossy(&row.path).into_owned(),
                    snippet: clip(by_id.remove(&row.id).as_deref().unwrap_or(""), 320),
                    byte_start: row.byte_start,
                    byte_end: row.byte_end,
                    line_start: row.line_start,
                    line_end: row.line_end,
                    score,
                    vector_score: Some(score),
                    lexical_score: None,
                }
            })
            .collect())
    }

    /// The vector list, shaped for [`fuse`].
    pub fn vector_ranked(&self, query: &str, limit: usize, under: Option<&[u8]>) -> Result<Vec<Ranked>, String> {
        Ok(self
            .search(query, limit, under)?
            .into_iter()
            .map(|h| Ranked {
                path: h.path,
                snippet: h.snippet,
                byte_start: h.byte_start,
                byte_end: h.byte_end,
                line_start: h.line_start,
                line_end: h.line_end,
                score: h.score,
            })
            .collect())
    }

    fn matrix(&self) -> Result<Arc<store::Matrix>, String> {
        let mut slot = self.cache.lock().unwrap();
        if slot.is_none() {
            *slot = Some(Arc::new(self.store.load_matrix()?));
        }
        Ok(slot.clone().unwrap())
    }

    fn invalidate(&self) {
        *self.cache.lock().unwrap() = None;
    }

    fn ensure_layout(&self) -> Result<(), String> {
        match self.store.layout()? {
            None => self.store.set_layout(self.cfg.dim, self.cfg.dtype),
            Some((dim, dtype)) if dim == self.cfg.dim && dtype == self.cfg.dtype => Ok(()),
            Some((dim, dtype)) => Err(format!(
                "semantic index is {dim} {}, config is {} {}; reset it with `fsearch semantic index --reset`",
                dtype.as_str(),
                self.cfg.dim,
                self.cfg.dtype.as_str()
            )),
        }
    }

    fn walk(&self, dir: &[u8], recursive: bool) -> Result<SyncStats, String> {
        let mut stats = SyncStats::default();
        let mut seen = Vec::new();
        self.walk_into(dir, recursive, &mut stats, &mut seen)?;
        let indexed = self.store.paths_under(dir, recursive)?;
        let stale: Vec<Vec<u8>> = indexed.into_iter().filter(|p| !seen.iter().any(|s| s == p)).collect();
        stats.removed = self.store.delete_paths(&stale)?;
        if stats.embedded > 0 || stats.removed > 0 {
            self.invalidate();
        }
        Ok(stats)
    }

    fn walk_into(&self, dir: &[u8], recursive: bool, stats: &mut SyncStats, seen: &mut Vec<Vec<u8>>) -> Result<(), String> {
        if dir != self.home.as_slice() && !crate::content::in_scope(dir, &self.home) {
            return Ok(());
        }
        let entries = std::fs::read_dir(std::ffi::OsStr::from_bytes(dir)).map_err(|e| format!("{}: {e}", String::from_utf8_lossy(dir)))?;
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        for ent in entries {
            let ent = ent.map_err(|e| e.to_string())?;
            let name = ent.file_name();
            let name = std::os::unix::ffi::OsStrExt::as_bytes(name.as_os_str());
            if name == b"." || name == b".." {
                continue;
            }
            let path = join(dir, name);
            let ft = ent.file_type().map_err(|e| e.to_string())?;
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                dirs.push(path);
            } else if ft.is_file() {
                files.push(path);
            }
        }
        files.sort();
        for path in files {
            self.consider(&path, stats, seen)?;
        }
        if recursive {
            dirs.sort();
            for sub in dirs {
                if descend(&sub, &self.home) {
                    self.walk_into(&sub, true, stats, seen)?;
                }
            }
        }
        Ok(())
    }

    /// Returns whether the file was (re)embedded, so the walker can pause.
    fn consider(&self, path: &[u8], stats: &mut SyncStats, seen: &mut Vec<Vec<u8>>) -> Result<bool, String> {
        let Some(meta) = file_meta(path) else { return Ok(false) };
        if !crate::content::eligible(path, meta.size, &self.home) {
            return Ok(false);
        }
        let Some((text, hash)) = read_text(path)? else {
            return Ok(false);
        };
        stats.seen += 1;
        seen.push(path.to_vec());
        if let Some(prev) = self.store.file_state(path)?
            && prev.hash == hash
        {
            if prev.mtime != meta.mtime || prev.size != meta.size {
                self.store.touch(path, &store::FileState { mtime: meta.mtime, size: meta.size, hash })?;
            }
            stats.unchanged += 1;
            return Ok(false);
        }
        let n = self.embed_file(path, &text, &store::FileState { mtime: meta.mtime, size: meta.size, hash })?;
        stats.embedded += 1;
        stats.chunks += n;
        if self.cfg.batch_pause_ms > 0 {
            std::thread::sleep(Duration::from_millis(self.cfg.batch_pause_ms));
        }
        Ok(true)
    }

    fn sync_one(&self, path: &[u8]) -> Result<SyncStats, String> {
        let mut stats = SyncStats::default();
        let mut seen = Vec::new();
        if !crate::content::in_scope(path, &self.home) && !path.starts_with(&self.home) {
            return Ok(stats);
        }
        self.consider(path, &mut stats, &mut seen)?;
        if seen.is_empty() {
            stats.removed = usize::from(self.store.delete_path(path)?);
        }
        if stats.embedded > 0 || stats.removed > 0 {
            self.invalidate();
        }
        Ok(stats)
    }

    fn embed_file(&self, path: &[u8], text: &str, state: &store::FileState) -> Result<usize, String> {
        let parts = chunk::chunk(text, &self.cfg.chunk_options());
        let title = embed::file_title(path);
        let mut stored = Vec::with_capacity(parts.len());
        for batch in parts.chunks(self.cfg.batch.max(1)) {
            let inputs: Vec<String> = batch.iter().map(|c| embed::format_document(&self.cfg.document_template, &title, &c.text)).collect();
            let vecs = self.client.embed(&inputs, self.cfg.dim)?;
            for (c, vector) in batch.iter().zip(vecs) {
                stored.push(store::StoredChunk {
                    byte_start: c.start as u64,
                    byte_end: c.end as u64,
                    line_start: c.line_start,
                    line_end: c.line_end,
                    text: c.text.clone(),
                    vector,
                });
            }
        }
        let n = stored.len();
        self.store.replace_file(path, state, &stored, self.cfg.dtype)?;
        Ok(n)
    }
}

struct Meta {
    mtime: u32,
    size: u64,
}

fn file_meta(path: &[u8]) -> Option<Meta> {
    let m = std::fs::symlink_metadata(std::ffi::OsStr::from_bytes(path)).ok()?;
    if !m.is_file() {
        return None;
    }
    let mtime = m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs().min(u32::MAX as u64) as u32).unwrap_or(0);
    Some(Meta { mtime, size: m.len() })
}

fn path_exists(path: &[u8]) -> bool {
    std::fs::symlink_metadata(std::ffi::OsStr::from_bytes(path)).is_ok()
}

fn path_is_file(path: &[u8]) -> bool {
    std::fs::symlink_metadata(std::ffi::OsStr::from_bytes(path)).is_ok_and(|m| m.file_type().is_file())
}

fn join(dir: &[u8], name: &[u8]) -> Vec<u8> {
    if dir.is_empty() || dir == b"/" {
        let mut p = Vec::with_capacity(name.len() + 1);
        p.push(b'/');
        p.extend_from_slice(name);
        p
    } else if dir.ends_with(b"/") {
        [dir, name].concat()
    } else {
        let mut p = Vec::with_capacity(dir.len() + 1 + name.len());
        p.extend_from_slice(dir);
        p.push(b'/');
        p.extend_from_slice(name);
        p
    }
}

fn inside(path: &[u8], root: &[u8]) -> bool {
    if root.is_empty() || root == b"/" {
        return true;
    }
    let root = root.strip_suffix(b"/").unwrap_or(root);
    path == root || path.starts_with(root) && path.get(root.len()) == Some(&b'/')
}

/// A directory the content index would open: in scope, and not gitignored
/// (a probe name of a kind `eligible` accepts is refused when the folder is).
fn descend(dir: &[u8], home: &[u8]) -> bool {
    if dir == home {
        return true;
    }
    if !crate::content::in_scope(dir, home) {
        return false;
    }
    let rs = join(dir, b"fsearch-probe.rs");
    let txt = join(dir, b"fsearch-probe.txt");
    crate::content::eligible(&rs, 1, home) || crate::content::eligible(&txt, 1, home)
}

fn read_text(path: &[u8]) -> Result<Option<(String, u64)>, String> {
    let mut file = match std::fs::File::open(std::ffi::OsStr::from_bytes(path)) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    let mut bytes = Vec::new();
    std::io::Read::take(&mut file, crate::content::MAX_REPO_FILE + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if !is_text(&bytes) {
        return Ok(None);
    }
    let hash = content_hash(&bytes);
    Ok(Some((String::from_utf8_lossy(&bytes).into_owned(), hash)))
}

fn is_text(buf: &[u8]) -> bool {
    !buf.contains(&0) && !long_lines(buf)
}

fn long_lines(buf: &[u8]) -> bool {
    let need = buf.len() / 1001;
    buf.len() >= 4096 && buf.iter().filter(|&&b| b == b'\n').take(need).count() < need
}

/// Same mix as `content`'s file hash, copied so a semantic rebuild does not
/// have to reach into the trigram index.
fn content_hash(b: &[u8]) -> u64 {
    #[inline]
    fn mix(x: u64, y: u64) -> u64 {
        let m = x as u128 * y as u128;
        m as u64 ^ (m >> 64) as u64
    }
    const K: [u64; 4] = [0x9E37_79B9_7F4A_7C15, 0xC2B2_AE3D_27D4_EB4F, 0x1656_67B1_9E37_79F9, 0x85EB_CA77_C2B2_AE63];
    let word = |w: &[u8], i: usize| u64::from_le_bytes(w[i * 8..i * 8 + 8].try_into().unwrap());
    let mut s = [K[0] ^ b.len() as u64, K[1], K[2], K[3] ^ (b.len() as u64).rotate_left(32)];
    let mut step = |w: &[u8]| {
        let x = [word(w, 0), word(w, 1), word(w, 2), word(w, 3)];
        s = [mix(x[0] ^ K[0], x[1] ^ s[0]), mix(x[1] ^ K[1], x[0] ^ s[1]), mix(x[2] ^ K[2], x[3] ^ s[2]), mix(x[3] ^ K[3], x[2] ^ s[3])];
    };
    let (blocks, rem) = b.as_chunks::<32>();
    for w in blocks {
        step(w);
    }
    let mut last = [0u8; 32];
    last[..rem.len()].copy_from_slice(rem);
    step(&last);
    mix(mix(s[0] ^ K[1], s[2] ^ K[2]) ^ s[1], s[3] ^ K[0])
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Shared {
    index: Index,
    inbox: Mutex<Vec<(Vec<u8>, bool)>>,
    cv: Condvar,
    stop: AtomicBool,
}

/// Background indexer. `note` only copies paths (it runs on the thread that
/// applies filesystem events). The worker persists them, then embeds.
pub struct Service {
    shared: Arc<Shared>,
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Service {
    pub fn start(dir: &Path, home: &str) -> Result<Arc<Service>, String> {
        let shared =
            Arc::new(Shared { index: Index::open(dir, home)?, inbox: Mutex::new(Vec::new()), cv: Condvar::new(), stop: AtomicBool::new(false) });
        let worker = shared.clone();
        let join = std::thread::Builder::new().name("fsearch-semantic".into()).spawn(move || worker_loop(worker)).map_err(|e| e.to_string())?;
        Ok(Arc::new(Service { shared, join: Mutex::new(Some(join)) }))
    }

    pub fn note(&self, changes: &[(&[u8], bool)]) {
        if changes.is_empty() {
            return;
        }
        let mut inbox = self.shared.inbox.lock().unwrap();
        for (p, r) in changes {
            inbox.push((p.to_vec(), *r));
        }
        self.shared.cv.notify_one();
    }

    pub fn index_path(&self, path: &[u8], recursive: bool) -> Result<(), String> {
        self.shared.index.enqueue(path, recursive)?;
        self.shared.cv.notify_one();
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize, under: Option<&[u8]>) -> Result<Vec<Hit>, String> {
        self.shared.index.search(query, limit, under)
    }

    pub fn vector_ranked(&self, query: &str, limit: usize, under: Option<&[u8]>) -> Result<Vec<Ranked>, String> {
        self.shared.index.vector_ranked(query, limit, under)
    }

    pub fn status(&self) -> Result<Status, String> {
        let mut s = self.shared.index.status()?;
        s.pending += self.shared.inbox.lock().unwrap().len();
        Ok(s)
    }

    pub fn reset(&self) -> Result<(), String> {
        self.shared.index.reset()
    }

    pub fn config(&self) -> &Config {
        self.shared.index.config()
    }

    pub fn wait_idle(&self, timeout: Duration) -> Result<(), String> {
        let start = Instant::now();
        loop {
            let queued = self.shared.inbox.lock().unwrap().len();
            let pending = self.shared.index.pending()?;
            if queued == 0 && pending == 0 {
                return Ok(());
            }
            if start.elapsed() > timeout {
                return Err(format!("semantic index still has {} jobs", queued + pending));
            }
            std::thread::sleep(Duration::from_millis(15));
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.cv.notify_all();
        if let Some(join) = self.join.lock().unwrap().take() {
            let _ = join.join();
        }
    }
}

fn worker_loop(shared: Arc<Shared>) {
    while !shared.stop.load(Ordering::Relaxed) {
        let batch = {
            let mut inbox = shared.inbox.lock().unwrap();
            if inbox.is_empty() {
                let (_guard, _) = shared.cv.wait_timeout(inbox, Duration::from_millis(200)).unwrap();
                inbox = _guard;
            }
            std::mem::take(&mut *inbox)
        };
        if shared.stop.load(Ordering::Relaxed) && batch.is_empty() {
            break;
        }
        for (path, recursive) in batch {
            if let Err(e) = shared.index.enqueue(&path, recursive) {
                log(format!("semantic enqueue: {e}"));
            }
        }
        match shared.index.drain_one() {
            Ok(true) => {}
            Ok(false) => {}
            Err(e) => {
                log(format!("semantic index: {e}"));
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    }
}

fn log(msg: String) {
    eprintln!("{} {msg}", crate::query::now_secs());
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::OnceLock;

    fn root(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fsearch-sem-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    struct Mock {
        port: u16,
    }

    fn mock() -> &'static Mock {
        static MOCK: OnceLock<Mock> = OnceLock::new();
        MOCK.get_or_init(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            std::thread::spawn(move || {
                for conn in listener.incoming().flatten() {
                    let _ = serve_one(conn);
                }
            });
            Mock { port }
        })
    }

    fn serve_one(mut conn: TcpStream) -> std::io::Result<()> {
        conn.set_read_timeout(Some(Duration::from_secs(2))).ok();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            let n = conn.read(&mut tmp)?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = std::str::from_utf8(&buf[..i]).unwrap_or("");
                let len: usize = head
                    .lines()
                    .find_map(|l| {
                        let l = l.trim_end_matches('\r');
                        l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).and_then(|(_, v)| v.trim().parse().ok())
                    })
                    .unwrap_or(0);
                if buf.len() >= i + 4 + len {
                    let v: Value = serde_json::from_slice(&buf[i + 4..i + 4 + len]).unwrap_or(json!({}));
                    let dim = v["dimensions"].as_u64().unwrap_or(16) as usize;
                    let inputs: Vec<String> = match &v["input"] {
                        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
                        Value::String(s) => vec![s.clone()],
                        _ => Vec::new(),
                    };
                    let data: Vec<Value> = inputs
                        .iter()
                        .enumerate()
                        .map(|(i, t)| json!({"object": "embedding", "index": i, "embedding": embed::hashed_embedding(t, dim)}))
                        .collect();
                    let resp = json!({"object": "list", "data": data}).to_string();
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

    fn write_cfg(dir: &Path, port: u16) {
        let cfg = json!({
            "enabled": true,
            "endpoint": format!("http://127.0.0.1:{port}/v1/embeddings"),
            "model": "test",
            "dim": 32,
            "dtype": "f16",
            "batch": 4,
            "batch_pause_ms": 0,
            "timeout_ms": 5000,
            "target_tokens": 48,
            "max_tokens": 64,
            "overlap_tokens": 8,
            "max_chunks": 8
        });
        std::fs::write(dir.join("semantic.json"), cfg.to_string()).unwrap();
    }

    #[test]
    fn indexes_text_skips_ignored_and_searches() {
        let port = mock().port;
        let home = root("home");
        write_cfg(&home, port);
        let notes = home.join("notes");
        std::fs::create_dir_all(&notes).unwrap();
        std::fs::write(notes.join("aurora.md"), "The northern lights are caused by charged particles from the sun.\n").unwrap();
        std::fs::write(notes.join("sky.md"), "北极光是太阳带电粒子进入大气层产生的现象。\n").unwrap();
        std::fs::write(notes.join("photo.png"), [0u8, 1, 2, 3]).unwrap();
        let repo = home.join("repo");
        std::fs::create_dir_all(repo.join(".git")).unwrap();
        std::fs::create_dir_all(repo.join("secret")).unwrap();
        std::fs::write(repo.join(".gitignore"), "secret/\n").unwrap();
        std::fs::write(repo.join("visible.md"), "A visible note about routers.\n").unwrap();
        std::fs::write(repo.join("secret").join("hidden.md"), "The wifi password is tumbleweed.\n").unwrap();
        std::fs::create_dir_all(home.join("node_modules").join("pkg")).unwrap();
        std::fs::write(home.join("node_modules").join("pkg").join("index.md"), "dependency text\n").unwrap();

        let index = Index::open(&home, home.to_str().unwrap()).unwrap();
        let stats = index.sync_path(home.as_os_str().as_encoded_bytes(), true).unwrap();
        assert!(stats.embedded >= 3, "{stats:?}");
        let (files, _, _) = index.store.counts().unwrap();
        let paths = index.store.paths_under(home.as_os_str().as_encoded_bytes(), true).unwrap();
        let has = |tail: &str| paths.iter().any(|p| String::from_utf8_lossy(p).ends_with(tail));
        assert!(has("aurora.md") && has("sky.md") && has("visible.md"), "{paths:?} files {files}");
        assert!(!has("hidden.md"), "gitignored: {paths:?}");
        assert!(!has("index.md"), "node_modules: {paths:?}");
        assert!(!has("photo.png"));

        let en = index.search("charged particles from the sun", 3, None).unwrap();
        assert!(en[0].path.ends_with("aurora.md"), "{:?}", en.iter().map(|h| &h.path).collect::<Vec<_>>());
        assert!(en[0].line_start >= 1);
        assert!(!en[0].snippet.is_empty());
        let zh = index.search("太阳带电粒子进入大气层", 3, None).unwrap();
        assert!(zh[0].path.ends_with("sky.md"), "{:?}", zh.iter().map(|h| &h.path).collect::<Vec<_>>());

        std::fs::write(notes.join("aurora.md"), "A completely different note about sourdough starter.\n").unwrap();
        index.sync_path(notes.join("aurora.md").as_os_str().as_encoded_bytes(), false).unwrap();
        let again = index.search("sourdough starter", 3, None).unwrap();
        assert!(again[0].path.ends_with("aurora.md"), "{:?}", again.iter().map(|h| &h.path).collect::<Vec<_>>());

        std::fs::remove_file(notes.join("sky.md")).unwrap();
        let removed = index.sync_path(notes.as_os_str().as_encoded_bytes(), false).unwrap();
        assert!(removed.removed >= 1, "{removed:?}");
        let paths = index.store.paths_under(home.as_os_str().as_encoded_bytes(), true).unwrap();
        assert!(!paths.iter().any(|p| String::from_utf8_lossy(p).ends_with("sky.md")));
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn background_worker_drains_a_note() {
        let port = mock().port;
        let home = root("worker");
        write_cfg(&home, port);
        std::fs::write(home.join("note.md"), "Key: tumbleweed-4417 lives in the drawer.\n").unwrap();
        let svc = Service::start(&home, home.to_str().unwrap()).unwrap();
        let path = home.as_os_str().as_encoded_bytes();
        svc.note(&[(path, true)]);
        svc.wait_idle(Duration::from_secs(10)).unwrap();
        let hits = svc.search("tumbleweed-4417", 2, None).unwrap();
        assert!(hits[0].path.ends_with("note.md"), "{hits:?}");
        drop(svc);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn default_config_is_disabled() {
        let dir = root("cfg");
        let cfg = Config::load(&dir).unwrap();
        assert!(!cfg.enabled);
        assert_eq!(cfg.dim, 256);
        assert_eq!(cfg.dtype, DType::F16);
        assert_eq!(cfg.endpoint, DEFAULT_ENDPOINT);
        assert!(cfg.document_template.contains("{title}") && cfg.query_template.contains("{query}"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
