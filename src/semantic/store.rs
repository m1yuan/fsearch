//! SQLite store for chunk vectors. Rows are the source of truth; search loads
//! them into a dense matrix and scans it. That scan is the stand-in an ANN
//! index would replace: same rows, same cosine, a different `nearest`.

use super::DType;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

pub struct Store {
    conn: Mutex<Connection>,
}

#[derive(Clone, Debug)]
pub struct FileState {
    pub mtime: u32,
    pub size: u64,
    pub hash: u64,
}

#[derive(Clone, Debug)]
pub struct StoredChunk {
    pub byte_start: u64,
    pub byte_end: u64,
    pub line_start: u32,
    pub line_end: u32,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub id: i64,
    pub path: Vec<u8>,
    pub byte_start: u64,
    pub byte_end: u64,
    pub line_start: u32,
    pub line_end: u32,
}

pub struct Matrix {
    pub dim: usize,
    pub rows: Vec<Row>,
    pub data: Vec<f32>,
}

pub struct Job {
    pub id: i64,
    pub path: Vec<u8>,
    pub recursive: bool,
}

impl Store {
    pub fn open(path: &Path) -> Result<Store, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        conn.busy_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())?;
        conn.pragma_update(None, "journal_mode", "WAL").map_err(|e| e.to_string())?;
        conn.pragma_update(None, "foreign_keys", "ON").map_err(|e| e.to_string())?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS files (
                path BLOB PRIMARY KEY,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL,
                hash INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path BLOB NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                byte_start INTEGER NOT NULL,
                byte_end INTEGER NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vec BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                path BLOB NOT NULL UNIQUE,
                recursive INTEGER NOT NULL,
                not_before INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )
        .map_err(|e| e.to_string())?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    pub fn reset(&self) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch("DELETE FROM files; DELETE FROM jobs; DELETE FROM meta;").map_err(|e| e.to_string())
    }

    pub fn layout(&self) -> Result<Option<(usize, DType)>, String> {
        let conn = self.conn.lock().unwrap();
        let dim = meta_get(&conn, "dim")?;
        let dtype = meta_get(&conn, "dtype")?;
        match (dim, dtype) {
            (Some(d), Some(t)) => Ok(Some((d.parse::<usize>().map_err(|e| e.to_string())?, DType::parse(&t)?))),
            _ => Ok(None),
        }
    }

    pub fn set_layout(&self, dim: usize, dtype: DType) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        meta_set(&conn, "dim", &dim.to_string())?;
        meta_set(&conn, "dtype", dtype.as_str())?;
        Ok(())
    }

    pub fn file_state(&self, path: &[u8]) -> Result<Option<FileState>, String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT mtime, size, hash FROM files WHERE path = ?1").map_err(|e| e.to_string())?;
        let row = stmt
            .query_row(params![path], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(row.map(|(mtime, size, hash)| FileState { mtime: mtime as u32, size: size as u64, hash: u64::from_ne_bytes(hash.to_ne_bytes()) }))
    }

    pub fn touch(&self, path: &[u8], state: &FileState) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE files SET mtime = ?2, size = ?3, hash = ?4 WHERE path = ?1",
            params![path, state.mtime as i64, state.size as i64, i64::from_ne_bytes(state.hash.to_ne_bytes())],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn replace_file(&self, path: &[u8], state: &FileState, chunks: &[StoredChunk], dtype: DType) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
        tx.execute("DELETE FROM files WHERE path = ?1", params![path]).map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO files (path, mtime, size, hash) VALUES (?1, ?2, ?3, ?4)",
            params![path, state.mtime as i64, state.size as i64, i64::from_ne_bytes(state.hash.to_ne_bytes())],
        )
        .map_err(|e| e.to_string())?;
        for c in chunks {
            tx.execute(
                "INSERT INTO chunks (path, byte_start, byte_end, line_start, line_end, text) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![path, c.byte_start as i64, c.byte_end as i64, c.line_start, c.line_end, c.text],
            )
            .map_err(|e| e.to_string())?;
            let id = tx.last_insert_rowid();
            let blob = encode_vec(&c.vector, dtype);
            tx.execute(
                "INSERT INTO vectors (chunk_id, dim, dtype, vec) VALUES (?1, ?2, ?3, ?4)",
                params![id, c.vector.len() as i64, dtype.as_str(), blob],
            )
            .map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn delete_path(&self, path: &[u8]) -> Result<bool, String> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute("DELETE FROM files WHERE path = ?1", params![path]).map_err(|e| e.to_string())?;
        Ok(n > 0)
    }

    /// Indexed files equal to `dir` or strictly inside it. `recursive` false
    /// keeps only the directory's own files, not nested ones.
    pub fn paths_under(&self, dir: &[u8], recursive: bool) -> Result<Vec<Vec<u8>>, String> {
        let prefix = if dir.ends_with(b"/") { dir.to_vec() } else { [dir, b"/"].concat() };
        let end = prefix_successor(&prefix);
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT path FROM files WHERE path = ?3 OR (path >= ?1 AND path < ?2)").map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map(params![prefix, end, dir], |r| r.get::<_, Vec<u8>>(0))
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        Ok(if recursive { rows } else { rows.into_iter().filter(|p| p == dir || !p[prefix.len()..].contains(&b'/')).collect() })
    }

    pub fn delete_paths(&self, paths: &[Vec<u8>]) -> Result<usize, String> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
        let mut n = 0;
        for p in paths {
            n += tx.execute("DELETE FROM files WHERE path = ?1", params![p]).map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())?;
        Ok(n)
    }

    pub fn load_matrix(&self) -> Result<Matrix, String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.path, c.byte_start, c.byte_end, c.line_start, c.line_end, v.vec, v.dtype
                 FROM chunks c JOIN vectors v ON v.chunk_id = c.id ORDER BY c.id",
            )
            .map_err(|e| e.to_string())?;
        let iter = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, Vec<u8>>(6)?,
                    r.get::<_, String>(7)?,
                ))
            })
            .map_err(|e| e.to_string())?;
        let mut rows = Vec::new();
        let mut data = Vec::new();
        let mut dim = 0;
        for item in iter {
            let (id, path, start, end, ls, le, blob, dtype) = item.map_err(|e| e.to_string())?;
            let dtype = DType::parse(&dtype)?;
            let vec = decode_vec(&blob, dtype)?;
            if dim == 0 {
                dim = vec.len();
            } else if vec.len() != dim {
                return Err(format!("chunk {id} has dim {}, index has {dim}", vec.len()));
            }
            data.extend_from_slice(&vec);
            rows.push(Row { id, path, byte_start: start as u64, byte_end: end as u64, line_start: ls as u32, line_end: le as u32 });
        }
        Ok(Matrix { dim, rows, data })
    }

    pub fn texts(&self, ids: &[i64]) -> Result<Vec<(i64, String)>, String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT text FROM chunks WHERE id = ?1").map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(text) = stmt.query_row(params![id], |r| r.get::<_, String>(0)).optional().map_err(|e| e.to_string())? {
                out.push((*id, text));
            }
        }
        Ok(out)
    }

    pub fn counts(&self) -> Result<(usize, usize, usize), String> {
        let conn = self.conn.lock().unwrap();
        let files: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0)).map_err(|e| e.to_string())?;
        let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0)).map_err(|e| e.to_string())?;
        let jobs: i64 = conn.query_row("SELECT COUNT(*) FROM jobs", [], |r| r.get(0)).map_err(|e| e.to_string())?;
        Ok((files as usize, chunks as usize, jobs as usize))
    }

    pub fn enqueue(&self, path: &[u8], recursive: bool) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO jobs (path, recursive, not_before) VALUES (?1, ?2, 0)
             ON CONFLICT(path) DO UPDATE SET recursive = MAX(jobs.recursive, excluded.recursive), not_before = MIN(jobs.not_before, excluded.not_before)",
            params![path, recursive as i64],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// The oldest job that may run now. It stays queued until `finish_job`.
    pub fn next_job(&self, now: u64) -> Result<Option<Job>, String> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT id, path, recursive FROM jobs WHERE not_before <= ?1 ORDER BY id LIMIT 1").map_err(|e| e.to_string())?;
        let row = stmt
            .query_row(params![now as i64], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?, r.get::<_, i64>(2)?)))
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(row.map(|(id, path, rec)| Job { id, path, recursive: rec != 0 }))
    }

    pub fn finish_job(&self, id: i64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM jobs WHERE id = ?1", params![id]).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn defer_job(&self, id: i64, not_before: u64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute("UPDATE jobs SET not_before = ?2 WHERE id = ?1", params![id, not_before as i64]).map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn meta_get(conn: &Connection, key: &str) -> Result<Option<String>, String> {
    let mut stmt = conn.prepare("SELECT value FROM meta WHERE key = ?1").map_err(|e| e.to_string())?;
    stmt.query_row(params![key], |r| r.get(0)).optional().map_err(|e| e.to_string())
}

fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<(), String> {
    conn.execute("INSERT INTO meta (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value", params![key, value])
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn prefix_successor(prefix: &[u8]) -> Vec<u8> {
    let mut b = prefix.to_vec();
    while let Some(last) = b.last_mut() {
        if *last != 255 {
            *last += 1;
            return b;
        }
        b.pop();
    }
    Vec::new()
}

pub fn encode_vec(v: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => {
            let mut out = Vec::with_capacity(v.len() * 2);
            for x in v {
                out.extend_from_slice(&f32_to_f16(*x).to_le_bytes());
            }
            out
        }
    }
}

pub fn decode_vec(blob: &[u8], dtype: DType) -> Result<Vec<f32>, String> {
    match dtype {
        DType::F32 => {
            if !blob.len().is_multiple_of(4) {
                return Err("f32 vector blob is truncated".into());
            }
            let (chunks, _) = blob.as_chunks::<4>();
            Ok(chunks.iter().copied().map(f32::from_le_bytes).collect())
        }
        DType::F16 => {
            if !blob.len().is_multiple_of(2) {
                return Err("f16 vector blob is truncated".into());
            }
            let (chunks, _) = blob.as_chunks::<2>();
            Ok(chunks.iter().copied().map(u16::from_le_bytes).map(f16_to_f32).collect())
        }
    }
}

/// IEEE binary16. Round to nearest, ties to even. Enough for a normalized
/// embedding: the search converts back to f32 before the dot product.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let rest = bits & 0x7fff_ffff;
    if rest >= 0x7f80_0000 {
        let nan = if rest == 0x7f80_0000 { 0 } else { 0x200 };
        return sign | 0x7c00 | nan;
    }
    if rest >= 0x4780_0000 {
        return sign | 0x7c00;
    }
    if rest < 0x3880_0000 {
        if rest < 0x3300_0000 {
            return sign;
        }
        let shift = (0x3880_0000 - rest) >> 23;
        let mant = (rest & 0x7f_ffff) | 0x80_0000;
        let shifted = mant.wrapping_shr(shift + 1);
        let round_bit = (mant >> shift) & 1;
        let sticky = mant & ((1u32 << shift) - 1);
        let mut h = (shifted >> 12) as u16;
        if round_bit == 1 && (sticky != 0 || h & 1 == 1) {
            h += 1;
        }
        return sign | h;
    }
    let exp = ((rest >> 23) as i32) - 127 + 15;
    let mant = (rest >> 13) & 0x3ff;
    let round = (rest & 0x1000) != 0 && ((rest & 0x2000) != 0 || (rest & 0xfff) != 0);
    let mut h = sign | ((exp as u16) << 10) | (mant as u16);
    if round {
        h += 1;
    }
    h
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e: i32 = -14;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Brute-force cosine over L2-normalized rows. `allow` drops rows outside
/// the query's folder. Replace this with an ANN index without changing the
/// row layout: the matrix is plain row-major f32.
pub fn nearest(matrix: &Matrix, query: &[f32], limit: usize, allow: impl Fn(&Row) -> bool) -> Vec<(usize, f32)> {
    if limit == 0 || query.len() != matrix.dim || matrix.dim == 0 {
        return Vec::new();
    }
    let mut heap: std::collections::BinaryHeap<Top> = std::collections::BinaryHeap::new();
    for (i, row) in matrix.rows.iter().enumerate() {
        if !allow(row) {
            continue;
        }
        let score = dot(query, &matrix.data[i * matrix.dim..(i + 1) * matrix.dim]);
        if heap.len() < limit {
            heap.push(Top(i, score));
        } else if let Some(worst) = heap.peek()
            && score > worst.1
        {
            heap.pop();
            heap.push(Top(i, score));
        }
    }
    let mut out: Vec<(usize, f32)> = heap.into_iter().map(|t| (t.0, t.1)).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let n = a.len() / 8;
    for i in 0..n {
        let o = i * 8;
        for k in 0..8 {
            acc[k] += a[o + k] * b[o + k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in n * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Min-heap by score: `Ord` is reversed so `BinaryHeap` pops the worst hit.
struct Top(usize, f32);

impl PartialEq for Top {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Top {}
impl PartialOrd for Top {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Top {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.1.total_cmp(&self.1).then_with(|| other.0.cmp(&self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("fsearch-sem-store-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn unit(seed: f32, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| (seed + i as f32).sin()).collect();
        super::super::embed::l2_normalize(&mut v);
        v
    }

    #[test]
    fn f16_roundtrip_known_values() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        for x in [0.1f32, -0.25, 3.5, 100.0, 0.001, -12.5] {
            let back = f16_to_f32(f32_to_f16(x));
            let rel = (back - x).abs() / x.abs().max(1e-6);
            assert!(rel < 1e-3, "{x} -> {back}");
        }
        let mut v = unit(1.0, 32);
        let back = decode_vec(&encode_vec(&v, DType::F16), DType::F16).unwrap();
        super::super::embed::l2_normalize(&mut v);
        let dot: f32 = v.iter().zip(&back).map(|(a, b)| a * b).sum();
        let nb = back.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(dot / nb > 0.999, "{dot}");
    }

    #[test]
    fn replace_search_and_delete() {
        let dir = tmp("db");
        let store = Store::open(&dir.join("semantic.sqlite")).unwrap();
        store.set_layout(8, DType::F32).unwrap();
        let a = b"/h/a.md".as_slice();
        let b = b"/h/sub/b.md".as_slice();
        let va = unit(0.2, 8);
        let vb = unit(2.2, 8);
        let chunk = |vec: Vec<f32>, text: &str| StoredChunk {
            byte_start: 0,
            byte_end: text.len() as u64,
            line_start: 1,
            line_end: 1,
            text: text.into(),
            vector: vec,
        };
        store.replace_file(a, &FileState { mtime: 1, size: 3, hash: 9 }, &[chunk(va.clone(), "alpha")], DType::F32).unwrap();
        store.replace_file(b, &FileState { mtime: 2, size: 4, hash: 8 }, &[chunk(vb, "beta")], DType::F32).unwrap();
        let matrix = store.load_matrix().unwrap();
        let hits = nearest(&matrix, &va, 1, |_| true);
        assert_eq!(matrix.rows[hits[0].0].path, a);
        assert!(hits[0].1 > 0.99);
        let under = store.paths_under(b"/h/sub", true).unwrap();
        assert_eq!(under, vec![b.to_vec()]);
        store.delete_paths(&[b.to_vec()]).unwrap();
        let (files, chunks, _) = store.counts().unwrap();
        assert_eq!((files, chunks), (1, 1));
        store.replace_file(a, &FileState { mtime: 3, size: 3, hash: 1 }, &[chunk(va, "alpha two")], DType::F32).unwrap();
        let (files, chunks, _) = store.counts().unwrap();
        assert_eq!((files, chunks), (1, 1), "replace drops the old chunk");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn jobs_are_resumable() {
        let dir = tmp("jobs");
        let store = Store::open(&dir.join("semantic.sqlite")).unwrap();
        store.enqueue(b"/h/notes", true).unwrap();
        store.enqueue(b"/h/notes", false).unwrap();
        let job = store.next_job(10).unwrap().unwrap();
        assert!(job.recursive, "a recursive enqueue upgrades the job");
        store.defer_job(job.id, 100).unwrap();
        assert!(store.next_job(10).unwrap().is_none());
        let again = store.next_job(100).unwrap().unwrap();
        store.finish_job(again.id).unwrap();
        assert!(store.next_job(1000).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
