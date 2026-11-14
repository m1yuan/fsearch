//! Reciprocal rank fusion. Ranks are 1-based; the constant `k` (60 here)
//! keeps a single first place from drowning out a document that is merely
//! good on every list. An ANN index is not involved: callers pass already
//! ordered lists.

pub const K: f64 = 60.0;

/// One hit in one ranked list. `score` is that list's own score (cosine,
/// name score, ...); fusion uses the position, not `score`.
#[derive(Clone, Debug)]
pub struct Ranked {
    pub path: String,
    pub snippet: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct Fused {
    pub path: String,
    pub snippet: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub line_start: u32,
    pub line_end: u32,
    /// Sum of `1 / (k + rank)` over the lists that named this path.
    pub score: f32,
    pub vector_score: Option<f32>,
    pub lexical_score: Option<f32>,
}

/// `lists[0]` is the vector list when it is present: its snippet wins, and
/// its score is recorded as `vector_score`. Later lists are lexical; the
/// best lexical score is kept. A path's rank in a list is its first row.
pub fn fuse(lists: &[Vec<Ranked>], k: f64, limit: usize) -> Vec<Fused> {
    if limit == 0 {
        return Vec::new();
    }
    let mut acc: Vec<Acc> = Vec::new();
    let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (li, list) in lists.iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        let mut rank = 0usize;
        for row in list {
            if !seen.insert(row.path.clone()) {
                continue;
            }
            rank += 1;
            let add = (1.0 / (k + rank as f64)) as f32;
            let idx = if let Some(&i) = at.get(&row.path) {
                i
            } else {
                let i = acc.len();
                at.insert(row.path.clone(), i);
                acc.push(Acc::from_row(row));
                i
            };
            acc[idx].score += add;
            if li == 0 {
                acc[idx].vector_score = Some(row.score);
                if !row.snippet.is_empty() {
                    acc[idx].take_snippet(row);
                }
            } else {
                acc[idx].lexical_score = Some(match acc[idx].lexical_score {
                    Some(s) => s.max(row.score),
                    None => row.score,
                });
                if acc[idx].snippet.is_empty() && !row.snippet.is_empty() {
                    acc[idx].take_snippet(row);
                }
            }
        }
    }
    acc.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
    acc.truncate(limit);
    acc.into_iter().map(Acc::into_fused).collect()
}

struct Acc {
    path: String,
    snippet: String,
    byte_start: u64,
    byte_end: u64,
    line_start: u32,
    line_end: u32,
    score: f32,
    vector_score: Option<f32>,
    lexical_score: Option<f32>,
}

impl Acc {
    fn from_row(row: &Ranked) -> Acc {
        Acc {
            path: row.path.clone(),
            snippet: String::new(),
            byte_start: 0,
            byte_end: 0,
            line_start: 0,
            line_end: 0,
            score: 0.0,
            vector_score: None,
            lexical_score: None,
        }
    }

    fn take_snippet(&mut self, row: &Ranked) {
        self.snippet.clone_from(&row.snippet);
        self.byte_start = row.byte_start;
        self.byte_end = row.byte_end;
        self.line_start = row.line_start;
        self.line_end = row.line_end;
    }

    fn into_fused(self) -> Fused {
        Fused {
            path: self.path,
            snippet: self.snippet,
            byte_start: self.byte_start,
            byte_end: self.byte_end,
            line_start: self.line_start,
            line_end: self.line_end,
            score: self.score,
            vector_score: self.vector_score,
            lexical_score: self.lexical_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(path: &str, score: f32) -> Ranked {
        Ranked { path: path.into(), snippet: path.into(), byte_start: 0, byte_end: 0, line_start: 1, line_end: 1, score }
    }

    #[test]
    fn k60_sums_reciprocal_ranks() {
        let a = vec![row("one", 3.0), row("two", 2.0), row("three", 1.0)];
        let b = vec![row("three", 9.0), row("one", 8.0)];
        let fused = fuse(&[a, b], K, 10);
        let score = |p: &str| fused.iter().find(|h| h.path == p).unwrap().score;
        let expect = |rank: usize| (1.0 / (K + rank as f64)) as f32;
        assert!((score("one") - (expect(1) + expect(2))).abs() < 1e-6);
        assert!((score("three") - (expect(3) + expect(1))).abs() < 1e-6);
        assert!((score("two") - expect(2)).abs() < 1e-6);
        assert!(score("one") > score("three") && score("three") > score("two"));
        assert_eq!(fused[0].vector_score, Some(3.0));
        assert_eq!(fused[0].lexical_score, Some(8.0));
    }

    #[test]
    fn duplicate_in_one_list_counts_once() {
        let a = vec![row("a", 1.0), row("a", 0.5), row("b", 0.2)];
        let fused = fuse(&[a], 60.0, 10);
        assert_eq!(fused.len(), 2);
        assert!((fused[0].score - (1.0 / 61.0) as f32).abs() < 1e-6);
    }

    #[test]
    fn vector_snippet_beats_a_lexical_one() {
        let mut v = row("a", 0.9);
        v.snippet = "from-vector".into();
        v.line_start = 4;
        let mut lex = row("a", 10.0);
        lex.snippet = "from-grep".into();
        lex.line_start = 9;
        let fused = fuse(&[vec![v], vec![lex]], 60.0, 5);
        assert_eq!(fused[0].snippet, "from-vector");
        assert_eq!(fused[0].line_start, 4);
    }
}
