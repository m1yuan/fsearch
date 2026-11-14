//! Split a text file into overlapping chunks of a few hundred tokens.
//!
//! The tokenizer is a heuristic, not the model's: whitespace words, with a
//! CJK character counting as its own token and a very long word split so a
//! minified line cannot become one chunk. Markdown headings and fenced code
//! blocks are boundaries, so a chunk prefers to start on a heading and to
//! keep a short fence intact.

#[derive(Clone, Debug)]
pub struct ChunkOptions {
    /// Soft size. A chunk closes once adding the next piece would pass this,
    /// when a later piece still fits in `max_tokens`.
    pub target_tokens: usize,
    /// Hard cap. A single piece longer than this is sliced on token boundaries.
    pub max_tokens: usize,
    /// Tokens repeated at the start of the next chunk. Not applied across a
    /// heading: a heading starts clean.
    pub overlap_tokens: usize,
    /// Stop after this many chunks (the rest of a huge file is left out).
    pub max_chunks: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self { target_tokens: 384, max_tokens: 512, overlap_tokens: 64, max_chunks: 64 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Byte offsets into the original text, on char boundaries.
    pub start: usize,
    pub end: usize,
    /// 1-based, inclusive.
    pub line_start: u32,
    pub line_end: u32,
    pub text: String,
}

pub fn chunk(text: &str, opt: &ChunkOptions) -> Vec<Chunk> {
    let target = opt.target_tokens.max(1);
    let max = opt.max_tokens.max(target);
    let overlap = opt.overlap_tokens.min(target.saturating_sub(1));
    let pieces = pieces(text);
    let mut out = Vec::new();
    let mut cur_start: Option<usize> = None;
    let mut cur_end = 0;
    let mut cur_tokens = 0usize;

    let flush = |out: &mut Vec<Chunk>, cur_start: &mut Option<usize>, cur_end: &mut usize, cur_tokens: &mut usize| {
        if let Some(start) = cur_start.take()
            && start < *cur_end
        {
            push_chunk(out, text, start, *cur_end);
        }
        *cur_tokens = 0;
    };

    for piece in &pieces {
        if out.len() >= opt.max_chunks {
            break;
        }
        let spans = token_spans(&text[piece.start..piece.end]);
        let n = spans.len();
        if n == 0 {
            continue;
        }
        if piece.hard && cur_tokens > 0 {
            flush(&mut out, &mut cur_start, &mut cur_end, &mut cur_tokens);
        }
        if (n > max || (cur_tokens > 0 && cur_tokens + n > max)) && cur_tokens > 0 && cur_tokens + n > target {
            flush(&mut out, &mut cur_start, &mut cur_end, &mut cur_tokens);
        }
        if n > max {
            if cur_tokens > 0 {
                flush(&mut out, &mut cur_start, &mut cur_end, &mut cur_tokens);
            }
            slice_long(text, piece.start, &spans, max, overlap, opt.max_chunks, &mut out);
            continue;
        }
        if cur_tokens > 0 && cur_tokens + n > target {
            let end = cur_end;
            flush(&mut out, &mut cur_start, &mut cur_end, &mut cur_tokens);
            if !piece.hard
                && overlap > 0
                && let Some((s, t)) = overlap_prefix(text, end, overlap)
            {
                cur_start = Some(s);
                cur_tokens = t;
            }
        }
        if cur_start.is_none() {
            cur_start = Some(piece.start);
        }
        cur_end = piece.end;
        cur_tokens += n;
    }
    if out.len() < opt.max_chunks {
        flush(&mut out, &mut cur_start, &mut cur_end, &mut cur_tokens);
    }
    out.truncate(opt.max_chunks);
    out
}

fn push_chunk(out: &mut Vec<Chunk>, text: &str, start: usize, end: usize) {
    let start = floor_char(text, start);
    let end = ceil_char(text, end.min(text.len()));
    if start >= end {
        return;
    }
    let slice = &text[start..end];
    if slice.trim().is_empty() {
        return;
    }
    out.push(Chunk { start, end, line_start: line_at(text, start), line_end: line_at(text, end.saturating_sub(1)), text: slice.to_string() });
}

/// Carry the last `overlap` tokens before `end` into the next chunk.
fn overlap_prefix(text: &str, end: usize, overlap: usize) -> Option<(usize, usize)> {
    let spans = token_spans(&text[..end]);
    if spans.len() <= 1 {
        return None;
    }
    let keep = overlap.min(spans.len().saturating_sub(1));
    let start = spans[spans.len() - keep].0;
    Some((start, keep))
}

fn slice_long(text: &str, base: usize, spans: &[(usize, usize)], max: usize, overlap: usize, limit: usize, out: &mut Vec<Chunk>) {
    if spans.is_empty() || max == 0 {
        return;
    }
    let step = max.saturating_sub(overlap).max(1);
    let mut i = 0;
    while i < spans.len() && out.len() < limit {
        let j = (i + max).min(spans.len());
        let start = base + spans[i].0;
        let end = base + spans[j - 1].1;
        push_chunk(out, text, start, end);
        if j == spans.len() {
            break;
        }
        i += step;
    }
}

struct Piece {
    start: usize,
    end: usize,
    /// A heading: close the current chunk before this piece.
    hard: bool,
}

fn pieces(text: &str) -> Vec<Piece> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut seg = 0;
    let mut hard = false;
    while i < b.len() {
        if b[i..].starts_with(b"```") {
            if i > seg {
                out.push(Piece { start: seg, end: i, hard });
            }
            let fence = fence_end(b, i);
            out.push(Piece { start: i, end: fence, hard: true });
            i = fence;
            seg = i;
            hard = false;
            continue;
        }
        if is_line_start(b, i) && b[i] == b'#' {
            if i > seg {
                out.push(Piece { start: seg, end: i, hard });
            }
            seg = i;
            hard = true;
        }
        i += 1;
    }
    if seg < b.len() {
        out.push(Piece { start: seg, end: b.len(), hard });
    } else if out.is_empty() && !text.is_empty() {
        out.push(Piece { start: 0, end: text.len(), hard: false });
    }
    out
}

fn is_line_start(b: &[u8], i: usize) -> bool {
    i == 0 || b[i - 1] == b'\n'
}

fn fence_end(b: &[u8], start: usize) -> usize {
    let mut i = start + 3;
    while i < b.len() && b[i] != b'\n' {
        i += 1;
    }
    if i < b.len() {
        i += 1;
    }
    while i < b.len() {
        if is_line_start(b, i) && b[i..].starts_with(b"```") {
            i += 3;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            if i < b.len() {
                i += 1;
            }
            return i;
        }
        i += 1;
    }
    b.len()
}

/// Byte ranges of tokens inside `s`, relative to `s`.
fn token_spans(s: &str) -> Vec<(usize, usize)> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let ch = s[i..].chars().next().unwrap();
        let w = ch.len_utf8();
        if ch.is_whitespace() {
            i += w;
            continue;
        }
        if is_cjk(ch) {
            out.push((i, i + w));
            i += w;
            continue;
        }
        let start = i;
        i += w;
        while i < b.len() {
            let c = s[i..].chars().next().unwrap();
            if c.is_whitespace() || is_cjk(c) {
                break;
            }
            i += c.len_utf8();
        }
        if i - start > 24 {
            let mut p = start;
            while p < i {
                let mut n = (p + 8).min(i);
                while n < i && !s.is_char_boundary(n) {
                    n += 1;
                }
                out.push((p, n));
                p = n;
            }
        } else {
            out.push((start, i));
        }
    }
    out
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{30FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{F900}'..='\u{FAFF}'
    )
}

fn line_at(text: &str, byte: usize) -> u32 {
    // `end - 1` can sit inside a multibyte character. Count newlines up to
    // the char that byte falls in.
    let byte = floor_char(text, byte.min(text.len()));
    text[..byte].bytes().filter(|&b| b == b'\n').count() as u32 + 1
}

fn floor_char(text: &str, mut i: usize) -> usize {
    i = i.min(text.len());
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char(text: &str, mut i: usize) -> usize {
    i = i.min(text.len());
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(target: usize, max: usize, overlap: usize) -> ChunkOptions {
        ChunkOptions { target_tokens: target, max_tokens: max, overlap_tokens: overlap, max_chunks: 32 }
    }

    #[test]
    fn short_text_is_one_chunk() {
        let t = "hello\nworld\n";
        let c = chunk(t, &opt(64, 128, 8));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].line_start, 1);
        assert_eq!(c[0].line_end, 2);
        assert_eq!(&t[c[0].start..c[0].end], c[0].text.as_str());
    }

    #[test]
    fn overlap_and_offsets_cover_a_long_paragraph() {
        let word = "alpha ";
        let t = word.repeat(400);
        let c = chunk(&t, &opt(40, 50, 10));
        assert!(c.len() > 2, "got {}", c.len());
        for (i, ch) in c.iter().enumerate() {
            assert!(t.is_char_boundary(ch.start) && t.is_char_boundary(ch.end));
            assert_eq!(&t[ch.start..ch.end], ch.text.as_str());
            assert!(ch.line_start >= 1 && ch.line_end >= ch.line_start);
            if i > 0 {
                assert!(ch.start < c[i - 1].end, "chunk {i} does not overlap");
                assert!(ch.start > c[i - 1].start);
            }
        }
    }

    #[test]
    fn heading_starts_a_new_chunk() {
        let t = "# One\n\nalpha beta gamma delta epsilon.\n\n# Two\n\nzeta eta theta iota kappa.\n";
        let c = chunk(t, &opt(8, 40, 0));
        assert!(c.len() >= 2, "{c:?}");
        let joined = c.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n");
        assert!(joined.contains("# One") && joined.contains("# Two"));
        assert!(c.iter().any(|c| c.text.starts_with("# Two") || c.text.contains("# Two")));
    }

    #[test]
    fn fence_stays_whole() {
        let t = "intro words here\n\n```\nfn main() {\n    let x = 1;\n}\n```\n\nafter the fence\n";
        let c = chunk(t, &opt(4, 80, 0));
        let fence = c.iter().find(|c| c.text.contains("fn main()")).unwrap();
        assert!(fence.text.contains("```"));
        assert!(fence.text.contains("let x = 1"));
        assert!(!fence.text.contains("after the fence"));
    }

    #[test]
    fn cjk_counts_per_character() {
        let t = "北极光".repeat(80);
        let c = chunk(&t, &opt(30, 40, 5));
        assert!(c.len() >= 3, "80 CJK chars should split, got {}", c.len());
        assert!(c[0].text.chars().count() <= 40);
    }

    #[test]
    fn empty_is_empty() {
        assert!(chunk("   \n\t", &ChunkOptions::default()).is_empty());
    }
}
