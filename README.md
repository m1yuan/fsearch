# FSearch

Whole-disk file search for macOS: names in about 0.1 ms, typo-tolerant,
plus indexed search inside files. A CLI with a small daemon, a Rust crate,
or an MCP server for coding agents.

```
cargo build --release && ./target/release/fsearch install   # -> ~/.local/bin/fsearch
fsearch fsearch main              # find files by name
fsearch 'ext:rs grep:apply_dir'   # search inside files
```

Or skip the Rust toolchain with a release build for Apple Silicon:

```
curl -L https://github.com/noahdunnagan/fsearch/releases/latest/download/fsearch-aarch64-apple-darwin.tar.gz | tar xz && ./fsearch install
```

## Speed

M4 Mac mini, a home holding the Linux or the Chromium source tree, vs the
previous version (b88f4e9). Medians; Linux / Chromium.

| | before | now |
|---|---:|---:|
| right file first, name search | 83% / 86% | 93% / 97% |
| right file first, two typos | 11% / 15% | 74% / 93% |
| find a file by name | 0.045 / 0.10 ms | 0.054 / 0.12 ms |
| an edited file greppable | 2.1 s | 1.3 ms |
| a rename or delete visible | 45 / 47 ms | 18 / 15 ms |
| first run, a name in a folder | 7.8 / 7.8 s | 0.18 / 1.2 s |
| first run, every match in the tree | 10.9 / 12.2 s | 2.4 / 10.8 s |
| files grep can search (rg's own set) | 96% / 92% | 99.9% / 99.8% |
| grep, first page | 0.29 / 0.48 ms | 0.33 / 0.65 ms |
| index on disk | 503 / 893 MB | 309 / 600 MB |
| peak memory while indexing | 1.13 / 1.37 GB | 0.77 / 0.94 GB |
| daemon memory | 19 / 100 MB | 21 / 35 MB |

The index is a third smaller while holding a third more text: one 3-bit
filter per file instead of two, compressed posting lists and paths, and
bit-packed name arrays read in place. Grep's first page reads a few more
files for it, and the deeper typo search costs a little on names.

## Queries

```
fsearch 'readme in:~/Developer'          # inside a folder
fsearch 'type:image size:>5mb mtime:<7d'
fsearch 'ext:rs regex:fn\s+\w+_dir'      # regex inside files
fsearch 'sym:apply_dir'                  # where it's defined
fsearch 'src/**/*.{c,h} !test/ init'     # globs, in or out
fsearch 'git:modified in:~/Developer/app'
```

Words are fuzzy and forgive a typo at any letter, two in words of 8+
(`mian.rs` finds `main.rs`); several words can spell one name. Accented
names match however they were stored. Also `'exact`, `^prefix`, `suffix$`
and `!exclude`. Filters: `ext:` `type:` `kind:` `in:` `size:` `mtime:`
`re:` `path:` `grep:` `regex:` `sym:` `limit:` and `git:` (`modified`
`staged` `untracked` `clean` `tracked` `ignored` `conflict`). Content search is
smart-case, and regexes match within a line, as in ripgrep.

## What grep covers

Text files up to 1 MB, and up to 10 MB inside git repos, minus what git
ignores. It leaves out lockfiles, JSON of 100 KB and up, minified or
bundled assets, linked git worktrees (their main checkout is indexed), and
build-output folders outside repos. To leave out more, list folders in
`~/Library/Application Support/FSearch/content-skip.txt`, one per line
(`~` and `#` comments work). Names are always indexed, and a grep inside
a file you name still reads it.

Churn folders — caches, logs, build output, `node_modules`, folders with a
`CACHEDIR.TAG` — are indexed like everything else, but their changes show
up only once they quiet down (a few seconds; a minute at most under
churn that never stops).

## Full Disk Access

From a terminal with Full Disk Access it indexes everything. As a login
item (`fsearch install --login`), grant `~/.local/bin/fsearch` access in
System Settings > Privacy & Security, again after each rebuild. Without
access it skips protected folders instead of prompting.

## API

JSON lines over `~/Library/Application Support/FSearch/fsearch.sock`, or
`fsearch stdio`:

```json
{"q": "fsearch main", "limit": 20, "offset": 20}
{"op": "grep", "pattern": "apply_dir", "in": "~/Developer", "context": 2}
{"q": "main.rs", "boost": true, "near": "~/Developer/app/src/lib.rs", "git": true}
{"op": "open", "path": "~/Developer/app/src/main.rs"}
{"op": "recent", "in": "~/Developer/app"}
```

`open` tells it what you opened; `"boost": true` ranks those higher for a
while, `near` favors files close to the one you're in, and `recent` lists
what you worked on lately. `"git": true` adds each hit's git status.

For coding agents, `fsearch mcp` is an MCP server on stdio with
`find_files`, `grep` and `multi_grep`. A build with `--features semantic`
also exposes `semantic_search` (meaning, not just the letters).

Or link the crate:

```rust
let engine = fsearch::Engine::start(fsearch::Options { dir: fsearch::default_dir(&home), home: home.clone(), skip: None })?;
let hits = engine.search(&fsearch::Query::parse("fsearch main", &home)?)?;
```

An app and the CLI share one index: the first process owns it, the rest
follow.

## How it works

- Crawls the disk once with `getattrlistbulk`, then follows FSEvents:
  changes show up in about 15 ms, and a restart replays only what changed.
  On a first run, a search in a folder indexes that folder first.
- Names live in one mmap'd file, folder by folder, so `in:` is a range. A
  bitmap of each name's letters rules out most names before scoring.
- Content search is a trigram index plus a per-file filter, so a search
  only reads files likely to match. Identical files are indexed once, and
  deleted files' entries are compacted away. Matches are read fresh from
  disk; files edited since the last index sync are read directly.
- The daemon runs with malloc's space-efficient mode, so it gives memory
  back when idle.

## Semantic search

Optional, and off unless you ask for it. `cargo build` does not link it.
`cargo build --features semantic` adds a SQLite vector index beside the
trigram one. The daemon still answers names and grep exactly as before
until `semantic.json` sets `"enabled": true` (that is what attaches the
indexer to FSEvents). A `semantic` request starts the index on its own,
in the background, and never waits on it to answer a name search.

It only embeds files the content index would already keep: the same size
caps, gitignore, `content-skip.txt`, and generated-file rules. Text is
split into chunks of a few hundred tokens (headings and fenced code are
boundaries, with overlap). Vectors are fp16 by default, 256 dimensions,
in `semantic.sqlite` under the same Application Support directory. Search
is a brute-force cosine over normalized rows; an ANN index can replace
that scan later without changing the rows.

Embeddings come from a local OpenAI-compatible server, default
`http://127.0.0.1:8765/v1/embeddings`. fsearch applies the prompt, so the
server embeds the string it is given:

- documents: `title: {filename or none} | text: {chunk}`
- queries: `task: search result | query: {q}`

Those two templates are the config keys `document_template` and
`query_template`. Point `endpoint` at another server and change the
templates to swap models. A hybrid query runs a filename search, a
content grep, and the vector search, then fuses the three lists with
reciprocal rank fusion (`k = 60`).

```
cargo build --release --features semantic
./target/release/fsearch semantic index ~/Documents --wait
./target/release/fsearch semantic "what causes the northern lights"
./target/release/fsearch semantic index --status
```

Socket / `fsearch stdio`:

```json
{"op": "semantic", "q": "wifi password", "limit": 10, "mode": "hybrid"}
{"op": "semantic", "action": "index", "path": "~/Documents", "wait": true}
{"op": "semantic", "action": "status"}
```

`mode` is `hybrid` (default) or `vector`. Each hit has `path`, `snippet`,
`line_start`, `line_end`, `score`, and when they exist `vector_score` and
`lexical_score`.

`~/Library/Application Support/FSearch/semantic.json`:

```json
{
  "enabled": true,
  "endpoint": "http://127.0.0.1:8765/v1/embeddings",
  "model": "embeddinggemma-2",
  "dim": 256,
  "dtype": "f16",
  "batch": 8,
  "document_template": "title: {title} | text: {text}",
  "query_template": "task: search result | query: {query}",
  "target_tokens": 384,
  "max_tokens": 512,
  "overlap_tokens": 64,
  "max_chunks": 64,
  "batch_pause_ms": 25
}
```

`enabled` is the FSEvents hook. `FSEARCH_SEMANTIC=1`, `FSEARCH_EMBED_URL`,
`FSEARCH_SEMANTIC_DIM` and `FSEARCH_SEMANTIC_DTYPE` override the file.
Changing `dim` or `dtype` needs `fsearch semantic index --reset`.

### Sidecar (EmbeddingGemma 2 on the Neural Engine)

`ane-embed/` is a Swift package for macOS 27. It serves `POST /v1/embeddings`
and `GET /health` on `127.0.0.1:8765`. Primary path: Apple Core AI (`.aimodel`),
`SpecializationOptions(preferredComputeUnitKind: .neuralEngine)`. Fallback:
Core ML `.mlpackage` with `computeUnits = .cpuAndNeuralEngine`. Neither path
enables the GPU. `.default` / `.all` are not used: Apple's macOS Core AI
embedding export is a 4-bit preset that prefers the GPU, and a model over
about 1 GiB falls off the ANE without an error.

Conversion (on the Mac, not in CI):

```
python3.13 -m venv .venv-eg2
.venv-eg2/bin/python -m pip install -r scripts/requirements-embeddinggemma2.txt
.venv-eg2/bin/python scripts/convert_embeddinggemma2.py \
  --out "$HOME/Library/Application Support/FSearch/models/embeddinggemma-2"
```

Pins: `coreai-torch==0.4.2`, `coreai-core==1.0.0b2`, `coreai-opt==0.2.1`,
`torch==2.11.0` (the Anemll / cowboycoderhq stack; later coreai-torch
mis-compiles the padding mask). The script exports fixed buckets 64 / 128 /
256 / 512, keeps the token table on the host, uses eager attention with
-1e4 masks, and runs RMSNorm in fp32. `--also-coreml` writes the fallback.
Weights are not in git. Copy `tokenizer.json` and `tokenizer_config.json`
from `google/embeddinggemma-2` into that directory.

```
scripts/install_ane_embed.sh          # release build + LaunchAgent
scripts/validate_ane.sh               # latency, powermetrics, optional accuracy
```

`validate_ane.sh` is the measurement. It prints `/health`, latency per
bucket, and, with passwordless sudo, `powermetrics --samplers
ane_power,gpu_power,cpu_power`. ANE power should rise and GPU power should
stay near zero. This repository's CI is Linux; it does not run that script
and it does not report ANE numbers. `scripts/semantic_accuracy.py` compares
the sidecar to fp32 sentence-transformers on a few English and Chinese
lines (target mean cosine >= 0.999) once both are installed on the Mac.

### Other embedding models

EmbeddingGemma 2 trails bge-m3 and Qwen3-Embedding-0.6B on C-MTEB. For
Chinese retrieval, run one of those behind the same HTTP shape and change
the templates (queries and documents must stay in one space):

bge-m3, often with no instruction (dim 1024, or a Matryoshka slice):

```json
{
  "model": "bge-m3",
  "dim": 1024,
  "document_template": "{text}",
  "query_template": "{query}"
}
```

Qwen3-Embedding-0.6B (dim 1024). Documents are raw; queries take the instruct prefix:

```json
{
  "model": "Qwen3-Embedding-0.6B",
  "dim": 1024,
  "document_template": "{text}",
  "query_template": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}"
}
```

Reset the index after changing `dim`. The sidecar in `ane-embed/` is
EmbeddingGemma-specific; these two are a different server that speaks
`/v1/embeddings`.

## 语义搜索

默认构建不含语义搜索。`cargo build --features semantic` 才会在三角函数索引旁边加一个 SQLite 向量库。名字搜索和 grep 的行为不变，除非 `semantic.json` 里 `"enabled": true`（这时才会挂上 FSEvents，后台增量嵌入）。直接发 `semantic` 请求也会把索引起来，但在后台跑，不会挡住词法搜索。

嵌入的文件就是内容索引本来会收的那些：同样的大小上限、gitignore、`content-skip.txt`、生成文件规则。文本切成几百 token 一段（标题和代码围栏是边界，段与段有重叠）。向量默认 fp16、256 维，存在同一 Application Support 目录的 `semantic.sqlite`。查询是归一化向量的暴力余弦；以后可以换成 ANN，行格式不用动。

向量来自本机的 OpenAI 兼容接口，默认 `http://127.0.0.1:8765/v1/embeddings`。提示词由 fsearch 加上，服务端只嵌入收到的字符串：

- 文档：`title: {文件名或 none} | text: {chunk}`
- 查询：`task: search result | query: {q}`

对应配置项 `document_template`、`query_template`。换模型就是换 `endpoint` 和这两段模板。混合查询会同时跑文件名、内容 grep 和向量，再用倒数排名融合（RRF，`k = 60`）。

```
cargo build --release --features semantic
./target/release/fsearch semantic index ~/Documents --wait
./target/release/fsearch semantic "北极光是怎么形成的"
./target/release/fsearch semantic index --status
```

套接字 / `fsearch stdio`：

```json
{"op": "semantic", "q": "无线网络密码", "limit": 10, "mode": "hybrid"}
{"op": "semantic", "action": "index", "path": "~/Documents", "wait": true}
{"op": "semantic", "action": "status"}
```

`mode` 为 `hybrid`（默认）或 `vector`。每条命中含 `path`、`snippet`、`line_start`、`line_end`、`score`，有的话还有 `vector_score` 和 `lexical_score`。

配置文件 `~/Library/Application Support/FSearch/semantic.json` 的字段见上面的英文示例。`enabled` 控制是否跟随文件系统事件。`FSEARCH_SEMANTIC=1`、`FSEARCH_EMBED_URL`、`FSEARCH_SEMANTIC_DIM`、`FSEARCH_SEMANTIC_DTYPE` 会覆盖文件。改了 `dim` 或 `dtype` 之后要 `fsearch semantic index --reset`。

### 侧车（EmbeddingGemma 2，神经网络引擎）

`ane-embed/` 是 macOS 27 的 Swift 包，在 `127.0.0.1:8765` 提供 `POST /v1/embeddings` 和 `GET /health`。优先走 Apple Core AI（`.aimodel`），`SpecializationOptions(preferredComputeUnitKind: .neuralEngine)`。不行再走 Core ML `.mlpackage`，`computeUnits = .cpuAndNeuralEngine`。两条路都不打开 GPU。不用 `.default` / `.all`：Apple 自己的 macOS Core AI 导出预设是 4-bit，运行时会偏向 GPU；模型超过大约 1 GiB 时还会悄悄掉出神经网络引擎。

在 Mac 上转换（不要在 Linux CI 里转）：

```
python3.13 -m venv .venv-eg2
.venv-eg2/bin/python -m pip install -r scripts/requirements-embeddinggemma2.txt
.venv-eg2/bin/python scripts/convert_embeddinggemma2.py \
  --out "$HOME/Library/Application Support/FSearch/models/embeddinggemma-2"
```

版本钉死为 `coreai-torch==0.4.2`、`coreai-core==1.0.0b2`、`coreai-opt==0.2.1`、`torch==2.11.0`（Anemll 和 cowboycoderhq 用的组合；更新的 coreai-torch 会把 padding mask 编错）。脚本按 64 / 128 / 256 / 512 的固定长度导出，词表留在主机上，注意力展开成 matmul，mask 用 -1e4 而不是 -inf，RMSNorm 用 fp32。`--also-coreml` 写出备用包。权重不进 git。把 `google/embeddinggemma-2` 的 `tokenizer.json` 和 `tokenizer_config.json` 拷到同一目录。

```
scripts/install_ane_embed.sh    # release 构建并装成 LaunchAgent
scripts/validate_ane.sh         # 延迟、powermetrics、可选精度
```

`validate_ane.sh` 才是实测。它打印 `/health`、各长度的延迟，并在免密 sudo 可用时采样 `powermetrics --samplers ane_power,gpu_power,cpu_power`。神经网络引擎功耗应上升，GPU 应接近 0。本仓库的 CI 是 Linux，不会跑这个脚本，也不会报任何 ANE 数字。`scripts/semantic_accuracy.py` 在 Mac 上把侧车和 fp32 的 sentence-transformers 比几条中英文（目标平均余弦 ≥ 0.999）。

### 换成更适合中文的模型

EmbeddingGemma 2 在 C-MTEB 上不如 bge-m3 和 Qwen3-Embedding-0.6B。中文检索可以把其中一个放到同样的 `/v1/embeddings` 后面，只改模板（查询和文档必须在同一向量空间）：

bge-m3 通常不加指令（维度 1024，或 Matryoshka 截断后的宽度）：

```json
{
  "model": "bge-m3",
  "dim": 1024,
  "document_template": "{text}",
  "query_template": "{query}"
}
```

Qwen3-Embedding-0.6B（维度 1024）。文档用原文，查询加 instruct 前缀：

```json
{
  "model": "Qwen3-Embedding-0.6B",
  "dim": 1024,
  "document_template": "{text}",
  "query_template": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{query}"
}
```

改了 `dim` 要重建索引。`ane-embed/` 只服务 EmbeddingGemma；这两个模型需要另一个兼容 `/v1/embeddings` 的进程。

