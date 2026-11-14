#!/bin/sh
# Exercise the embeddings sidecar on a Mac. This VM is Linux and cannot run
# it; the script is what you run on the M4 Pro. It does not invent ANE numbers:
# it prints /health, a latency table, and (when sudo is available) powermetrics
# samples so you can see whether ANE power rises and GPU power stays near zero.
set -eu

URL="${FSEARCH_EMBED_URL:-http://127.0.0.1:8765}"
PORT="${FSEARCH_EMBED_PORT:-8765}"
BIN="${ANE_EMBED_BIN:-}"
LOG="${TMPDIR:-/tmp}/fsearch-ane-validate.log"
POWER="${TMPDIR:-/tmp}/fsearch-ane-powermetrics.txt"
SAMPLE="${TMPDIR:-/tmp}/fsearch-semantic-sample"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "validate_ane.sh: this machine is $(uname -s). The Neural Engine check has to run on the Mac." >&2
  exit 2
fi

health() {
  curl -fsS --max-time 2 "$URL/health" 2>/dev/null || true
}

if [ -z "$(health)" ]; then
  if [ -z "$BIN" ]; then
    echo "sidecar is not up at $URL and ANE_EMBED_BIN is unset." >&2
    echo "build it with: cd ane-embed && swift build -c release" >&2
    echo "then: ANE_EMBED_BIN=ane-embed/.build/release/ane-embed $0" >&2
    exit 1
  fi
  echo "starting $BIN (first Core AI / Core ML load can take minutes)"
  "$BIN" --port "$PORT" >"$LOG" 2>&1 &
  PID=$!
  trap 'kill "$PID" 2>/dev/null || true' EXIT
  i=0
  while [ "$i" -lt 180 ]; do
    if [ -n "$(health)" ]; then
      break
    fi
    i=$((i + 1))
    sleep 2
  done
fi

echo "=== /health ==="
HEALTH="$(health)"
if [ -z "$HEALTH" ]; then
  echo "sidecar did not become healthy. log: $LOG" >&2
  exit 1
fi
printf '%s\n' "$HEALTH"
echo "$HEALTH" | grep -q '"ok"[[:space:]]*:[[:space:]]*true' || {
  echo "sidecar is up but the model is not loaded. see the health JSON above." >&2
  exit 1
}

bench_one() {
  chars="$1"
  python3 - "$URL" "$chars" <<'PY'
import json, sys, time, urllib.request
url, n = sys.argv[1], int(sys.argv[2])
text = ("token " * (n // 2 + 4))[: max(n, 8)]
body = json.dumps({"input": [text], "model": "embeddinggemma-2", "dimensions": 256}).encode()
# one warmup
urllib.request.urlopen(urllib.request.Request(url + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"}), timeout=120).read()
times = []
for _ in range(8):
    t = time.perf_counter()
    urllib.request.urlopen(urllib.request.Request(url + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"}), timeout=120).read()
    times.append((time.perf_counter() - t) * 1000)
times.sort()
print(f"chars~{n:4d}  median {times[len(times)//2]:7.2f} ms  min {times[0]:7.2f} ms")
PY
}

echo "=== latency (single text, after one warmup) ==="
bench_one 64
bench_one 128
bench_one 256
bench_one 512

echo "=== powermetrics ==="
if sudo -n true 2>/dev/null; then
  sudo -n powermetrics --samplers ane_power,gpu_power,cpu_power -i 500 -n 6 >"$POWER" 2>/dev/null &
  PM=$!
  bench_one 256
  wait "$PM" || true
  echo "raw samples: $POWER"
  python3 - "$POWER" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
def avg(label):
    vals = [float(x) for x in re.findall(label + r"[^0-9\-]{0,16}([0-9]+(?:\.[0-9]+)?)\s*mW", text, flags=re.I)]
    if not vals:
        return None
    return sum(vals) / len(vals)
for name, pat in (("ANE", r"ANE\s*Power"), ("GPU", r"GPU\s*Power"), ("CPU", r"CPU\s*Power")):
    v = avg(pat)
    print(f"{name} average: {v:.1f} mW" if v is not None else f"{name} average: (no mW samples matched; read the raw file)")
print("ANE should rise under load. GPU near zero means the graph did not fall onto the GPU.")
print("A low ANE number with a high CPU number means the graph fell onto the CPU. /health placement is a hint; these samples are the measurement.")
PY
else
  echo "skipped: powermetrics needs passwordless sudo (sudo -n). Re-run under sudo to sample ANE/GPU/CPU power."
fi

echo "=== accuracy vs sentence-transformers (optional) ==="
if python3 -c "import sentence_transformers" 2>/dev/null; then
  python3 "$(dirname "$0")/semantic_accuracy.py" --url "$URL/v1/embeddings" || true
else
  echo "skipped: pip install sentence-transformers torch (fp32 reference). Target mean cosine >= 0.999."
fi

echo "=== end-to-end smoke ==="
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SAMPLE/notes"
printf '%s\n' "The northern lights are caused by charged particles from the sun." > "$SAMPLE/notes/aurora.md"
printf '%s\n' "北极光是太阳带电粒子进入大气层产生的现象。" > "$SAMPLE/notes/sky.md"
printf '%s\n' "The wifi password is tumbleweed-4417." > "$SAMPLE/notes/wifi.md"
FSEARCH="${FSEARCH_BIN:-}"
if [ -z "$FSEARCH" ]; then
  if [ -x "$ROOT/target/release/fsearch" ]; then
    FSEARCH="$ROOT/target/release/fsearch"
  elif command -v fsearch >/dev/null 2>&1; then
    FSEARCH="$(command -v fsearch)"
  fi
fi
if [ -z "$FSEARCH" ]; then
  echo "no fsearch binary. On the Mac: cargo build --release --features semantic" >&2
  echo "sample folder is ready at $SAMPLE" >&2
  exit 0
fi
HOME_BAK="${HOME:-}"
# Index only the sample. fsearch embeds files under \$HOME, so point HOME at it
# for this smoke and keep the real data dir via FSEARCH_DATA if you set one.
export HOME="$SAMPLE"
mkdir -p "$SAMPLE/Library/Application Support/FSearch"
printf '%s\n' "{\"enabled\":false,\"endpoint\":\"$URL/v1/embeddings\",\"dim\":256,\"dtype\":\"f16\",\"batch_pause_ms\":0}" > "$SAMPLE/Library/Application Support/FSearch/semantic.json"
"$FSEARCH" semantic index "$SAMPLE" --wait --reset || {
  echo "fsearch semantic index failed. The binary must be built with --features semantic." >&2
  exit 1
}
echo "-- northern lights --"
"$FSEARCH" semantic "what causes the northern lights" --mode hybrid --limit 3 || true
echo "-- 北极光 --"
"$FSEARCH" semantic "北极光是怎么形成的" --mode hybrid --limit 3 || true
echo "-- wifi --"
"$FSEARCH" semantic "wifi password" --mode vector --limit 3 || true
export HOME="$HOME_BAK"
echo "sample folder: $SAMPLE"
