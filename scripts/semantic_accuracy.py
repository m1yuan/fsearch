#!/usr/bin/env python3
"""Compare the local sidecar to the fp32 sentence-transformers reference.

The sidecar must already be serving POST /v1/embeddings. Texts are sent with
the same EmbeddingGemma prompts fsearch uses. The reference encodes the raw
text with prompt_name, so the prompts are not applied twice.

Exit status is 0 when the mean cosine is at least --min (default 0.999).
This script does not measure the Neural Engine; scripts/validate_ane.sh does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request

SAMPLES = [
    ("What causes the northern lights?", "The northern lights are caused by charged particles from the sun hitting the atmosphere."),
    ("wifi password", "The wifi password is tumbleweed-4417."),
    ("北极光是怎么形成的", "北极光是太阳带电粒子进入大气层产生的现象。"),
    ("如何重置路由器", "按住路由器背面的重置按钮十秒，然后重新设置无线网络。"),
]


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def post_embeddings(url: str, texts: list[str], dim: int) -> list[list[float]]:
    body = json.dumps({"input": texts, "model": "embeddinggemma-2", "dimensions": dim}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
    return [row["embedding"] for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/v1/embeddings")
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--min", type=float, default=0.999, dest="minimum")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed; skipping the accuracy check", file=sys.stderr)
        return 2

    model = SentenceTransformer(args.model, config_kwargs={"vision_config": None, "audio_config": None})
    queries = [q for q, _ in SAMPLES]
    docs = [d for _, d in SAMPLES]
    # Sidecar embeds the string it is given. fsearch sends the prompt itself,
    # so the accuracy check sends the same strings and asks the reference to
    # apply the matching prompt_name on the raw text only.
    q_prompt = [f"task: search result | query: {q}" for q in queries]
    d_prompt = [f"title: none | text: {d}" for d in docs]
    side_q = post_embeddings(args.url, q_prompt, args.dim)
    side_d = post_embeddings(args.url, d_prompt, args.dim)
    ref_q = model.encode(queries, prompt_name="query", normalize_embeddings=True, truncate_dim=args.dim)
    ref_d = model.encode(docs, prompt_name="document", normalize_embeddings=True, truncate_dim=args.dim)

    scores: list[float] = []
    for i, (q, _) in enumerate(SAMPLES):
        cq = cosine(side_q[i], [float(x) for x in ref_q[i]])
        cd = cosine(side_d[i], [float(x) for x in ref_d[i]])
        scores.extend((cq, cd))
        print(f"{cq:.6f}  query  {q}")
        print(f"{cd:.6f}  doc    {docs[i][:48]}")
    mean = sum(scores) / len(scores)
    worst = min(scores)
    print(f"mean {mean:.6f}  min {worst:.6f}  target {args.minimum}")
    if mean < args.minimum:
        print("ACCURACY: FAIL", file=sys.stderr)
        return 1
    print("ACCURACY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
