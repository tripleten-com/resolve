"""Measure retrieval: recall@k and MRR at the article level.

    python evals/retrieval_eval.py --set dev                 # compare all three modes
    python evals/retrieval_eval.py --set heldout --mode hybrid --write

Each query in data/eval/retrieval_{set}.jsonl has "query" and "gold_doc_ids". A hit
is any gold article appearing among the articles of the top-k chunks; the rank is
the position of the first chunk from a gold article.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from resolve.kb.index import load_config
from resolve.kb.search import search_kb


def load_queries(name: str) -> list[dict]:
    path = Path(f"data/eval/retrieval_{name}.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(queries: list[dict], k: int, mode: str) -> dict:
    hits, rr, t0 = 0, 0.0, time.perf_counter()
    for q in queries:
        gold = set(q["gold_doc_ids"])
        if not gold:
            continue
        ranked_docs = []
        for h in search_kb(q["query"], k, mode):
            if h["doc_id"] not in ranked_docs:
                ranked_docs.append(h["doc_id"])
        for rank, doc in enumerate(ranked_docs[:k], start=1):
            if doc in gold:
                hits += 1
                rr += 1.0 / rank
                break
    n = sum(1 for q in queries if q["gold_doc_ids"])
    return {"mode": mode, "k": k, "n": n, "recall_at_k": round(hits / n, 4),
            "mrr": round(rr / n, 4), "seconds": round(time.perf_counter() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--mode", default=None, help="keyword, vector, or hybrid; default: all three")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--write", action="store_true", help="write evals/results/retrieval.json")
    args = ap.parse_args()
    cfg = load_config()
    k = args.k or int(cfg.get("k", 5))
    queries = load_queries(args.set)
    modes = [args.mode] if args.mode else ["keyword", "vector", "hybrid"]
    results = [evaluate(queries, k, m) for m in modes]
    for r in results:
        print(f"{args.set:8} {r['mode']:8} k={r['k']}  recall@k={r['recall_at_k']:.3f}  mrr={r['mrr']:.3f}  (n={r['n']}, {r['seconds']}s)")
    if args.write:
        out = Path("evals/results/retrieval.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"set": args.set, "config": cfg, "results": results}, indent=2))
        print(f"wrote {out}")
