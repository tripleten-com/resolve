"""Search the knowledge base three ways: keyword (BM25), vector (k-NN), or hybrid.

    from resolve.kb.search import search_kb, read_doc
    hits = search_kb("my API key stopped working with QS-1042", k=5)

Hybrid merges the two ranked lists with reciprocal rank fusion (RRF): each chunk
scores 1 / (60 + rank) in every list it appears in, and the sums are ranked.
"""
from __future__ import annotations

from functools import lru_cache

from resolve.kb.index import get_client, get_embedder, load_config
from resolve.kb.load import article_by_id, load_articles

_cfg = load_config()
INDEX = _cfg.get("index", "resolve-kb")
MODEL_NAME = _cfg.get("embedding_model", "all-MiniLM-L6-v2")
DEFAULT_MODE = _cfg.get("mode", "hybrid")
RRF_K = int(_cfg.get("rrf_k", 60))
CANDIDATES = int(_cfg.get("candidates", 20))
FIELDS = ["chunk_id", "doc_id", "title", "heading", "space", "text"]


@lru_cache(maxsize=1)
def _embedder():
    return get_embedder(MODEL_NAME)


def _hits(resp) -> list[dict]:
    out = []
    for h in resp["hits"]["hits"]:
        d = {f: h["_source"].get(f) for f in FIELDS}
        d["score"] = float(h["_score"])
        out.append(d)
    return out


def keyword_search(query: str, k: int = CANDIDATES) -> list[dict]:
    body = {"size": k, "_source": FIELDS,
            "query": {"multi_match": {"query": query, "fields": ["text", "title^2", "heading"]}}}
    return _hits(get_client().search(index=INDEX, body=body))


def vector_search(query: str, k: int = CANDIDATES) -> list[dict]:
    vec = _embedder().encode(query, normalize_embeddings=True).tolist()
    body = {"size": k, "_source": FIELDS, "query": {"knn": {"vector": {"vector": vec, "k": k}}}}
    return _hits(get_client().search(index=INDEX, body=body))


def rrf(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            seen.setdefault(cid, hit)
    merged = []
    for cid, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        d = dict(seen[cid])
        d["score"] = s
        merged.append(d)
    return merged


def search_kb(query: str, k: int = 5, mode: str | None = None) -> list[dict]:
    """Return the top-k chunks for a query. mode: 'keyword', 'vector', or 'hybrid'."""
    mode = mode or DEFAULT_MODE
    if mode == "keyword":
        return keyword_search(query, k)
    if mode == "vector":
        return vector_search(query, k)
    return rrf(keyword_search(query, CANDIDATES), vector_search(query, CANDIDATES))[:k]


@lru_cache(maxsize=1)
def _articles() -> dict[str, dict]:
    return article_by_id(load_articles())


def read_doc(doc_id: str) -> dict:
    """Return one whole article by id (title, space, body), or an error dict."""
    a = _articles().get(doc_id)
    if not a:
        return {"error": f"no article with id {doc_id!r}"}
    return {"doc_id": a["id"], "title": a["title"], "space": a["space"], "body": a["body"]}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "my api key stopped working"
    for mode in ("keyword", "vector", "hybrid"):
        print(f"== {mode}")
        for h in search_kb(q, 5, mode):
            print(f"  {h['score']:.4f} {h['chunk_id']:14} {h['space']:8} {h['title']} > {h['heading']}")
