"""Build the `resolve-kb` index: chunk every article, embed each chunk, bulk-load.

    python -m resolve.kb.index            # build or rebuild
    python -m resolve.kb.index --count    # how many chunks are indexed

The index has a `text` field (BM25 keyword search), a `vector` field (k-NN over
the embedding), and keyword fields for the ids, title, heading, and space. Chunk
ids are used as document ids, so running this twice updates rather than duplicates.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from opensearchpy import OpenSearch, helpers

from resolve.kb.chunk import chunk_all
from resolve.kb.load import load_articles

CONFIG_PATH = Path(os.environ.get("RETRIEVAL_CONFIG", "configs/retrieval.yaml"))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def get_client() -> OpenSearch:
    url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
    return OpenSearch(hosts=[url], timeout=60)


def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def mapping(dim: int) -> dict:
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {"properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "title": {"type": "text"},
            "heading": {"type": "text"},
            "space": {"type": "keyword"},
            "labels": {"type": "keyword"},
            "text": {"type": "text"},
            "vector": {"type": "knn_vector", "dimension": dim,
                       "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}},
        }},
    }


def build(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    index_name = cfg.get("index", "resolve-kb")
    model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")
    max_words = int(cfg.get("chunk_max_words", 220))
    overlap = int(cfg.get("chunk_overlap_words", 30))

    articles = load_articles()
    chunks = chunk_all(articles, max_words=max_words, overlap_words=overlap)
    embedder = get_embedder(model_name)
    dim = embedder.get_embedding_dimension()

    client = get_client()
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=mapping(dim))

    vectors = embedder.encode([c["text"] for c in chunks], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=False)
    actions = ({"_index": index_name, "_id": c["chunk_id"], "_source": {**c, "vector": v.tolist()}}
               for c, v in zip(chunks, vectors))
    ok, errors = helpers.bulk(client, actions, raise_on_error=False)
    client.indices.refresh(index=index_name)
    print(f"{len(articles)} articles -> {len(chunks)} chunks; indexed {ok}, errors {len(errors) if isinstance(errors, list) else errors}")
    return ok


def count(cfg: dict | None = None) -> int:
    cfg = cfg or load_config()
    return get_client().count(index=cfg.get("index", "resolve-kb"))["count"]


if __name__ == "__main__":
    if "--count" in sys.argv:
        print(count())
    else:
        build()
