The retriever from the last lesson lives inside one Python process. Restart the process and it is gone; ask it about an error code and it blurs. Resolve needs a store that outlives the process, ranks by words as well as by meaning, and filters on the fields the chunker carried through. That is a search engine, and the one this course uses is **OpenSearch**, running in a single Docker container on your laptop.

## What OpenSearch is

OpenSearch is an open-source search engine built on Lucene, the same library that gave lesson 02 its BM25 defaults. It started as a fork of Elasticsearch in 2021, and it does what that lineage suggests: you send it JSON documents, it builds an inverted index over their text fields, and a `match` query returns the documents ranked by BM25. It then grew a k-NN plugin, so a document can also carry a `knn_vector` field, and a `knn` query returns the documents whose vectors are nearest a query vector, using an HNSW graph. Both kinds of query run over the same documents in the same index. That is the property Resolve needs and it is why this chapter did not reach for a dedicated vector database.

You talk to it over HTTP on port 9200, and from Python through the `opensearch-py` client:

```bash
pip install opensearch-py==3.2.0
```

## The Compose service

`compose.yaml` in the project root defines it. This is the relevant service; the same file also runs the helpdesk that the agent talks to later.

```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:2.19.0
    environment:
      - discovery.type=single-node
      - DISABLE_SECURITY_PLUGIN=true
      - DISABLE_INSTALL_DEMO_CONFIG=true
      - OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g
      - bootstrap.memory_lock=true
    ports:
      - "9200:9200"
    volumes:
      - resolve_osdata:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 30s
```

Four settings are worth reading. `discovery.type=single-node` tells the engine it is the whole cluster, so it stops waiting for peers. `DISABLE_SECURITY_PLUGIN=true` turns off authentication and TLS, which is the right call for a container that only your laptop can reach and the wrong call for anything else; with it on, every request would need a username, a password and a certificate. `OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g` pins the Java heap to 1 GB, which is plenty for a few thousand chunks and keeps the container from taking half your machine. The `healthcheck` polls the cluster health endpoint every ten seconds and gives it thirty seconds to start; Compose reports the service as `healthy` once that succeeds, and anything that depends on it waits until then. The named volume keeps the index on disk between restarts, which is the persistence the last lesson asked for.

Start it and check it.

```bash
docker compose up -d opensearch
curl -s http://localhost:9200/_cluster/health
```

```plain text
{"cluster_name":"docker-cluster","status":"green","timed_out":false,"number_of_nodes":1, ...}
```

A status of `green` or `yellow` is fine on one node; `yellow` only means a replica shard has nowhere to go, and this index asks for zero replicas.

## The mapping

An index in OpenSearch is a named collection of documents plus a **mapping**, which declares each field's type. The type decides how a field is indexed and therefore which queries can use it. `resolve.kb.index` builds the mapping in one function, and the dimension is read off the model rather than typed in.

```python
import json
from sentence_transformers import SentenceTransformer
from resolve.kb.index import mapping

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_embedding_dimension()
print(dim)
print(json.dumps(mapping(dim)["mappings"]["properties"]["vector"], indent=2))
```

```plain text
384
{
  "type": "knn_vector",
  "dimension": 384,
  "method": {
    "name": "hnsw",
    "engine": "lucene",
    "space_type": "cosinesimil"
  }
}
```

The full mapping has eight fields, in three types.

`vector` is `knn_vector` with `dimension` 384. The `method` block chooses HNSW, the graph index from lesson 04, built by the `lucene` engine (OpenSearch also offers `faiss` and `nmslib`; Lucene is the one that needs no native library and supports filtered k-NN queries directly). `space_type` is `cosinesimil`, so the engine ranks by cosine similarity. The vectors are normalized before indexing anyway, which lesson 03 showed makes cosine and Euclidean rank identically; saying `cosinesimil` keeps the setting honest about what the scores mean. The index settings also set `index.knn: true`, without which the field type is refused.

`text`, `title` and `heading` are `text` fields. A `text` field is analyzed: lowercased, split into terms, and written into the inverted index, so a `match` query on it is a BM25 ranking. That is the lexical half of the retriever, and it costs nothing beyond declaring the type.

`chunk_id`, `doc_id`, `space` and `labels` are `keyword` fields. A `keyword` field is stored as one exact value, unanalyzed, which is what you want for filtering (`space` is `public`) and for reading back an id. Declare `space` as `text` by mistake and a filter on it still works, until an id or label contains a character the analyzer splits on, and then it silently matches the wrong things.

## Bulk loading with chunk ids

`build()` in `resolve.kb.index` does the whole load. Read it in three parts.

```python
# from resolve.kb.index.build, abridged
articles = load_articles()
chunks = chunk_all(articles, max_words=max_words, overlap_words=overlap)
embedder = get_embedder(model_name)
dim = embedder.get_embedding_dimension()

client = get_client()
if client.indices.exists(index=index_name):
    client.indices.delete(index=index_name)
client.indices.create(index=index_name, body=mapping(dim))
```

The names come from `configs/retrieval.yaml`: the index is `resolve-kb`, the model is `all-MiniLM-L6-v2`, the ceiling is 220 words with 30 of overlap. `get_client()` reads `OPENSEARCH_URL` from `.env`, defaulting to `http://localhost:9200`. If the index already exists it is deleted and recreated, so a rebuild always starts from the current mapping and the current articles; there is no way to end up with last week's chunks beside this week's.

```python
# from resolve.kb.index.build, abridged, continued
vectors = embedder.encode([c["text"] for c in chunks], batch_size=64,
                          normalize_embeddings=True, show_progress_bar=False)
actions = ({"_index": index_name, "_id": c["chunk_id"], "_source": {**c, "vector": v.tolist()}}
           for c, v in zip(chunks, vectors))
```

One `encode` call embeds all 558 chunks in batches. Then each chunk becomes a bulk action: which index, which document id, and the source document, which is the chunk dictionary plus its vector as a plain list. The `_id` is the chunk id, `kb-0046#03`. That choice is the idempotency story in one line: a document id is unique within an index, so writing `kb-0046#03` a second time replaces the first copy instead of adding a second. Together with the delete-and-recreate above, that means running the build twice, or ten times, leaves exactly one copy of every chunk.

```python
# from resolve.kb.index.build, abridged, continued
ok, errors = helpers.bulk(client, actions, raise_on_error=False)
client.indices.refresh(index=index_name)
print(f"{len(articles)} articles -> {len(chunks)} chunks; indexed {ok}, errors {len(errors)}")
```

`helpers.bulk` sends the actions in a few large HTTP requests instead of 558 small ones. `raise_on_error=False` is deliberate: with the default, the first failed action raises `BulkIndexError` and you never see the tuple; with it, `ok` counts what landed and `errors` lists what did not, so you can print both and decide. The `refresh` call is the subject of the second error below.

Run it from the repository root, then count.

```bash
PYTHONPATH=src python -m resolve.kb.index
PYTHONPATH=src python -m resolve.kb.index --count
```

```plain text
85 articles -> 558 chunks; indexed 558, errors 0
558
```

Every chunk from the last lesson is now in `resolve-kb`, on disk, with its vector and its fields.

## A match query by hand

The client is one object. Every query is a JSON body sent to `search`, and the response is JSON back. Read the raw response once so the shape is familiar before `resolve.kb.search` hides it.

```python
import json
from opensearchpy import OpenSearch

client = OpenSearch(hosts=["http://localhost:9200"])

# 1. BM25 over the analyzed text field. Ask for two hits and three source fields.
body = {"size": 2, "_source": ["chunk_id", "title", "heading"],
        "query": {"match": {"text": "API key expired QS-1042"}}}
resp = client.search(index="resolve-kb", body=body)
print(json.dumps(resp, indent=1))
```

```plain text
{
 "took": 1,
 "timed_out": false,
 "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
 "hits": {
  "total": {"value": 176, "relation": "eq"},
  "max_score": 22.066736,
  "hits": [
   {"_index": "resolve-kb", "_id": "kb-0046#03", "_score": 22.066736,
    "_source": {"heading": "Error QS-1042: API key has expired",
                "title": "API key expiry and error QS-1042", "chunk_id": "kb-0046#03"}},
   {"_index": "resolve-kb", "_id": "kb-0109#03", "_score": 19.293995,
    "_source": {"heading": "Send an API key expiry macro",
                "title": "Support macros and reply templates", "chunk_id": "kb-0109#03"}}
  ]
 }
}
```

`took` is milliseconds. `hits.total` says 176 chunks matched at least one query term, which is what a `match` query means: any term, OR-ed together, ranked by BM25. `size` cut that to two. Each hit carries `_id`, which is the chunk id because that is what the load used, `_score`, which is the BM25 sum and has no fixed range, and `_source`, trimmed to the three fields the query asked for so that 384 floats are not shipped back on every hit. The top hit is the `QS-1042` section itself at 22.07, well clear of the internal macro page beneath it. The matrix retriever in the last lesson put that section fourth.

## A knn query by hand

Same client, same index, different query clause. The query vector has to come from the same model that embedded the chunks.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("money taken twice", normalize_embeddings=True).tolist()

# 1. Nearest two vectors. The outer "vector" names the field; the inner one is the query.
body = {"size": 2, "_source": ["chunk_id", "title", "heading"],
        "query": {"knn": {"vector": {"vector": vec, "k": 2}}}}
resp = client.search(index="resolve-kb", body=body)
for h in resp["hits"]["hits"]:
    print(h["_id"], round(h["_score"], 4), h["_source"]["title"], ">", h["_source"]["heading"])
```

```plain text
kb-0034#01 0.7142 I was charged twice: duplicate charges > Before you start
kb-0102#03 0.7106 Refund approval policy > Duplicate charges
```

Two things differ from the match response. `hits.total` is now 2, because a `knn` query returns exactly `k` candidates rather than everything that matches. And the score is on a fixed scale: for `cosinesimil` the engine reports `(1 + cosine) / 2`, so 0.7142 is a cosine of 0.428, the same number the numpy retriever printed for this chunk. The ranking is identical to the matrix version, because it is the same vectors, the same query and the same measure; what changed is where the vectors live.

The `_source` fields are what makes a hit usable downstream. `resolve.kb.search` reads six of them off every hit and returns dictionaries with `chunk_id`, `doc_id`, `title`, `heading`, `space`, `text` and `score`, which is the shape the `docs` field in the agent's state expects.

## The two errors everyone hits

Dimension mismatch. Change the embedding model, keep the index, and the first query fails:

```python
try:
    client.search(index="resolve-kb", body={"size": 1,
                  "query": {"knn": {"vector": {"vector": [0.1] * 768, "k": 1}}}})
except Exception as exc:
    print(exc)
```

```plain text
RequestError(400, 'search_phase_execution_exception', 'failed to create query: Query vector has invalid dimension: 768. Dimension should be: 384')
```

That is the good version of the error: clear, and at query time. The bad version is at index time. A 768-dimensional vector written into a 384-dimensional mapping is rejected with `mapper_parsing_exception: failed to parse field [vector] of type [knn_vector]`, and because the bulk load runs with `raise_on_error=False`, that rejection lands in the `errors` list rather than stopping anything. If the build prints `errors 558`, read the first entry. Reading `dim` off the model, as `build()` does, means the mapping follows the model; what it cannot do is rebuild the index for you, so a model change means `python -m resolve.kb.index` again.

Querying before refresh. OpenSearch makes new documents searchable on a schedule, about once a second by default, rather than on every write. Write a document and search for it in the same breath and you get nothing:

```python
from resolve.kb.index import mapping
client.indices.create(index="scratch", body=mapping(384))
client.index(index="scratch", id="x#00", body={"chunk_id": "x#00", "text": "hello world", "vector": vec})
print(client.count(index="scratch")["count"])
client.indices.refresh(index="scratch")
print(client.count(index="scratch")["count"])
client.indices.delete(index="scratch")
```

```plain text
0
1
```

Nothing was lost; it was not visible yet. This is why `build()` calls `refresh` after the bulk load and why `--count` reports 558 immediately afterwards. A test that indexes a document and asserts on a search without a refresh in between fails once in a while, which is the most confusing kind of failure to chase.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Index mappings and bulk-load loops are good things to have an assistant draft: the JSON shapes are fiddly and well documented. Review two things on sight. First, where does `dimension` come from? If it is a literal number, ask for it to be read from the model instead, so a model swap cannot leave the mapping behind. Second, what is `_id`? If the answer is a random UUID or a content hash, a rebuild after editing one article leaves stale chunks in the index. It should be an identity, the chunk id, so a rewrite lands on the same document.
</callout>

You now have the same 558 chunks in a store that survives restarts, answers a `match` query with BM25 and a `knn` query with cosine, and carries the fields to filter on. The two queries disagree about what matters, one blind to paraphrase and the other to error codes. The next lesson, Hybrid Search and Retrieval Quality, runs both, merges the rankings, and measures the result against a labelled query set.
