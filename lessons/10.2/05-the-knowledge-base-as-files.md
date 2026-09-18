Quillstack's help center is a folder of markdown files. Before a search engine, before an agent, it is worth spending one lesson with the files themselves: what an article looks like, what the chunker makes of it, and whether the pieces it produces are pieces you would want handed to a model. By the end you will have a working vector retriever that is one numpy matrix and one `argsort`, which is the whole idea of vector search with nothing hidden.

## The article format

The knowledge base lives at `data/kb/` in two folders: `public/` for the articles customers can read and `internal/` for runbooks, the refund approval policy and admin-tool notes that support staff use and that must never be cited to a customer. Every file has the same shape: a **front matter** block of YAML between two `---` lines, then a markdown body.

```plain text
---
id: kb-0046
title: API key expiry and error QS-1042
space: public
labels: [api, keys, errors]
updated: 2026-07-13
version: 3
---

API keys expire after a fixed period that depends on your Quillstack plan. ...

## Before you start
...
## Steps
...
## Error QS-1042: API key has expired
...
## Related
- [Create an API key](kb-0044)
```

Six fields, and each one has a job later. `id` is the stable name the agent cites and the eval set uses as gold. `title` is what the customer sees. `space` is `public` or `internal`, and it matches the folder the file sits in. `labels` are short topic tags. `updated` and `version` are what you would compare against a live documentation site to know whether the index is stale. The body is ordinary markdown: `##` headings, numbered steps, tables, fenced code blocks with example JSON responses, and exact strings such as error codes, config keys and menu paths.

## Loading articles

`resolve.kb.load` turns that folder into a list of dictionaries. Run this and everything else in this lesson from the repository root with `PYTHONPATH=src`, so that `resolve` imports.

```python
from resolve.kb.load import load_articles

# 1. Every article in data/kb/public and data/kb/internal, in id order.
articles = load_articles()
print(len(articles), sorted(articles[0].keys()))

# 2. The front matter becomes top-level keys; the body is one string.
a = next(x for x in articles if x["id"] == "kb-0046")
print({k: a[k] for k in ("id", "title", "space", "labels", "updated", "version")})
print(a["body"][:120])
```

```plain text
85 ['body', 'id', 'labels', 'path', 'space', 'updated', 'version']
{'id': 'kb-0046', 'title': 'API key expiry and error QS-1042', 'space': 'public', 'labels': ['api', 'keys', 'errors'], 'updated': '2026-07-13', 'version': 3}
API keys expire after a fixed period that depends on your Quillstack plan. When a key expires, requests using that key fail w
```

Eighty-five articles, 75 public and 10 internal. The loader reads the front matter with `yaml.safe_load`, refuses any file that lacks `id`, `title` or `space`, and refuses any file whose `space` disagrees with its folder. Those checks are cheap and they catch the mistake that matters most: an internal page that ends up marked public reaches customers.

## Chunking all of them

`chunk_all` applies the heading-based chunker from the previous lesson to every article.

```python
from resolve.kb.chunk import chunk_all

# 1. One chunk per heading section, long sections split on paragraphs.
chunks = chunk_all(articles)
print(len(chunks), "chunks from", len(articles), "articles")

# 2. The first dozen, with their headings and sizes.
for c in chunks[:12]:
    print(c["chunk_id"], "|", c["heading"] or "(intro)", "|", len(c["text"].split()), "words")
```

```plain text
558 chunks from 85 articles
kb-0001#00 | (intro) | 48 words
kb-0001#01 | Before you start | 64 words
kb-0001#02 | Steps | 150 words
kb-0001#03 | What to do next | 148 words
kb-0001#04 | Troubleshooting | 118 words
kb-0002#00 | (intro) | 55 words
kb-0002#01 | Before you start | 91 words
kb-0002#02 | Steps | 182 words
kb-0002#03 | Workspace settings | 141 words
kb-0002#04 | Troubleshooting | 124 words
kb-0003#00 | (intro) | 54 words
kb-0003#01 | Before you start | 88 words
```

Each chunk is a dictionary with `chunk_id`, `doc_id`, `title`, `heading`, `space`, `labels` and `text`. The chunk id is the article id plus a two-digit position, so `kb-0001#02` is the third chunk of the first article, and the ids are the same every time you run this. Notice what is missing from the list: no "Related" chunk for any article, because the chunker drops link lists.

## Reading chunks and judging them

Before you embed anything, read some. This is the step people skip, and it is where most retrieval problems are visible to the naked eye.

```python
# 1. Every chunk of the API-key-expiry article, then one in full.
for c in chunks:
    if c["doc_id"] == "kb-0046":
        print(c["chunk_id"], repr(c["heading"]), len(c["text"].split()))
print(next(c for c in chunks if c["chunk_id"] == "kb-0046#02")["text"])
```

```plain text
kb-0046#00 '' 45
kb-0046#01 'Before you start' 81
kb-0046#02 'Steps' 125
kb-0046#03 'Error QS-1042: API key has expired' 96
kb-0046#04 'Error QS-1041: API key is invalid or revoked' 80
kb-0046#05 'How to avoid expiry issues' 129
kb-0046#06 'Troubleshooting' 125
API key expiry and error QS-1042 > Steps

Follow these steps to rotate an expiring key:

1. Sign in to app.quillstack.com and navigate to Settings > Integrations > API keys.
2. Click Create API key and give the new key a descriptive name.
3. Copy the new key immediately; it is shown in full only once.
4. Update your integration to use the new key. Test that it works before revoking the old key.
5. Once you confirm the new key works, revoke the old key by clicking Revoke next to it in the list.

If you wait until after a key expires, you must create a new key and update your integration immediately. There is no way to extend or renew an expired key.
```

Three questions to ask of a chunk like that. Does it stand alone? Someone handed only this text knows which article it is from, which section, and what to do; the `Title > Heading` line does that work. Does it contain the fact a ticket would need? The full rotation procedure is here, including the detail that the key is shown once, which is the thing customers get wrong. Is it the right size? 125 words is short enough to be precise and long enough to be complete. Now look at `#03` and `#04`: the two error codes got their own chunks, because the author gave each its own heading, so a ticket quoting `QS-1041` can retrieve the section about `QS-1041` rather than the whole article.

Read a dozen across different articles and you develop a sense for the corpus. The `Before you start` chunks carry the plan limits and the facts. The `Steps` chunks carry menu paths. The `Troubleshooting` chunks are where the ticket-shaped sentences are, phrased the way a customer would phrase them, which is why they retrieve well.

## A retriever in one matrix

Everything needed for vector search is now in a list. Embed every chunk once, stack the vectors into a matrix, and a query becomes one multiplication.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. The same model the whole system uses: 384 dimensions, normalized.
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. One row per chunk. Batched, normalized to unit length, kept in memory.
M = model.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True)
print(M.shape, M.dtype)
```

```plain text
(558, 384) float32
```

That takes under a minute on a laptop CPU. `M` is 558 rows of 384 numbers, one row per chunk, every row at length 1. Because every row is unit length, the dot product of a row with a unit-length query vector is their cosine similarity, and the dot product of the whole matrix with the query is 558 cosines in one call.

```python
def search(query, k=5):
    # 1. Embed the query with the same model, normalized the same way.
    q = model.encode(query, normalize_embeddings=True)
    # 2. One cosine per chunk, then the k largest, largest first.
    scores = M @ q
    top = np.argsort(-scores)[:k]
    return [(round(float(scores[i]), 3), chunks[i]["chunk_id"],
             chunks[i]["title"], chunks[i]["heading"]) for i in top]

for hit in search("money taken twice"):
    print(hit)
```

```plain text
(0.428, 'kb-0034#01', 'I was charged twice: duplicate charges', 'Before you start')
(0.421, 'kb-0102#03', 'Refund approval policy', 'Duplicate charges')
(0.419, 'kb-0034#03', 'I was charged twice: duplicate charges', 'What causes duplicate charges')
(0.399, 'kb-0034#02', 'I was charged twice: duplicate charges', 'Steps')
(0.398, 'kb-0034#05', 'I was charged twice: duplicate charges', 'Troubleshooting')
```

`np.argsort(-scores)` sorts the negated scores ascending, which is the same as sorting the scores descending, and returns positions rather than values; `[:k]` keeps the first five positions, and those positions index straight back into `chunks`. That is the entire retriever. No engine, no network, twelve lines.

Read the hits. Four of the five come from the duplicate-charges article, which never uses the words "money" or "taken". The second hit is `kb-0102`, the internal refund approval policy. It is a correct match on meaning and a chunk that must never be quoted to a customer, which is the first sign that the retriever needs a `space` filter and that the fields you carried through chunking are about to matter.

Now a query with an error code in it.

```python
for hit in search("my api key stopped working with QS-1042"):
    print(hit)
```

```plain text
(0.816, 'kb-0046#04', 'API key expiry and error QS-1042', 'Error QS-1041: API key is invalid or revoked')
(0.768, 'kb-0056#01', 'Revoke a compromised API key', 'Before you start')
(0.751, 'kb-0046#06', 'API key expiry and error QS-1042', 'Troubleshooting')
(0.747, 'kb-0046#03', 'API key expiry and error QS-1042', 'Error QS-1042: API key has expired')
(0.666, 'kb-0044#04', 'Create an API key', 'Troubleshooting')
```

The right article, four times out of five. But the top hit is the `QS-1041` section, and the `QS-1042` section, which is the one that answers the ticket, sits at rank four. To an embedding model, `QS-1041` and `QS-1042` are nearly the same string in nearly the same context, and the model blurs them. Lesson 02 said exact codes are the job of lexical search. Here is the evidence, and lesson 07 fixes it by running both.

## What an engine adds

The matrix retriever is real vector search, and for 558 chunks on one laptop it is fast enough. Four things push it into a search engine.

Persistence. `M` lives in one Python process. Restart it and you re-embed the corpus; run two processes and each holds its own copy. An engine keeps the vectors on disk and serves every caller from one index.

Keyword search. The matrix has nothing to say about `QS-1042` as a string. An engine keeps an inverted index over the same chunks and ranks them with BM25 in the same request, which is the other half of the retriever.

Filtering. Restricting a search to `space: public`, or to chunks with a given label, is a line in the query when the fields are indexed alongside the vector. In the matrix version it is a mask you write and maintain by hand.

Scale. `argsort` over 558 rows is instant. Over a few million it is not, and that is where the HNSW graph from the previous lesson comes in, built and maintained by the engine.

None of these change what a hit is: a chunk, a score, and the fields that make it citable. The next lesson, OpenSearch on Your Laptop, starts that engine in Docker, loads these same 558 chunks into it with their ids, and runs the two kinds of query by hand.
