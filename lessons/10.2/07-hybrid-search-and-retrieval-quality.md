The `knn` query is good at meaning. Ask it about a customer charged twice and it finds the duplicate-charges article without the word "twice" appearing in the query. Now hand it a real ticket: the customer has pasted `QS-1042` and nothing else. To an embedding, one error code looks much like another, so the vector search returns the `QS-1099` article and leaves the one document that names `QS-1042` off the list. This lesson closes that blind spot, then does the thing most RAG walkthroughs skip and measures whether the retrieval is any good.

## Two searches, three results

`resolve.kb.search` wraps the two queries from the last lesson into two functions with the same return shape: a list of hit dictionaries carrying `chunk_id`, `doc_id`, `title`, `heading`, `space`, `text` and `score`. Run both on the bare error code.

```python
from resolve.kb.search import keyword_search, vector_search

for name, fn in (("keyword", keyword_search), ("vector", vector_search)):
    print(name)
    for h in fn("QS-1042", 5):
        print(f"  {h['score']:8.4f} {h['chunk_id']:12} {h['title']} > {h['heading']}")
```

```plain text
keyword
   12.3893 kb-0046#00   API key expiry and error QS-1042 > 
   12.3893 kb-0046#01   API key expiry and error QS-1042 > Before you start
   12.3893 kb-0046#02   API key expiry and error QS-1042 > Steps
   12.3893 kb-0046#03   API key expiry and error QS-1042 > Error QS-1042: API key has expired
   12.3893 kb-0046#04   API key expiry and error QS-1042 > Error QS-1041: API key is invalid or revoked
vector
    0.7383 kb-0055#01   Handling QS-1099 internal errors > Before you start
    0.7011 kb-0052#00   API error code reference (QS-1001 to QS-1099) > 
    0.6962 kb-0052#08   API error code reference (QS-1001 to QS-1099) > Troubleshooting
    0.6949 kb-0055#00   Handling QS-1099 internal errors > 
    0.6879 kb-0055#04   Handling QS-1099 internal errors > Contact support
```

The keyword side nails it and the vector side misses entirely: five chunks about the wrong error codes. `keyword_search` is a `multi_match` query over `text`, `title` boosted by two, and `heading`, which is why the five hits tie: every chunk of `kb-0046` carries the code in its title, and the title match outscores anything in the body. Ties like that are a reminder that the unit you retrieve is a chunk while the unit you care about is often the article.

<table header-row="true">
<tr>
<td>The query contains</td>
<td>Vector k-NN</td>
<td>Keyword BM25</td>
</tr>
<tr>
<td>A paraphrased concept ("money taken twice")</td>
<td>Strong, finds synonyms and rewordings</td>
<td>Weak, misses text that says it differently</td>
</tr>
<tr>
<td>An exact error code (`QS-1042`)</td>
<td>Weak, treats one code much like another</td>
<td>Strong, a rare exact term is what IDF rewards</td>
</tr>
<tr>
<td>A config key or menu path (`sso.enforce`)</td>
<td>Mixed</td>
<td>Strong</td>
</tr>
<tr>
<td>A vague symptom description</td>
<td>Strong</td>
<td>Weak</td>
</tr>
</table>

Neither ranker wins everywhere, so run both and merge.

## Fusing two rankings

The trap is adding the scores. A k-NN score of 0.74 and a BM25 score of 12.4 are not on a common scale, and if you sum them the ranker that happens to emit bigger numbers wins every contest by accident rather than by relevance. **Reciprocal rank fusion** (RRF) sidesteps scales entirely by scoring each chunk on its position in each list:

$`\text{RRF}(d)=\sum_{r \in R}\frac{1}{k + \text{rank}_r(d)}`$

`rrf` in `resolve.kb.search` is that formula over any number of hit lists. Check it on two lists small enough to do in your head.

```python
from resolve.kb.search import rrf

a = [{"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}]
b = [{"chunk_id": "C"}, {"chunk_id": "D"}, {"chunk_id": "A"}]
print([(h["chunk_id"], round(h["score"], 4)) for h in rrf(a, b)])
```

```plain text
[('A', 0.0323), ('C', 0.0323), ('B', 0.0161), ('D', 0.0161)]
```

A is first in one list and third in the other: 1/61 plus 1/63, or 0.0323. C is the mirror image and ties it. B and D each appear once, at rank two, and score 1/62. So a chunk that both rankers place gets more than a chunk that one ranker places first, which is exactly the behavior you want when one ranker is blind to error codes and the other to paraphrase. The constant `k`, 60 by default and `rrf_k` in `configs/retrieval.yaml`, flattens the top of the curve so that rank 1 and rank 2 are not wildly far apart.

*Diagram idea: two ranked columns, "Keyword" and "Vector", for the QS-1042 query, with each chunk's rank feeding 1/(60 + rank) and merging into one fused column where kb-0046 chunks rise above the QS-1099 chunks that only the vector side liked.*

`search_kb` is the function the agent calls. It runs both searches at a wider `candidates` depth (20 by default), fuses them, and returns the top `k`.

```python
from resolve.kb.search import search_kb

for h in search_kb("QS-1042", k=5):
    print(f"  {h['score']:.4f} {h['chunk_id']:12} {h['title']} > {h['heading']}")
print(sorted(search_kb("QS-1042", k=1)[0].keys()))
```

```plain text
  0.0298 kb-0052#00   API error code reference (QS-1001 to QS-1099) > 
  0.0298 kb-0052#08   API error code reference (QS-1001 to QS-1099) > Troubleshooting
  0.0291 kb-0046#00   API key expiry and error QS-1042 > 
  0.0288 kb-0046#03   API key expiry and error QS-1042 > Error QS-1042: API key has expired
  0.0287 kb-0046#04   API key expiry and error QS-1042 > Error QS-1041: API key is invalid or revoked
['chunk_id', 'doc_id', 'heading', 'score', 'space', 'text', 'title']
```

The fused list carries both the error-code reference, which the vector side liked and which does define `QS-1042` in its table, and the dedicated article, which the keyword side found. The `mode` argument switches to `"keyword"` or `"vector"` alone, and `configs/retrieval.yaml` sets the default to `hybrid`. Notice the depth: two lists of five give RRF almost nothing to agree about, so `search_kb` retrieves 20 from each side, fuses, then truncates to the five it will hand the model.

OpenSearch can also fuse server-side, with a `hybrid` query and a search pipeline that normalizes each sub-query's scores before blending them. Client-side RRF is a dozen lines you can read, and it is what Resolve ships.

## A gold set and two numbers

Which is better, keyword, vector or the fusion? Is 220 words the right ceiling? Should `k` be 5 or 10? You cannot answer any of those by reading a few results and nodding. You need labels and a number.

A **gold set** is a list of realistic queries, each labelled with the article ids that answer it. Resolve ships two, in `data/eval/`: `retrieval_dev.jsonl` with 80 queries for building and tuning, and `retrieval_heldout.jsonl` with 40 that you run once at the end. One line per query:

```plain text
{"query": "forgot my password how do I get back in", "gold_doc_ids": ["kb-0019"]}
{"query": "QS-1050 read-only mode", "gold_doc_ids": ["kb-0052"]}
{"query": "configure saml single sign on with our identity provider", "gold_doc_ids": ["kb-0024", "kb-0025"]}
```

The labels are article ids, and the metrics are computed at the article level. That is a deliberate choice: two chunks from the right article are both right, and a labeller can say which article answers a question far more reliably than which of its seven chunks does. `evals/retrieval_eval.py` does the scoring. The core of it:

```python
# from evals/retrieval_eval.py, abridged
ranked_docs = []
for h in search_kb(q["query"], k, mode):
    if h["doc_id"] not in ranked_docs:
        ranked_docs.append(h["doc_id"])          # distinct articles, best chunk first
for rank, doc in enumerate(ranked_docs[:k], start=1):
    if doc in gold:
        hits += 1                                # recall@k: a gold article made the list
        rr += 1.0 / rank                         # MRR: how high the first one landed
        break
```

Two metrics come out. **Recall@k** is the fraction of queries for which at least one gold article appears among the articles of the top-k chunks. With one gold article per query, which is most of the set, that is the same as hit rate, and it is the ceiling on everything downstream: if the article is not in the prompt, no prompt engineering recovers it. **MRR**, mean reciprocal rank, averages 1 over the rank of the first gold article, so a hit at rank 1 scores 1, at rank 2 scores 0.5, at rank 5 scores 0.2. It matters because of the lost-in-the-middle effect from lesson 01: a relevant chunk at the bottom of five is in the context and may as well not be.

Run it against the index you built.

```bash
PYTHONPATH=src python evals/retrieval_eval.py --set dev
```

```plain text
dev      keyword  k=5  recall@k=0.787  mrr=0.732  (n=80, 1.5s)
dev      vector   k=5  recall@k=0.838  mrr=0.794  (n=80, 12.0s)
dev      hybrid   k=5  recall@k=0.925  mrr=0.839  (n=80, 4.2s)
```

Read the scorecard. Keyword search alone puts the right article in the top five for 79% of the dev queries and vector search for 84%, which is the sort of gap that would tempt you to declare a winner and move on. The fusion reaches 92.5%, past either one, and the only way for that to happen is if the two rankers fail on different queries. That is the whole argument for hybrid search, now a measurement someone else can reproduce. The timings are dominated by embedding each query on the CPU; the keyword run has no model in the loop.

## Dev versus held-out

Now the set you have not touched.

```bash
PYTHONPATH=src python evals/retrieval_eval.py --set heldout
```

```plain text
heldout  keyword  k=5  recall@k=0.775  mrr=0.652  (n=40)
heldout  vector   k=5  recall@k=0.975  mrr=0.887  (n=40)
heldout  hybrid   k=5  recall@k=0.950  mrr=0.827  (n=40)
```

Hybrid holds up at 95%, and vector alone edges past it. Two lessons in that. First, forty queries is a small set: one query is 2.5 points of recall, so a one-query difference is noise, and the honest reading is that vector and hybrid are tied on this set while keyword trails. Second, this is why the sets are separate. You choose modes, ceilings and `k` on dev, where you are allowed to look as often as you like; you run held-out once, to report a number nobody tuned against. Run held-out after every change and it turns into a second dev set, and the number you report at the end stops meaning anything. Task 1 asks for the dev numbers in the report and the held-out numbers once, at the end, and `--write` saves the run to `evals/results/retrieval.json` with the config that produced it.

## Tuning against the scorecard

Every knob in this chapter now has an answer, and they all follow the same shape: change one thing in `configs/retrieval.yaml`, rebuild if the change touched chunking, re-run the dev set, compare.

`chunk_max_words`: lower it and re-index. If recall rises, sections were being truncated by the embedding model's 256-word-piece limit. If recall holds and MRR falls, the right article is still found and its answer is spread across more chunks.

`k`: raising it almost always raises recall and spends more of the context window on every ticket. On dev, `--k 10` lifts hybrid recall from 0.925 to 0.975 at the cost of five more chunks in every prompt; that is a real trade, and the finance stakeholder has an opinion about it.

`candidates`: the depth each side retrieves before fusion. Too shallow and RRF has nothing to agree on; deeper costs a little latency.

Filters: the `space` field is a `keyword` in the mapping, and both query types accept a filter on it. A `knn` clause takes `"filter": {"term": {"space": "public"}}` beside `"k"`, and a `match` goes inside a `bool` with the same `filter`. Restricting a customer-facing search to public chunks removes the internal refund policy from the results before the reviewer has to catch it, and now you can measure what that costs in recall instead of guessing.

Reranking: retrieve a wide top 20 with hybrid search, then have a **cross-encoder**, a model that reads the query and a passage together and outputs one relevance score, re-order those 20 and keep the best 5. It is more accurate and far more expensive per candidate, the right next lever once the first stage is solid, and the wrong first move when recall@20 is already poor, because a reranker cannot promote an article the retriever never fetched.

An abstain threshold: when the top fused result is weak, the honest answer is that the knowledge base does not cover the question. RRF scores are ranks, so the threshold is better set on the underlying scores, by looking at where known-good hits sit on dev. This is the abstention lesson 01 promised, and in **LangChain & LangGraph** it becomes an input to the agent's `route` node, which only auto-resolves a ticket when the specialist's confidence clears a threshold; a retrieval that found nothing convincing is the first reason for that confidence to be low.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Drafting gold-set queries is a good use of an assistant, which is fast at inventing realistic phrasings and covers angles you would miss at query 30. The labels are a different matter: never let the same model both write the queries and decide which article answers them, or it marks its own guesses correct and every tuning decision downstream inherits the fiction. Review a generated set on three points: every `gold_doc_ids` list confirmed by you opening the article; phrasings that vary between paraphrase, exact codes and vague symptoms; and a few queries the corpus genuinely cannot answer, so abstention is measurable too.
</callout>

## What you can now claim

Two rankers, because they fail differently: keyword search owns the error code and the config key, vector search owns the paraphrase, and on dev the fusion reaches 92.5% recall@5 where neither alone passes 84%.

Rank, not score: fusion works on positions because raw scores from different rankers live on incomparable scales.

A number, not an impression: 80 labelled queries are the entire difference between tuning and guessing, and 40 more that you touch once are what make the final number believable.

Every change to this retriever from here on, a new embedding model, a different ceiling, a filter, a reranker, gets checked against the dev scorecard before it ships. That closes the chapter. **RAG Warm-Up Build** takes this retriever and puts a model on the end of it, so that a Quillstack ticket goes in and a cited draft reply comes out.
