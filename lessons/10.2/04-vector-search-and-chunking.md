You can turn a passage into a vector and you can score two vectors against each other. Two problems still stand between that and a retriever: how to search a pile of vectors that keeps growing without comparing the query against every one of them, and how to decide what a passage even is. The first is what a search engine solves on your behalf, and knowing how it solves it is what lets you choose its settings instead of accepting them. The second costs teams more retrieval quality than the first ever will.

## Brute force, and where it stops being fine

Frame the work. A query arrives, you embed it, and you want the ten nearest out of a million stored vectors at 384 dimensions each.

```python
# 1. Exact search: one multiply-and-add per dimension per stored vector.
n_vectors, dims, bytes_per_float = 1_000_000, 384, 4

ops = n_vectors * dims
ram_gb = n_vectors * dims * bytes_per_float / 1e9

print(f"{ops / 1e6:.0f}M operations per query, {ram_gb:.1f} GB resident")
```

```plain text
384M operations per query, 1.5 GB resident
```

The operation count is the number people quote, and the memory line is the harder wall: ten million vectors need 15 GB, a billion need 1.5 TB, and every concurrent user pays the full traversal again.

For a few hundred or a few thousand vectors, brute force is genuinely fine, and Quillstack's knowledge base is one of those cases: the next lesson searches it with one matrix multiplication. What follows is what you reach for an order of magnitude later, and what the search engine in lesson 06 is doing underneath the setting you pick.

## Approximate neighbors

The formal name for the task is nearest-neighbor search. Exact search guarantees the true top *k*. **Approximate nearest neighbor** search, universally shortened to ANN, gives up the guarantee in exchange for speed, typically a hundredfold or more, at 95% to 99% recall on a tuned index.

For retrieval that trade is close to free, because the ranking was already fuzzy: the fifth-best passage and the seventh-best passage are usually about equally useful to the model reading them.

Simple spatial trees do not rescue you here. In two or three dimensions a bounding box excludes most candidates cheaply, but as dimensions climb into the hundreds points drift toward being roughly equidistant and that exclusion stops working, which is the curse of dimensionality. Three families of algorithm work around it.

<table header-row="true">
<tr>
<td>Method</td>
<td>How it avoids comparing everything</td>
<td>What it costs</td>
</tr>
<tr>
<td>IVF (inverted file index)</td>
<td>Clusters the space with k-means, then searches only the cells whose centroids are nearest the query</td>
<td>A true neighbor sitting in a skipped cell is missed; probing more cells trades the speed back</td>
</tr>
<tr>
<td>HNSW (hierarchical navigable small world)</td>
<td>Builds a layered proximity graph and walks it greedily from a sparse top layer down to the full bottom layer</td>
<td>The graph itself is large in memory, and more links per node means better recall for more RAM</td>
</tr>
<tr>
<td>PQ (product quantization)</td>
<td>Splits each vector into subvectors and stores a codebook index per subvector instead of raw floats</td>
<td>Distances are computed on lossy codes, so accuracy drops a few points</td>
</tr>
</table>

Put numbers on the first and third rows.

```python
# 1. IVF: probe 10 of 1,000 cells instead of scanning the whole index.
n_vectors, n_cells, n_probe = 1_000_000, 1_000, 10
compared = n_vectors / n_cells * n_probe

# 2. PQ: 48 one-byte codes in place of 384 float32 values.
raw_bytes, pq_bytes = 384 * 4, 48

print(f"IVF compares {compared:,.0f} vectors; PQ stores {raw_bytes} B as {pq_bytes} B")
```

```plain text
IVF compares 10,000 vectors; PQ stores 1536 B as 48 B
```

A hundredfold cut in comparisons, and a thirty-twofold cut in memory that turns a 1.5 TB index into something one machine holds. Both are ceilings rather than promises: the IVF figure assumes the true neighbor sits in one of the ten cells you probed, and the PQ figure assumes you can live with distances computed on lossy codes.

**HNSW** is the default almost everywhere, including in OpenSearch, and reaches sub-millisecond search over millions of vectors. Its three knobs are worth recognizing when you see them: `m` is links per node, `ef_construction` is how many candidates the builder considers, and `ef_search` is how many the query considers. All three buy recall with memory or latency. Start with defaults, measure, then tune. Lesson 07 gives you the measurement.

OpenSearch, the engine this chapter ends up using, belongs to a different lineage from the purpose-built vector databases. It is a search engine that grew k-NN support, rather than a vector store that grew text search, and that history is exactly why it can rank by BM25 and by vector similarity over the same documents. That is why this chapter ends up there.

## What you index is what you retrieve

Now the decision that matters more than any of the above.

A whole help-center article cannot be one vector. A single embedding cannot represent seven sections on seven sub-topics, and even if it somehow could, retrieving the whole article to answer one question would fill the context window with the six sections that do not apply. Documents have to be cut into pieces, and the process is called **chunking**. A **chunk** is one such piece: the unit that gets embedded, indexed, retrieved and cited.

The tension is simple to state and impossible to resolve in general. Small chunks are precise: a 100-word passage about rotating an API key matches a key-rotation question exactly. But if the full procedure runs 300 words and you retrieve one 100-word piece of it, the answer the model gives is confidently incomplete. Large chunks preserve context: a whole section keeps every related sentence together. But if 50 of its 1,000 words are relevant, the other 950 dilute the embedding and eat the context budget you were trying to protect.

Underneath both is one rule worth stating outright: whatever you chunk is what you retrieve. Chunk by paragraph and you retrieve paragraphs. Chunk by section and you retrieve sections. So the question is what unit of information answers the questions your users ask.

### The strategies, cheapest first

Fixed size takes *N* words at a time and ignores the document entirely. Predictable, works on any format, and cuts sentences in half.

Overlap repairs the worst of that. Instead of taking words 0 to 180, then 180 to 360, take 0 to 180, then 150 to 330. Ten to twenty percent of chunk size is the usual amount. A sentence severed at a boundary now appears whole in at least one chunk, and content near boundaries gets two chances at retrieval. The cost is more chunks to store and the occasional near-duplicate in your results.

Sentence and paragraph boundaries give you complete thoughts, at the price of uneven sizes.

Section boundaries align chunks with how the author organized the material, and headings make excellent embedding context. This needs the structure to exist and to be detectable, which rules out most PDFs and suits markdown perfectly.

Recursive chunking combines them. Try splitting by section; if a section is over budget, split it by paragraph; if a paragraph is still over budget, split by sentence. It preserves structure where there is structure and degrades gracefully where there is not.

Semantic chunking is the most expensive. Embed each sentence, split where the similarity between consecutive sentences drops, and pay by embedding the corpus twice. It earns its cost on long documents with no headings to mark the seams.

*Diagram idea: one short article shown four times, with chunk boundaries drawn for fixed-size, fixed-size with overlap, section-based, and recursive splitting, so the reader sees where each strategy cuts.*

## What the Resolve chunker does

Quillstack's articles are markdown with `##` headings, and every one of them follows the same shape: a short intro, "Before you start", "Steps", sometimes one extra section, "Troubleshooting", and a "Related" list of links. The chunker in `resolve.kb.chunk` is written for that shape. It is the recursive strategy, stopped at the paragraph level.

A chunk is one heading's section. The intro before the first heading counts as a section too, with an empty heading. Sections with fewer than fifteen words are dropped, and so are sections headed "Related" or "See also": a list of four links retrieves well, because it is dense with titles, and answers nothing. A section longer than the ceiling (`max_words`, 220 by default from `configs/retrieval.yaml`) is split on blank lines into pieces that stay under the ceiling, and each new piece starts with the last 30 words of the previous one (`overlap_words`). Every chunk's text is prefixed with the article title and the heading, joined by `>`, so that the passage carries its own context into the embedding and into the prompt. And every chunk gets an id built from the article id and its position, `kb-0046#03`, which the next two lessons rely on.

Run it on a small handmade article to see the boundaries.

```python
from resolve.kb.chunk import chunk_article

article = {
    "id": "kb-9001", "title": "Rotate an API key", "space": "public", "labels": ["api"],
    "body": """Keys expire on a schedule that depends on your plan, so rotate them before the date shown in Settings.

## Steps

1. Open Settings > Integrations > API keys and click Create API key.
2. Copy the new key. It is shown in full only once.
3. Update your integration, test it, then revoke the old key.

## Tip

Set a reminder ten days before expiry.

## Related

- [Create an API key](kb-0044)
- [API key expiry and error QS-1042](kb-0046)
""",
}
for c in chunk_article(article):
    print(c["chunk_id"], repr(c["heading"]), len(c["text"].split()), "words")
print(chunk_article(article)[1]["text"])
```

```plain text
kb-9001#00 '' 23 words
kb-9001#01 'Steps' 42 words
Rotate an API key > Steps

1. Open Settings > Integrations > API keys and click Create API key.
2. Copy the new key. It is shown in full only once.
3. Update your integration, test it, then revoke the old key.
```

Four sections went in and two chunks came out. The intro became `#00` with an empty heading. "Tip" was six words long and was dropped. "Related" was skipped by name. And the "Steps" chunk begins with `Rotate an API key > Steps`, which is the title-and-heading prefix: a query about rotating keys now matches this chunk on its first line even though the word "rotate" never appears in the steps themselves.

This fits support documentation for a reason that generalizes. A help-center section is written to answer one question, and its heading names the question. Chunking on headings means the retrieval unit and the authoring unit are the same thing, tables and code blocks stay whole because they never contain a heading, and the heading in the prefix does the job that a metadata field would otherwise do.

One limit to know about. The splitter only ever cuts on blank lines. A single paragraph, a table, or a numbered list written without blank lines is never cut, however long it is, so a few chunks run past the ceiling. On the current knowledge base the longest chunk is 547 words, from an internal runbook whose "Steps" section is one long list. The embedding model reads only the first 256 word pieces of such a chunk; keyword search still sees all of it. The lever for that is `chunk_max_words`, and lesson 07 shows how to tell whether moving it helps.

## Sizing chunks against three ceilings

Three ceilings decide where the word ceiling should sit.

The embedding model's input limit is the one that bites silently. `all-MiniLM-L6-v2` truncates at 256 word pieces. Quillstack's articles run about 1.45 word pieces per word once you count the `>` characters, code, and error codes, so a 220-word ceiling is above the limit on paper. In practice the mean chunk is 110 words, because most sections are short, and only about one chunk in nine reaches the truncation point. That ratio is a measurement, and the reason to measure rather than assume is that the model raises nothing when it truncates. You get a valid vector for two thirds of a chunk and never find out.

The LLM's context window is the second. Retrieve five chunks at 800 tokens each and you have spent 4,000 tokens before the system prompt and the ticket. Smaller chunks let you retrieve from more sources within the same budget; larger chunks give you more complete information from fewer.

What users actually ask is the third and the most neglected. "What is the maximum file size?" is answered by one small focused passage. "How does SSO work?" needs a whole explanatory section. Document type follows the same logic: dense technical documentation does well at 200 to 400 tokens, while narrative and legal text want the surrounding clauses kept together.

Run the arithmetic on Quillstack's corpus. About 85 articles at 500 to 900 words each, chunked on headings, come out at 558 chunks averaging 110 words, or about six and a half chunks per article. At 384 dimensions in float32 that is 558 times 384 times 4 bytes, under a megabyte of vectors. The index is not the interesting problem here. Chunk quality is.

## Metadata turns a chunk into a citation

Consider this retrieved chunk: "This setting is available on Business only."

Useful where it was written. Useless where it landed. Which setting, in which article?

Two repairs, and you want both. Prepend the heading path into the chunk text, so the embedding captures what the chunk is about as well as what it says; the Resolve chunker does this with `Title > Heading`. And store the source data beside the chunk as separate fields: `doc_id`, `title`, `heading`, `space` and `labels`. The first improves retrieval. The second is what makes filtering and attribution possible at all, and it is how the agent's `citations` field can name `kb-0046` and the reviewer can check that no `internal` chunk was cited to a customer. Lesson 06 puts those fields into the index mapping.

One habit sits alongside them. When you assemble retrieved chunks into a prompt, put the strongest ones first and last, because of the lost-in-the-middle effect from lesson 01.

## What decides retrieval quality here

Three claims to carry forward.

1. The index decides speed; chunking decides whether the right thing is findable at all. An HNSW index over badly cut passages returns the wrong passage very quickly.
2. Every chunking parameter is empirical. The word ceiling, the overlap, whether to prefix headings: none of them has a correct answer, and all of them have a measurable one.
3. Fix the metadata schema before you index. Adding a filterable field afterward means re-chunking, re-embedding and re-indexing the whole corpus.

So far the knowledge base has been a stand-in. The next lesson, The Knowledge Base as Files, opens Quillstack's real articles, loads and chunks all of them, and builds a working retriever out of one numpy matrix before any search engine is involved.
