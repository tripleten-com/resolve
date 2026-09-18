Last lesson ended on a score of exactly zero: "money taken out two times" against a sentence about a card charged twice, and the lexical scorer had nothing to say, because the two share no words. The repair is to stop treating words as arbitrary symbols and give them coordinates, so that closeness in space means closeness in meaning. **Embeddings** are those coordinates: a list of a few hundred numbers that a model produces for a piece of text, placed so that texts with similar meaning land near each other. The surprising part is that nobody defines what the dimensions mean.

## Sparse counts, dense vectors

TF-IDF over a 50,000-word vocabulary hands each document 50,000 dimensions, nearly all of them zero. One dimension per word, and the dimensions know nothing about each other.

The purest version of that problem is one-hot encoding, where a word is a single 1 in a field of zeros. Measure the geometry that produces.

```python
import numpy as np

# 1. One-hot vectors: one dimension per vocabulary word.
vocab = ["invoice", "receipt", "charge", "webhook"]
onehot = np.eye(len(vocab))

# 2. Euclidean distance between every pair of distinct words.
pairs = [(i, j) for i in range(len(vocab)) for j in range(i + 1, len(vocab))]
print([round(float(np.linalg.norm(onehot[i] - onehot[j])), 3) for i, j in pairs])
```

```plain text
[1.414, 1.414, 1.414, 1.414, 1.414, 1.414]
```

Six pairs, one distance. "Invoice" is no closer to "receipt" than it is to "webhook", because the representation records identity and nothing else.

Embeddings invert the shape: a few hundred dimensions, every one of them carrying a value. Compression is a side effect. What those few hundred numbers mean is the point.

## Meaning comes from company

Fill in the blank: "the billing team reviewed the ___ and reversed the second charge."

You thought invoice, payment, transaction, receipt. You did not think webhook, timeline or workspace. Nothing in the sentence names a billing document, yet "billing", "reviewed" and "charge" narrowed the field to a handful of words that could all substitute for each other. The linguist J.R. Firth put the principle in one line: you shall know a word by the company it keeps.

Word2Vec, released by Google in 2013, turned that principle into a training objective. Train a small neural network to predict words from their neighbors, and useful vectors fall out as a byproduct. It comes in two flavors that flip the prediction. **CBOW** predicts the center word from the surrounding words: given "billing", "reviewed", "reversed", guess "invoice". **Skip-gram** predicts the surroundings from the center: given "invoice", guess that "billing" and "charge" are nearby. Skip-gram does better on rare words and smaller corpora, CBOW trains faster on large ones, and skip-gram is what you see most often.

Every word starts with a random vector, so the space begins meaningless: "invoice" might sit next to "webhook" by chance. Training nudges. Words that co-occur move closer, words that do not stay apart, and after billions of pairs, all of them extracted from raw text with no labels, the randomness has organized itself.

*Diagram idea: a two-dimensional scatter of word vectors with one visible cluster of billing words (invoice, receipt, charge, refund, proration) and a separate cluster of API words (key, token, webhook, endpoint, header).*

Compress the space to two dimensions and the structure is visible. A cluster of invoice, receipt, charge, refund and proration was never labelled "billing". It formed because those words keep the same company. Elsewhere in the space, API vocabulary groups, and so does security vocabulary.

The stranger consequence is that relationships become arithmetic. Take the vector for "king", subtract "man", add "woman", and the nearest word to the result is "queen". The same trick works for capitals (Paris minus France plus Italy lands near Rome) and for verb forms (walking minus walk plus swim lands near swimming). Gender, nationality and tense each turned into a consistent direction in the space, learned from usage patterns by a model that was never told any grammar.

### One vector per word is not enough

Every model above gives each word exactly one vector, which breaks on words that mean more than one thing. "Key" gets a single vector averaging the API sense and the door sense, useful for neither. Same for "charge", the payment and the accusation. Same for "python".

**Transformer** models solved it by making the vector depend on the sentence. A transformer is a neural network built from **attention** layers, where each token's representation is computed by looking at every other token in the input. "Rotate the API key" and "the key to the office" produce two different vectors for the same three letters. The word is fixed; the representation moves with its context. That is the property retrieval needs, because a support ticket rarely disambiguates itself.

## One vector for a whole sentence

A retriever does not index words. It indexes passages, so you need one vector per passage.

Averaging the word vectors is the cheap answer, and it works better than it has any right to: a passage full of billing words averages out somewhere in the billing region. It also throws away word order completely, which means "not refunded" and "refunded not" are identical. Weighting each word by its TF-IDF score before averaging helps, since content words then outvote function words.

**Sentence transformers** do the job properly, in four moves. Tokenize the text into subword pieces. Run it through the transformer stack, where attention lets every token's representation see every other token. **Pool** the token vectors into one, either by averaging them or by taking the vector at the `[CLS]` position, a special token that BERT-style models prepend to every input so there is one place to read a summary from. Then optionally normalize the result to unit length. What comes out is a fixed-width vector no matter how long the input was: 384 numbers for the MiniLM models, 768 for mpnet-base, 1024 for the large BGE models.

Training is what separates them from plain **BERT**, the 2018 transformer that most of these models descend from. BERT is trained to fill in masked words, which is a different objective from "put paraphrases near each other", and a BERT that has never been asked to do the second does the second badly. Sentence transformers are fine-tuned contrastively: show the model pairs, pull related ones together, push unrelated ones apart. The pairs come from **natural language inference** data (NLI), where a pair of sentences is labelled entailment, contradiction or neutral, and from semantic-textual-similarity data, where humans scored pairs from 0 to 5.

## Encoding a corpus and a query

The library is `sentence-transformers`, which downloads a model on first use and runs it locally with PyTorch. Install the pinned version once:

```bash
pip install sentence-transformers==6.0.1
```

Two lines of setup, one line of work.

```python
from sentence_transformers import SentenceTransformer

# 1. One model. 384 numbers out, whatever the input length.
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "the customer was charged twice for one invoice",
    "a duplicate payment appeared on the card statement",
    "rotate your API key before it expires",
]

# 2. Batch the encode call; normalize so cosine similarity is a plain dot product.
vectors = model.encode(sentences, batch_size=32, normalize_embeddings=True)

print(vectors.shape)
print(round(float(vectors[0] @ vectors[1]), 3), round(float(vectors[0] @ vectors[2]), 3))
print(vectors[0] @ vectors[1] > vectors[0] @ vectors[2])
```

```plain text
(3, 384)
0.522 0.045
True
```

The shape line says it plainly: three sentences in, three vectors of 384 numbers each out, and the three input sentences were 8, 8 and 6 words long. Length went away.

The second line is the whole promise of semantic retrieval in two numbers. Sentences one and two share no content word at all, and they score 0.522; sentence one against the API sentence scores 0.045. `normalize_embeddings=True` scales every vector to unit length, which is what lets the `@` operator stand in for cosine similarity with no division.

Now the question that scored zero last lesson.

```python
# 1. Embed the query with the same model, normalized the same way.
query = model.encode("money taken twice", normalize_embeddings=True)

# 2. One dot product per sentence.
print((vectors @ query).round(3))
```

```plain text
[0.481 0.501 0.07 ]
```

"Money taken twice" shares not one word with "a duplicate payment appeared on the card statement", and they score 0.501. The API sentence sits at 0.07. That is the vocabulary gap closed, and the number that lesson 02 could not produce.

## Cosine similarity, and its alternatives

The measure doing the work above:

$`\text{cos}(A,B)=\frac{A \cdot B}{\lVert A\rVert\,\lVert B\rVert}`$

It ranges from 1 for vectors pointing the same direction, through 0 for perpendicular ones, to -1 for opposite directions. Text embeddings rarely go negative in practice. For this model, paraphrases land somewhere around 0.5 and above, related topics between roughly 0.3 and 0.5, and different subjects below that. The exact bands depend on the model, which is one reason lesson 07 measures rankings instead of trusting raw scores.

Why not straight-line distance? Because magnitude carries information you do not want. A three-word answer and a 300-word article on the same topic point in nearly the same direction at very different lengths, and cosine calls them similar while Euclidean distance may not. There is one exception, and it is worth checking rather than taking on trust, because it explains why the index setting in lesson 06 barely matters once vectors are normalized.

```python
import numpy as np

# 1. Two unit-length vectors, so the dot product is the cosine.
a, b = np.array([0.6, 0.8]), np.array([0.8, 0.6])
cosine = float(a @ b)

# 2. Squared Euclidean distance, then the identity it should match.
print(round(cosine, 3), round(float(np.sum((a - b) ** 2)), 3), round(2 - 2 * cosine, 3))
```

```plain text
0.96 0.08 0.08
```

The last two numbers agree because on unit-length vectors squared Euclidean distance is exactly 2 minus twice the cosine. That mapping only ever runs one way, so the two measures put documents in the same order, and an index configured for Euclidean distance over normalized vectors is ranking by cosine whether its settings say so or not.

<table header-row="true">
<tr>
<td>Measure</td>
<td>What it compares</td>
<td>When to reach for it</td>
</tr>
<tr>
<td>Cosine similarity</td>
<td>The angle between two vectors, ignoring length</td>
<td>The default for text retrieval</td>
</tr>
<tr>
<td>Dot product</td>
<td>Angle and magnitude together</td>
<td>Vectors already normalized, so it saves the division</td>
</tr>
<tr>
<td>Euclidean distance</td>
<td>Straight-line distance between two points</td>
<td>Magnitude carries real meaning in your space</td>
</tr>
</table>

## Choosing a model, and then not changing it

`all-MiniLM-L6-v2` is the model Resolve uses everywhere. It produces 384 dimensions from six transformer layers in about 90 MB, reads at most 256 word pieces of input and silently drops the rest, and embeds Quillstack's whole knowledge base on a laptop in under a minute. Larger local models exist: `all-mpnet-base-v2` produces 768 dimensions with a 384-token limit and noticeably better quality, and `bge-large-en-v1.5` produces 1024 dimensions and expects an instruction prefix on queries, which costs you accuracy if you skip it, so read the model card. Hosted alternatives such as OpenAI's `text-embedding-3-small` replace local compute with an API call billed per token, and everything in this chapter works the same way with them.

Then one hard rule: use the same model for documents and for queries, always. An embedding is a position in a space that one particular model learned. A query embedded by model A compared against documents embedded by model B produces coordinates from two different rooms, and the similarity numbers come back looking exactly like signal. Nothing raises an error. The retrieval is simply wrong, quietly, on every query. The same rule is why changing your embedding model later means re-embedding the entire corpus, which is a decision worth making before you index rather than after.

Two habits that follow from the cost of embedding. Batch the encode call rather than looping one text at a time, which is what `batch_size=32` above is doing. And cache the vectors, because recomputing an embedding you already have is pure waste. From lesson 06 on, the index does that caching for you.

## What the geometry buys you

Nobody designed this space. It falls out of counting company at scale, and it arrives with synonyms in the same neighborhood, antonyms held apart, analogies as parallel directions, and one region per topic. Which is why "money taken twice" now finds the duplicate-charge sentence, and why last lesson's triple zero is gone.

Two vectors and a cosine work fine on three sentences. A knowledge base needs somewhere to keep those vectors that a query can search as the corpus grows, and the passages it holds have to be the right size in the first place. Both of those are the next lesson, Vector Search and Document Chunking.
