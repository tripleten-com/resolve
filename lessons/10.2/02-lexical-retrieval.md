Retrieval starts with a question search engines have been answering for fifty years: given this query, which document in the pile is most about it? The oldest workable answer counts words, and it survives in production because it is fast, interpretable and unbeatable at one specific job. **TF-IDF** is where that answer begins. You will rank three Quillstack sentences with it in a moment, then watch it score a perfect zero on a question it should have got right.

## Not every matched word is worth the same

Counting is the naive version. A 200-word help-center article might use "the" fifteen times and "refund" twice. Match a query on "the" and you have matched every article, which is the same as matching none.

Two quantities fix that together. **Term frequency** asks how much of this document is this word:

$`\text{TF}=\frac{t}{n}`$

where *t* is how often the word appears in the document and *n* is the document's length in words. **Inverse document frequency** asks how unusual the word is across the whole collection:

$`\text{IDF}=\log\!\left(\frac{D}{d}\right)`$

where *D* is the number of documents and *d* is how many of them contain the word. Multiply the two and you get a score that rises with local density and falls with global commonness.

Put numbers on it. Take a collection of 20 help-center articles in which "refund" appears in 2 of them, and one 40-word article that uses "refund" 5 times. TF is 5 divided by 40, or 0.125. IDF is the base-10 log of 20 over 2, which is the log of 10, which is 1. TF-IDF is 0.125. In the same article, "the" appears 8 times, so its TF is 0.2, higher than "refund". But "the" appears in all 20 articles, so its IDF is the log of 1, which is 0, and its TF-IDF is 0. The most frequent word in the document contributes nothing to the score.

The logarithm is doing real work there. Without it, a word appearing in 1 document out of 10,000 would carry an IDF of 10,000 and drown out everything else in the query. With a base-10 log it carries 4. Which base you use shifts the absolute numbers and leaves the ranking alone, so implementations differ freely; most use the natural log.

## Three sentences, by hand

Small enough to check on paper, and small enough to run. The three sentences are the kind of thing Quillstack's billing and API articles say.

```python
import math

# 1. Three "articles", preprocessed to lowercase, as word lists.
articles = [
    "the api key expired and the request failed with error qs-1042".split(),
    "the card was charged twice and the billing team reversed the second charge".split(),
    "refund requests go to the billing team and a refund is reviewed within three business days".split(),
]

# 2. Textbook TF-IDF: count over document length, times log(D / d).
def tfidf(term, doc):
    tf = doc.count(term) / len(doc)
    df = sum(term in a for a in articles)
    return tf * math.log10(len(articles) / df)

# 3. A rare term, a universal one, an error code, and a shared one.
for term in ("refund", "the", "qs-1042", "billing"):
    print(term, [round(tfidf(term, a), 3) for a in articles])
```

```plain text
refund [0.0, 0.0, 0.06]
the [0.0, 0.0, 0.0]
qs-1042 [0.043, 0.0, 0.0]
billing [0.0, 0.014, 0.011]
```

Walk those numbers back to the formula. The three sentences are 11, 13 and 16 words long. "Refund" appears twice in the third sentence's sixteen words, so TF is 0.125, and it sits in 1 of the 3 articles, so IDF is the base-10 log of 3, or 0.477; the product is 0.060. "The" is present in all three, so its IDF is the log of 1 and its whole row is zero however often it occurs. `qs-1042` occurs once, in the first sentence, on the same IDF of 0.477, and lands at 0.043. "Billing" is in two of the three sentences, so its IDF drops to the log of 1.5, or 0.176, and its scores are small in both.

Read the `qs-1042` and `billing` rows side by side. One mention of an error code outscores a mention of "billing" three times over, because distinctiveness beats volume. That is exactly the property a retriever wants: a ticket containing `QS-1042` should point hard at the one article about that code and nowhere else.

## Ranking a query against the corpus

The same maths, done by a library, on a query instead of a single term. The library is scikit-learn, which you have used since the first project.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. The same three sentences stand in for the knowledge base.
corpus = [
    "the api key expired and the request failed with error qs-1042",
    "the card was charged twice and the billing team reversed the second charge",
    "refund requests go to the billing team and a refund is reviewed within three business days",
]

# 2. Fit on the corpus only. The query must not contribute to the IDF statistics.
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(corpus)

# 3. Rank the corpus for a query phrased the way the documents are.
q1 = cosine_similarity(vectorizer.transform(["my request failed with qs-1042"]), doc_vectors)[0]
print("scores:", q1.round(3))
for i in q1.argsort()[::-1]:
    print("  ", corpus[i])

# 4. Now ask about the second sentence in different words.
q2 = cosine_similarity(vectorizer.transform(["money taken out two times"]), doc_vectors)[0]
print("scores:", q2.round(3))

# 5. The IDF table the library actually built, for three of the terms.
idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
print("idf:", round(idf["the"], 3), round(idf["qs"], 3), round(idf["billing"], 3))
```

```plain text
scores: [0.682 0.    0.   ]
   the api key expired and the request failed with error qs-1042
   refund requests go to the billing team and a refund is reviewed within three business days
   the card was charged twice and the billing team reversed the second charge
scores: [0. 0. 0.]
idf: 1.0 1.693 1.288
```

Step 2 is the line that matters most and looks least important. `fit_transform` builds the vocabulary and the IDF table from the corpus; `transform` on the query reuses both. Fitting on the query as well would let the query's own words influence the statistics that are supposed to judge it.

Step 3 scores by **cosine similarity**: the cosine of the angle between the query vector and each document vector, which is 1 when they point the same way and 0 when they share no terms at all. Cosine ignores vector length, and that matters here, because a long article should not win for being long. The first sentence shares "request", "failed" and the error code with the query and scores 0.682; the other two share only "the" and "with", which the IDF table has made nearly worthless, and they tie at zero.

One detail hides in step 5. The vectorizer's default tokenizer splits `qs-1042` on the hyphen, so the vocabulary holds `qs` and `1042` as two terms, each of them rare, each carrying the top IDF of 1.693. Search engines do the same thing with their analyzers, and it is why a pasted error code still works as a query: both halves are distinctive.

Step 4 is the interesting one. "Money taken out two times" is a real support question, and the corpus contains a sentence about a card charged twice and reversed. The score is `[0. 0. 0.]`. Not low. Zero, three times, because the query and the documents have no word in common. Passing `ngram_range=(1, 2)` would match short phrases as well as single words, which helps with "billing team" and does nothing at all for this.

## scikit-learn's formula is not the textbook one

Worth knowing before you compare your hand arithmetic to library output and conclude one of them is broken. `TfidfVectorizer` makes three choices that differ from the definitions above.

Term frequency is the raw count, with no division by document length. Length gets handled at the very end instead, by **L2 normalization**: each document's row is divided by its Euclidean length, so every row ends up at length 1. That is the `norm="l2"` default, and it is why a cosine between two rows is a plain dot product.

IDF is smoothed, as if one document containing every term had been added to the collection:

$`\text{idf}(t)=\ln\!\left(\frac{1+n}{1+\text{df}(t)}\right)+1`$

That trailing 1 is a floor, and step 5 above printed what it buys. `the`, which appears in all three sentences, comes back with an IDF of 1.0 instead of the textbook 0: the natural log of 4 over 4 is zero, and the floor lifts it to 1. `qs`, in exactly one sentence, gets the log of 4 over 2 plus 1, or 1.693. `billing`, in two, gets 1.288. So the gap between a unique term and a universal one narrows from everything-to-nothing down to 1.693 against 1. Common words get pushed down rather than deleted, and the ordering of terms by informativeness is unchanged.

## BM25 adds saturation and length

TF-IDF's term frequency grows without limit. The twentieth "refund" in an article does not make it twice as much about refunds as the tenth did. Linear TF says it does. **BM25** is the ranking function that encodes that intuition, and it is the default in every modern search engine:

$`\text{BM25}(q,d)=\sum_{t \in q}\text{IDF}(t)\cdot\frac{f_{t,d}\,(k_1+1)}{f_{t,d}+k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)}`$

Two parameters carry the behavior. Lift the term-frequency fraction out on its own and count what each extra occurrence actually buys.

```python
# 1. The BM25 term-frequency factor, with Lucene's default parameters.
k1, b, avgdl = 1.2, 0.75, 13

def tf_factor(f, doclen):
    length_norm = 1 - b + b * (doclen / avgdl)
    return f * (k1 + 1) / (f + k1 * length_norm)

# 2. One term occurring 1, 2, 5 and 20 times in an average-length document.
print([round(tf_factor(f, avgdl), 3) for f in (1, 2, 5, 20)])
```

```plain text
[1.0, 1.375, 1.774, 2.075]
```

The second occurrence of the term adds 0.375. The fifteen occurrences that separate the fifth from the twentieth add 0.3 in total, and no count at all pushes the factor past *k₁+1*, which is 2.2 here. That flattening is **saturation**, and *k₁* sets how quickly it arrives. The *b* term compares this document's length to the collection average (`avgdl`, 13 words for the three sentences above), and at *b* equal to 1 the score is fully length-normalized while at 0 it is not normalized at all. The defaults, 1.2 and 0.75, come from **Lucene**, the open-source search library that Elasticsearch and OpenSearch are built on. The IDF factor is a variant of the one above, computed as the natural log of 1 plus (*N* minus *n* plus 0.5) over (*n* plus 0.5).

You will not implement this. A search engine stores every term in an **inverted index**, a table from each term to the list of documents that contain it and how often, which is what makes "which documents mention `qs-1042`" a lookup instead of a scan. OpenSearch computes BM25 over that index for any `match` query, so the lexical half of Resolve's retriever costs one line of query JSON. That is the `match` query in lesson 06, OpenSearch on Your Laptop.

## What lexical retrieval can and cannot do

It owns exact terms. Error codes like `QS-1042`, config keys like `sso.enforce`, invoice numbers like `INV-202608-00417`, header names, command flags. A rare token appearing exactly is precisely what IDF rewards most, and nothing that works on meaning matches it as reliably. Keep that in mind through the next two lessons, when meaning starts looking like the answer to everything.

It cannot cross a vocabulary gap. "Money taken out two times" against a sentence about a card charged twice scored zero, and no parameter fixes that. The representation has no concept of meaning at all, so two ways of saying the same thing are, to it, unrelated.

Closing that gap means giving up on words as arbitrary symbols and representing each one by the company it keeps. The next lesson, Semantic Retrieval: Sentence Embeddings, does that, and the same duplicate-charge question comes back with a score.
