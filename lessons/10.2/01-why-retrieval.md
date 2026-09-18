Ask a language model about the French Revolution and you get a careful, accurate answer. Ask it how long a Quillstack customer has to request a refund and you get an equally careful, equally confident answer that happens to be invented. Nothing in the model's behavior separates those two cases, and that missing separation is the problem this chapter exists to solve. The fix has a name: **retrieval-augmented generation**, or RAG.

## The confident wrong answer

A model that has never seen Quillstack's help center does not know that it has never seen it. Asked about a document it cannot reach, it does exactly what training rewarded: it produces fluent text shaped like an answer. Right register, right level of detail, hedges in the right places. Every fact in it fiction.

The field calls this **hallucination**, and it arrives in four recognizable shapes. Fabricated facts: "refunds are available within 30 days of any charge." Invented citations: "see the article Refund exceptions for annual plans." False specifications: "error `QS-1042` means the rate limit was exceeded." Plausible statistics: "about 73% of duplicate charges clear themselves within a week." None of them look like errors, which is why they cost so much. A crash announces itself. A wrong refund window quoted to a customer announces itself a week later, through the customer, and lands on Priya's queue as an angry follow-up.

## Three gaps, and why training closes none of them

Hallucination is the symptom. Three gaps in what a model can know produce it.

<table header-row="true">
<tr>
<td>Gap</td>
<td>What it looks like</td>
<td>Why retraining does not close it</td>
</tr>
<tr>
<td>Knowledge cutoff</td>
<td>Nothing after the training data's end date exists: a new plan price, a policy change, last week's incident</td>
<td>Every update means another training run, and the next day reopens the gap</td>
</tr>
<tr>
<td>Private data</td>
<td>Quillstack's help center, its internal runbooks and its customer records were never in the corpus</td>
<td>The data you most need answers about is the data you least want inside a training set</td>
</tr>
<tr>
<td>No way to abstain</td>
<td>An unanswerable question gets an answer anyway, at full confidence</td>
<td>Fluency was the objective, and "I don't know" scores worse than a plausible guess</td>
</tr>
</table>

The first gap gets the most attention and matters the least. The second one is where the money is. Every question worth asking Resolve is local: what is the refund window, why does this customer see two charges, what does `QS-1042` mean. A general-purpose model cannot answer any of them accurately, and no amount of extra capability changes that. The model lacks access, and access is a systems problem.

## Why pasting everything into the prompt fails

The obvious move is to stop asking the model to remember and hand it everything instead:

```plain text
Here is the complete Quillstack help center:
[about 80 articles, roughly 60,000 words]

Now answer: what does error QS-1042 mean?
```

Quillstack's help center is small: about 80 articles in two spaces, a public one that customers can read and an internal one that only support staff can. At roughly 700 words each, the whole thing is around 80,000 tokens, and a modern context window can hold that. So the ceiling on context size is the weakest of the objections here. The others are not weak.

Pricing is per token, so every ticket pays for all 80 articles, including the 79 that are irrelevant. At 400 tickets a day that is 32 million input tokens a day for a question that one article answers, and the finance stakeholder who is counting cost per ticket will notice. Long contexts also degrade: models attend more closely to the beginning and the end of what you send than to the middle, an effect known as **lost in the middle**, so burying the answer at article 60 of 80 is a reliable way to lose it. Latency scales with everything you send. And the help center grows; the arrangement that barely fits today does not fit next year.

More context is not the same as more accuracy. Relevant context is.

## Retrieve, augment, generate

Retrieval-augmented generation keeps the model's reasoning and takes away its obligation to remember. Three steps, in order.

Retrieve: search an external store for the passages most relevant to this question.

Augment: put those passages into the prompt, alongside the question.

Generate: let the model answer from what is in front of it.

*Diagram idea: three boxes left to right, Retrieve (a question going into a store of articles, a few passages coming out), Augment (the passages and the question assembled into one prompt), Generate (the prompt going into the model, a cited answer coming out).*

Watch it work on one question. Quillstack's public help center holds a refund policy article that says, among other things:

```plain text
Refunds are only possible within 14 days of a charge. Refunds are never issued
automatically or by the support assistant; every refund request is reviewed by
the billing team, which replies within 3 business days.
```

Without retrieval, "how long do I have to ask for a refund?" produces an answer averaged over every refund policy on the public internet, probably "typically 14 to 30 days," delivered with no hint that it is a guess. With retrieval, the search step finds that article, the prompt carries those three lines, and the answer comes back as 14 days, reviewed by the billing team, reply within 3 business days. Specific, correct, and traceable to an article you can open.

Look at what each gap cost and what closed it. The cutoff closes because a store is a store: edit the article and the next question sees the edit, with no retraining. Private data closes because the documents never leave Quillstack's systems; they are read at query time and passed as context. And abstention becomes possible for the first time, because now there is a checkable condition. If nothing relevant comes back, the system can say so instead of composing something.

## Where this sits inside Resolve

Resolve is the internal agent that takes the first pass on every ticket. Its specialists call a tool named `search_kb` that returns the best few chunks of the help center for a query, and the state field `docs` holds what came back. The draft reply is written from `docs`, the `citations` field lists the article ids it used, and a reviewer checks every claim in the draft against `docs` before anything goes out. Everything downstream of retrieval, from the reply to the reviewer's verdict, is only as good as what `search_kb` returned. The agent side of that is the subject of **Agent Fundamentals** and **LangChain & LangGraph**; this chapter builds the retrieval side.

## The pieces this chapter builds

Six components sit between a question and a grounded answer, and the rest of the chapter builds them in order.

A **knowledge base** of source documents, which for Quillstack is a folder of markdown files (lesson 05, The Knowledge Base as Files). A **chunker** that cuts each article into retrievable pieces, following the article's headings (lesson 04, Vector Search and Document Chunking). A scoring method, which is where most of the work lives: lexical scoring that matches words (lesson 02, Lexical Retrieval) and embeddings that match meaning (lesson 03, Semantic Retrieval). An **index** that keeps search fast as the corpus grows, first in memory (lesson 05) and then in a search engine running in Docker on your laptop (lesson 06, OpenSearch on Your Laptop). A **retriever** that runs both kinds of search, merges them, and is measured against a labelled query set (lesson 07, Hybrid Search and Retrieval Quality). And the model, doing the part it was always good at.

*Diagram idea: the six components in a row from articles on the left to the model on the right, each labelled with the lesson that builds it.*

## What retrieval buys you

Two things worth holding onto before the mechanics start.

Retrieval quality is a ceiling on answer quality. If the right passage is not in the prompt, no prompt engineering recovers it. The reasoning, the tone, the citations, all of it is capped by what the retriever found. That is why this chapter spends six lessons on search and none on wording.

RAG is a pattern, and the name gets attached to products. Underneath, every one of them runs the three steps above with some choice of chunker, embedding model and index. Knowing those choices is what lets you debug the thing when it returns confident nonsense out of the wrong article.

The cheapest of those scoring methods is also the oldest, and it is still one half of what production retrieval runs on. It counts words. The next lesson, Lexical Retrieval: TF-IDF and BM25, builds it by hand on three Quillstack sentences and shows exactly where it fails.
