In **Agent Fundamentals** you built an agent by hand: a `while` loop over `resolve.llm.chat`, tool schemas written out as dicts, a step cap, and a fallback string. It worked, and now you know the machine underneath. This lesson swaps that hand-rolled glue for **LangChain**, a library whose design rests on one interface that every piece implements. Learn the interface once and a prompt, a model, a parser, and a retriever snap together with a single character.

<callout icon="📖" color="gray_bg">
	Every code block in this chapter runs from the Resolve repository with `PYTHONPATH=src`, the virtual environment built from `requirements.txt`, OpenSearch at `http://localhost:9200` holding the `resolve-kb` index from **Retrieval and RAG**, and your key in `.env`. Model replies shown are real and shortened; yours will differ in wording.
</callout>

## What LangChain is, and what it is not

LangChain is two things bolted together.

- Integrations: wrappers for model providers, vector stores, document loaders, and tools, so you do not rewrite the same glue for every service.
- Abstractions: one small composable interface, the Runnable, for wiring those pieces into pipelines.

It is not a reasoning engine, and it will not make a weak model smart. It adds no speed over a raw client either. Treat it as plumbing that lets you swap a model, a prompt, or a parser without rewriting the code around them.

The library's shape moves between releases. It is split into `langchain-core` (the stable interfaces), `langchain` (chains and agents), and provider packages such as `langchain-openai`. Import paths drift, so `requirements.txt` pins `langchain-core==1.6.3` and `langchain-openai==1.6.2`, the same discipline you applied to Docker base images. Resolve needs only those two and `langgraph`. There is no provider-specific package, because every provider this course names speaks the OpenAI-compatible chat protocol, and `ChatOpenAI` speaks it back.

## The core objects

<table header-row="true">
<tr>
<td>Object</td>
<td>What it does</td>
<td>What it replaces from **Agent Fundamentals**</td>
</tr>
<tr>
<td>Chat model</td>
<td>Wraps a hosted model behind one interface</td>
<td>`resolve.llm.chat` and the raw `openai` client</td>
</tr>
<tr>
<td>Prompt template</td>
<td>Turns variables into a list of messages</td>
<td>f-strings glued together by hand</td>
</tr>
<tr>
<td>Output parser</td>
<td>Turns model output into a string, JSON, or a Pydantic object</td>
<td>A manual `json.loads` plus validation</td>
</tr>
<tr>
<td>Tool</td>
<td>A callable carrying a schema the model can request</td>
<td>Your hand-written tool dict</td>
</tr>
<tr>
<td>Runnable</td>
<td>The interface (`invoke`, `batch`, `stream`) every piece shares</td>
<td>Your loop's control flow</td>
</tr>
</table>

The last row is the one the rest of this chapter leans on. Every object above is a **Runnable**, which means they all answer to the same methods and compose with one operator.

*Diagram idea: a horizontal swap panel. On the left, the hand-rolled stack from Agent Fundamentals (raw client, f-string, json.loads, hand-written tool dict); each item has an arrow to its LangChain replacement; one bar labelled "Runnable interface" spans all four replacements. Caption: "LangChain replaces the glue. The model still does the thinking."*

## Your first model call

Here is the client from **Agent Fundamentals**, rebuilt as a LangChain chat model. The endpoint, key, and model come from `.env`, the same three variables `resolve.llm` reads, and any OpenAI-compatible provider works: OpenRouter, OpenAI, Anthropic's compatibility endpoint, or a local server.

```python
# pip install langchain-core==1.6.3 langchain-openai==1.6.2
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()                      # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from .env
model = ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                   model=os.environ["LLM_MODEL"], temperature=0, max_tokens=200)

resp = model.invoke("A Quillstack customer writes: 'my 2FA code is rejected every time, "
                    "my phone clock is synced'. Is this a how-to question or a bug report? One sentence.")
print(type(resp).__name__)
print(resp.content)
print(resp.usage_metadata["input_tokens"], resp.usage_metadata["output_tokens"])
```

```plain text
AIMessage
This is a bug report, as the customer is describing a malfunction (codes being rejected despite proper setup) rather than asking how to perform a task.
49 34
```

`ChatOpenAI` is named after the protocol rather than a vendor. `base_url` points it at whatever endpoint `.env` names, and `model` is that provider's model id, which is how one class fronts a model from any vendor. `temperature=0` earns its place because triage decisions should be reproducible: the same ticket twice should classify the same way, or your evaluations measure noise. `max_tokens` bounds what one call can spend. `invoke` returns an `AIMessage`, so the text lives on `.content` rather than three levels down a JSON response, and `usage_metadata` carries the token counts that **Running the Agent as a Service** turns into a cost per ticket.

## The Runnable interface

Every LangChain component, model or prompt or parser or retriever, answers to the same small set of calls:

- `invoke(input)` runs once and returns the output.
- `batch([inputs])` runs many inputs concurrently.
- `stream(input)` yields output incrementally, token by token for a model.
- `ainvoke`, `abatch`, and `astream` are the async twins, which is what a FastAPI endpoint calls.

Because they share that interface, the pipe operator composes them left to right. LangChain calls this **LCEL**, the LangChain Expression Language, and the rule is simple: each stage's output is the next stage's input.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Resolve, Quillstack's support assistant. Answer only from the context. "
               "If the context does not answer, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | model | StrOutputParser()

print(chain.invoke({
    "context": "API keys expire after 30 days on Free, 90 days on Team, and 365 days on Business, "
               "and must be rotated before they expire.",
    "question": "How often do we need to rotate our key? We are on Team.",
}))
```

```plain text
Based on the context provided, if you're on the Team plan, your API keys expire after **90 days**, so you need to rotate your keys before that 90-day period expires.
```

Three lines do real work here. The `("system", ...)` and `("human", ...)` tuples are shorthand: `from_messages` builds the `SystemMessage` and `HumanMessage` objects the model expects and fills `{context}` and `{question}` from the dict you hand to `invoke`. `StrOutputParser` sits at the end so the chain returns a plain string instead of an `AIMessage` the next stage would have to unwrap. And `chain` is itself a Runnable, which is why it has `invoke`, `batch`, and `stream` without you writing them, and why it can drop into a bigger chain as a single stage. The context is a line from `data/facts.yaml`, so the reply is checkable: 90 days on Team.

When a chain misbehaves, call one stage on its own and look at what it hands to the next:

```python
msgs = prompt.invoke({"context": "c", "question": "q"})
print(type(msgs).__name__, [m.type for m in msgs.messages])
```

```plain text
ChatPromptValue ['system', 'human']
```

That is the cheapest debugging move in LangChain, and it works at every pipe.

## Assembling multi-part inputs

Real pipelines rarely run in a straight line, because a prompt usually needs several variables built from one input. Three helpers cover it. `RunnableParallel`, or a plain dict of Runnables, runs its branches concurrently and collects the results into a dict. `RunnablePassthrough` forwards the input unchanged, so the original value survives alongside the computed ones. `RunnableLambda` wraps any ordinary function so it can join the pipeline at all.

That is enough to express the retrieval flow from **Retrieval and RAG** declaratively. `search_kb` below is the real hybrid search from `src/resolve/kb/search.py`, over the `resolve-kb` index, returning chunks with `doc_id`, `chunk_id`, `title`, `space`, `text`, and `score`:

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from resolve.kb.search import search_kb

def retrieve(question: str) -> str:
    hits = search_kb(question, k=4)
    return "\n\n".join(f"[{h['doc_id']}] {h['text']}" for h in hits)

rag_chain = (
    {"context": RunnableLambda(retrieve), "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
print(rag_chain.invoke("How do I rotate an API key?"))
```

```plain text
# How to Rotate an API Key
Based on the knowledge base, here's the rotation process:
## Basic Steps
1. **Create a new API key** (you need Admin or Owner permissions)
2. **Update every service** that calls the API with the new key
3. **Revoke the old key** once all services are updated
## Recommended Approach (Zero-Downtime)
For production environments, use **overlapping keys**:
...
```

Read the dict out loud and the whole thing decodes: run `retrieve` on the input to produce `context`, pass the input straight through as `question`, then hand both to the prompt. The two branches run at the same time rather than one after the other. Notice what the model did with a plain "answer only from the context" prompt: markdown headings and bold, which is fine in a terminal and wrong in a ticket reply. Resolve's specialist prompts say "plain text, no markdown headings" for exactly this reason. Format is steered in the prompt, and a chain does nothing about it on its own.

What you get back is one object, and that object inherits the rest of the interface for free:

```python
answers = rag_chain.batch(["How do I archive a project?", "Is SSO available on the Team plan?"])
for a in answers:
    print("-", a[:160].replace("\n", " "))
```

```plain text
- To archive a project, follow these steps:  1. Go to **Projects** and open the project you want to archive. 2. Click **Settings > Archive project**. 3. Confirm t
- No, SSO (SAML/OIDC) is not available on the Team plan. According to the documentation, SSO is available only on the Business plan. The Team and Free plans do no
```

`batch` fires the calls concurrently instead of serializing the network round trips, and `rag_chain.stream(...)` yields partial tokens so a chat UI shows text immediately. Same chain, no extra code. Both answers are grounded, and the second is a fact from `facts.yaml` (SSO is Business only) that the model could only have got from the chunks.

## Tools are Runnables too

A **tool** is a Python function the model may ask for, described to the model by a name, a sentence about when to use it, and a typed argument schema. In **Agent Fundamentals** you wrote that schema by hand in the OpenAI function format. LangChain's `@tool` decorator builds the same JSON Schema from a function's signature and docstring. Resolve's two retrieval tools live in `src/resolve/tools/kb.py`, and this is the whole of the first one:

```python
from langchain_core.tools import tool
from resolve.kb import search as kb_search

@tool
def search_kb(query: str, k: int = 5) -> list[dict]:
    """Search the Quillstack help center for chunks that answer a question. Use it first, with the
    customer's own words or an exact error code (QS-1042), and again with different words if the
    first results do not answer. Returns chunks with doc_id, chunk_id, title, space, text, score."""
    return kb_search.search_kb(query, k=int(k))
```

The docstring is the description the model reads, so every word of it is prompt engineering, the same lesson as the hand-written `description` field. The type hints become the schema. Import the finished tool and look at what the decorator built:

```python
from resolve.tools.kb import search_kb as search_kb_tool

print(search_kb_tool.name)
print(search_kb_tool.args)
print(search_kb_tool.description[:120])
hits = search_kb_tool.invoke({"query": "rotate an API key", "k": 2})
print([(h["doc_id"], h["title"], round(h["score"], 3)) for h in hits])
```

```plain text
search_kb
{'query': {'title': 'Query', 'type': 'string'}, 'k': {'default': 5, 'title': 'K', 'type': 'integer'}}
Search the Quillstack help center for chunks that answer a question. Use it first, with the
    customer's own words or
[('kb-0045', 'Rotate an API key', 0.033), ('kb-0045', 'Rotate an API key', 0.032)]
```

`name` and `args` are what the model will be sent; `description` is the docstring. A tool is a Runnable, so `invoke` works on it too, taking a dict of arguments. The two hits are two chunks of the same article, kb-0045, and the small scores are reciprocal rank fusion scores from the hybrid search rather than probabilities. The lesson after next binds tools like this to a model and puts them in a loop.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Scaffolding LangChain code is decent delegation, mechanical and easy to check, with one specific trap: an assistant trained on older material will emit imports that no longer exist, usually the flat `from langchain import ...` form from before the package split, or a provider package you do not have installed. Have it draft, then review three things on sight: every import resolves against the version you installed, which `pip show langchain-core` tells you; each pipe stage receives the shape the previous one emits, which you confirm by calling `invoke` on a sub-chain and printing the value; and no `RunnablePassthrough` has been dropped from a dict branch, because a missing one silently starves a prompt variable rather than raising.
</callout>

## Where a chain stops

A chain is a pipeline, and that is both its strength and its ceiling. Input enters at one end, comes out the other, and every stage runs exactly once in a fixed order. For retrieval and answering, that is the right shape.

Resolve needs three things a pipe cannot express. It has to branch, sending a how-to ticket to `knowledge_agent` and a bug to `bug_agent`, then choosing between `resolve` and `escalate` at the end. It has to loop, when `review` sends a draft back. And it has to pause, waiting for a human to approve a write before `create_issue` runs. All three are control flow, and control flow is exactly what a straight pipe gives up in exchange for its terseness.

LCEL is for straight lines. The next lesson, **LangGraph State Machines, Routing, and Conditional Edges**, is for graphs: it takes the retrieve-and-answer flow you just piped, rebuilds it as nodes and edges, and then adds the branch Resolve's router needs and the loop the reviewer needs.
