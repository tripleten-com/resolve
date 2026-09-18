Run `python -m resolve.run T-10001` and you get a summary: the guard's risk score, the classifier's label, the specialist's citations, the reviewer's verdict, the decision, and one line of usage. Nine lines for a run that made six model calls and four tool calls. When a ticket ends in the wrong place, those nine lines tell you that something went wrong without telling you which call did it. What you need is a recording shaped like the run itself. A **trace** is that recording: a tree with one entry per node, per model call, and per tool call, each carrying its inputs, its outputs, its timing, and its token count. LangSmith stores one for every ticket Resolve handles, and reading one is the difference between debugging the graph and guessing at it.

<callout icon="📖" color="gray_bg">
	Every code block in this chapter runs from the Resolve repository with `PYTHONPATH=src`, the venv from `requirements.txt`, OpenSearch at `http://localhost:9200`, the helpdesk at the port in `.env`, and a `.env` holding `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. The endpoint and model come from `.env` and any OpenAI-compatible provider works. The blocks that talk to LangSmith also need `LANGSMITH_API_KEY`; the ones that only read local files do not.
</callout>

## Why the summary line is not enough

A single prompt is easy to reason about. One input, one output, and if the output is wrong the prompt is the only suspect. Resolve is a loop of model calls, tool calls, and state updates whose path depends on the ticket, so the suspect list grows with every step. Three questions come up on every bad run, and none of them survive `print` debugging:

- The reviewer sent the draft back. Was the draft wrong, or did the reviewer misread the chunks?
- `knowledge_agent` called `search_kb` four times. Did any of those searches return the article the draft needed?
- This ticket cost $0.0657 and took 58 seconds when the median is $0.0223 and 17 seconds. Which node spent it?

A trace answers all three by recording every step as a row in a tree. You read the answer instead of reproducing the bug, and an agent bug often refuses to reproduce.

## Turning tracing on

LangSmith traces any LangGraph app as soon as three environment variables are set. Your graph code does not change:

```bash
pip install langsmith==0.12.6

# in .env, next to the LLM_ variables
LANGSMITH_TRACING=true        # the switch
LANGSMITH_API_KEY=lsv2_pt_... # a personal key from smith.langchain.com, free tier
LANGSMITH_PROJECT=resolve     # the project every trace is filed under
```

The first variable is the switch. With it unset the SDK stays installed and silent, which is what you want inside `pytest` and in CI. The second is a credential, so it lives in `.env` next to `LLM_API_KEY` and never in the repository. The third is a label; every trace lands in the project it names, so `resolve` for your laptop and `resolve-ci` for the workflow keeps the two from mixing.

Resolve keeps the whole question of "is tracing on?" in one place, `src/resolve/tracing.py`:

```python
import os


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes"}
```

LangGraph and the LangSmith client both read `LANGSMITH_TRACING` themselves, so this function exists for Resolve's own code to ask the same question and get the same answer. Nothing else in the module runs unless it returns `True`. That is the discipline worth copying: tracing is a property of the environment, never a line in the graph that says `tracing=True`. With the switch off, `python -m resolve.run` prints its summary and sends nothing, with no warnings. With the switch on and no key, the run still completes; LangSmith's background uploader fails quietly and you see the traces only once the key is in place.

## What LangGraph traces on its own

Turn the switch on, run one ticket, and open the project. The graph was compiled with `name="resolve"`, so the root of the trace is the graph, and every node underneath it is a row: `input_guard`, `classify`, `knowledge_agent`, `review`, and so on, in the order they ran, each with the state it received as input and the fields it returned as output. LangChain tools are traced the same way. `search_kb` and `read_doc` are `@tool` functions, and the helpdesk tools arrive through the MCP adapter as LangChain tools too, so every call the specialist makes shows up as a `tool` row under the node that made it, with its arguments and its result.

What LangGraph cannot see is the model. Resolve calls the model through the raw `openai` client in `resolve.llm`, the same `chat()` helper you met in **Agent Fundamentals**, and a plain HTTP client is invisible to a tracer. Without one more step, the tree shows `classify` as a node that took 1.2 seconds and returned a label, with no prompt, no completion, and no token count inside it.

## Wrapping the model client

`wrap_openai` fixes that. It takes an OpenAI client and hands back the same client with every `chat.completions.create` call recorded as an `llm` run carrying the messages that went over the wire, the completion, and the usage the provider reported. Nothing downstream changes, because the wrapped client has the same methods:

```python
import os

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import OpenAI

load_dotenv()

# the endpoint and model come from .env; any OpenAI-compatible provider works
client = wrap_openai(OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"]))
```

Pass the key explicitly. The `openai` package looks for `OPENAI_API_KEY` when you do not, and Resolve's key is under `LLM_API_KEY`, so an unwrapped default client raises before the first call. Resolve wraps its shared client once, at graph build time, in `wrap_llm_client()`:

```python
def wrap_llm_client() -> bool:
    if not tracing_enabled():
        return False
    from langsmith.wrappers import wrap_openai

    import resolve.llm as llm
    client = llm.client()
    if not getattr(client, "_resolve_traced", False):
        wrapped = wrap_openai(client)
        wrapped._resolve_traced = True
        llm._client = wrapped
    return True
```

The first line of `build_graph()` in `src/resolve/graph/build.py` is `wrap_llm_client()`. Every node calls `chat()`, `chat()` calls `client()`, and `client()` returns whatever `_client` holds, so swapping the module-level client is enough to trace every model call in every node. The `_resolve_traced` flag stops a second `build_graph()` from wrapping the wrapper. And because the function returns early when tracing is off, `pytest` never imports the wrapper at all. The `llm` rows appear under the name `ChatOpenAI` by default; `wrap_openai(client, chat_name="resolve-llm")` renames them if you prefer.

## Tracing a plain function

The last gap is Python that is neither a node nor a LangChain tool: the fusion step inside the hybrid search, or the injection scorer in the input guard. If you want one of those as its own row, decorate it:

```python
from langsmith import traceable

from resolve.kb import search as kb_search


@traceable(run_type="tool")
def search_kb_traced(query: str, k: int = 5) -> list[dict]:
    return kb_search.search_kb(query, k=k)


hits = search_kb_traced("how do i archive a project?", k=3)
print([(h["chunk_id"], round(h["score"], 4)) for h in hits])
```

```plain text
[('kb-0015#00', 0.032), ('kb-0015#03', 0.032), ('kb-0015#02', 0.0317)]
```

`run_type` decides the icon, the filters, and the arithmetic. LangSmith rolls token counts and cost up from `llm` rows only, so a model call filed as a `chain` drops out of the cost column. Nesting is implicit: a decorated function called inside a node lands under that node in the tree. With tracing off the decorator is a no-op and the function returns exactly what it would have returned anyway, which is what the output above shows: that block ran with `LANGSMITH_TRACING` unset.

## Tags and metadata, so you can find the run again

Two hundred dev tickets is two hundred traces, and "the billing one the reviewer sent back twice" is not a search query. Attach the identifiers you will filter on at invoke time. Resolve builds them in `run_config()`:

```python
def run_config(ticket_id: str, ticket_type: str | None = None, **configurable) -> dict:
    tags = [ticket_id] + ([ticket_type] if ticket_type else [])
    return {
        "configurable": {"thread_id": ticket_id, **configurable},
        "tags": tags,
        "metadata": {"ticket_id": ticket_id, "ticket_type": ticket_type or "unknown"},
        "run_name": f"resolve:{ticket_id}",
    }
```

```python
print(run_config("T-10001", "how_to", approval="dry_run"))
```

```plain text
{'configurable': {'thread_id': 'T-10001', 'approval': 'dry_run'}, 'tags': ['T-10001', 'how_to'], 'metadata': {'ticket_id': 'T-10001', 'ticket_type': 'how_to'}, 'run_name': 'resolve:T-10001'}
```

`run_ticket()` in `build.py` passes that dict as the second argument of `graph.invoke(initial_state(ticket_id, ticket), cfg)`. One dict does three jobs. `configurable.thread_id` is the checkpointer key you met in **LangChain & LangGraph**, so the trace and the checkpoint share the ticket id. `tags` is a flat list for slicing populations: filter on `how_to` and you see every how-to run; filter on `T-10038` and you see every attempt at that ticket, including the resumes. `metadata` holds the same identifiers as key-value pairs, which is what the SDK's filter syntax and the UI's metadata search read. `run_name` is the label on the root row, so the project list reads `resolve:T-10001` rather than a column of `resolve`.

`evals/judge_eval.py` passes `ticket_type=label["type"]` for every labelled ticket. `python -m resolve.run` does not know the type before `classify` runs, so its metadata says `unknown` and you filter those runs by the classifier's output instead. Keep customer text out of metadata; it is stored and searchable, and the ticket body is already in the root run's inputs.

## Reading the tree

The reference run of T-10001, "how do i archive a project?", left this shape behind. The local export in `traces/T-10001.json` holds the node sequence from the checkpointer and the final state, so the listing below is built from that file; a LangSmith trace has the same rows with tokens and timing on each one.

```python
import json

export = json.load(open("traces/T-10001.json", encoding="utf-8"))
state = export["state"]
print(f"root  resolve:{export['ticket_id']}   decision={state['decision']}")
for node in export["node_sequence"][1:]:
    print(f"  node  {node}")
print("tool calls, in order:")
for c in state["tool_calls"]:
    print(f"  {c['node']:<16} {c['name']:<20} {json.dumps(c['args'])[:44]}")
u = state["usage"]
print(f"usage in={u['input_tokens']} out={u['output_tokens']} cost=${u['cost_usd']:.4f} latency={u['latency_ms']}ms")
```

```plain text
root  resolve:T-10001   decision=auto_resolve
  node  input_guard
  node  classify
  node  knowledge_agent
  node  review
  node  knowledge_agent
  node  review
  node  output_guard
  node  route
  node  resolve
tool calls, in order:
  knowledge_agent  search_kb            {"query": "archive project"}
  knowledge_agent  search_kb            {"query": "unarchive restore project"}
  knowledge_agent  search_kb            {"query": "archive project"}
  knowledge_agent  search_kb            {"query": "restore unarchive project"}
  resolve          add_ticket_comment   {"ticket_id": "T-10001", "body": "Hi,\n\nGre
  resolve          set_ticket_status    {"ticket_id": "T-10001", "status": "resolved
usage in=16310 out=1175 cost=$0.0336 latency=29911ms
```

Read the shape first. `knowledge_agent` appears twice with `review` between them, so the reviewer sent the first draft back and the specialist ran again; `revision_count` in `review` is 1. The second pass repeated the same two searches, which is one reason a revision costs what it does. Then read the numbers. In the LangSmith tree each `llm` row carries its own prompt and completion tokens and each node rolls up the rows beneath it, so the root's 16,310 input tokens split across the classify call, the specialist's model turns, and the two review calls. On the reference build about 60 percent of a normal ticket's cost sits in the specialist loop, about 30 percent in the reviewer, and about 5 percent in `classify`; a revision adds roughly $0.02 and 15 seconds. T-10012, a how-to approved first time, cost $0.0223 and took 20.4 seconds against this ticket's $0.0336 and 29.9. The tree tells you where a cost fix has to land before you have formed an opinion about quality.

Then open an `llm` row, because that is where the most useful debugging artifact lives: the **resolved prompt**, the exact messages that went over the wire, with the ticket between its markers, every tool result appended, and the tool schemas attached. It is what the model saw, and that diverges from what you believe your code assembled more often than anyone expects.

## Walking a bad run

With the tree open, debugging follows one discipline: look at the whole, narrow, then read the detail.

1. Find the first node whose output looks wrong. Later nodes are usually downstream victims.
2. Look at that node's input. If the state arriving was already wrong, the real problem sits above it.
3. Read the resolved prompt and the raw completion at that node, in that order.
4. Check the tool rows feeding it. What did `search_kb` and `get_account` return?

The reference build's one routing miss on the dev tickets shows the four steps in order. T-10038 is a billing question about a $4.80 prorated charge for a seat added mid-cycle, labelled `auto_resolve`. It ended in `escalate`. Step one: the last node that changed anything is `route`, and its `escalation_reason` says the review verdict was `revise`, so `route` did its job and the problem is upstream. Step two: the `review` row's input holds the draft, and the draft says "According to our system, there were 12 days remaining when you added the seat, which resulted in the $4.80 charge." The reviewer's `reasons` say the tool results show no such number. Step three: open the `billing_agent` row's last `llm` call. The resolved prompt contains the chunks from kb-0031 with worked proration examples using 20 days and 10 days, and the completion does the arithmetic for 12 days itself. Step four: the `get_account` row returned the last five charges, June through the first of September, and the September 15th charge the customer is asking about is not among them. The model had no source for "12 days" and produced one anyway. The reviewer caught it, sent it back, caught it again on the revision, and the ticket escalated. The judge later scored the draft 3 on correctness and 3 on groundedness, and the claim-level check marked two of six claims unsupported.

That is the design working as specified, and the trace shows what to fix next: a line in the billing prompt that says to point customers at Settings > Billing > Invoices for the exact calculation rather than computing one, which is what kb-0031 itself recommends. Without the trace you would have a `MISS` in the eval output and a guess.

## Exporting a trace

Task pages ask for one exported trace per ticket type in `traces/`. The UI exports any run as JSON from the trace view. The SDK does the same thing when you want the export to be reproducible:

```python
from langsmith import Client

client = Client()   # reads LANGSMITH_API_KEY

root = next(client.list_runs(project_name="resolve", is_root=True, filter='has(tags, "T-10001")', limit=1))
run = client.read_run(root.id, load_child_runs=True)
```

`read_run(..., load_child_runs=True)` returns the root with every child nested under `child_runs`, which is the whole tree in one object. This block needs a key; it is the only one on this page that does. langsmith 0.12 marks `list_runs` and `read_run` as deprecated in favour of `client.runs.query` and `client.runs.retrieve`, which take a project id rather than a name and need a LangSmith server at 0.16 or later; both older calls keep working until 2027 and are the ones this chapter uses.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Adding instrumentation to a graph that already works is good delegation: mechanical and verifiable by re-running one ticket. Hand over the `@traceable` decorators and the tags and metadata plumbing, then review on sight for four things: `run_type` set correctly, since a model call filed as a `chain` drops out of the cost math; no customer text and no keys in `metadata`; tracing gated on `LANGSMITH_TRACING` rather than hard-coded on; and nesting that matches the call graph, checked by looking at the tree rather than at the diff. Do not delegate the diagnosis. An assistant reading a pasted prompt cannot see the upstream state or the tool results, so treat its explanation as a hypothesis and confirm it against the trace.
</callout>

A trace is a record of one run, and one run cannot tell you whether a new billing prompt is better or merely different. The next lesson, Building Evals in LangSmith, measures a change across the held-out tickets at once, with a frozen dataset and evaluators that turn each run into a score.
