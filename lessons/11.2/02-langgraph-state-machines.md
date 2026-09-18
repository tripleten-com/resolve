A chain only goes forward. Resolve does not: it branches to pick a specialist, loops when the reviewer sends a draft back, and pauses to wait for a person. **LangGraph** makes that control flow explicit by turning it into a state machine you can draw, compile, and inspect. Four moving parts do all the work, and one of them is where a pipeline stops being a pipeline.

## Why a graph instead of a pipe

Recall the orchestration patterns from **Agent Fundamentals**: the router, the reflection loop, the human checkpoint. A linear LCEL chain cannot express any of the three, for the same reason each time. A pipe only moves forward. A graph moves wherever its edges point, which buys you:

- Cycles: an edge can point back at an earlier node, which is what review-and-revise needs.
- Branches: one node can hand control to different successors depending on what is in state.
- Checkpoints: the machine can stop between nodes and resume later, for human approval or after a crash.

None of this changes the model calls. The same prompts go to the same model as they would from a chain. What changes is that the wiring becomes data you control instead of a fixed pipe, and control flow you can inspect is control flow you can debug.

## The four moving parts

- State: a typed dict handed to every node. It is the graph's shared memory.
- Nodes: plain functions that read state and return a partial update.
- Edges: the connections that decide which node runs next.
- The compiled graph: the runnable machine you `invoke` like any other Runnable.

State is a `TypedDict`. Each node receives the whole thing and returns only the keys it changed, which LangGraph merges back in. The default merge rule overwrites: return `{"docs": ...}` and `docs` is replaced while every other key survives untouched. That rule is called a **reducer**, and swapping it for an appending one is how message history works, which the lesson after next covers.

Resolve's full state is `TicketState` in `src/resolve/graph/state.py`, fifteen fields, one per thing some node writes. This lesson uses a slice of it with the same names, so nothing you write here gets renamed later. The model client is the one from the previous lesson; endpoint and model come from `.env`, and any OpenAI-compatible provider works.

```python
# pip install langgraph==1.2.11
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from resolve.kb.search import search_kb

load_dotenv()
model = ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                   model=os.environ["LLM_MODEL"], temperature=0, max_tokens=300)

class State(TypedDict, total=False):
    ticket_id: str
    ticket: dict            # subject, body, ...
    docs: list[dict]        # doc_id, chunk_id, title, space, text, score
    draft: str

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Resolve, Quillstack's support assistant. Answer only from the chunks. "
               "Cite the doc_id of each chunk you used."),
    ("human", "Chunks:\n{chunks}\n\nTicket subject: {subject}\n{body}"),
])

def retrieve(state: State) -> dict:
    t = state["ticket"]
    return {"docs": search_kb(f"{t['subject']} {t['body']}", k=4)}

def answer(state: State) -> dict:
    chunks = "\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in state["docs"])
    msg = prompt.invoke({"chunks": chunks, **state["ticket"]})
    return {"draft": model.invoke(msg).content}
```

There is no framework machinery in either function. They are the kind of Python you would write anyway: take a dict, return a dict. `retrieve` writes `docs` as the list of chunk dicts `search_kb` returns, the same shape `TicketState.docs` declares, and `answer` reads it. Returning a single key is the normal case, and it is worth being precise about why it is safe: `ticket` and `draft` are absent from `retrieve`'s return because this node has no opinion about them. Nodes that return the full state instead are how two nodes end up quietly clobbering each other's work. `total=False` says every key is optional, because state starts as whatever the caller passes and each node adds its own.

## Edges, compilation, and watching state flow

Edges wire the nodes into a flow. `START` and `END` are built-in sentinels marking where execution enters and leaves the machine:

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)

builder.add_edge(START, "retrieve")       # edges name their endpoints by the registered string
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

graph = builder.compile()

ticket = {"subject": "how do i archive a project?",
          "body": "we have a project thats finished and i want to archive it. can we bring it back later?"}
for step in graph.stream({"ticket_id": "T-10001", "ticket": ticket}, stream_mode="values"):
    print(sorted(step))                   # the whole state after every node
print([d["doc_id"] for d in step["docs"]])
print(step["draft"][:300])
```

```plain text
['ticket', 'ticket_id']
['docs', 'ticket', 'ticket_id']
['docs', 'draft', 'ticket', 'ticket_id']
['kb-0015', 'kb-0015', 'kb-0015', 'kb-0015']
# Archiving and Restoring Projects
Great question! You can definitely archive a finished project and bring it back later if needed.
## How to Archive a Project
Here are the steps [kb-0015]:
1. Go to Projects and open the project you want to archive
2. Click **Settings > Archive project**
...
```

`compile()` hands back a Runnable, so `invoke`, `batch`, and `stream` all behave exactly as they did on a chain. The three printed lines are the state growing one node at a time: the input on its own, then `docs` after `retrieve`, then `draft` after `answer`. That printout is the cheapest debugging tool in the chapter, and it is the same data **LangSmith Observability** renders as a trace. All four chunks came from kb-0015, "Archive and restore a project", which is the gold article for this ticket in `data/tickets/dev.jsonl`.

Notice what this graph actually is. Two nodes in a row with no choice between them is a chain wearing a different coat, and it took more lines than the pipe version. The structure only starts paying for itself when the edges stop being straight.

*Diagram idea: START into retrieve into answer into END as solid arrows, plus a ghosted dashed back-edge curving from a faint decision point to retrieve. A side panel shows the State dict with ticket, docs, draft, highlighting which single key each node writes. Caption: "Nodes write to shared state; edges decide who runs next."*

## A conditional edge is Resolve's router

Here is the part a chain cannot do. A **conditional edge** is an ordinary function that reads state and returns a string naming the next node, plus a map telling LangGraph what each string means. Resolve's first one sits after `classify`. It reads two fields of `classification`, `type` and `aimed_at_agent`, and names a specialist or `escalate`:

```python
class TicketState(TypedDict, total=False):
    ticket_id: str
    ticket: dict
    classification: dict    # type, urgency, aimed_at_agent, confidence
    docs: list[dict]
    draft: str
    citations: list[str]
    confidence: float
    review: dict            # verdict, reasons, revision_count
    guard_output: dict      # flags, ok
    decision: str
    escalation_reason: str

SPECIALISTS = {"how_to": "knowledge_agent", "billing": "billing_agent", "bug": "bug_agent"}

def after_classify(state: TicketState) -> str:
    c = state["classification"]
    if c["type"] == "security" or c["aimed_at_agent"]:
        return "escalate"
    return SPECIALISTS.get(c["type"], "escalate")
```

That is the spec's rule word for word: `security`, or any text aimed at the agent, goes to `escalate` before any retrieval happens; the other three types go to their specialist. The `.get(..., "escalate")` default means a type the map does not know escalates instead of crashing the run.

To exercise the wiring without paying for model calls, the nodes are stubs. `classify` reads a label the test ticket carries, the specialist returns a fixed draft, and the reviewer approves. `route` is the real rule from the spec, because the decision logic should be tested from the first run:

```python
def classify(state: TicketState) -> dict:
    return {"classification": state["ticket"]["_label"]}   # a model call in the real build

def specialist(state: TicketState) -> dict:
    return {"draft": "Open the project, choose Archive from the menu...", "citations": ["kb-0015"], "confidence": 0.9}

def review(state: TicketState) -> dict:
    return {"review": {"verdict": "approve", "reasons": [], "revision_count": 0}}

def output_guard(state: TicketState) -> dict:
    return {"guard_output": {"flags": [], "ok": True}}

AUTO_RESOLVE_TYPES = ("how_to", "billing")
THRESHOLD = 0.7

def route(state: TicketState) -> dict:
    reasons = []
    if state["review"]["verdict"] != "approve":
        reasons.append("review did not approve")
    if state["confidence"] < THRESHOLD:
        reasons.append(f"confidence {state['confidence']} below {THRESHOLD}")
    if state["guard_output"]["flags"]:
        reasons.append("output flags")
    if state["classification"]["type"] not in AUTO_RESOLVE_TYPES:
        reasons.append(f"type {state['classification']['type']} never auto-resolves")
    if reasons:
        return {"decision": "escalate", "escalation_reason": "; ".join(reasons)}
    return {"decision": "auto_resolve", "escalation_reason": ""}

def after_route(state: TicketState) -> str:
    return "resolve" if state["decision"] == "auto_resolve" else "escalate"

def resolve(state: TicketState) -> dict:
    return {"decision": "auto_resolve"}

def escalate(state: TicketState) -> dict:
    return {"decision": "escalate"}
```

`route` is a node, and it is code with no model in it. It writes `decision` after checking the four conditions the spec names: the reviewer approved, the confidence clears the threshold (`confidence_threshold: 0.7` in `configs/graph.yaml`), the output guard raised no flags, and the type is one that may auto-resolve at all. It collects every reason it found rather than stopping at the first, so the escalation note a human reads later says everything that was wrong. `after_route` is the second conditional edge, and it only reads `decision`.

Now the wiring. Every string a router may return appears on the left of its map, and the node it means on the right:

```python
builder = StateGraph(TicketState)
for name, fn in [("classify", classify), ("knowledge_agent", specialist), ("billing_agent", specialist),
                 ("bug_agent", specialist), ("review", review), ("output_guard", output_guard),
                 ("route", route), ("resolve", resolve), ("escalate", escalate)]:
    builder.add_node(name, fn)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", after_classify, {
    "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent",
    "bug_agent": "bug_agent", "escalate": "escalate"})
for s in ("knowledge_agent", "billing_agent", "bug_agent"):
    builder.add_edge(s, "review")
builder.add_edge("review", "output_guard")
builder.add_edge("output_guard", "route")
builder.add_conditional_edges("route", after_route, {"resolve": "resolve", "escalate": "escalate"})
builder.add_edge("resolve", END)
builder.add_edge("escalate", END)
graph = builder.compile()

def run(label: dict) -> None:
    path, final = [], {}
    for update in graph.stream({"ticket_id": "T-0", "ticket": {"_label": label}}, stream_mode="updates"):
        for node, delta in update.items():    # one entry per node that ran
            path.append(node)
            final.update(delta)
    print(" -> ".join(path), "|", final["decision"], final.get("escalation_reason", ""))

run({"type": "how_to", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.95})
run({"type": "bug", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.9})
run({"type": "how_to", "urgency": "normal", "aimed_at_agent": True, "confidence": 0.9})
```

```plain text
classify -> knowledge_agent -> review -> output_guard -> route -> resolve | auto_resolve
classify -> bug_agent -> review -> output_guard -> route -> escalate | escalate type bug never auto-resolves
classify -> escalate | escalate
```

The third argument to `add_conditional_edges` is a **routing map**, and being explicit there is what lets LangGraph check the wiring before anything runs. `compile()` validates destinations and reachability: a typo on the right-hand side of the map names a node that was never registered and is caught there, as is any node with no path from `START`. What no compiler can catch is the string the router hands back, because that value does not exist until the graph is running. A router that returns `"escalte"` fails on the one ticket that takes that branch, in production, map or no map. The map still earns its lines: every legal destination sits in one readable place.

`stream_mode="updates"` emits one dict per node that ran, keyed by the node's name, which is how `run` reconstructs the path. Read the three paths against the spec. The how-to ticket went through its specialist and the reviewer and resolved. The bug ticket went through exactly the same nodes and still escalated, because `route` says a `bug` never auto-resolves; the draft it produced is for the human. The ticket aimed at the agent never reached a specialist at all.

*Diagram idea: the triage graph with a diamond after classify fanning out to three specialists and escalate, all three specialists converging on review, then output_guard, route, and a second diamond into resolve or escalate. Caption: "A conditional edge is a function returning the next node's name."*

## Loops that cannot run forever

A conditional edge can also point backward, and a backward edge is a cycle. That is how the reviewer works: draft, review, redraft. It is also how an agent burns a month of budget in an afternoon, because a cycle with no exit condition is a `while True` with no `break`. Resolve bounds it with `revision_count`, a counter inside `review` that the reviewer node increments and the edge after it reads:

```python
REVISION_LIMIT = 1                                    # revision_limit in configs/graph.yaml

def review2(state: TicketState) -> dict:
    prev = state.get("review") or {}
    count = prev.get("revision_count", 0)
    verdict = "revise" if "12 days" in state["draft"] else "approve"   # a model call in the real build
    if verdict == "revise":
        count += 1
    return {"review": {"verdict": verdict, "reasons": ["unsupported claim: 12 days"] if verdict == "revise" else [],
                       "revision_count": count}}

def after_review(state: TicketState) -> str:
    r = state["review"]
    if r["verdict"] == "revise" and r["revision_count"] <= REVISION_LIMIT:
        return SPECIALISTS[state["classification"]["type"]]    # back to the same specialist
    return "output_guard"

def specialist2(state: TicketState) -> dict:
    if (state.get("review") or {}).get("verdict") == "revise":
        return {"draft": "Your next invoice is prorated; the billing team can confirm the amount.", "confidence": 0.85}
    return {"draft": "You have 12 days remaining on the current cycle.", "confidence": 0.85}
```

The stub reviewer objects to one specific claim, an arithmetic result no document supports, which is the kind of thing the real reviewer catches. The stub specialist notices it is being asked again, because `review.verdict` is `revise` when it runs the second time, and drops the claim. Two details carry this block. `revision_count` has to be a state field rather than a local variable, because state is the only thing that survives from one node invocation to the next. And the reviewer has to increment it on every `revise`, or the counter reads zero forever and the cap never fires. That failure is quiet: nothing raises, the graph cycles until the provider's rate limiter stops it, and the symptom you see is a hung ticket and a bill.

Only one edge changes in the wiring. The plain edge from `review` to `output_guard` becomes a conditional one whose map includes the three specialists:

```python
builder = StateGraph(TicketState)
for name, fn in [("classify", classify), ("knowledge_agent", specialist2), ("billing_agent", specialist2),
                 ("bug_agent", specialist2), ("review", review2), ("output_guard", output_guard),
                 ("route", route), ("resolve", resolve), ("escalate", escalate)]:
    builder.add_node(name, fn)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", after_classify, {
    "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent",
    "bug_agent": "bug_agent", "escalate": "escalate"})
for s in ("knowledge_agent", "billing_agent", "bug_agent"):
    builder.add_edge(s, "review")
builder.add_conditional_edges("review", after_review, {           # the cycle
    "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent",
    "bug_agent": "bug_agent", "output_guard": "output_guard"})
builder.add_edge("output_guard", "route")
builder.add_conditional_edges("route", after_route, {"resolve": "resolve", "escalate": "escalate"})
builder.add_edge("resolve", END)
builder.add_edge("escalate", END)
graph = builder.compile()

label = {"type": "billing", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.9}
path = []
for update in graph.stream({"ticket_id": "T-0", "ticket": {"_label": label}}, stream_mode="updates"):
    for node, delta in update.items():
        path.append(node)
        if node in ("review", "route"):
            print(f"{node}:", delta)
print(" -> ".join(path))
```

```plain text
review: {'review': {'verdict': 'revise', 'reasons': ['unsupported claim: 12 days'], 'revision_count': 1}}
review: {'review': {'verdict': 'approve', 'reasons': [], 'revision_count': 1}}
route: {'decision': 'auto_resolve', 'escalation_reason': ''}
classify -> billing_agent -> review -> billing_agent -> review -> output_guard -> route -> resolve
```

`billing_agent` appears twice in the path, once for the first draft and once for the revision, and `review` twice, with the counter at 1 both times: it went up on the `revise` and stayed on the `approve`. Had the second verdict been `revise` as well, the count would have reached 2, the edge would have gone on to `output_guard` anyway, and `route` would have escalated because the verdict was not `approve`. The loop ends either way, and the ticket reaches a human with the reviewer's reasons attached.

Three routing shapes cover most of what Resolve needs:

<table header-row="true">
<tr>
<td>Pattern</td>
<td>What the router returns</td>
<td>Where it is in Resolve</td>
</tr>
<tr>
<td>Category dispatch</td>
<td>One handler node per category</td>
<td>`after_classify`: type to specialist</td>
</tr>
<tr>
<td>Confidence gate</td>
<td>`resolve` or `escalate`</td>
<td>`route` plus `after_route`</td>
</tr>
<tr>
<td>Reflect and retry</td>
<td>Back to the specialist, or forward</td>
<td>`after_review`, capped by `revision_count`</td>
</tr>
</table>

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Graph wiring is good delegation, since it is structural code with a compiler that checks it, and an assistant makes the same three mistakes reliably. Have it draft the nodes and edges, then review on sight: every node has a path from `START` and a path to `END`, because an unreachable node raises at `compile()` and the error names it, so read the error instead of re-prompting; every string a router can return appears in its routing map; and every cycle has both a cap in the router and the matching increment in the node it loops through, which is the one an assistant leaves out most often.
</callout>

## What the wiring buys you

1. A graph is state plus nodes plus edges, compiled. Nodes return partial updates, edges decide who runs next, `START` and `END` bound the flow, and `compile()` gives you back something that behaves like every other Runnable.
2. Conditional edges are where the decisions live. A router is a function returning a node name, declared against an explicit map so the graph can be validated before it ever runs. Resolve has four: after the input guard, after `classify`, after `review`, and after `route`.
3. Every cycle needs a cap and an advancing counter. The cap goes in the router, the increment goes in the node, and both live in state.

What the graph still cannot do is think for itself inside a node: `knowledge_agent` above returns a canned draft. The next lesson, **The Agent Loop as a Graph**, takes the tool-calling loop from **Agent Fundamentals** and puts it inside that node, first with LangGraph's prebuilt pieces and then the way Resolve's `agents.py` really does it.
