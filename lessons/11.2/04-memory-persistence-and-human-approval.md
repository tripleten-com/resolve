Your graph runs once and forgets everything. Resolve cannot work that way. A run has to stop before `create_issue` and wait, possibly for hours, for a support engineer to say yes. The process may restart in between. And afterwards someone has to be able to open the run and see what it did. All three need **memory**, and LangGraph draws a hard line between two kinds that people routinely confuse.

## Two kinds of memory

<table header-row="true">
<tr>
<td></td>
<td>Conversation memory</td>
<td>Durable persistence</td>
</tr>
<tr>
<td>What it is</td>
<td>Message history kept inside state</td>
<td>State written to a store after every node</td>
</tr>
<tr>
<td>How far it reaches</td>
<td>Within one run</td>
<td>Across runs, pauses, and crashes</td>
</tr>
<tr>
<td>What implements it</td>
<td>An appending reducer on a messages field</td>
<td>A checkpointer</td>
</tr>
<tr>
<td>What it makes possible</td>
<td>Multi-turn context</td>
<td>Pause and resume, human approval, recovery, inspection</td>
</tr>
</table>

The two are independent. You can have either, both, or neither, and picking the wrong one is the reason an agent that "has memory" still starts from scratch after a restart.

## Conversation memory: the appending reducer

The default merge rule overwrites: a node returning a key replaces whatever was there. For a running message history you want the opposite. LangGraph's `add_messages` reducer does that, attached to a state field through `Annotated`. The previous lesson used it through `MessagesState`; this is what that class is made of:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # appends
    ticket_id: str                                         # overwrites

def echo(state: ChatState) -> dict:
    n = len(state["messages"])
    return {"messages": [AIMessage(f"I have seen {n} messages on {state['ticket_id']}")]}

builder = StateGraph(ChatState)
builder.add_node("echo", echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
```

The whole mechanism is the second slot in `Annotated`. `list[AnyMessage]` is the type, which is all a type checker cares about, and `add_messages` is the merge instruction, which is all LangGraph cares about. A node returning `{"messages": [reply]}` now appends `reply`, while `ticket_id`, sitting there without a reducer, keeps the overwrite behaviour. Two fields, two merge rules, one dict. The node needs no model; it reports how many messages it can see, which is the only thing this section is measuring.

## Checkpointers and the thread id

A **checkpointer** saves the full state after every node. Attach one when you compile, and every run is keyed by a `thread_id`, the identity of one conversation:

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())          # compile-time, not call-time

cfg = {"configurable": {"thread_id": "T-10018"}}              # the thread is the ticket
out = graph.invoke({"ticket_id": "T-10018", "messages": [HumanMessage("two identical charges on August 4th")]}, cfg)
print(out["messages"][-1].content)
out = graph.invoke({"ticket_id": "T-10018", "messages": [HumanMessage("any update?")]}, cfg)
print(out["messages"][-1].content)
out = graph.invoke({"ticket_id": "T-10006", "messages": [HumanMessage("refund please")]},
                   {"configurable": {"thread_id": "T-10006"}})
print(out["messages"][-1].content)
```

```plain text
I have seen 1 messages on T-10018
I have seen 3 messages on T-10018
I have seen 1 messages on T-10006
```

Two arguments carry that block. `checkpointer=` at compile time is what makes state durable at all, and `thread_id` at call time decides whose state gets loaded. The second call on T-10018 loaded the two saved messages, appended the new one, and the node saw three. The call on T-10006 started clean. In Resolve the thread id is always the ticket id, which is what lets `python -m resolve.approve T-10089` find a run that paused an hour ago in another process.

## SqliteSaver: the one the build uses

`MemorySaver` keeps checkpoints in RAM, which is fine for a test and gone the moment the process exits, so a graph that "has persistence" via `MemorySaver` loses every paused ticket on restart. Resolve uses `SqliteSaver` from the `langgraph-checkpoint-sqlite` package, a file on disk behind the same interface:

```python
# pip install langgraph-checkpoint-sqlite==3.1.1
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

path = Path("scratch/checkpoints.db")
path.parent.mkdir(parents=True, exist_ok=True)
saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
graph = builder.compile(checkpointer=saver)
graph.invoke({"ticket_id": "T-10018", "messages": [HumanMessage("two identical charges")]}, cfg)
print(graph.get_state(cfg).values["messages"][-1].content)
print(path.exists(), path.stat().st_size, "bytes")
```

```plain text
I have seen 1 messages on T-10018
True 4096 bytes
```

The graph code did not change. Only the object passed to `compile()` did, and now the file outlives the process. This is, line for line, `default_checkpointer` in `src/resolve/graph/build.py`:

```python
def default_checkpointer() -> SqliteSaver:
    path = ROOT / graph_config().get("checkpoint_path", "var/checkpoints.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
```

The path comes from `checkpoint_path: var/checkpoints.db` in `configs/graph.yaml`, and `var/` is gitignored. `check_same_thread=False` matters because the FastAPI wrapper in **Running the Agent as a Service** calls the graph from worker threads, and SQLite refuses to share a connection across threads unless told otherwise. One connection for the process, opened once in `get_graph()`.

The thread id is set in one place, `run_config` in `src/resolve/tracing.py`, and every entry point goes through it:

```python
def run_config(ticket_id: str, ticket_type: str | None = None, **configurable) -> dict:
    """The RunnableConfig for one ticket: thread_id = ticket_id, plus tags a trace can be found by."""
    tags = [ticket_id] + ([ticket_type] if ticket_type else [])
    return {
        "configurable": {"thread_id": ticket_id, **configurable},
        "tags": tags,
        "metadata": {"ticket_id": ticket_id, "ticket_type": ticket_type or "unknown"},
        "run_name": f"resolve:{ticket_id}",
    }
```

`configurable` also carries the approval mode (`human`, `auto`, or `dry_run`), which the loop reads with `get_config()`. The tags and metadata are for **LangSmith Observability**, where they let you find one ticket's trace among hundreds.

## Pausing before a write

The engineer's condition from the project brief: see the write before it happens. In LangGraph a pause is `interrupt()`. Call it inside a node with a payload, and the run stops right there, the checkpointer saves the state, and `invoke` returns with the payload under `__interrupt__`. Because the pause is a function call rather than a compile option, it can sit in the middle of a loop, in front of one specific tool call, which is where Resolve needs it. (`compile(interrupt_before=[...])` exists too and stops before a whole node, which is too coarse here: the bug agent should search freely and pause only at `create_issue`.)

```python
from langgraph.types import interrupt, Command

class WriteState(TypedDict, total=False):
    ticket_id: str
    tool_calls: list[dict]

def create_issue(title: str, ticket_id: str) -> dict:          # stand-in for the helpdesk write
    return {"issue_id": "ISS-4711", "title": title, "ticket_id": ticket_id}

runs = {"n": 0}

def bug_agent(state: WriteState) -> dict:
    runs["n"] += 1
    print(f"  bug_agent started (execution {runs['n']})")
    proposed = {"tool": "create_issue", "args": {"title": "QS-1099 on /v2/pages", "ticket_id": state["ticket_id"]}}
    decision = interrupt(proposed)                      # pause here; nothing below runs yet
    if decision["approved"]:
        result = create_issue(**decision["call"]["args"])
    else:
        result = {"declined": True}
    return {"tool_calls": [{"name": "create_issue", "result": result}]}

builder = StateGraph(WriteState)
builder.add_node("bug_agent", bug_agent)
builder.add_edge(START, "bug_agent")
builder.add_edge("bug_agent", END)
graph = builder.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "T-10089"}}
out = graph.invoke({"ticket_id": "T-10089"}, cfg)
print(sorted(out))
print(out["__interrupt__"][0].value)
```

```plain text
  bug_agent started (execution 1)
['__interrupt__', 'ticket_id']
{'tool': 'create_issue', 'args': {'title': 'QS-1099 on /v2/pages', 'ticket_id': 'T-10089'}}
```

`interrupt(proposed)` is the pause, and `proposed` is what the human will be shown. Look at the keys of `out`: `ticket_id`, which the caller passed, and `__interrupt__`. The node did not finish, so nothing it would have returned is in state. This is why the Task 3 page's phrase "write the proposed call to `pending_write` and interrupt" cannot be done literally: a node has no way to persist a partial update before `interrupt()` stops it. In Resolve the proposal travels in the interrupt payload, and `run_ticket` in `build.py` copies it into `pending_write` of the state it returns, so the rest of the code sees the field the spec names.

## Inspecting a paused run

A paused thread can be opened from any process that has the same checkpointer. `get_state` returns a snapshot:

```python
snap = graph.get_state(cfg)
print("next:", snap.next)
print("pending:", snap.interrupts[0].value["tool"])
print("values:", snap.values)
```

```plain text
next: ('bug_agent',)
pending: create_issue
values: {'ticket_id': 'T-10089'}
```

`next` names the node that will run when the thread resumes, `interrupts` holds the payloads waiting for an answer, and `values` is the saved state. `python -m resolve.approve` starts exactly here, through `pending_write` in `build.py`:

```python
def pending_write(graph, ticket_id: str) -> dict | None:
    """The write the paused run is waiting on, or None."""
    snap = graph.get_state({"configurable": {"thread_id": ticket_id}})
    for intr in snap.interrupts:
        return intr.value
    return None
```

The checkpointer also keeps every earlier snapshot of the thread, and `get_state_history(cfg)` walks them newest first. The worked example at the end of this chapter uses that to print the path a finished run took, with no model call.

## Resuming with the human's answer

`Command(resume=value)` continues the thread and makes `interrupt()` return `value` inside the node:

```python
pending = snap.interrupts[0].value
out = graph.invoke(Command(resume={"approved": True, "call": pending}), cfg)
print(out["tool_calls"])
print("next after resume:", graph.get_state(cfg).next)
```

```plain text
  bug_agent started (execution 2)
[{'name': 'create_issue', 'result': {'issue_id': 'ISS-4711', 'title': 'QS-1099 on /v2/pages', 'ticket_id': 'T-10089'}}]
next after resume: ()
```

Read the first line of the output. The node re-executed from its start. LangGraph does not freeze a Python frame; on resume it runs the node again and, when it reaches the same `interrupt()` call, hands back the resume value instead of stopping. Everything before the pause runs twice. In Resolve that means the model calls the bug agent made before proposing the write run again on resume, two to four of them, a few cents. Two consequences follow. The code before the pause has to be safe to repeat, which reads, model calls, and appends to local lists are. And the resume value carries the exact call the human saw, so the write that executes is the approved one with the approved idempotency key, whatever the model proposes on its second pass. This is the part of `execute_write` in `agents.py` that does it:

```python
    key = idempotency_key(ticket_id, name, args)          # f"{ticket_id}:{tool}:{sha256(args)[:8]}"
    proposed = {"tool": name, "args": args, "idempotency_key": key, "ticket_id": ticket_id,
                "node": node, "summary": describe_write(name, args)}
    mode = approval_mode()                                # human | auto | dry_run, from the run config
    if mode == "auto":
        decision = {"approved": True, "call": proposed}
    elif mode == "dry_run":
        decision = {"approved": False, "call": proposed, "dry_run": True}
    else:
        decision = interrupt(proposed)                    # the pause
    call = decision.get("call") or proposed               # the write the human saw, never a fresh one
    ...
    elif decision.get("approved"):
        result = _parse_result(safe_invoke(tool, call_args))
    else:
        result = {"declined": True, "message": "A human reviewer declined this write. Continue without it and say "
                                               "in your reply that a support engineer will handle it."}
```

`summary` is the plain-words line the approval script prints. The three modes exist because the evaluation suite has to run hundreds of tickets without a human at the keyboard: `auto` executes every write, `dry_run` records it and executes nothing. On a decline the tool result tells the model what happened, and the loop continues so the draft can say a support engineer will handle it. The other side, in `build.py`, sends the answer back with the pending call attached:

```python
def resume_ticket(graph, ticket_id: str, approved: bool, approval: str = "human",
                  pending: dict | None = None, config: dict | None = None) -> dict:
    pending = pending or pending_write(graph, ticket_id)
    cfg = config or run_config(ticket_id, approval=approval)
    return _strip(graph.invoke(Command(resume={"approved": bool(approved), "call": pending}), cfg))
```

A run can pause more than once. A bug ticket pauses at `create_issue`, then at the escalation note, then at the status change, so `approve.py` loops: show the pending write, ask, resume, and stop when the returned state has no `pending_write` or the human says no.

*Diagram idea: a timeline for thread T-10089. bug_agent runs, calls interrupt, a checkpoint icon saves state, the process exits. Later a new process loads the checkpoint, bug_agent runs again from its start, reaches the same interrupt, receives the approval, executes the write, and the run continues to review. Caption: "The node re-executes from its start; the approved call is the one that runs."*

## One thread is not the whole customer

Everything above is thread-scoped, which is a precise limit worth stating plainly: the checkpointer remembers this ticket. It has nothing to say about the ticket the same customer filed last week. Memory that spans threads is a different mechanism, LangGraph's `Store`, a key-value space namespaced however you choose, with `InMemoryStore` as the local implementation and `store=` as a second argument to `compile()` so nodes can read it. Resolve does not use one: a support engineer reads the customer's history in the helpdesk, and giving the agent a memory of past tickets raises retention and isolation questions the project does not take on. Know it exists, and reach for it only when a requirement genuinely spans tickets.

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(("customers", "acct-4471"), "notes", {"reported": ["T-10004", "T-10089"], "plan": "Team"})
print(store.get(("customers", "acct-4471"), "notes").value)
```

```plain text
{'reported': ['T-10004', 'T-10089'], 'plan': 'Team'}
```

## Which mechanism to reach for

Two questions separate the kinds cleanly.

- Does it need to remember earlier turns of this conversation? That is the `add_messages` reducer, a state-schema decision that costs one line.
- Does it need to survive the end of the process, pause for a person, or be inspected afterwards? That is a checkpointer, a compile-time decision, and the backend you pass is what decides whether "survive" means until the next restart or until the ticket closes.

Resolve uses both: `add_messages` is not in `TicketState` because the specialist's conversation is rebuilt from state on every pass, and the checkpointer is `SqliteSaver` under `thread_id = ticket_id`. The pause is `interrupt()` inside the shared loop, the resume is `Command`, and the node re-executes from its start when it comes back.

You now have the loop and the pause. The next lesson, **Specialists and a Reviewer: Multi-Agent in One Graph**, explains why Resolve runs three specialists through that one loop, differing only by prompt and tool scope, how the tool-less reviewer and the router fit in the same graph, and what the arrangement costs per ticket.
