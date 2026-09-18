The loop you wrote in **Agent Fundamentals** was about twenty lines: a `messages` list, a call to `chat` with the tool schemas attached, a dispatch table, one `tool` message per result, a step cap, and a fallback string. Every one of those lines has a LangGraph counterpart, and this lesson maps them one to one. Then it opens `src/resolve/graph/agents.py`, where Resolve's three specialists share exactly this loop, and shows what the production version adds and why.

## Messages as state

In the raw loop, `messages` was a local list you appended to. In a graph it has to live in state, and the default reducer from the previous lesson would overwrite it every time a node returned. LangGraph ships a state class for this case. `MessagesState` is a `TypedDict` with one field, `messages`, whose reducer is `add_messages`: a node that returns `{"messages": [reply]}` appends `reply` to the list instead of replacing it. Subclass it to add a step counter, because a counter that is not in state resets every time a node runs.

Binding tools is the other half. `bind_tools` takes a list of `@tool` functions, converts each to the JSON Schema you wrote by hand in **Agent Fundamentals**, and attaches them to every call, which is what `tools=tool_schemas` did in `chat`. The tools are Resolve's real ones from `resolve.tools.kb`, and the endpoint and model come from `.env` as in every lesson in this chapter.

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from resolve.tools.kb import search_kb, read_doc

load_dotenv()
model = ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                   model=os.environ["LLM_MODEL"], temperature=0, max_tokens=900)

TOOLS = [search_kb, read_doc]
model_with_tools = model.bind_tools(TOOLS)

class AgentState(MessagesState):
    steps: int                        # messages comes from MessagesState, with add_messages

MAX_STEPS = 6

def agent(state: AgentState) -> dict:
    resp = model_with_tools.invoke(state["messages"])
    return {"messages": [resp], "steps": state.get("steps", 0) + 1}
```

`agent` is the think half of the cycle: one model call, one appended message, one increment. `model_with_tools` is a Runnable like everything else, and the `AIMessage` it returns carries `tool_calls` as a list of dicts with `name`, `args`, and `id`, which is the same three-part request you unpacked from `msg.tool_calls` by hand.

## The act half: ToolNode and tools_condition

The dispatch table, the `try/except` around it, and the `tool` message with the matching `tool_call_id` all collapse into one prebuilt node. `ToolNode` reads the last message in state, runs every tool call it carries, and returns one `ToolMessage` per call with the right id, which is the tool-calling contract from **Agent Fundamentals** implemented for you. `tools_condition` is the prebuilt conditional edge: it returns `"tools"` if the last message has tool calls and `END` otherwise, which is the `if not msg.tool_calls: return` line from the raw loop.

The step cap goes in the router, and it is checked first:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def after_agent(state: AgentState) -> str:
    if state["steps"] >= MAX_STEPS:
        return END                       # the cap wins, whatever the model asked for
    return tools_condition(state)        # "tools" if the last message has tool calls, else END

builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")       # the cycle
graph = builder.compile()
```

Two nodes and a cycle. `agent` runs, the edge asks whether it wants a tool, `tools` runs the tool and loops back, and the whole thing ends when the model answers without a tool call or the cap fires. The map declares both destinations, so `compile()` can check them. Note what `after_agent` does when the cap wins with a tool call still pending: it ends the run with an unanswered request as the last message. That is fine for a demo and wrong for Resolve, which is one of the gaps the real loop closes.

## Running it on a real ticket

The ticket is T-10001 from `data/tickets/dev.jsonl`, a how-to about archiving a project. The system prompt is a short version of the knowledge agent's:

```python
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM = ("You are Resolve, an internal assistant for Quillstack's support team. Search the help center "
          "with search_kb before answering, read the whole article with read_doc if a chunk is cut off, "
          "state only facts that appear in what the tools returned, and end with the doc_ids you relied on.")
ticket = HumanMessage("Ticket T-10001\nSubject: how do i archive a project?\n\nwe have a project thats "
                      "finished and i want to archive it but cant find the option anywhere. also can we "
                      "bring it back later if we need to? im an admin")

out = graph.invoke({"messages": [SystemMessage(SYSTEM), ticket], "steps": 0})
for m in out["messages"]:
    if m.type == "ai" and m.tool_calls:
        print("ai  ->", [(tc["name"], tc["args"]) for tc in m.tool_calls])
    elif m.type == "tool":
        print("tool<-", m.name, str(m.content)[:80].replace("\n", " "))
print("steps:", out["steps"])
print(out["messages"][-1].content[:600])
```

```plain text
ai  -> [('search_kb', {'query': 'archive project'})]
tool<- search_kb [{"chunk_id": "kb-0015#00", "doc_id": "kb-0015", "title": "Archive and restore a
steps: 2
Perfect! I found the information. Here's what you need to know:
## How to Archive a Project
1. Go to **Projects** and open the project you want to archive
2. Click **Settings > Archive project**
3. Confirm the action in the dialog
4. The project will disappear from the active project list and move to **Projects > Archived**
## Can You Restore It Later?
Yes! Archived projects can be restored at any time. Since you're an admin, you have the permissions needed.
...
```

One search, one answer, two steps. Where a chain would have stuffed the top chunks into the prompt whether they fit or not, here the model chose one query, judged that the chunks answered both halves of the question, and stopped. The trace is the full `messages` list in state: the tool request, the `ToolMessage` with the chunk, the answer.

Now list what this graph is missing relative to Resolve, because each item is a reason the real loop is longer:

- `ToolNode` runs whatever the model asks for. Nothing scopes the tool list per specialist, nothing pauses before a write, nothing screens a result for instructions aimed at the model.
- The answer is free text with markdown in it. Resolve needs `draft`, `citations`, and `confidence` as separate fields, because `review` and `route` read them.
- Every chunk the tools returned lives only inside a `ToolMessage`. Resolve needs `docs` as a state field so the reviewer and the output guard can check citations against it.
- When the cap wins, the run just stops. Resolve returns a draft that says so at confidence 0.0, so the graph escalates rather than hangs.

## The same loop in agents.py

Resolve does not use `ToolNode`. Its specialists are plain nodes, three lines each, that call one shared function. This is the whole of `src/resolve/graph/nodes/knowledge_agent.py`:

```python
"""knowledge_agent: the shared loop with the how-to prompt and the two retrieval tools."""
from resolve.graph.agents import run_agent
from resolve.graph.prompts import KNOWLEDGE_AGENT

def knowledge_agent(state: dict) -> dict:
    return run_agent("knowledge_agent", KNOWLEDGE_AGENT, state)
```

`billing_agent.py` and `bug_agent.py` are the same three lines with a different prompt and, for billing, a `call_check` function. Here is the shape of `run_agent`, abridged; the comments mark what is cut:

```python
def run_agent(agent: str, system_prompt: str, state: dict,
              call_check: Callable[[str, dict, dict], str | None] | None = None) -> dict:
    """Run one specialist over the ticket. Returns the fields it changed."""
    max_steps = int(graph_config().get("max_steps", 6))
    tools = registry.get_tools(tool_scope(agent))          # only this specialist's tools
    by_name = {t.name: t for t in tools}
    schemas = [convert_to_openai_tool(t) for t in tools]   # @tool -> OpenAI function format
    writes = write_tools()
    docs = list(state.get("docs") or [])
    tool_calls = list(state.get("tool_calls") or [])
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": spotlight_ticket(state["ticket"], state.get("ticket_id"))}]
    # ... if review.verdict is "revise": append the old draft and the reviewer's reasons
    final = None
    for _step in range(max_steps):
        resp = chat(messages, model=model, tools=schemas, max_tokens=max_tokens)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = parse_final(msg.content or "", docs)   # the JSON draft, or a fallback
            break
        messages.append(...)                               # the assistant turn, tool calls included
        for tc in msg.tool_calls:
            # ... refuse a tool outside the scope, or one call_check rejects
            # ... a write goes through execute_write, which pauses for a human
            # ... anything else runs, is screened, and search_kb chunks are added to docs
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
    if final is None:                                      # the cap fired
        final = FinalAnswer(draft="I could not finish researching this ticket within ... steps. "
                                  "A support engineer will follow up with a full answer.",
                            citations=[], confidence=0.0)
    return {"docs": docs, "draft": final.draft, "citations": list(dict.fromkeys(final.citations)),
            "confidence": float(final.confidence), "tool_calls": tool_calls, "pending_write": None,
            "usage": meter.merge(state)}
```

Read it against the raw loop and the raw loop is all still there: the messages list starts with a system prompt and the ticket, `chat` goes out with the tool schemas, the `for` over `max_steps` is the cap, a message with no tool calls ends the loop, and each tool result goes back as a `tool` message with its id. What surrounds it is what a graph node needs that a script did not:

- The tool list comes from `registry.get_tools(tool_scope(agent))`, so each specialist sees only its own tools, as listed in `configs/tools.yaml`. `convert_to_openai_tool` turns a `@tool` into the schema `chat` expects, the same format you wrote by hand.
- `spotlight_ticket` wraps the ticket in `<<<TICKET>>>` markers so the prompt can say that marked text is data. That is a preference the model may ignore; **Defenses & Red-Teaming** adds the guards that do not depend on it.
- `call_check` is a hook the billing agent uses to refuse a `get_account` call for any email except the ticket's own customer, in code, before the tool runs.
- A tool on the writes list goes through `execute_write`, which pauses the run and waits for a human. That is the next lesson.
- Chunks from `search_kb` are appended to `docs`, deduplicated by `chunk_id`, and the final message is parsed as JSON into `FinalAnswer` with `draft`, `citations`, and `confidence`. If the model ignored the format, `parse_final` falls back to the text with whatever doc ids it can find in it and a confidence of 0.3.
- The cap fallback is a draft with confidence 0.0, so `route` escalates the ticket instead of the run hanging.
- `Meter` adds the tokens and cost of every call to `usage`.

Why `chat` from `resolve.llm` inside a graph node rather than `ChatOpenAI` and `ToolNode`? LangGraph does not care what runs inside a node. A node is plain Python that takes a dict and returns a dict, and the raw client gives the loop the exact tool-call ids and the provider's own usage figures that the meter reads. The graph supplies what the script could not: state that survives between nodes, the edges to `review` and back, a checkpointer, and the pause.

Run the real loop on its own, outside the graph, the way a unit test would:

```python
from resolve.graph import registry
from resolve.graph.agents import run_agent
from resolve.graph.prompts import KNOWLEDGE_AGENT
from resolve.graph.state import initial_state
from resolve.tickets import load_ticket
from resolve.tools.kb import KB_TOOLS

registry.register(KB_TOOLS)                                   # build.py does this at startup
state = initial_state("T-10001", load_ticket("T-10001"))
update = run_agent("knowledge_agent", KNOWLEDGE_AGENT, state)
print(sorted(update))
print([t["name"] for t in update["tool_calls"]])
print(update["citations"], update["confidence"], len(update["docs"]), "chunks")
print(update["draft"][:300])
print(update["usage"])
```

```plain text
['citations', 'confidence', 'docs', 'draft', 'pending_write', 'tool_calls', 'usage']
['search_kb', 'search_kb']
['kb-0015'] 0.95 6 chunks
Hi Hanna,
Great question! You can archive your finished project in just a few clicks, and yes, you can bring it back anytime if needed.
To archive the project:
1. Go to Projects and open the project you want to archive.
2. Click Settings > Archive project.
...
{'input_tokens': 4717, 'output_tokens': 444, 'cost_usd': 0.006937, 'latency_ms': 11000}
```

`registry.register(KB_TOOLS)` is the line `build.py` runs at startup; `initial_state` builds a complete `TicketState` with empty fields; and the update contains exactly the keys the node changed, nothing else. Two searches, one cited article, confidence 0.95, six chunks in `docs`. The draft is plain text and greets the customer by first name because `KNOWLEDGE_AGENT` asks for both, which is the difference a real system prompt makes over the short one above. The whole loop cost under a cent and took about eleven seconds, and that is the number a reviewer and a classifier get added to in the lessons that follow.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Assistants invent LangGraph prebuilt names with total confidence, and older ones reach for helpers that have been renamed or removed. Before you trust a draft, run `python -c "import langgraph.prebuilt as p; print(dir(p))"` and check every name against it, and read `inspect.signature` for any argument you did not expect. Then review the loop the same way you reviewed the raw one: the step cap is in the router and checked before `tools_condition`, every tool call gets a `ToolMessage` with its id, and the last message on the cap path is something a human can act on.
</callout>

The loop now lives in a node, and the node has a pending problem: `execute_write` needs somewhere to stop. The next lesson, **Memory, Persistence, and Human Approval**, adds the checkpointer that saves state between nodes, the thread id that names a ticket, and the `interrupt()` call that pauses the loop before a write and resumes it with the human's answer.
