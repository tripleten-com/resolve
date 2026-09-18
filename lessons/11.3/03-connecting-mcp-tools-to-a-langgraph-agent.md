The server from the last lesson answers JSON-RPC. The graph from **LangChain & LangGraph** calls LangChain tools: objects with a `name`, a `description`, an `args` schema, and an `invoke` method, the shape the `@tool` decorator gave `search_kb` and `read_doc`. This lesson closes that gap, then puts three controls around the tools once they are in: each specialist sees only its own, the billing lookup is checked in code, and every write waits for a human and carries a key that makes a replay harmless. Every block below is the code in `src/resolve/graph/build.py`, `src/resolve/tools/helpdesk.py`, and `src/resolve/graph/agents.py`.

## From an MCP server to LangChain tools

The `langchain-mcp-adapters` package does the translation. Install the pinned version:

```bash
pip install langchain-mcp-adapters==0.3.2
```

Its client takes a dict of named server connections in the same terms as the launch recipe from the first lesson, and `get_tools()` returns one LangChain tool per MCP tool:

```python
import asyncio, os, sys
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "helpdesk": {"transport": "stdio", "command": sys.executable,
                 "args": ["-m", "resolve.mcp_server"],
                 "env": {**os.environ, "PYTHONPATH": "src"}},
})

async def main():
    tools = await client.get_tools()
    for t in tools:
        print(f"{t.name}: {sorted(t.args)}")
    print(type(tools[0]).__name__)

asyncio.run(main())
```

```plain text
get_ticket: ['ticket_id']
get_account: ['email']
search_issues: ['query']
create_issue: ['body', 'idempotency_key', 'ticket_id', 'title']
add_ticket_comment: ['body', 'idempotency_key', 'internal', 'ticket_id']
set_ticket_status: ['idempotency_key', 'status', 'ticket_id']
StructuredTool
```

Each entry is a `StructuredTool`, the same class `@tool` builds, so the rest of the graph cannot tell an MCP tool from a hand-written one. Two things differ underneath. A tool from the adapter is asynchronous: `await tool.ainvoke(args)` works and `tool.invoke(args)` has no synchronous body to run. And what comes back is a list of content blocks, `[{"type": "text", "text": "{...}"}]`, the same shape the wire carried in the first lesson, so the caller parses the text before reading a field.

Used this way the client is **stateless**: every `get_tools()` and every tool call opens a fresh session, which for stdio means a fresh server process. That is fine for a script and wasteful inside a graph that calls tools a dozen times per ticket.

## One session for the life of the process

`src/resolve/tools/helpdesk.py` keeps one session open instead. It has to solve a second problem too: the graph's nodes are ordinary synchronous functions, because `SqliteSaver` and `interrupt()` are synchronous, and the MCP session is asynchronous. The file starts one background thread that runs an `asyncio` event loop forever, and every MCP call is handed to that loop and waited on:

```python
def run_async(coro, timeout: float = 120.0) -> Any:
    """Run a coroutine on the background loop from any thread and return its result."""
    return asyncio.run_coroutine_threadsafe(coro, _ensure_loop()).result(timeout=timeout)


def server_connection() -> dict:
    cfg = tools_config().get("mcp_server", {})
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return {"transport": "stdio", "command": sys.executable,
            "args": ["-m", cfg.get("module", "resolve.mcp_server")], "env": env}
```

`server_connection()` is the launch recipe, with the server module's name read from `configs/tools.yaml` and the environment passed on purpose, `PYTHONPATH` included, so the subprocess can import `resolve.mcp_server`. The loader opens the session once and caches the tools:

```python
def load_helpdesk_tools() -> list[BaseTool]:
    """Start the MCP server once, open one session, and return its tools as LangChain tools."""
    global _session_cm, _tools
    with _lock:
        if _tools is not None:
            return _tools
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        name = tools_config().get("mcp_server", {}).get("name", "helpdesk")
        client = MultiServerMCPClient({name: server_connection()})

        async def _open():
            cm = client.session(name)
            session = await cm.__aenter__()
            return cm, await load_mcp_tools(session, server_name=name)

        _session_cm, _tools = run_async(_open(), timeout=60)
        atexit.register(close_helpdesk_tools)
        return _tools
```

`client.session(name)` is an async context manager that starts the subprocess and performs the handshake; entering it by hand and keeping the manager in a module variable is what keeps the session open after the function returns. `load_mcp_tools(session, ...)` is the adapter's lower-level call: given an open session, list the tools and wrap them. `atexit` closes the session when the process exits so the server subprocess does not outlive it. The graph calls tools through one more helper, `invoke_tool(tool, args)`, which runs a hand-written tool directly and an adapter tool on the background loop.

`build.py` then puts both kinds of tool into one registry when the graph is assembled:

```python
def load_tools(with_helpdesk: bool = True) -> None:
    """Retrieval tools from Python, helpdesk tools from the MCP server, into one registry."""
    registry.register(KB_TOOLS)
    if with_helpdesk:
        from resolve.tools.helpdesk import load_helpdesk_tools
        registry.register(load_helpdesk_tools())
```

```plain text
['add_ticket_comment', 'create_issue', 'get_account', 'get_ticket', 'read_doc', 'search_issues', 'search_kb', 'set_ticket_status']
```

Eight tools, by name, in `resolve.graph.registry.TOOLS`. `with_helpdesk=False` exists for tests that need the graph without a server.

## Each specialist sees only its own

Which specialist gets which tool is data, in `configs/tools.yaml`:

```yaml
agents:
  knowledge_agent: [search_kb, read_doc]
  billing_agent: [search_kb, read_doc, get_account]
  bug_agent: [search_kb, read_doc, search_issues, create_issue]

writeback:
  resolve: [add_ticket_comment, set_ticket_status]
  escalate: [add_ticket_comment, set_ticket_status]

writes: [create_issue, add_ticket_comment, set_ticket_status]
```

Two helpers in `agents.py` read it, `tool_scope(agent)` and `write_tools()`, and the shared loop starts by asking the registry for exactly that list:

```python
tools = registry.get_tools(tool_scope(agent))
by_name = {t.name: t for t in tools}
schemas = [convert_to_openai_tool(t) for t in tools]
writes = write_tools()
```

```plain text
knowledge_agent ['search_kb', 'read_doc']
billing_agent ['search_kb', 'read_doc', 'get_account']
bug_agent ['search_kb', 'read_doc', 'search_issues', 'create_issue']
writes ['add_ticket_comment', 'create_issue', 'set_ticket_status']
```

`convert_to_openai_tool` is the LangChain function that turns any tool into the OpenAI function format the loop sends to the model, the same rename the first lesson did by hand. The model for `bug_agent` sees four schemas and no others; if it asks for `set_ticket_status` anyway, the loop's `name not in by_name` check refuses it before anything runs. `tests/test_tools.py` pins this: the two writeback tools appear in no specialist's list, so no specialist can close a ticket.

## The billing rule lives in code

The billing specialist may look up the ticket's own customer and nobody else. The prompt says so, and a prompt is a preference. The control is a function in `src/resolve/graph/nodes/billing_agent.py`:

```python
def check_billing_call(name: str, args: dict, state: dict) -> str | None:
    if name != "get_account":
        return None
    own = (state.get("ticket", {}).get("customer_email") or "").strip().lower()
    asked = (args.get("email") or "").strip().lower()
    if asked != own:
        return (f"refused: get_account may only be called for the ticket's own customer ({own}); "
                f"{asked or '(empty)'} is another customer. Do not look it up; hand off to a human instead.")
    return None


def billing_agent(state: dict) -> dict:
    return run_agent("billing_agent", BILLING_AGENT, state, call_check=check_billing_call)
```

The node passes it into the shared loop as `call_check`, and the loop runs it on every tool call before dispatch:

```python
refusal = None
if name not in by_name:
    refusal = f"tool {name!r} is not available to {agent}"
elif call_check is not None:
    refusal = call_check(name, args, state)
if refusal:
    result = {"error": refusal}
    entry["confirmed"] = False
    entry["result_summary"] = f"refused: {refusal}"
elif name in writes:
    result, entry = execute_write(name, args, state, agent)
else:
    result = _parse_result(safe_invoke(by_name[name], args))
```

A refused call never reaches the MCP server; the model gets the refusal string as its tool result and the `tool_calls` entry records it with `confirmed: False`. Called directly:

```plain text
check_billing_call("get_account", {"email": "quentin.byrne10@sable-systems.ca"}, state) -> None
check_billing_call("get_account", {"email": "dana@pinecone-health.org"}, state) -> "refused: get_account may only be called for the ticket's own customer (quentin.byrne10@sable-systems.ca); dana@pinecone-health.org is another customer. ..."
check_billing_call("search_kb", {"query": "duplicate charge"}, state) -> None
```

The comparison strips and lowercases both sides, because the ticket file has mixed-case emails and the model may normalise on its own. `_parse_result` on the last line is the helper that turns the adapter's content-block list back into a dict.

## A write waits for a human

The third branch above is where a tool on the `writes` list goes. `execute_write` proposes the call, pauses, and executes it once if approved:

```python
def execute_write(name: str, args: dict, state: dict, node: str) -> tuple[dict, dict]:
    tool = registry.TOOLS.get(name)
    ticket_id = state["ticket_id"]
    if tool is not None and "ticket_id" in (tool.args or {}):
        args = {**args, "ticket_id": ticket_id}  # a write may only ever touch this run's own ticket
    key = idempotency_key(ticket_id, name, args)
    proposed = {"tool": name, "args": args, "idempotency_key": key, "ticket_id": ticket_id,
                "node": node, "summary": describe_write(name, args)}
    mode = approval_mode()
    if mode == "auto":
        decision: dict = {"approved": True, "call": proposed}
    elif mode == "dry_run":
        decision = {"approved": False, "call": proposed, "dry_run": True}
    else:
        decision = interrupt(proposed)
    ...
```

Read the first `if`: whatever `ticket_id` the model put in the arguments is overwritten with the run's own. A write can only touch the ticket the graph is working on. Then the key is computed and a `proposed` dict is built with a plain-words `summary` for the human. `approval_mode()` reads `configurable.approval` from the run config: `human` pauses, `auto` executes without asking (demos only), and `dry_run` records the write and executes nothing (the evaluation runs use it).

In `human` mode, `interrupt(proposed)` from `langgraph.types` stops the run. The checkpointer saves the state, the graph returns with an `__interrupt__` entry carrying `proposed`, and `run_ticket` in `build.py` mirrors that value into `pending_write` in the state it returns. Resuming is one call:

```python
def resume_ticket(graph, ticket_id: str, approved: bool, approval: str = "human",
                  pending: dict | None = None, config: dict | None = None) -> dict:
    pending = pending or pending_write(graph, ticket_id)
    cfg = config or run_config(ticket_id, approval=approval)
    return _strip(graph.invoke(Command(resume={"approved": bool(approved), "call": pending}), cfg))
```

`Command(resume=...)` is how LangGraph hands a value back to a paused `interrupt()`. The value carries the exact write the human saw, and `execute_write` executes `decision["call"]`, so what runs is the approved write with the approved key, whatever the model proposes the second time around. That last clause matters because `interrupt()` re-executes the node from its start on resume: the model calls before the pause run again, at the cost of a few more calls. `python -m resolve.approve <ticket_id>` prints the `summary` and the key, asks yes or no, and calls `resume_ticket`; a declined write returns a message telling the model a reviewer declined it and to say so in the reply.

## The key

`src/resolve/tools/idempotency.py` is four lines:

```python
def idempotency_key(ticket_id: str, tool: str, args: dict) -> str:
    payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    return f"{ticket_id}:{tool}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]}"
```

```plain text
idempotency_key("T-10004", "create_issue", {"title": "QS-2FA code rejected", "body": "steps", "ticket_id": "T-10004"})
-> T-10004:create_issue:b6b1bbf4
same args with body "steps."     -> T-10004:create_issue:90beea5a
same args in a different order   -> T-10004:create_issue:b6b1bbf4
```

The key names the ticket, the tool, and the first eight hex characters of a SHA-256 hash of the arguments, serialised with sorted keys so argument order does not matter. Two writes with the same intent get the same key; change one character of the body and the key changes. A random key would defeat the purpose: a replay would carry a new key, the helpdesk would see it as a new write, and you would have two issues. A key with no argument hash would be wrong the other way, refusing a second, different issue for the same ticket.

## The replay test, with the real run

`tests/test_tools.py::test_replay_of_an_approved_run_files_one_issue` is the proof the Task 3 page asks for. It builds a graph on an in-memory `SqliteSaver`, invents a fresh ticket id and a fresh error code each run, and runs a bug ticket in `human` mode:

```python
state = run_ticket(graph, ticket_id, ticket=ticket, approval="human", ticket_type="bug")
assert is_paused(state)
pending = state["pending_write"]
assert pending["tool"] == "create_issue"
assert pending["idempotency_key"].startswith(f"{ticket_id}:create_issue:")
paused_checkpoint = graph.get_state(cfg).config  # the checkpoint the human approved from

after_first = resume_ticket(graph, ticket_id, True, pending=pending)
calls = [c for c in after_first["tool_calls"] if c["name"] == "create_issue"]
assert calls and calls[0]["confirmed"] and '"created": true' in calls[0]["result_summary"]

# crash-and-restart: resume from the same checkpoint again with the same approval
graph.invoke(Command(resume={"approved": True, "call": pending}), paused_checkpoint)

issues = [i for i in httpx.get(f"{HD}/issues").json() if i.get("ticket_id") == ticket_id]
assert len(issues) == 1
```

The second `graph.invoke` is the crash. It resumes the pre-approval checkpoint a second time with the same approval, as if the process had died after the human said yes and been restarted from the saved state. The test then counts issues for that ticket in the helpdesk. Run today:

```plain text
python -m pytest -q tests/test_tools.py -k replay
1 passed in 72.95s
```

```plain text
2026-09-18T05:24:30+00:00 POST /issues {"issue_id": "ISS-1011", "created": true, "ticket_id": "T-REPLAY-21a2d3", "idempotency_key": "T-REPLAY-21a2d3:create_issue:ff18676e", "title": "QS-9938: Export fails for large Team workspace (4,000 pages, 3 GB attachments)"}
```

One issue, `ISS-1011`, with the key the human approved. In this run the second resume never reached the helpdesk at all: the re-executed bug agent searched the tracker first, found `ISS-1011` by its error code, and referenced it instead of proposing a write, so no second `POST` appears in the log. That is the first line of defence, the search-before-file habit in the prompt. When the model does propose the write again, which happens when its second search uses different words, the key is the backstop, and the helpdesk answers `created: false` for the same key, as the pair of audit lines in the last lesson showed. Both lines of the test's history are worth knowing. The first version reused the same bug text every run, so the agent found the issue filed the run before and correctly did not file again, and the test proved nothing; the fresh error code is what makes the first resume a real write.

<callout icon="📖" color="gray_bg">
	Optional aside. `MultiServerMCPClient` takes any number of servers, and a hosted one is one more entry with `"transport": "streamable_http"` and a URL. If you have a Jira or Confluence site, Atlassian publishes an MCP server for it, and the same client would load its issue and page tools next to the helpdesk's with no change to the graph. Nothing in this course requires it; the helpdesk stands in for those systems, and every task runs on your laptop.
</callout>

The next lesson, **The Agent's Toolbelt End to End**, runs a real bug ticket through this graph and reads what happened: the specialist's tool choices, the pause, the approval, the issue in the helpdesk, and the audit log, then a billing ticket where the check above refuses a lookup.
