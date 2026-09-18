The last lesson called a server that already existed. This one builds it. `src/resolve/mcp_server.py` is about a hundred lines, and every line answers one of four questions: how does a Python function become a tool the model can see, what does the model read to choose it, what comes back, and what happens when the helpdesk says no. The file is the reference for Task 3 and you read it block by block here.

## What the helpdesk offers

The helpdesk is a small FastAPI app that stands in for three internal systems: tickets and comments, the bug tracker, and the customer-account lookup. Its README lists the endpoints. Six of them become tools:

<table header-row="true">
<tr>
<td>Tool</td>
<td>Endpoint</td>
<td>Kind</td>
</tr>
<tr>
<td>`get_ticket(ticket_id)`</td>
<td>`GET /tickets/{id}`</td>
<td>read</td>
</tr>
<tr>
<td>`get_account(email)`</td>
<td>`GET /accounts/{email}`</td>
<td>read</td>
</tr>
<tr>
<td>`search_issues(query)`</td>
<td>`GET /issues?q=`</td>
<td>read</td>
</tr>
<tr>
<td>`create_issue(title, body, ticket_id, idempotency_key)`</td>
<td>`POST /issues`</td>
<td>write</td>
</tr>
<tr>
<td>`add_ticket_comment(ticket_id, body, internal)`</td>
<td>`POST /tickets/{id}/comments`</td>
<td>write</td>
</tr>
<tr>
<td>`set_ticket_status(ticket_id, status)`</td>
<td>`PATCH /tickets/{id}/status`</td>
<td>write</td>
</tr>
</table>

The server does no thinking. It describes these six operations so a model can pick one, calls the HTTP API when a tool is invoked, and shapes what comes back.

## FastMCP turns functions into tools

The `mcp` package ships a high-level server called **FastMCP**. You give it a name, decorate ordinary Python functions, and it derives each tool's name from the function name, its description from the docstring, and its input schema from the type hints. It also speaks the protocol from the last lesson, so `initialize`, `tools/list`, and `tools/call` all work without you writing a line of JSON-RPC.

```bash
pip install mcp==1.30.0 httpx==0.28.1
```

The top of the file:

```python
import os
import sys

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
HELPDESK_URL = os.environ.get("HELPDESK_URL", "http://localhost:8100").rstrip("/")
STATUSES = ("open", "pending_customer", "escalated", "resolved")
AUTHOR = "resolve"

mcp = FastMCP("helpdesk", instructions="Quillstack helpdesk: tickets, comments, issues, accounts.")
```

`load_dotenv()` reads `.env`, so `HELPDESK_URL` comes from the environment with the Compose default as a fallback; nothing about where the helpdesk lives is in code. `STATUSES` is the allowed set from the README, kept here so the server can refuse a bad status before an HTTP call. `AUTHOR` is the name every comment is posted under, which matters later for deduplication. The `instructions` string is sent to the client during the handshake as a one-line summary of what the server is for.

One rule about this file before anything else: never print to stdout. Over stdio, stdout is the protocol channel, and a stray `print` corrupts the stream and kills the session. Anything you want to see goes to stderr, which is what the `sys` import is for.

## One HTTP helper that returns errors

Every tool calls the helpdesk through the same function:

```python
def _client() -> httpx.Client:
    return httpx.Client(base_url=HELPDESK_URL, timeout=15)


def _call(method: str, path: str, **kw) -> dict | list:
    try:
        with _client() as c:
            r = c.request(method, path, **kw)
    except httpx.HTTPError as exc:
        return {"error": f"helpdesk unreachable at {HELPDESK_URL}: {exc.__class__.__name__}", "status": 0}
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return {"error": str(detail)[:300], "status": r.status_code}
    return r.json()
```

`httpx` is the HTTP client library the rest of Resolve uses; `httpx.Client(base_url=...)` lets every call pass a path such as `/tickets/T-10004` instead of a full URL. The two branches that return `{"error": ..., "status": ...}` are the design decision of the file. A tool that raises an exception ends the agent's turn with a traceback the model never sees. A tool that returns an error dict hands the model a sentence it can act on: retry with a different id, or tell the customer a support engineer will follow up. The helpdesk puts its reason under `detail` in the JSON body, so the helper pulls that out and caps it at 300 characters, because an HTML error page pasted into the model's context is tokens spent on nothing.

## A read tool, compact

The first tool wraps `GET /tickets/{id}`:

```python
def _short(text: str | None, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


@mcp.tool()
def get_ticket(ticket_id: str) -> dict:
    """Read one helpdesk ticket by id (for example T-10004) with its status and the last five
    comments. Use it when you need the ticket's history, not for the ticket text you already have."""
    t = _call("GET", f"/tickets/{ticket_id}")
    if "error" in t:
        return t
    comments = t.get("comments") or []
    return {
        "ticket_id": t["ticket_id"], "status": t.get("status"), "plan": t.get("plan"),
        "customer_email": t.get("customer_email"), "subject": t.get("subject"),
        "body": _short(t.get("body"), 1500),
        "comments": [{"author": c["author"], "internal": c["internal"], "body": _short(c["body"], 300)}
                     for c in comments[-5:]],
    }
```

`@mcp.tool()` registers the function. Its name becomes the tool name, the `ticket_id: str` hint becomes `{"ticket_id": {"type": "string"}}` with `ticket_id` in `required`, and the docstring becomes the description. That docstring is the most important prose in the file. The model reads it to decide whether to call the tool, so it is written as an instruction: what the tool does, when to use it, and when not to. The second sentence exists because the specialist already has the ticket text in its prompt, and a model that re-reads what it already has is spending money.

The return value is trimmed on purpose. The helpdesk sends every comment and the full body; the tool keeps the fields a specialist would read, the last five comments, and caps every text. A tool result re-enters the model's context on every later turn of the loop, so a ticket with fifty comments would be paid for five or six times. Compact results are a cost control and a focus control at once.

`search_issues` and `get_account` follow the same pattern. The issue search caps at five hits and returns a `count`, and its docstring tells the model to use it before filing anything:

```python
@mcp.tool()
def search_issues(query: str) -> dict:
    """Search the bug tracker for existing issues whose title or body contains the query (an error
    code such as QS-1042 works best). Use it before filing a new issue so you never file a duplicate."""
    issues = _call("GET", "/issues", params={"q": query})
    if isinstance(issues, dict) and "error" in issues:
        return issues
    return {"query": query, "count": len(issues), "issues": [
        {"issue_id": i["issue_id"], "title": i["title"], "status": i["status"], "body": _short(i["body"], 240)}
        for i in issues[:5]]}
```

`GET /issues` returns a list on success and a dict on error, hence the `isinstance` check. The docstring's hint about error codes is a fact about the helpdesk: its search is a substring match, so `QS-1042` finds the issue and a paraphrased sentence finds nothing. `get_account` lowercases and strips the email and keeps the last three invoices and five charges; the rule that the agent may only look up the ticket's own customer is enforced in the graph, in the next lesson, and the server does not know about it.

## A write tool with an idempotency key

The write tools change the helpdesk. `create_issue` takes one extra argument:

```python
@mcp.tool()
def create_issue(title: str, body: str, ticket_id: str, idempotency_key: str = "") -> dict:
    """File a new bug issue in the tracker, linked to the ticket. Use it only when search_issues
    found no matching issue. Title: one line with the error code; body: steps, plan, what changed."""
    payload = {"title": title, "body": body, "ticket_id": ticket_id}
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    r = _call("POST", "/issues", json=payload)
    if "error" in r:
        return r
    return {"issue_id": r["issue_id"], "created": r.get("created"), "title": r["title"], "ticket_id": r.get("ticket_id")}
```

An **idempotency key** is a string that identifies one intended write. The helpdesk remembers every key it has seen on `POST /issues`; a request with a new key creates an issue and answers `created: true`, and a request with a key it has seen before creates nothing and answers `created: false` with the existing issue. The key has a default of an empty string so the tool is still valid when called by hand, and the graph fills it in for every real write. Where the key comes from and why it is built the way it is belongs to the next lesson; the server's only job is to pass it through and hand `created` back so the caller can tell a first write from a replay.

The other two writes have no key on the helpdesk side, so the server makes them safe to repeat itself. `add_ticket_comment` reads the ticket first and, if a comment by `resolve` with the same body and the same `internal` flag already exists, returns that comment with `created: false` instead of posting again. `set_ticket_status` checks the value locally and skips the call when nothing would change:

```python
@mcp.tool()
def set_ticket_status(ticket_id: str, status: str, idempotency_key: str = "") -> dict:
    """Change a ticket's status to one of: open, pending_customer, escalated, resolved. Setting the
    status it already has changes nothing."""
    if status not in STATUSES:
        return {"error": f"status must be one of {list(STATUSES)}, got {status!r}", "status": 422}
    t = _call("GET", f"/tickets/{ticket_id}")
    if "error" in t:
        return t
    if t.get("status") == status:
        return {"ticket_id": ticket_id, "status": status, "changed": False, "idempotency_key": idempotency_key}
    r = _call("PATCH", f"/tickets/{ticket_id}/status", json={"status": status})
    if "error" in r:
        return r
    return {"ticket_id": ticket_id, "status": r["status"], "changed": True, "idempotency_key": idempotency_key}
```

The first `if` is the check-yourself box on the Task 3 page: a status outside the allowed set comes back as an error result with the same shape the helpdesk would have used, and the server keeps running. Both write tools accept the key and echo it back so the graph's audit trail can match a result to the write the human approved.

## Running the server

The last block of the file starts the server:

```python
if __name__ == "__main__":
    print(f"[mcp_server] helpdesk tools over stdio, target {HELPDESK_URL}", file=sys.stderr)
    mcp.run(transport="stdio")
```

```bash
python -m resolve.mcp_server
```

That prints one line to stderr and then waits on stdin for JSON-RPC. Run on its own it looks like it has hung; it has not, it is a server with no client. The graph starts it as a subprocess and never runs it by hand, and FastMCP writes an `INFO` line to stderr for every request it processes, which is the log to read when a tool call goes wrong.

## A short client script

The client from the last lesson is enough to check the server. This lists every tool with its required arguments, runs the two checks from the Task 3 page, and files the same issue twice:

```python
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(
    command=sys.executable, args=["-m", "resolve.mcp_server"],
    env={**os.environ, "PYTHONPATH": "src"})

async def call(session, name, args):
    res = await session.call_tool(name, args)
    return json.loads(res.content[0].text)

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for t in (await session.list_tools()).tools:
                print(f"{t.name:20} required={t.inputSchema['required']}")
            print(await call(session, "search_issues", {"query": "QS-1042"}))
            print(await call(session, "set_ticket_status", {"ticket_id": "T-10003", "status": "closed"}))
            args = {"title": "QS-1777 lesson duplicate check", "body": "filed twice on purpose",
                    "ticket_id": "T-LESSON-02", "idempotency_key": "T-LESSON-02:create_issue:0badc0de"}
            print(await call(session, "create_issue", args))
            print(await call(session, "create_issue", args))

asyncio.run(main())
```

```plain text
get_ticket           required=['ticket_id']
get_account          required=['email']
search_issues        required=['query']
create_issue         required=['title', 'body', 'ticket_id']
add_ticket_comment   required=['ticket_id', 'body']
set_ticket_status    required=['ticket_id', 'status']
{'query': 'QS-1042', 'count': 1, 'issues': [{'issue_id': 'ISS-1001', 'title': 'QS-1042 returned for keys rotated less than 90 days ago', 'status': 'open', 'body': 'Team-plan workspaces get QS-1042 (key rotation not due) when calling POST /v1/api-keys/rotate, ...'}]}
{'error': "status must be one of ['open', 'pending_customer', 'escalated', 'resolved'], got 'closed'", 'status': 422}
{'issue_id': 'ISS-1016', 'created': True, 'title': 'QS-1777 lesson duplicate check', 'ticket_id': 'T-LESSON-02'}
{'issue_id': 'ISS-1016', 'created': False, 'title': 'QS-1777 lesson duplicate check', 'ticket_id': 'T-LESSON-02'}
```

The `idempotency_key` argument is absent from `required` for the three writes because it has a default; the model may omit it, and the graph never does. The bad status came back as data and the session survived to make the next call. The same key filed one issue, `ISS-1016`, and the second call said so. The helpdesk's audit log has both lines:

```plain text
2026-09-18T05:30:14+00:00 POST /issues {"issue_id": "ISS-1016", "created": true, "ticket_id": "T-LESSON-02", "idempotency_key": "T-LESSON-02:create_issue:0badc0de", "title": "QS-1777 lesson duplicate check"}
2026-09-18T05:30:15+00:00 POST /issues {"issue_id": "ISS-1016", "created": false, "idempotency_key": "T-LESSON-02:create_issue:0badc0de"}
```

With the helpdesk in Compose the log is inside the container, at `/app/db/audit.log`; `docker compose exec helpdesk tail -n 5 /app/db/audit.log` reads it. That pair of lines is the evidence the Task 3 page asks for, produced at the tool level. The next lesson, **Connecting MCP Tools to a LangGraph Agent**, loads these six tools into the graph next to the two retrieval tools, hands each specialist only its own, and puts a human in front of every write.

<callout icon="🤖" color="blue_bg">
	**Working with AI** An assistant will write a server like this from the helpdesk README in a minute, and it will probably run. Read three things in what it produces. The docstrings: it will describe what each tool does and leave out when to use it and when not to, and that second half is what steers the model. The error path: generated tools tend to let `httpx` raise, which ends the agent's turn instead of giving the model a sentence to recover from. And the result shape: it will return the helpdesk's whole response, comments and all, which is tokens the specialist pays for on every turn.
</callout>
