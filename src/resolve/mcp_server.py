"""The helpdesk as MCP tools: six operations over the helpdesk HTTP API, served over stdio.

    python -m resolve.mcp_server            # runs the server (the graph starts it as a subprocess)

Reads HELPDESK_URL (default http://localhost:8100). Every tool returns a compact dict; an HTTP
error comes back as {"error": ..., "status": ...} rather than an exception, so the model can
recover. Never print to stdout here: stdout is the protocol channel.
"""
from __future__ import annotations

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


@mcp.tool()
def get_account(email: str) -> dict:
    """Look up the billing account for a customer email: plan, seats, the last three invoices, and
    the last five charges. Use it only for the ticket's own customer, to check what was charged."""
    a = _call("GET", f"/accounts/{email.strip().lower()}")
    if "error" in a:
        return a
    return {
        "email": a["email"], "plan": a["plan"], "seats": a["seats"],
        "invoices": a.get("invoices", [])[-3:], "charges": a.get("charges", [])[-5:],
    }


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


@mcp.tool()
def add_ticket_comment(ticket_id: str, body: str, internal: bool = False, idempotency_key: str = "") -> dict:
    """Post a comment on a ticket: the customer-facing reply (internal=false) or a note for the
    support team (internal=true). Posting the same body twice adds nothing the second time."""
    t = _call("GET", f"/tickets/{ticket_id}")
    if "error" in t:
        return t
    for c in t.get("comments") or []:
        if c.get("author") == AUTHOR and c.get("body") == body and bool(c.get("internal")) == bool(internal):
            return {"comment_id": c["id"], "ticket_id": ticket_id, "internal": bool(internal), "created": False,
                    "idempotency_key": idempotency_key}
    r = _call("POST", f"/tickets/{ticket_id}/comments", json={"body": body, "author": AUTHOR, "internal": bool(internal)})
    if "error" in r:
        return r
    return {"comment_id": r["id"], "ticket_id": ticket_id, "internal": bool(internal), "created": True,
            "idempotency_key": idempotency_key}


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


if __name__ == "__main__":
    print(f"[mcp_server] helpdesk tools over stdio, target {HELPDESK_URL}", file=sys.stderr)
    mcp.run(transport="stdio")
