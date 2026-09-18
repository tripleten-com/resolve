"""Quillstack helpdesk: a small FastAPI + SQLite stand-in for a ticketing
system, a bug tracker, and a customer-account lookup.

Students never edit this file. They write an MCP server over its HTTP API.

Environment:
  HELPDESK_DB         path to the SQLite file   (default helpdesk/helpdesk.db)
  HELPDESK_AUDIT_LOG  path to the audit log     (default helpdesk/audit.log)
  HELPDESK_DATA_DIR   where tickets/*.jsonl and accounts.jsonl live
                      (default ../data next to this package)
"""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

from helpdesk import seed

HERE = Path(__file__).resolve().parent
STATUSES = ("open", "pending_customer", "escalated", "resolved")
TICKET_COLUMNS = "ticket_id, created_at, channel, customer_email, plan, subject, status, urgency"


def db_path() -> Path:
    return Path(os.environ.get("HELPDESK_DB", HERE / "helpdesk.db"))


def audit_path() -> Path:
    return Path(os.environ.get("HELPDESK_AUDIT_LOG", HERE / "audit.log"))


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def audit(request: Request, summary: dict) -> None:
    """Append one line per write so students can show what the agent did."""
    line = f"{now()} {request.method} {request.url.path} {json.dumps(summary)}\n"
    audit_path().parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path(), "a", encoding="utf-8") as f:
        f.write(line)


def short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not db_path().exists():
        seed.seed(db_path())
    yield


app = FastAPI(title="Quillstack helpdesk", version="1.0", lifespan=lifespan)


# --- request bodies ---------------------------------------------------------

class CommentIn(BaseModel):
    body: str
    author: str = "resolve"
    internal: bool = False


class StatusIn(BaseModel):
    status: Literal["open", "pending_customer", "escalated", "resolved"]


class IssueIn(BaseModel):
    title: str
    body: str
    ticket_id: str | None = None
    idempotency_key: str | None = None


# --- helpers ----------------------------------------------------------------

def fetch_ticket(conn: sqlite3.Connection, ticket_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"ticket {ticket_id} not found")
    return row


def next_issue_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(CAST(SUBSTR(issue_id, 5) AS INTEGER)) AS n FROM issues").fetchone()
    return f"ISS-{(row['n'] or 1000) + 1}"


# --- read endpoints ---------------------------------------------------------

@app.get("/health")
def health():
    with connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    return {"status": "ok", "tickets": n}


@app.get("/tickets")
def list_tickets(status: str | None = None, limit: int = Query(50, ge=1, le=500)):
    sql = f"SELECT {TICKET_COLUMNS} FROM tickets"
    args: list = []
    if status:
        if status not in STATUSES:
            raise HTTPException(status_code=422, detail=f"status must be one of {list(STATUSES)}")
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY created_at DESC, ticket_id DESC LIMIT ?"
    args.append(limit)
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args)]


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    with connect() as conn:
        ticket = dict(fetch_ticket(conn, ticket_id))
        rows = conn.execute(
            "SELECT id, ticket_id, created_at, author, body, internal FROM comments "
            "WHERE ticket_id = ? ORDER BY id", (ticket_id,))
        ticket["comments"] = [dict(r) | {"internal": bool(r["internal"])} for r in rows]
    return ticket


@app.get("/issues")
def search_issues(q: str = ""):
    q = q.strip()
    with connect() as conn:
        if q:
            rows = conn.execute(
                "SELECT * FROM issues WHERE title LIKE ? OR body LIKE ? ORDER BY issue_id LIMIT 20",
                (f"%{q}%", f"%{q}%"))
        else:
            rows = conn.execute("SELECT * FROM issues ORDER BY issue_id DESC LIMIT 20")
        return [dict(r) for r in rows]


@app.get("/accounts/{email}")
def get_account(email: str):
    with connect() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE email = ?", (email.lower(),)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no account for {email}")
    return {
        "email": row["email"], "plan": row["plan"], "seats": row["seats"],
        "invoices": json.loads(row["invoices_json"]), "charges": json.loads(row["charges_json"]),
    }


# --- write endpoints (every one appends to the audit log) --------------------

@app.post("/tickets/{ticket_id}/comments", status_code=201)
def add_comment(ticket_id: str, body: CommentIn, request: Request):
    with connect() as conn:
        fetch_ticket(conn, ticket_id)
        cur = conn.execute(
            "INSERT INTO comments (ticket_id, created_at, author, body, internal) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, now(), body.author, body.body, int(body.internal)))
        row = conn.execute("SELECT * FROM comments WHERE id = ?", (cur.lastrowid,)).fetchone()
    audit(request, {"author": body.author, "internal": body.internal, "body": short(body.body)})
    return dict(row) | {"internal": bool(row["internal"])}


@app.patch("/tickets/{ticket_id}/status")
def set_status(ticket_id: str, body: StatusIn, request: Request):
    with connect() as conn:
        before = fetch_ticket(conn, ticket_id)["status"]
        conn.execute("UPDATE tickets SET status = ? WHERE ticket_id = ?", (body.status, ticket_id))
        row = conn.execute(f"SELECT {TICKET_COLUMNS} FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    audit(request, {"from": before, "to": body.status})
    return dict(row)


@app.post("/issues")
def create_issue(body: IssueIn, request: Request, response: Response):
    with connect() as conn:
        if body.idempotency_key:
            existing = conn.execute(
                "SELECT * FROM issues WHERE idempotency_key = ?", (body.idempotency_key,)).fetchone()
            if existing is not None:
                audit(request, {"issue_id": existing["issue_id"], "created": False,
                                "idempotency_key": body.idempotency_key})
                response.status_code = 200
                return dict(existing) | {"created": False}
        issue_id = next_issue_id(conn)
        conn.execute(
            "INSERT INTO issues (issue_id, created_at, title, body, ticket_id, status, idempotency_key) "
            "VALUES (?, ?, ?, ?, ?, 'open', ?)",
            (issue_id, now(), body.title, body.body, body.ticket_id, body.idempotency_key))
        row = conn.execute("SELECT * FROM issues WHERE issue_id = ?", (issue_id,)).fetchone()
    audit(request, {"issue_id": issue_id, "created": True, "ticket_id": body.ticket_id,
                    "idempotency_key": body.idempotency_key, "title": short(body.title)})
    response.status_code = 201
    return dict(row) | {"created": True}


@app.post("/admin/reset")
def reset(request: Request):
    counts = seed.seed(db_path())
    audit(request, counts)
    return {"reset": True} | counts
