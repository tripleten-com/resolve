"""Drop and recreate the helpdesk database, then load tickets and accounts.

    python -m helpdesk.seed              # seed the default DB
    HELPDESK_DB=/tmp/x.db python -m helpdesk.seed

Tickets come from data/tickets/dev.jsonl and heldout.jsonl (label fields are
ignored), accounts from data/accounts.jsonl. When those files are missing the
three tickets and one account in helpdesk/fixtures.jsonl are used instead, so
the app still runs. Eight pre-existing bug issues are always seeded.
"""

import json
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
TICKET_FIELDS = ("ticket_id", "created_at", "channel", "customer_email", "plan", "subject", "body")

SCHEMA = """
DROP TABLE IF EXISTS comments;
DROP TABLE IF EXISTS issues;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS tickets;
CREATE TABLE tickets (
    ticket_id      TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    channel        TEXT NOT NULL,
    customer_email TEXT NOT NULL,
    plan           TEXT NOT NULL,
    subject        TEXT NOT NULL,
    body           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'open',
    urgency        TEXT
);
CREATE TABLE comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  TEXT NOT NULL REFERENCES tickets(ticket_id),
    created_at TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    internal   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE issues (
    issue_id        TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    ticket_id       TEXT,
    status          TEXT NOT NULL DEFAULT 'open',
    idempotency_key TEXT UNIQUE
);
CREATE TABLE accounts (
    email         TEXT PRIMARY KEY,
    plan          TEXT NOT NULL,
    seats         INTEGER NOT NULL,
    invoices_json TEXT NOT NULL,
    charges_json  TEXT NOT NULL
);
"""

# Bugs the tracker already knows about, so search_issues has something to find.
ISSUES = [
    ("ISS-1001", "2026-07-02", "QS-1042 returned for keys rotated less than 90 days ago",
     "Team-plan workspaces get QS-1042 (key rotation not due) when calling POST /v1/api-keys/rotate, "
     "even though the docs say the 90-day window is a maximum, not a minimum. Rotation should be allowed "
     "at any time. Reproduced on three workspaces. Workaround: create a new key and revoke the old one."),
    ("ISS-1002", "2026-07-15", "Rate limiter counts rejected 429 retries against the 600 rpm budget",
     "Requests that are already rejected with QS-1090 still increment the per-key counter, so a client "
     "that retries immediately never recovers within the minute. Counter should only advance on accepted "
     "requests. Seen with the GitHub integration polling loop."),
    ("ISS-1003", "2026-07-28", "Workspace export stuck at 'preparing' past the 24 hour window",
     "Exports of workspaces with more than 5,000 pages sit at status 'preparing' for days. The job "
     "times out at 6 hours and is silently re-queued. Customers see no error and no email. Add a "
     "visible failure state and raise the job timeout."),
    ("ISS-1004", "2026-08-05", "SSO login loops back to the sign-in page when the IdP returns a lowercase NameID",
     "Business plan, Okta and Entra. If the identity provider sends the NameID in lowercase but the "
     "Quillstack user was invited with mixed case, the SAML assertion is accepted and then the session "
     "is dropped, so the browser bounces back to sign-in. Match on a case-folded email."),
    ("ISS-1005", "2026-08-12", "Slack integration posts page updates twice after a reconnect",
     "Reconnecting the Slack integration registers a second webhook without removing the first. "
     "Every page update then posts two messages to the channel until the integration is removed and "
     "re-added. Deduplicate on the Slack team id."),
    ("ISS-1006", "2026-08-19", "GitHub integration shows 'unknown repo' on linked pull requests after a repository rename",
     "Pull request links on project cards are stored by repository name rather than id. After the repo is "
     "renamed on GitHub the card shows 'unknown repo' and the link 404s. Store the numeric repository id "
     "and refresh the name on webhook events."),
    ("ISS-1007", "2026-08-26", "Invoice PDF shows the seat count from before the last downgrade",
     "When a Team workspace reduces seats mid-cycle, the next invoice amount is correct but the PDF "
     "header still lists the old seat count. Customers read this as being overcharged and open billing "
     "tickets. The PDF template reads seats from the cached plan snapshot."),
    ("ISS-1008", "2026-09-03", "QS-1207 on page search when the query ends with a backslash",
     "Searching pages with a query that ends in a single backslash returns QS-1207 (bad query) instead "
     "of escaping it. Same for a lone double quote. The search box should escape user input before "
     "building the query."),
]


def data_dir() -> Path:
    return Path(os.environ.get("HELPDESK_DATA_DIR", HERE.parent / "data"))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sources() -> tuple[list[dict], list[dict], str]:
    """Return (tickets, accounts, source) from the data files or the fixtures."""
    fixtures = read_jsonl(HERE / "fixtures.jsonl")
    ticket_files = [p for p in (data_dir() / "tickets" / "dev.jsonl", data_dir() / "tickets" / "heldout.jsonl") if p.exists()]
    account_file = data_dir() / "accounts.jsonl"
    tickets = [t for p in ticket_files for t in read_jsonl(p)]
    accounts = read_jsonl(account_file) if account_file.exists() else []
    source = "data" if ticket_files else "fixtures"
    if not ticket_files:
        tickets = [r for r in fixtures if r.get("kind") == "ticket"]
    if not accounts:
        accounts = [r for r in fixtures if r.get("kind") == "account"]
    return tickets, accounts, source


def seed(db_path: Path | str) -> dict:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tickets, accounts, source = load_sources()
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO tickets (ticket_id, created_at, channel, customer_email, plan, subject, body) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [tuple(t[f] for f in TICKET_FIELDS) for t in tickets])
        conn.executemany(
            "INSERT INTO accounts (email, plan, seats, invoices_json, charges_json) VALUES (?, ?, ?, ?, ?)",
            [(a["email"].lower(), a["plan"], int(a.get("seats", 1)),
              json.dumps(a.get("invoices", [])), json.dumps(a.get("charges", []))) for a in accounts])
        conn.executemany(
            "INSERT INTO issues (issue_id, created_at, title, body, status) VALUES (?, ?, ?, ?, 'open')",
            [(i, f"{d}T09:00:00+00:00", t, b) for i, d, t, b in ISSUES])
    conn.close()
    return {"source": source, "tickets": len(tickets), "accounts": len(accounts), "issues": len(ISSUES)}


if __name__ == "__main__":
    target = Path(os.environ.get("HELPDESK_DB", HERE / "helpdesk.db"))
    print(f"seeding {target}: {seed(target)}")
