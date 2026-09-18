"""Find a ticket by id: the labelled JSONL files first, then the helpdesk over HTTP.

    from resolve.tickets import load_ticket, load_labelled
    t = load_ticket("T-10004")        # dict with subject, body, customer_email, plan, created_at, channel
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from resolve.config import ROOT

TICKET_FIELDS = ("subject", "body", "customer_email", "plan", "created_at", "channel")
TICKET_FILES = [ROOT / "data" / "tickets" / "dev.jsonl", ROOT / "data" / "tickets" / "heldout.jsonl"]


def helpdesk_url() -> str:
    return os.environ.get("HELPDESK_URL", "http://localhost:8100").rstrip("/")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labelled(name: str = "dev") -> list[dict]:
    """Every labelled ticket record from data/tickets/{name}.jsonl."""
    return read_jsonl(ROOT / "data" / "tickets" / f"{name}.jsonl")


def ticket_fields(record: dict) -> dict:
    return {k: record.get(k) for k in TICKET_FIELDS}


def load_ticket(ticket_id: str, prefer_helpdesk: bool = False) -> dict:
    """The ticket fields the graph needs. Raises KeyError if nowhere has it."""
    sources = [_from_helpdesk, _from_files] if prefer_helpdesk else [_from_files, _from_helpdesk]
    for source in sources:
        t = source(ticket_id)
        if t:
            return t
    raise KeyError(f"ticket {ticket_id} not found in data/tickets or at {helpdesk_url()}")


def _from_files(ticket_id: str) -> dict | None:
    for path in TICKET_FILES:
        for rec in read_jsonl(path):
            if rec.get("ticket_id") == ticket_id:
                return ticket_fields(rec)
    return None


def _from_helpdesk(ticket_id: str) -> dict | None:
    try:
        r = httpx.get(f"{helpdesk_url()}/tickets/{ticket_id}", timeout=10)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    return ticket_fields(r.json())
