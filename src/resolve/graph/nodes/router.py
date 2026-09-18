"""The conditional edges. Each reads one or two fields and returns the next node's name."""
from __future__ import annotations

from resolve.config import graph_config

SPECIALISTS = {"how_to": "knowledge_agent", "billing": "billing_agent", "bug": "bug_agent"}


def specialist_for(ticket_type: str | None) -> str:
    return SPECIALISTS.get(ticket_type or "", "escalate")


def after_input_guard(state: dict) -> str:
    return "escalate" if (state.get("guard_input") or {}).get("quarantined") else "classify"


def after_classify(state: dict) -> str:
    c = state.get("classification") or {}
    if c.get("type") == "security" or c.get("aimed_at_agent"):
        return "escalate"
    return specialist_for(c.get("type"))


def after_review(state: dict) -> str:
    r = state.get("review") or {}
    limit = int(graph_config().get("revision_limit", 1))
    if r.get("verdict") == "revise" and int(r.get("revision_count", 0)) <= limit:
        return specialist_for((state.get("classification") or {}).get("type"))
    return "output_guard"


def after_route(state: dict) -> str:
    return "resolve" if state.get("decision") == "auto_resolve" else "escalate"
