"""TicketState: the one object every node reads and writes (SPEC section 3, nothing more)."""
from __future__ import annotations

from typing import TypedDict


class TicketState(TypedDict, total=False):
    ticket_id: str
    ticket: dict            # subject, body, customer_email, plan, created_at, channel
    guard_input: dict       # flags: list[str], risk: float, quarantined: bool
    classification: dict    # type, urgency, aimed_at_agent, confidence
    docs: list[dict]        # doc_id, chunk_id, title, space, text, score
    draft: str
    citations: list[str]    # doc_ids that appear in docs
    confidence: float
    tool_calls: list[dict]  # name, args, result_summary, confirmed: bool, idempotency_key
    pending_write: dict | None
    review: dict            # verdict, reasons: list[str], revision_count: int
    guard_output: dict      # flags: list[str], ok: bool
    decision: str           # auto_resolve | escalate
    escalation_reason: str
    usage: dict             # input_tokens, output_tokens, cost_usd, latency_ms


def initial_state(ticket_id: str, ticket: dict) -> TicketState:
    return TicketState(
        ticket_id=ticket_id, ticket=ticket,
        guard_input={"flags": [], "risk": 0.0, "quarantined": False},
        classification={}, docs=[], draft="", citations=[], confidence=0.0,
        tool_calls=[], pending_write=None,
        review={"verdict": "", "reasons": [], "revision_count": 0},
        guard_output={"flags": [], "ok": True},
        decision="", escalation_reason="",
        usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0},
    )
