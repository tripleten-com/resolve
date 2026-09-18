"""route: code. auto_resolve only if review approved, confidence >= threshold, no output flags,
and the type is how_to or billing. Everything else escalates, with the reason written down."""
from __future__ import annotations

import re

from resolve.config import graph_config

AUTO_RESOLVE_TYPES = ("how_to", "billing")
# Refunds are a human decision at Quillstack (SPEC: refunds always escalate). A prompt asks the
# billing agent to hand off; this is the control that makes sure it happens.
REFUND_SUBJECT = re.compile(r"\brefund|\breversal\b|\breverse\b|\bchargeback|\bmoney back\b", re.I)
DUPLICATE_CHARGE = re.compile(r"\b(charged|billed) twice\b|\bduplicate charge|\bdouble[- ]charged?\b", re.I)


def refund_request(ticket: dict) -> bool:
    subject, body = ticket.get("subject") or "", ticket.get("body") or ""
    return bool(REFUND_SUBJECT.search(subject) or DUPLICATE_CHARGE.search(subject + " " + body))


def decide(state: dict) -> tuple[str, str]:
    threshold = float(graph_config().get("confidence_threshold", 0.7))
    c = state.get("classification") or {}
    review = state.get("review") or {}
    guard_out = state.get("guard_output") or {}
    reasons: list[str] = []
    if c.get("type") == "billing" and refund_request(state.get("ticket") or {}):
        reasons.append("refund or duplicate charge: the billing team decides")
    if (state.get("guard_input") or {}).get("quarantined"):
        reasons.append("input quarantined")
    if review.get("verdict") != "approve":
        reasons.append(f"review verdict is {review.get('verdict') or 'missing'}"
                       + (f" ({'; '.join(review.get('reasons', []))[:200]})" if review.get("reasons") else ""))
    if float(state.get("confidence") or 0.0) < threshold:
        reasons.append(f"confidence {float(state.get('confidence') or 0.0):.2f} below threshold {threshold}")
    flags = [f for f in guard_out.get("flags", []) if f != "guards_disabled"]
    if flags:
        reasons.append(f"output flags: {', '.join(flags)}")
    if c.get("type") not in AUTO_RESOLVE_TYPES:
        reasons.append(f"type {c.get('type')} never auto-resolves")
    if reasons:
        return "escalate", "; ".join(reasons)
    return "auto_resolve", ""


def route(state: dict) -> dict:
    decision, reason = decide(state)
    return {"decision": decision, "escalation_reason": reason or state.get("escalation_reason", "")}
