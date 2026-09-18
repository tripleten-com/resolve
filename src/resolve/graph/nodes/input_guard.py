"""input_guard: code, no model. Length cap, secret scrub, encoded-text flags, injection risk.

Writes guard_input. If quarantined, the edge after this node goes straight to escalate.
"""
from __future__ import annotations

from resolve.graph.usage import Meter
from resolve.guards.input import guard_ticket


def input_guard(state: dict) -> dict:
    meter = Meter()
    ticket, guard = guard_ticket(state["ticket"])
    out = {"ticket": ticket, "guard_input": guard, "usage": meter.merge(state)}
    if guard["quarantined"]:
        out["escalation_reason"] = (f"input_guard quarantined the ticket: risk {guard['risk']:.2f} "
                                    f"(flags: {', '.join(guard['flags'])})")
    return out
