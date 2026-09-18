"""resolve and escalate: the two exits. Each writes a comment and a status through the helpdesk
tools, behind the same approval pause as every other write."""
from __future__ import annotations

from resolve.graph.agents import execute_write
from resolve.graph.usage import Meter

NOTE_MAX = 3500


def _writeback(state: dict, node: str, body: str, internal: bool, status: str) -> dict:
    meter = Meter()
    ticket_id = state["ticket_id"]
    tool_calls = list(state.get("tool_calls") or [])
    _, entry = execute_write("add_ticket_comment", {"ticket_id": ticket_id, "body": body, "internal": internal}, state, node)
    tool_calls.append(entry)
    _, entry = execute_write("set_ticket_status", {"ticket_id": ticket_id, "status": status}, state, node)
    tool_calls.append(entry)
    return {"tool_calls": tool_calls, "pending_write": None, "usage": meter.merge(state)}


def resolve(state: dict) -> dict:
    out = _writeback(state, "resolve", state.get("draft") or "", internal=False, status="resolved")
    out["decision"] = "auto_resolve"
    return out


def escalation_note(state: dict) -> str:
    c = state.get("classification") or {}
    review = state.get("review") or {}
    gi, go = state.get("guard_input") or {}, state.get("guard_output") or {}
    lines = [
        "Resolve escalated this ticket.",
        f"Reason: {state.get('escalation_reason') or 'not recorded'}",
        f"Classification: type={c.get('type', '-')} urgency={c.get('urgency', '-')} "
        f"aimed_at_agent={c.get('aimed_at_agent', '-')} confidence={c.get('confidence', '-')}",
        f"Input guard: risk={gi.get('risk', 0)} flags={gi.get('flags', [])}",
        f"Output guard: flags={go.get('flags', [])}",
    ]
    if review.get("verdict"):
        lines.append(f"Reviewer: {review['verdict']}; reasons={review.get('reasons', [])}; "
                     f"unsupported={review.get('unsupported_claims', [])}")
    calls = [f"  - {t['name']}({', '.join(f'{k}={str(v)[:40]!r}' for k, v in (t.get('args') or {}).items())}) -> "
             f"{t.get('result_summary', '')[:120]}" for t in state.get("tool_calls") or []]
    if calls:
        lines.append("Tool calls:\n" + "\n".join(calls))
    docs = state.get("docs") or []
    if docs:
        seen = dict.fromkeys(f"{d.get('doc_id')} {d.get('title')} ({d.get('space')})" for d in docs)
        lines.append("Sources tried:\n" + "\n".join(f"  - {s}" for s in seen))
    if state.get("draft"):
        lines.append(f"Draft reply (confidence {state.get('confidence', 0):.2f}, citations {state.get('citations', [])}):\n"
                     f"{state['draft']}")
    note = "\n".join(lines)
    return note if len(note) <= NOTE_MAX else note[:NOTE_MAX - 15] + "\n[truncated]"


def escalate(state: dict) -> dict:
    out = _writeback(state, "escalate", escalation_note(state), internal=True, status="escalated")
    out["decision"] = "escalate"
    if not state.get("escalation_reason"):
        out["escalation_reason"] = "escalated without a recorded reason"
    return out
