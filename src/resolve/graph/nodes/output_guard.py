"""output_guard: code. Secrets, citations, links, schema, and the write flag. Writes guard_output."""
from __future__ import annotations

from resolve.graph.usage import Meter
from resolve.guards.output import guard_draft


def output_guard(state: dict) -> dict:
    meter = Meter()
    draft, guard = guard_draft(state.get("draft") or "", state.get("citations") or [],
                               state.get("docs") or [], state.get("tool_calls") or [])
    return {"draft": draft, "guard_output": guard, "usage": meter.merge(state)}
