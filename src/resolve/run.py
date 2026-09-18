"""Run one ticket through the graph and print the decision.

    python -m resolve.run T-10004                   # pauses before the first write
    python -m resolve.run T-10004 --approval auto   # executes writes without asking (demo only)
    python -m resolve.run T-10004 --approval dry_run --json
"""
from __future__ import annotations

import argparse
import json
import sys

from resolve.graph.build import get_graph, is_paused, run_ticket


def summary(state: dict) -> str:
    c = state.get("classification") or {}
    r = state.get("review") or {}
    u = state.get("usage") or {}
    lines = [
        f"ticket        {state.get('ticket_id')}  ({state.get('ticket', {}).get('subject', '')[:60]})",
        f"input_guard   risk={state.get('guard_input', {}).get('risk')} flags={state.get('guard_input', {}).get('flags')}",
        f"classify      type={c.get('type')} urgency={c.get('urgency')} aimed_at_agent={c.get('aimed_at_agent')} "
        f"confidence={c.get('confidence')}",
        f"specialist    confidence={state.get('confidence')} citations={state.get('citations')} "
        f"docs={len(state.get('docs') or [])} tool_calls={[t['name'] for t in state.get('tool_calls') or []]}",
        f"review        {r.get('verdict') or '-'} revision_count={r.get('revision_count', 0)} reasons={r.get('reasons', [])}",
        f"output_guard  flags={state.get('guard_output', {}).get('flags')}",
        f"usage         in={u.get('input_tokens')} out={u.get('output_tokens')} cost=${u.get('cost_usd', 0):.4f} "
        f"latency={u.get('latency_ms')}ms",
    ]
    if is_paused(state):
        lines.append(f"PAUSED        {state['pending_write'].get('summary')}")
        lines.append(f"              run: python -m resolve.approve {state.get('ticket_id')}")
    else:
        lines.append(f"DECISION      {state.get('decision')}  {state.get('escalation_reason') or ''}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticket_id")
    ap.add_argument("--approval", default="human", choices=["human", "auto", "dry_run"])
    ap.add_argument("--json", action="store_true", help="print the full final state as JSON")
    args = ap.parse_args(argv)
    graph = get_graph()
    state = run_ticket(graph, args.ticket_id, approval=args.approval)
    if args.json:
        print(json.dumps(state, indent=2, default=str))
    else:
        print(summary(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
