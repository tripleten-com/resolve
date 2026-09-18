"""Show the pending write of a paused run in plain words, ask yes or no, resume.

    python -m resolve.approve T-10004          # interactive
    python -m resolve.approve T-10004 --yes    # approve every pending write without asking
    python -m resolve.approve T-10004 --no     # decline every pending write

A run can pause more than once (the bug agent's issue, then the exit's comment and status), so
this keeps going until the run finishes or you say no to a write.
"""
from __future__ import annotations

import argparse
import sys

from resolve.graph.build import get_graph, is_paused, pending_write, resume_ticket
from resolve.run import summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticket_id")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--yes", action="store_true")
    g.add_argument("--no", action="store_true")
    args = ap.parse_args(argv)
    graph = get_graph()

    pending = pending_write(graph, args.ticket_id)
    if pending is None:
        print(f"{args.ticket_id}: nothing is waiting for approval.")
        return 0
    while pending is not None:
        print(f"\nPending write for {args.ticket_id} (from {pending.get('node')}):")
        print(f"  {pending.get('summary')}")
        print(f"  idempotency key: {pending.get('idempotency_key')}")
        if args.yes:
            answer = "y"
        elif args.no:
            answer = "n"
        else:
            answer = input("Approve? [y/N] ").strip().lower()
        approved = answer in {"y", "yes"}
        print("  -> approved, executing once" if approved else "  -> declined")
        state = resume_ticket(graph, args.ticket_id, approved)
        pending = state.get("pending_write") if is_paused(state) else None
    print()
    print(summary(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
