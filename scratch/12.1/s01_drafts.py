# Run three dev tickets through the graph with dry-run writes and keep what the judge needs.
import json
from resolve.graph.build import get_graph, run_ticket

KEEP = ("ticket_id", "ticket", "docs", "draft", "citations", "decision", "review", "usage")
graph = get_graph()
states = {}
for tid in ("T-10001", "T-10038", "T-10004"):
    s = run_ticket(graph, tid, approval="dry_run")
    states[tid] = {k: s.get(k) for k in KEEP}
    print(tid, s["decision"], f"${s['usage']['cost_usd']:.4f}", len(s["draft"]), "chars")

json.dump(states, open("evals/results/lesson_drafts.json", "w", encoding="utf-8"), indent=2)
