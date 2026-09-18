import json, statistics
from resolve.graph.build import get_graph
from resolve.config import judge_config

print("=== B1 usage in state")
graph = get_graph()
cfg = {"configurable": {"thread_id": "T-10083"}}
snap = graph.get_state(cfg)
u = snap.values["usage"]
print(u)
print(f"${u['cost_usd']:.4f} for {u['input_tokens']:,} input and {u['output_tokens']:,} output tokens in {u['latency_ms'] / 1000:.1f} s")

print("=== B2 per node")
history = list(reversed(list(graph.get_state_history(cfg))))   # oldest first
prev = {"cost_usd": 0.0, "latency_ms": 0}
for before, after in zip(history, history[1:]):
    node = before.next[0]                      # the node that ran between these two checkpoints
    u = after.values["usage"]
    if node != "__start__":
        print(f"{node:16} ${u['cost_usd'] - prev['cost_usd']:.4f}  {u['latency_ms'] - prev['latency_ms']:>6} ms")
    prev = u

print("=== B3 percentiles")
records = json.load(open("evals/results/judge_dev.json", encoding="utf-8"))["records"]
costs = [r["cost_usd"] for r in records]
lats = [r["latency_ms"] for r in records]

def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]

print(f"n={len(records)}  cost mean ${statistics.mean(costs):.4f}  p50 ${pct(costs, 0.5):.4f}  p95 ${pct(costs, 0.95):.4f}")
print(f"latency p50 {pct(lats, 0.5) / 1000:.1f} s  p95 {pct(lats, 0.95) / 1000:.1f} s")
print(f"400 tickets a day at the mean: ${400 * statistics.mean(costs):.2f}")

print("=== B4 floors")
floors = judge_config()["floors"]
summary = json.load(open("evals/results/judge_dev.json", encoding="utf-8"))["summary"]
suite = json.load(open("security/results/guards_on.json", encoding="utf-8"))
measured = {
    "routing_accuracy": summary["routing_accuracy"],
    "groundedness_rate": summary["groundedness_rate"],
    "detection_rate": suite["detection_rate"],
    "cost_per_ticket_usd_max": summary["cost_usd"]["mean"],
}
for name, floor in floors.items():
    value = measured[name]
    ok = value <= floor if name.endswith("_max") else value >= floor
    print(f"{name:24} {value:<8} floor {floor:<6} {'ok' if ok else 'MISSED'}")

print("=== B5 budget")
def over_budget(usage: dict, max_usd: float = 0.10, max_ms: int = 60_000) -> str | None:
    if usage["cost_usd"] > max_usd:
        return f"cost ${usage['cost_usd']:.4f} is over the ${max_usd:.2f} budget"
    if usage["latency_ms"] > max_ms:
        return f"latency {usage['latency_ms']} ms is over the {max_ms} ms budget"
    return None

print(over_budget(snap.values["usage"]))
print(over_budget(snap.values["usage"], max_usd=0.03))
