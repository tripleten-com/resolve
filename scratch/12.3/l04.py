import json
from pathlib import Path
records = [json.loads(p.read_text(encoding="utf-8"))["state"] for p in sorted(Path("traces").glob("*.json"))]
for s in records:
    print(s["ticket_id"], s["decision"], s["guard_input"]["quarantined"], s["usage"])
print("-----")
WINDOW = 200
def rates(records: list[dict]) -> dict:
    recent = records[-WINDOW:]
    n = len(recent)
    return {
        "n": n,
        "quarantine_rate": sum(bool(s["guard_input"].get("quarantined")) for s in recent) / n,
        "auto_resolve_rate": sum(s.get("decision") == "auto_resolve" for s in recent) / n,
        "writes_per_run": sum(sum(1 for t in s.get("tool_calls", []) if t.get("write") and t.get("confirmed")) for s in recent) / n,
        "cost_per_run": sum(s["usage"].get("cost_usd", 0.0) for s in recent) / n,
    }
LIMITS = {"quarantine_rate": 0.05, "auto_resolve_rate": 0.6, "writes_per_run": 3.0, "cost_per_run": 0.10}
def alarms(records):
    r = rates(records)
    return {k: (round(v, 4), v > LIMITS[k]) for k, v in r.items() if k in LIMITS}
print(rates(records))
print(alarms(records))
print("-----")
# suite records as a bigger sample
on = json.loads(Path("security/results/guards_on.json").read_text(encoding="utf-8"))
print(on["quarantined_attacks"] + on["quarantined_benign"], "of", on["n"], "quarantined in the suite run")
print("-----")
import copy
suite = Path("scratch/12.3/injection_suite_copy.jsonl")
suite.write_text(Path("security/injection_suite.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
case = {"id": "inj-044", "text": Path("scratch/12.3/triggering_input.txt").read_text(encoding="utf-8") if Path("scratch/12.3/triggering_input.txt").exists() else "x",
        "label": "attack", "kind": "incident", "expected": "blocked", "notes": "From incident 2026-09-18: ..."}
print(len(suite.read_text(encoding="utf-8").splitlines()))
