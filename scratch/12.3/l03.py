import json
from pathlib import Path
on = json.loads(Path("security/results/guards_on.json").read_text(encoding="utf-8"))
off = json.loads(Path("security/results/guards_off.json").read_text(encoding="utf-8"))
keys = ["n", "attacks", "benign", "detection_rate", "false_positive_rate", "quarantined_attacks",
        "quarantined_benign", "attacks_reaching_a_specialist", "attacks_flagged_by_classify", "total_cost_usd"]
print(f"{'':32}{'guards off':>12}{'guards on':>12}")
for k in keys:
    print(f"{k:32}{str(off[k]):>12}{str(on[k]):>12}")
print("-----")
by_id = {r["id"]: r for r in off["results"]}
for r in on["results"]:
    if not r["pass"] or not by_id[r["id"]]["pass"]:
        print(f"{r['id']}  {r['label']:6} {r['kind']:20} off: {by_id[r['id']]['why']:45} on: {r['why']}")
print("-----")
r = next(x for x in off["results"] if x["id"] == "inj-041")
print(json.dumps({k: r["observed"][k] for k in ("quarantined", "type", "decision", "write_attempted", "outside_link_hosts", "planted_chunk_in_docs", "chunks_dropped", "output_flags", "cost_usd")}, indent=2))
r = next(x for x in on["results"] if x["id"] == "inj-041")
print(json.dumps({k: r["observed"][k] for k in ("quarantined", "type", "decision", "write_attempted", "outside_link_hosts", "planted_chunk_in_docs", "chunks_dropped", "output_flags", "cost_usd")}, indent=2))
print("-----")
for f, d in (("off", off), ("on", on)):
    r = next(x for x in d["results"] if x["id"] == "inj-043")
    print(f, r["pass"], r["why"], {k: r["observed"][k] for k in ("write_attempted", "outside_link_hosts", "canary_leaked", "chunks_dropped", "output_flags")})
print("-----")
for f, d in (("off", off), ("on", on)):
    r = next(x for x in d["results"] if x["id"] == "inj-001")
    print(f, {k: r["observed"][k] for k in ("quarantined", "risk", "aimed_at_agent", "type", "decision", "reached_specialist", "model_calls_tokens", "cost_usd")})
print("-----")
import statistics
for f, d in (("off", off), ("on", on)):
    costs = [r["observed"]["cost_usd"] for r in d["results"] if r["label"] == "attack" and r["kind"] not in ("indirect_doc", "indirect_issue")]
    print(f, "direct attack cost total", round(sum(costs), 4), "max", max(costs))
