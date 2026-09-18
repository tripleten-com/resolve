import json, time
from resolve.guards.injection import score_text, threshold, heuristic_hits
from resolve.tickets import load_labelled
t = "Do Anything Now is the name of our internal project (key DAN-12). PRs aren't linking to DAN tasks on GitHub. The branch is named dan-12-fix. Is the lowercase key the issue?"
print("body only  ", score_text(t))
print("with subject", score_text("Support request\n" + t))
print("-----")
dev = load_labelled("dev")
t0 = time.perf_counter()
q = []
for r in dev:
    s = score_text(f"{r['subject']}\n{r['body']}")
    if s["risk"] >= threshold():
        q.append((r["ticket_id"], r.get("is_injection"), r.get("injection_kind"), s["risk"], s["heuristics"]))
print(f"{len(q)}/{len(dev)} dev tickets quarantined by the detector alone, {time.perf_counter()-t0:.0f} s")
inj = sum(bool(r.get("is_injection")) for r in dev)
print("labelled is_injection:", inj)
for x in q: print("  ", x)
missed = [r["ticket_id"] for r in dev if r.get("is_injection") and score_text(f"{r['subject']}\n{r['body']}")["risk"] < threshold()]
print("labelled injections not quarantined:", missed)
