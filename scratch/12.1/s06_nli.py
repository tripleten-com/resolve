import json, time
from transformers import pipeline

t0 = time.perf_counter()
nli = pipeline("text-classification", model="microsoft/deberta-large-mnli")   # labels: CONTRADICTION, NEUTRAL, ENTAILMENT
print(f"loaded in {time.perf_counter() - t0:.0f} s")

def supported_by_nli(claim: str, docs: list[dict]) -> bool:
    """A claim counts as supported when at least one retrieved chunk entails it."""
    pairs = [{"text": d["text"][:1500], "text_pair": claim} for d in docs]   # premise = chunk, hypothesis = claim
    return any(r["label"] == "ENTAILMENT" for r in nli(pairs, truncation=True))

states = json.load(open("evals/results/lesson_drafts.json", encoding="utf-8"))
claims = json.load(open("scratch/12.1/claims.json", encoding="utf-8"))
for tid in ("T-10038", "T-10004"):
    t1 = time.perf_counter()
    verdicts = [(c["claim"], supported_by_nli(c["claim"], states[tid]["docs"])) for c in claims[tid]["claims"]]
    n_ok = sum(ok for _, ok in verdicts)
    print(f"{tid}: {n_ok}/{len(verdicts)} entailed in {time.perf_counter() - t1:.1f} s")
    for claim, ok in verdicts:
        print(f"   {'ok ' if ok else 'NO '} {claim[:90]}".encode("ascii", "replace").decode())
