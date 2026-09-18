import json
from collections import Counter
from pathlib import Path

def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Agreement between two raters beyond what chance alone would give."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n            # share of rows where both agree
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] / n * cb[k] / n for k in set(a) | set(b))  # agreement if both guessed at random
    return (observed - expected) / (1 - expected)

rows = [json.loads(line) for line in Path("evals/calibration.jsonl").read_text().splitlines() if line.strip()]
for line in ("correctness", "groundedness", "completeness", "action_safety"):
    human = [r["human"][line] for r in rows]
    judge = [r["judge"][line] for r in rows]
    exact = sum(h == j for h, j in zip(human, judge))
    print(f"{line:14} kappa={cohen_kappa(human, judge):.2f}  exact {exact}/{len(rows)}")
