import json
import yaml
from resolve.llm import ask, MODEL_REVIEW

# --- block A: the rubric from configs/judge.yaml ---
cfg = yaml.safe_load(open("configs/judge.yaml", encoding="utf-8"))
lines = []
for name, spec in cfg["rubric"].items():
    lines.append(f"{name}: {spec['question']}")
    for score, anchor in sorted(spec["anchors"].items()):
        lines.append(f"  {score} = {anchor}")
RUBRIC = "\n".join(lines)
print(RUBRIC[:600])
print("....")

# --- block B: strip_fences ---
def strip_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]        # drop the opening fence line
        text = text.rsplit("```", 1)[0]      # drop the closing fence
    return text.strip()

print(json.loads(strip_fences('```json\n{"score": 4}\n```')))

# --- block C: the pointwise judge ---
SYSTEM = ("You are a strict judge of support replies drafted by an AI assistant. Score with the rubric below. "
          "Length, confidence, and politeness earn nothing; only correctness against the retrieved chunks does.\n\n" + RUBRIC)

def format_chunks(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['chunk_id']}] {d['title']}\n{d['text']}" for d in docs) or "(nothing retrieved)"

def judge(state: dict) -> dict:
    prompt = (f"Ticket:\nSubject: {state['ticket']['subject']}\n{state['ticket']['body']}\n\n"
              f"Retrieved chunks:\n{format_chunks(state['docs'])}\n\n"
              f"Citations: {state['citations']}\n\nDRAFT:\n{state['draft']}\n\n"
              'Reply with JSON only, keys in this order: {"reasoning": "<two to five sentences>", '
              '"correctness": <1-5>, "groundedness": <1-5>, "completeness": <1-5>, "action_safety": <1-5>}')
    raw = ask(prompt, system=SYSTEM, model=MODEL_REVIEW, max_tokens=700)
    return json.loads(strip_fences(raw))

states = json.load(open("evals/results/lesson_drafts.json", encoding="utf-8"))
verdicts = {}
for tid, state in states.items():
    v = judge(state)
    verdicts[tid] = v
    print(tid, [v[k] for k in ("correctness", "groundedness", "completeness", "action_safety")])
    print("   ", v["reasoning"])
json.dump(verdicts, open("scratch/12.1/verdicts.json", "w"), indent=2)
