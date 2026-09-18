Resolve writes a draft reply for every ticket that reaches a specialist. Priya wants to know how many of those drafts are right. Eight support engineers can read a few dozen drafts a week and say so, and Resolve produces hundreds a day. An **LLM-as-judge** is the way out: a model, prompted with a rubric you wrote, reads another model's output and scores it. The judge is a measuring instrument. This lesson is about building one that measures what you think it measures.

<callout icon="📖" color="gray_bg">
	Every code block in this chapter runs from the Resolve repository with `PYTHONPATH=src`, the virtual environment built from `requirements.txt`, OpenSearch and the helpdesk up in Compose, and a `.env` file holding `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and `LLM_MODEL_REVIEW`. The model client is `resolve.llm` (`src/resolve/llm.py`), the one every part of Resolve uses: it reads the endpoint and both model names from `.env`, and any OpenAI-compatible provider works. This lesson also reads a YAML file, so `pyyaml==6.0.3` from `requirements.txt` is in play.
</callout>

## Three judging patterns

A judge can be asked three different questions, and each needs different inputs.

<table header-row="true">
<tr>
<td>Pattern</td>
<td>What the judge sees</td>
<td>Use it for</td>
</tr>
<tr>
<td>**Pointwise**</td>
<td>One draft plus a rubric</td>
<td>An absolute score you can chart over time, 1 to 5 per rubric line</td>
</tr>
<tr>
<td>**Pairwise**</td>
<td>Two drafts for the same ticket and the question "which is better?"</td>
<td>Deciding whether a new prompt or model beats the one you have</td>
</tr>
<tr>
<td>**Reference-based**</td>
<td>One draft plus something known to be true</td>
<td>Grading where ground truth exists: the gold articles and the `key_facts` in the ticket labels</td>
</tr>
</table>

Pairwise verdicts are the most reliable of the three, because a model ranks two candidates better than it pins a number on one. Pointwise is what you chart, because a win rate against last week's prompt says nothing about whether either version is good enough to ship. Resolve's eval suite is pointwise with a reference: the judge gets the rubric and the `key_facts` from the labelled tickets, so it scores against something known rather than against its own taste.

*Diagram idea: three panels side by side. Pointwise (one draft, an arrow to a score), pairwise (draft A against draft B, an arrow to a winner), reference-based (draft against key facts, an arrow to a match).*

## Drafts to judge

A judge needs something to read. The block below runs three dev tickets through the finished graph with `approval="dry_run"`, the mode from **LangChain & LangGraph** in which a write tool is recorded in the state as if approved and never executed, so the helpdesk does not change. It keeps the fields a judge needs and saves them to a file the rest of this chapter reads.

```python
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
```

```plain text
T-10001 auto_resolve $0.0170 1056 chars
T-10038 escalate $0.0530 689 chars
T-10004 escalate $0.0604 812 chars
```

Three different tickets on purpose. T-10001 is a how-to ("how do i archive a project?") that the graph auto-resolved. T-10038 is a billing question about a prorated charge; the reviewer sent the draft back twice for an unsupported claim and the graph escalated it. T-10004 is a bug ("2FA code not working") that always escalates. Keep the drafts in mind: the judge should tell these three apart.

## Four rules for a judge you can trust

A judge that returns plausible numbers is easy. A judge that tracks quality follows four rules.

1. Spell out the rubric. Name every criterion. The judge does not share your idea of a good support reply unless you write it down.
2. Anchor the scale. Say what a 1, a 3, and a 5 look like, in terms a grader can point at in the text. A bare "rate this 1 to 5" gets you the average of everyone else's 1 to 5.
3. Ask for reasoning before the score. A score emitted first turns the explanation into a defence of a number the model already picked. This is the structured-reasoning move from **Prompt Engineering**.
4. Return parseable output. JSON with fixed keys, so a few hundred verdicts aggregate without a regex.

Resolve keeps its rubric in `configs/judge.yaml`, where a reviewer can read it and a student can edit it without touching code. Four lines, each with a question and three anchors.

```python
import yaml

cfg = yaml.safe_load(open("configs/judge.yaml", encoding="utf-8"))
lines = []
for name, spec in cfg["rubric"].items():
    lines.append(f"{name}: {spec['question']}")
    for score, anchor in sorted(spec["anchors"].items()):
        lines.append(f"  {score} = {anchor}")
RUBRIC = "\n".join(lines)
print(RUBRIC[:600])
```

```plain text
correctness: Are the facts and steps in the draft right for this ticket, judged against the key facts and the retrieved chunks?
  1 = A central fact or step is wrong, or the draft answers a different question.
  3 = The main answer is right but a detail is wrong, missing a condition, or vague where the docs are exact.
  5 = Every fact and step matches the documents and the key facts, including exact values, error codes, and menu paths.
groundedness: Is every factual claim in the draft supported by a retrieved chunk, and does the draft cite the articles it relied on?
  1 = The draft states fact
```

The four lines are `correctness`, `groundedness`, `completeness`, and `action_safety`. Read the whole file once. Every anchor names something visible: an exact value, a missing citation, a promise of a refund. The groundedness anchor for a 5 ends with "Length and confidence earn nothing here", which is a bias guard written into the rubric itself, and the next section explains why it needs to be there.

## The JSON comes back in a fence

Rule four says parseable output, and there is a snag. Asked for "JSON only", the judge model usually wraps its answer in a Markdown code fence, so `json.loads` on the raw reply fails. A small helper strips the fence when it is there and leaves the text alone when it is not.

```python
import json

def strip_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]        # drop the opening fence line
        text = text.rsplit("```", 1)[0]      # drop the closing fence
    return text.strip()

print(json.loads(strip_fences('```json\n{"score": 4}\n```')))
```

```plain text
{'score': 4}
```

`evals/judge_eval.py` avoids the problem another way: it forces a tool call with a Pydantic schema, so the provider returns the arguments as JSON with no prose around them. That is the structured-output move from **Agent Fundamentals**, and the next lesson uses it. Plain text plus `strip_fences` is enough here, and it works with providers that do not support forced tool calls.

## A pointwise judge

Everything so far assembles into one function. The judge sees the ticket, the retrieved chunks, the citations, and the draft, in that order, and answers with reasoning first and the four scores after. The model is `MODEL_REVIEW`: the graph's specialists draft with `LLM_MODEL`, and the reviewer and the judge use `LLM_MODEL_REVIEW`, the stronger of the two.

```python
from resolve.llm import ask, MODEL_REVIEW

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
for tid, state in states.items():
    v = judge(state)
    print(tid, [v[k] for k in ("correctness", "groundedness", "completeness", "action_safety")])
```

```plain text
T-10001 [5, 5, 5, 5]
T-10038 [3, 3, 5, 3]
T-10004 [5, 5, 5, 5]
```

`ask` is the simplest call in `resolve.llm`: one prompt in, the reply text out, at temperature 0 by default. Temperature 0 matters more for a judge than for a drafter. An instrument that returns a 4 and then a 3 for the same draft has no reading to give you.

Now read the verdicts against the drafts. T-10001 scores 5 across the board, and the reasoning says why: every menu path and permission claim traces to a chunk of kb-0015. T-10038 gets a 3 on correctness, groundedness, and action safety. The judge's reasoning quotes the sentence that did it: the draft says "there were 12 days remaining in the cycle" and "your charge has been confirmed in our system", and no chunk says either. That is the same sentence the in-graph reviewer flagged when it sent the draft back, so the offline judge and the online reviewer agree about this ticket. T-10004 scores 5 on groundedness even though the reasoning itself notes that "sign out and back in" is not in the chunks. The rubric's 3 anchor says "one minor claim is unsupported", so a strict reading gives a 3 or a 4. The judge was lenient. Keep that example; calibration, below, is how you find out whether it is lenient in general.

## Pairwise, run in both orders

**Position bias** is the judge's tendency to prefer whichever draft sits in the first slot. Put the incumbent first every time and you measure the slot. The fix is mechanical: run every comparison twice with the candidates swapped, and count only a verdict that survives the swap.

```python
PAIRWISE = """Two assistants drafted a reply to the same Quillstack support ticket. Which draft is better?
Better means correct and supported by the retrieved chunks. Length is not quality.

Ticket:
{ticket}

Retrieved chunks:
{chunks}

Draft A:
{a}

Draft B:
{b}

Reply with JSON only, keys in this order: {{"reasoning": "<two sentences>", "winner": "A" or "B"}}"""

def winner(ticket: str, chunks: str, a: str, b: str) -> str:
    raw = ask(PAIRWISE.format(ticket=ticket, chunks=chunks, a=a, b=b), model=MODEL_REVIEW, max_tokens=300)
    return json.loads(strip_fences(raw))["winner"]

def compare(ticket: str, chunks: str, current: str, candidate: str) -> str:
    first = winner(ticket, chunks, current, candidate)     # candidate in slot B
    second = winner(ticket, chunks, candidate, current)    # candidate in slot A
    if first == "B" and second == "A":
        return "candidate"
    if first == "A" and second == "B":
        return "current"
    return "tie"
```

Try it on T-10038. The current draft is the one with the invented "12 days remaining". The candidate is a three-sentence reply that says what the docs say and points the customer at the invoice for the exact calculation.

```python
s = states["T-10038"]
ticket = f"Subject: {s['ticket']['subject']}\n{s['ticket']['body']}"
chunks = format_chunks(s["docs"])
short = ("Hi Zara,\n\nThat charge is the prorated cost of the seat you added mid-cycle: on the Team plan a seat is "
         "$12 a month and you pay only for the days left in the cycle. Settings > Billing > Invoices shows the "
         "exact calculation.\n\nQuillstack Support")
print(compare(ticket, chunks, current=s["draft"], candidate=short))
```

```plain text
tie
```

Both single calls returned `A`. With the agent's draft in slot A the judge preferred the agent's draft; with the short reply in slot A it preferred the short reply. One run in a fixed order would have reported a winner, and which winner would have depended on how you happened to order the arguments. `compare` throws that away, and "tie" is the honest reading. On a 40-ticket comparison you report three numbers, candidate wins, current wins, and ties, and a large tie bucket tells you the judge cannot tell your two versions apart.

## Two biases the swap does not fix

**Verbosity bias** is the judge's tendency to score a longer, more confident, better-formatted reply above a shorter correct one. The rubric line "length and confidence earn nothing" helps, and so does a length cap on the specialist prompts. Test it instead of trusting it: hand the judge a padded, confident, wrong reply and see what comes back. The T-10038 tie above is verbosity at work too. The agent's draft is longer and sounds more certain, and the judge could not decide it was worse even though it contains a number no document supports.

**Self-preference bias** is the judge's tendency to prefer text from its own model family. The fix is to judge with a different model or family than the one that drafts. In the course the specialists use `LLM_MODEL` and the judge uses `LLM_MODEL_REVIEW`; `configs/judge.yaml` names `LLM_MODEL_REVIEW` as the judge model, so pointing that variable at another family is a one-line change in `.env`, and the judge code does not change.

## Calibration with Cohen's kappa

None of the guards above tells you whether the judge agrees with a person. **Calibration** does: you score a sample of drafts yourself against the same rubric, blind to the judge, then run the judge on the same sample and measure the agreement. Task 5 asks for about twenty such rows in `evals/calibration.jsonl`, one JSON object per line with your scores and the judge's.

```plain text
{"ticket_id": "T-10201", "human": {"correctness": 4, "groundedness": 4, "completeness": 5, "action_safety": 5}, "judge": {"correctness": 4, "groundedness": 4, "completeness": 5, "action_safety": 5}}
```

Raw agreement, the share of rows where the two scores match, overstates how good the judge is. If almost every draft is a 5 and the judge says 5 to everything, the two of you agree on almost every row and the judge has learned nothing. **Cohen's kappa** corrects for that. It compares the agreement you observed with the agreement two raters would reach by chance, given how often each of them uses each score:

kappa = (observed agreement - chance agreement) / (1 - chance agreement)

A kappa of 1 means perfect agreement. A kappa of 0 means the judge agrees with you exactly as often as a random rater with the same habits would. Values below 0 are possible (worse than chance) and rare. The usual reading: below 0.4 the judge is not tracking you; 0.4 to 0.6 is moderate; above 0.6 is substantial agreement, and 0.6 is the bar most teams set before they let a judge run unsupervised.

```python
from collections import Counter
from pathlib import Path

def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Agreement between two raters beyond what chance alone would give."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n                 # share of rows where both agree
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] / n * cb[k] / n for k in set(a) | set(b))  # agreement if both scored at random
    return (observed - expected) / (1 - expected)

rows = [json.loads(line) for line in Path("evals/calibration.jsonl").read_text().splitlines() if line.strip()]
for line in ("correctness", "groundedness", "completeness", "action_safety"):
    human = [r["human"][line] for r in rows]
    judge_scores = [r["judge"][line] for r in rows]
    exact = sum(h == j for h, j in zip(human, judge_scores))
    print(f"{line:14} kappa={cohen_kappa(human, judge_scores):.2f}  exact {exact}/{len(rows)}")
```

```plain text
correctness    kappa=0.70  exact 17/20
groundedness   kappa=0.75  exact 17/20
completeness   kappa=0.11  exact 16/20
action_safety  kappa=0.57  exact 17/20
```

The twenty rows behind that output are a sample written to show the file's shape; your own calibration file replaces it in Task 5. The reading is what matters. Correctness and groundedness clear 0.6. Completeness has almost the same exact-match count as the others and a kappa near zero, because nearly every row is a 5 for both raters, so chance agreement is already high and the few disagreements are all the signal there is. Action safety sits at 0.57 with the same 17 of 20 matches, which is a coin flip away from the bar. This is why you read kappa per rubric line and never as one averaged number.

Two things follow. First, kappa is a property of the pair (this judge, this rubric, your labels), so rerun it after every rubric edit or model change. Second, when a line comes in low, read the disagreements rather than the number. Where you and the judge differ by two points or more, one of you is reading an anchor differently, and the fix is a sharper anchor in `configs/judge.yaml`. For an ordered 1 to 5 scale there is also a weighted variant of kappa that charges a 5-against-4 miss less than a 5-against-1 (`sklearn.metrics.cohen_kappa_score` with `weights="quadratic"`); the plain version above is the one to understand first.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Drafting rubric anchors is good delegation. Ask an assistant for three anchors per line, then test what comes back before you trust a single score: feed the judge a padded, confident, wrong draft and check it lands at a 1 or a 2; feed it a two-sentence correct draft and check brevity is not punished; check every anchor names something you can point at in the text. Then run the kappa block against your own labels. The assistant can write the rubric; whether it tracks quality is yours to establish.
</callout>

The judge's reasoning for T-10038 quoted the sentence no document supports, and that is a list of words rather than a number. The next lesson, Hallucination Detection Metrics, turns "which claims are unsupported" into two numbers with a threshold: the share of claims supported and the hallucination rate.
