The judge in the previous lesson gave T-10038 a 3 on groundedness and quoted the sentence that earned it. That is a useful verdict and a poor metric: a quoted sentence does not aggregate, and a 3 on a 1 to 5 scale does not say how much of the draft was invented. This lesson defines **hallucination** narrowly, the way a retrieval-augmented system needs it defined, and turns the judge's list of quoted sentences into two numbers you can put a floor under.

<callout icon="📖" color="gray_bg">
	This lesson reads `evals/results/lesson_drafts.json`, the three dry-run states saved in the previous lesson, and calls the judge model through `resolve.llm` (`model=MODEL_REVIEW`). The structured call uses `pydantic==2.13.5` and `langchain-core==1.6.3`; the local alternative at the end uses `transformers==5.17.0` and `torch==2.14.0`. All four are pinned in `requirements.txt`.
</callout>

## Ungrounded rather than wrong

For Resolve, a hallucination is a factual claim in the draft that no retrieved chunk supports. Notice what the definition leaves out. It says nothing about the wider world. If the retrieved article says a workspace export can take up to 24 hours and the draft says up to 24 hours, the claim is grounded, even if the product changed last week and nobody updated the article. Grounding is measured against the chunks in `docs`, because those chunks are the only thing the specialist was allowed to rely on.

That narrowness is the point. A stale article is a content problem. A retriever that never found the right article is a retrieval problem, measured with recall in **Retrieval and RAG**. A draft that says something the chunks do not say is a generation problem, and it is the one this lesson measures. Put the three in one bucket and the fix lands in the wrong team's backlog.

Two words for the rest of the lesson. A **claim** is one checkable statement pulled out of the draft: a number, a limit, a step, a menu path, an error code, a promise about what happens next. **Faithfulness**, which Resolve's code calls groundedness, is the share of a draft's claims that a retrieved chunk supports.

*Diagram idea: one draft sentence split into three claim chips, each with an arrow toward the retrieved chunks. Two arrows land on a chunk (supported), one points at empty space (unsupported). Caption: "share supported = 2/3; one unsupported claim makes the reply a hallucination."*

## The check, claim by claim

`evals/judge_eval.py` implements the check in one model call. The judge model reads the retrieved chunks and the draft, splits the draft into claims, and for each claim says whether a chunk supports it and which one. The reply comes back as a forced tool call against a Pydantic schema, the structured-output move from **Agent Fundamentals**: the provider returns the arguments as JSON, and Pydantic validates the shape.

```python
import json
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from resolve.llm import chat, MODEL_REVIEW

class Claim(BaseModel):
    claim: str
    supported: bool
    evidence: str | None = Field(default=None, description="chunk_id of the supporting chunk, or null")

class Claims(BaseModel):
    """Every factual claim in the draft and whether a retrieved chunk supports it."""
    claims: list[Claim]

SYSTEM = ("Split the draft into its individual factual claims (a number, a limit, a step, a menu path, an error code, a "
          "promise about what happens next). Greetings and sign-offs are not claims. For each claim say whether one of "
          "the retrieved chunks supports it, naming the chunk_id. A claim that only restates the customer's message "
          "is supported. Be literal: 'up to 24 hours' is not supported by a chunk that says '48 hours'.")
```

Read the system prompt slowly, because the decomposition decides whether the number underneath means anything. "Greetings and sign-offs are not claims" keeps "Hi Zara" out of the denominator. "A claim that only restates the customer's message is supported" stops the check from penalising "you added a seat on September 15th" when the customer said exactly that. "Be literal" is the instruction that makes an unsupported number count as unsupported even when it is close. The `evidence` field makes the judge show its work: a chunk id you can open.

```python
def format_chunks(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['chunk_id']}] {d['title']}\n{d['text']}" for d in docs) or "(nothing retrieved)"

def groundedness(state: dict) -> dict:
    schema = convert_to_openai_tool(Claims)                       # the Pydantic class as an OpenAI tool schema
    resp = chat([{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"Retrieved chunks:\n{format_chunks(state['docs'])}\n\n"
                                             f"DRAFT:\n{state['draft']}\n\nCall Claims."}],
                model=MODEL_REVIEW, tools=[schema], max_tokens=900,
                tool_choice={"type": "function", "function": {"name": "Claims"}})   # the model must call Claims
    c = Claims.model_validate_json(resp.choices[0].message.tool_calls[0].function.arguments)
    n, supported = len(c.claims), sum(x.supported for x in c.claims)
    return {"claims": [x.model_dump() for x in c.claims], "n_claims": n, "supported": supported,
            "share_supported": round(supported / n, 4) if n else None,
            "hallucination": n > 0 and supported < n}
```

`convert_to_openai_tool` turns the `Claims` class into the function schema the chat API expects, and `tool_choice` forces the model to call that one function, so there is no prose and no code fence to strip. The last three keys are the metric: `share_supported` is the faithfulness of this draft, and `hallucination` is true when any claim failed. `judge_eval.py` has exactly this function, with a meter attached to record the cost.

## Three drafts, real numbers

```python
states = json.load(open("evals/results/lesson_drafts.json", encoding="utf-8"))
results = {tid: groundedness(s) for tid, s in states.items()}
for tid, g in results.items():
    print(tid, f"{g['supported']}/{g['n_claims']} supported, share {g['share_supported']}, hallucination {g['hallucination']}")
    for c in g["claims"]:
        if not c["supported"]:
            print("    unsupported:", c["claim"])
shares = [g["share_supported"] for g in results.values()]
print("mean share supported:", round(sum(shares) / len(shares), 3))
print("hallucination rate:", sum(g["hallucination"] for g in results.values()), "/", len(results))
```

```plain text
T-10001 16/16 supported, share 1.0, hallucination False
T-10038 5/7 supported, share 0.7143, hallucination True
    unsupported: which resulted in the $4.80 charge
    unsupported: Your charge has been confirmed in our system
T-10004 3/4 supported, share 0.75, hallucination True
    unsupported: sometimes the server and client can be slightly out of sync, and using the next code in the sequence resolves it
mean share supported: 0.821
hallucination rate: 2 / 3
```

Sixteen claims for the archive draft, every one traced to a chunk of kb-0015. Seven claims for the proration draft, two unsupported: no chunk mentions this customer's charge, and nothing in the docs lets an agent say a charge is "confirmed in our system". Four claims for the 2FA draft, and the one that fails is the explanation the specialist added on its own: the docs say to try the next code, and say nothing about servers and clients drifting out of sync. Each unsupported claim is a sentence a support engineer would have to delete before sending.

Now look at what the check let through. The T-10038 claim list contains "there were 12 days remaining in the cycle when you added the seat", marked supported with `evidence` null. The customer never said twelve days; the specialist worked it backwards from $4.80. The judge in the previous lesson called this the invented fact, and this check waved it past under the restatement rule while catching the two claims that followed from it. This is a model call, and it makes model-call mistakes. A one-line filter shows you where to look:

```python
for tid, g in results.items():
    for c in g["claims"]:
        if c["supported"] and not c["evidence"]:
            print(tid, "supported with no chunk:", c["claim"])
```

```plain text
T-10038 supported with no chunk: that charge is a prorated amount for the seat you added on September 15th
T-10038 supported with no chunk: there were 12 days remaining in the cycle when you added the seat
```

The first line is a real restatement. The second is the hallucination. Read every "supported with no chunk" line by hand before you trust a groundedness number, and if the list is long, tighten the prompt so that only claims quoting the ticket may pass without evidence.

## The two numbers you report

Run the check over a set of tickets and report both of these, because they answer different questions.

- Mean share supported (the groundedness rate in `judge_eval.py`): the average of the per-draft fractions. Here 0.821 across three drafts.
- Hallucination rate: the share of drafts with at least one unsupported claim. Here two of three.

The mean says how much of what the agent writes is backed by a document. The rate says how many customers would have received at least one sentence nobody can back up, and it is the number Priya can act on: it decides whether Resolve may close tickets on its own or whether every reply goes past a person. The two can move apart. A nine-ticket dev run in the reference build gave a mean share of about 0.83 and a hallucination rate of three drafts in seven, and a set where every draft carries exactly one small unsupported aside would show a high mean and a hallucination rate near 1. `configs/judge.yaml` puts a floor of 0.7 under the groundedness rate for CI; the worked example in the next lesson shows where that floor is checked.

Cost is one judge call per draft, at whatever `LLM_MODEL_REVIEW` charges. The three calls above cost a few cents in total. Over a hundred held-out tickets, expect the groundedness check and the rubric judge together to cost more than the graph runs they are scoring, which is fine for a final evaluation and is why CI runs ten tickets rather than a hundred.

## A local alternative: entailment without an API

The verify step does not need a hosted model. A **natural language inference** (NLI) model is a small classifier that reads a premise and a hypothesis and says whether the premise entails the hypothesis, contradicts it, or neither. That is the groundedness question with the chunk as the premise and the claim as the hypothesis. `microsoft/deberta-large-mnli` is a public checkpoint that does it; the first run downloads about 1.6 GB, and after that it runs on the CPU with no key.

```python
import time
from transformers import pipeline

nli = pipeline("text-classification", model="microsoft/deberta-large-mnli")   # labels: CONTRADICTION, NEUTRAL, ENTAILMENT

def supported_by_nli(claim: str, docs: list[dict]) -> bool:
    """A claim counts as supported when at least one retrieved chunk entails it."""
    pairs = [{"text": d["text"][:1500], "text_pair": claim} for d in docs]   # premise = chunk, hypothesis = claim
    return any(r["label"] == "ENTAILMENT" for r in nli(pairs, truncation=True))

for tid in ("T-10038", "T-10004"):
    t0 = time.perf_counter()
    verdicts = [(c["claim"], supported_by_nli(c["claim"], states[tid]["docs"])) for c in results[tid]["claims"]]
    print(f"{tid}: {sum(ok for _, ok in verdicts)}/{len(verdicts)} entailed in {time.perf_counter() - t0:.1f} s")
    for claim, ok in verdicts:
        print(f"   {'ok ' if ok else 'NO '} {claim[:90]}")
```

```plain text
T-10038: 5/7 entailed in 47.7 s
   NO  that charge is a prorated amount for the seat you added on September 15th
   ok  On the Team plan, each seat costs $12 per month
   ok  When you add a seat mid-cycle, you're charged only for the days remaining in your current
   ok  there were 12 days remaining in the cycle when you added the seat
   ok  which resulted in the $4.80 charge
   ok  The new seat will be billed at the full $12 monthly rate on every invoice after that, star
   NO  Your charge has been confirmed in our system
T-10004: 3/4 entailed in 10.9 s
   ok  TOTP codes are time-based
   ok  even a few minutes of clock drift can cause codes to fail
   ok  The next step to try is entering the next code that appears after the current one refreshe
   NO  sometimes the server and client can be slightly out of sync, and using the next code in th
```

The claims come from the judge's decomposition above; the NLI model only replaces the verify step. Each claim is checked against every chunk, one pair at a time, and counts as supported if any chunk entails it. The totals match the judge, five of seven and three of four, and the claims behind them do not all match. The NLI model fails the restatement about September 15th, because it has no restatement rule and no chunk mentions that date. It passes "12 days remaining" and "$4.80", because the proration chunk holds a table of similar numbers and a small model reads a table loosely. It agrees with the judge on "confirmed in our system" and on the out-of-sync explanation.

So treat it as a second instrument, cheaper and deterministic, that is wrong in different places. NLI models truncate at 512 tokens, which is why the block cuts each chunk at 1,500 characters and checks chunk by chunk. A claim that needs two chunks combined comes back neutral, where a large judge reasons across them. Run the NLI check when you have no API key, or as a second opinion on a groundedness number that looks too good; do not expect it to flag the same sentences.

## RAGAS, in one paragraph

RAGAS is a library that packages these ideas as named metrics: faithfulness (the claim-level check above), answer relevance, context precision, and context recall, the last two grading retrieval rather than generation. Its `evaluate()` takes a table of questions, answers, and contexts and calls a model it must be configured with, so it needs its own LLM settings and its own budget. You can reach for it on a later project. For Resolve, the hand-built check is the one you can read, defend, and calibrate.

## What the number cannot tell you

1. Grounded does not mean complete. A draft that is faithful to a chunk that never held the answer scores 1.0 and leaves the customer stuck. That failure is retrieval's, and the retrieval hit rate in the next lesson is the evaluator that catches it.
2. The decomposition sets the ceiling. Under-split claims and misapplied restatement rules inflate every number you report, and nothing in the pipeline warns you. Read ten claim lists by hand each time the prompt or the judge model changes.
3. Grounded does not mean the right action. A grounded, well-scored draft that auto-resolves a ticket labelled `escalate` is still a failure, and only an evaluator that reads the decision catches it.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Have an assistant draft the decomposition prompt, then check its work on a draft whose facts you know cold. Does it split "the seat costs $12 and you pay for the days remaining" into two claims or one? Does it treat "there were 12 days remaining" as a restatement or as a claim? Does it invent a claim the draft never made? Under-decomposition hides hallucinations behind a comfortable number, so verify the granularity yourself before any groundedness figure leaves your machine.
</callout>

Two metrics, and neither says whether the agent did the right thing with the ticket. The next lesson, Worked Example: Building an Agent Eval Suite, walks through `evals/judge_eval.py`, where the judge and this check run beside routing accuracy, retrieval hit rate, action safety, cost, and latency over the labelled tickets, and where the floors in `configs/judge.yaml` turn the scorecard into a CI gate.
