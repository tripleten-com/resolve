Resolve runs on your laptop, on tickets you chose, for a reviewer who reads a results file. Nothing on this page is a deliverable, and none of the code below is something the project asks you to run. It is here because the question Priya will ask on the day Resolve handles the real queue is one an offline eval cannot answer: is the thing we shipped still behaving? **Monitoring** is observability over live traffic, and this lesson is a read-only tour of what it adds, using the same traces, tags, and client you already have.

<callout icon="📖" color="gray_bg">
	The blocks on this page use `langsmith==0.12.6` and read runs from a LangSmith project, so each one needs `LANGSMITH_API_KEY` and a project with traffic in it. Read them for shape. The one function you can run locally is the percentile helper, and it is checked below against `evals/results/judge_dev.json`.
</callout>

## What monitoring adds over an offline eval

An offline eval runs before a change ships, on a held-out dataset with labels, and answers "did this change help". Production sends tickets nobody labelled: typos, three questions in one message, a stack trace pasted into the subject line, and the occasional attack. Monitoring runs continuously after the change ships, on that traffic, and answers a different question.

<table header-row="true">
<tr>
<td>Dimension</td>
<td>Offline eval</td>
<td>Monitoring</td>
</tr>
<tr>
<td>What it runs on</td>
<td>`resolve-heldout`, 100 labelled tickets</td>
<td>Every ticket the queue sends</td>
</tr>
<tr>
<td>When it runs</td>
<td>In CI, before a change ships</td>
<td>Continuously, after it ships</td>
</tr>
<tr>
<td>What grades it</td>
<td>The labels: `expected_action`, `gold_doc_ids`, `key_facts`</td>
<td>Rates from the traces, feedback from people, a judge on a sample</td>
</tr>
<tr>
<td>What it decides</td>
<td>Did this change help?</td>
<td>Is the live graph still healthy?</td>
</tr>
</table>

Neither replaces the other. The eval gates a release because there you control the inputs and know the answers. Monitoring catches what the dataset never covered, which is most of what a support queue contains. The useful part is the arrow between the two columns, and it points from production back into the dataset.

## The numbers a dashboard shows

Every trace already carries latency, token counts, and cost, and every root run carries Resolve's final state as its output. A dashboard is mostly a matter of deciding what to look at. The SDK can compute the whole set before anyone builds a chart:

```python
from datetime import datetime, timedelta, timezone

from langsmith import Client

client = Client()   # reads LANGSMITH_API_KEY

since = datetime.now(timezone.utc) - timedelta(days=1)
runs = list(client.list_runs(project_name="resolve", is_root=True, start_time=since))

latency_s = sorted((r.end_time - r.start_time).total_seconds() for r in runs if r.end_time)
costs = [float(r.total_cost or 0) for r in runs]
decisions = [(r.outputs or {}).get("decision") for r in runs]
quarantined = sum(1 for r in runs if ((r.outputs or {}).get("guard_input") or {}).get("quarantined"))
```

`is_root=True` gives one run per ticket rather than one per node. `r.outputs` is the final `TicketState`, so `decision` and `guard_input` are right there, and `r.total_cost` is LangSmith's own roll-up of the `llm` rows beneath the root. As the previous lessons noted, 0.12 marks `list_runs` deprecated in favour of `client.runs.query`, which takes a project id and returns pages; the shape of the loop is the same. Then read the tail, never the mean:

```python
def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]


print(f"tickets          {len(runs):>7}")
print(f"p50 latency      {pct(latency_s, 0.50):>6.1f}s")
print(f"p95 latency      {pct(latency_s, 0.95):>6.1f}s")
print(f"cost / ticket    ${sum(costs) / len(costs):>6.4f}")
print(f"escalation rate  {decisions.count('escalate') / len(runs):>6.1%}")
print(f"quarantine rate  {quarantined / len(runs):>6.1%}")
```

That helper is the `pct()` from `evals/judge_eval.py`. Fed the nine records in `evals/results/judge_dev.json` instead of a day of runs, it prints the same numbers the reference notes report:

```plain text
tickets                9
p50 latency        17.0s
p95 latency        30.6s
cost / ticket    $0.0311
escalation rate   55.6%
```

Read those as a support manager would. A p50 of 17 seconds is fine for a first pass nobody is waiting on; the p95 of 30.6 seconds says one ticket in twenty takes twice as long, and on the reference build the slowest tickets were all ones the reviewer sent back. Cost per ticket is the number a prompt bloat or a runaway loop moves first, and it is cheap to watch because `total_cost` is already on every run; at $0.0311 a ticket, 400 tickets a day is $12 to $15 for the graph. The escalation rate is the domain metric and the subtle one. Rising means Resolve is auto-resolving less than it did, which costs Priya's team hours; falling sharply may mean it has started auto-resolving things it should hand over, which is the failure Marcus cares about. On the held-out labels 45 percent of tickets expect a human, so a live rate far from that band is a signal either way.

## Feedback keys

None of those numbers say whether a reply was any good. Without labels, quality signal comes from **feedback**: a score attached to the run that produced the reply, under a named key.

```python
client.create_feedback(
    run_id=run_id,             # the root run for the ticket
    key="engineer_edited_draft",
    score=1,                   # 1 = the engineer rewrote it before sending
    comment="removed a promised refund timeline",
)
```

The `key` is what makes this chartable. Feedback filed under one stable key becomes a rate you can watch; feedback scattered across `edited`, `rewrote`, and `bad_draft` becomes three sparse charts. Score it numerically so it can be averaged and alerted on.

Resolve has three implicit signals that are stronger than any thumbs button, and all three arrive without asking anyone. A support engineer who edits a draft before sending it has said the draft was wrong. A customer who reopens a ticket Resolve auto-resolved has said the reply did not work. A human who overrides a `pending_write` with a no has said the proposed action was unsafe. Each is a labelled failure, each maps to one key, and each one lands on the run by its ticket id, because the tag on every root run is the same `ticket_id` the helpdesk knows.

## Sampled online judges

Feedback is sparse, so a judge fills the gap. The rubric judge from the previous lesson costs about $0.028 per ticket on the reference build, two calls to the stronger model. Judging every one of 400 tickets a day is about $11 a day, close to what the graph itself costs. So sample, and sample where the risk is.

In LangSmith this is an automation rule on the tracing project: a filter that selects which traces qualify, a sampling rate, and an evaluator that runs as traces arrive, filing its result as feedback under the evaluator's key. For Resolve the filter writes itself. Judge every run whose `decision` is `auto_resolve`, because those are the replies that reached a customer without a human reading them; judge 5 percent of the escalated ones, because those a human reads anyway. On the dev mix that is roughly half the tickets plus a sliver, about $6 a day, and the population it watches is exactly the one a wrong answer can hurt.

A sampled judge score is a trend, never a verdict on one ticket. The number to chart is the share of judged auto-resolves scoring below 4 on groundedness, week over week.

## Alerts on rates

Dashboards only help someone who is looking at them. The rates above become useful when a threshold is written down and something fires when it is crossed. LangSmith's automation rules and its metrics views cover the mechanics; the harder part is choosing the rates and giving each one two edges.

- Quarantine rate above 5 percent of tickets in an hour: either an attack wave or a heuristic that started matching customers, and the reference build's detector quarantined none of its 20 benign lookalikes, so anything near 5 percent is news.
- Escalation rate outside a band, say 35 to 60 percent: below it Resolve may be auto-resolving tickets it should not; above it the support engineers are back to reading everything.
- p95 latency above 45 seconds: a provider slowdown, or the reviewer sending everything back.
- Cost per ticket above $0.10, the same ceiling `configs/judge.yaml` enforces in CI: a runaway specialist loop, or a model switch nobody announced.
- Error rate above zero: the one line where zero is the target, because a failed run is a ticket nobody answered.

The band matters. An escalation alert with only a ceiling is silent when Resolve escalates nothing, which is the worse of the two failures. State both edges, before the incident rather than during it, or every regression turns into an argument about whether the numbers are bad yet.

## Promoting a failed run into the dataset

Here is the part that makes monitoring worth the effort. A production failure is an example you did not have. When a reopen or an edited draft flags a run, read it, decide what the right outcome was, and promote it:

```python
run = client.read_run(run_id)

client.create_example(
    inputs=run.inputs,                       # the ticket exactly as the customer wrote it
    outputs={"type": "billing", "expected_action": "escalate",
             "gold_doc_ids": ["kb-0035"], "key_facts": ["F009", "F010"]},   # your corrected label
    dataset_name="resolve-dev",
    source_run_id=run.id,
    metadata={"origin": "production", "signal": "customer_reopened"},
)
```

`inputs=run.inputs` keeps this honest: the example carries the ticket with its typos and its buried question, never a cleaned-up paraphrase the graph would find easier. `outputs` is your label in the same shape the held-out labels use, so the evaluators from the previous lesson score it without changes. `source_run_id` preserves the link to the trace, so six months later you can still open the run that earned this example its place. It goes into `resolve-dev`, never `resolve-heldout`; the held-out set stays frozen so the final measurement stays comparable. Then the eval covers the case, the fix is verified before it ships, and the same failure cannot come back quietly.

*Diagram idea: a loop from production traces, to flagged runs (reopens, edited drafts, low judge scores), to new examples in `resolve-dev`, to an eval run, to a fix, to a redeploy, and back to traces.*

## The standard to hold a deployed graph to

Four things, and Resolve in production should have all four before anyone calls it monitored.

1. Percentiles on every latency and cost chart. A healthy p50 hides the tail, and the tail is where complaints come from.
2. A written threshold with two edges on every rate you would act on.
3. Feedback captured under stable keys, implicit signals included. Reopens, edited drafts, and declined writes are labelled failures you get for free.
4. A sampling rule and a promotion path, so a slice of traffic is always being judged and every confirmed failure ends up in the dataset.

Section 7 of your report asks what happens on the day Resolve auto-resolves a ticket wrongly: how anyone notices, who is told, what gets switched off first, and what evidence you would pull from the traces. This page is the material for that answer. The reopen or the edited draft is how anyone notices; the alert on the feedback rate is who is told; `LANGSMITH_TRACING` stays on and `auto_resolve` is what gets switched off, by raising `confidence_threshold` in `configs/graph.yaml` past 1.0 so `route` escalates everything; and the trace tagged with the ticket id is the evidence.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Triaging a week of flagged traces is pattern-finding over material you can check, which makes it good delegation. Ask an assistant to cluster the reopened and edited runs and name the themes, and it will surface things like "most of these are proration questions where the draft computed a number". Then review for three things: every theme names runs you can open, the pattern survives your reading two or three of those traces yourself, and the corrected label on any example it proposes is yours. The assistant proposes; you own what becomes a regression test.
</callout>

This is the end of the chapter. The judge that scores a sampled trace here is the same one that scores the held-out set, and whether its scores deserve trust is the question **LLM-as-Judge & Hallucination Metrics** takes up next: judge patterns, the biases a rubric has to defend against, and the claim-level groundedness check behind Resolve's hallucination rate.
