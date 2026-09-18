Finance's line in the brief was blunt: an agent that burns tokens on a password reset is not a win. Priya's engineers care about how long a ticket waits. Neither of them will read a trace. They want two numbers per ticket, cost and latency, and a promise that both stay under a line. This lesson shows where those numbers come from in the state, where the tokens go inside one run, what a day of tickets costs, which knobs bound the spend, and how the floors in `configs/judge.yaml` turn all of it into a gate that fails a push.

<callout icon="📖" color="gray_bg">
	The blocks below read the checkpoints and results files from the last two lessons and make no model calls, so they cost nothing to run. The dollar figures are from the reference build with `anthropic/claude-haiku-4.5` drafting and `anthropic/claude-sonnet-4.5` reviewing through OpenRouter; your provider and models will move them, and the method is what matters.
</callout>

## How usage is recorded

`TicketState` has one field for accounting, `usage`, with four keys: `input_tokens`, `output_tokens`, `cost_usd`, and `latency_ms`. No node writes it directly. Each node that calls a model opens a **Meter**, from `src/resolve/graph/usage.py`, and merges it into the state at the end:

```python
class Meter:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def add(self, resp, model: str) -> None:
        u = usage_of(resp)
        self.input_tokens += u["input_tokens"]
        self.output_tokens += u["output_tokens"]
        self.cost += cost_usd(resp, model)

    def merge(self, state: dict) -> dict:
        prev = state.get("usage") or {}
        return {
            "input_tokens": int(prev.get("input_tokens", 0)) + self.input_tokens,
            "output_tokens": int(prev.get("output_tokens", 0)) + self.output_tokens,
            "cost_usd": round(float(prev.get("cost_usd", 0.0)) + self.cost, 6),
            "latency_ms": int(prev.get("latency_ms", 0)) + int((time.perf_counter() - self.t0) * 1000),
        }
```

`add` runs once per model response. `usage_of` reads the token counts every OpenAI-compatible provider returns, and `cost_usd` prefers the provider's own cost figure when it sends one (OpenRouter does, in `usage.cost`) and otherwise multiplies the tokens by the per-million prices in `configs/graph.yaml`. `merge` returns the running total, so `usage` in the state is always the sum for the whole run so far, and the clock in `latency_ms` is wall time inside nodes, which excludes the seconds a run spends paused waiting for a human. The code nodes `input_guard` and `output_guard` open a Meter too; they add zero tokens and a few milliseconds, and `route` touches `usage` not at all.

Because the state is checkpointed, the numbers are still there after the run. For the ticket from the first lesson:

```python
from resolve.graph.build import get_graph

graph = get_graph()
cfg = {"configurable": {"thread_id": "T-10083"}}
snap = graph.get_state(cfg)
u = snap.values["usage"]
print(u)
print(f"${u['cost_usd']:.4f} for {u['input_tokens']:,} input and {u['output_tokens']:,} output tokens in {u['latency_ms'] / 1000:.1f} s")
```

```plain text
{'input_tokens': 19563, 'output_tokens': 1295, 'cost_usd': 0.038766, 'latency_ms': 22877}
$0.0388 for 19,563 input and 1,295 output tokens in 22.9 s
```

Fifteen input tokens for every output token. That ratio is normal for a tool-calling agent: every turn resends the system prompt, the ticket, and every tool result so far, and the retrieved chunks are the bulk of it.

## Where the tokens go

The total says what a ticket cost. The checkpoint history says which node spent it, because the checkpointer saved `usage` after every node. Each snapshot's `next` names the node that ran between it and the following one, so the difference in `usage` between neighbours is that node's bill:

```python
history = list(reversed(list(graph.get_state_history(cfg))))   # oldest first
prev = {"cost_usd": 0.0, "latency_ms": 0}
for before, after in zip(history, history[1:]):
    node = before.next[0]
    u = after.values["usage"]
    if node != "__start__":
        print(f"{node:16} ${u['cost_usd'] - prev['cost_usd']:.4f}  {u['latency_ms'] - prev['latency_ms']:>6} ms")
    prev = u
```

```plain text
input_guard      $0.0000      59 ms
classify         $0.0020    1241 ms
knowledge_agent  $0.0096    9288 ms
review           $0.0098    4521 ms
knowledge_agent  $0.0081    5311 ms
review           $0.0092    1903 ms
output_guard     $0.0000       0 ms
route            $0.0000       0 ms
resolve          $0.0000     554 ms
```

Read the shape rather than the cents. The guards and the router are free. `classify` is one short structured call, about five percent of the ticket. The specialist loop is the largest single item: several calls, each carrying the growing conversation. The reviewer is one call per pass with no tools, and it still costs as much as a specialist pass, because it runs on the stronger model at several times the price per token and reads every retrieved chunk. On a ticket with no revision the split in the reference build is roughly sixty percent specialist, thirty percent reviewer, five percent classify. This ticket had one revision, which added a second specialist pass and a second review, so the reviewer's share is closer to half, and the revision cost more than the first draft. That is the cheapest optimisation in the whole system: a prompt change that makes the first draft pass review saves a specialist loop and a Sonnet call per affected ticket, and the `revision_count` field tells you how many tickets that is.

## Cost per ticket, and per day

One ticket is an anecdote. `evals/judge_eval.py` records `cost_usd` and `latency_ms` for every ticket it runs, and `evals/results/judge_dev.json` holds the reference run over nine dev tickets:

```python
import json, statistics

records = json.load(open("evals/results/judge_dev.json", encoding="utf-8"))["records"]
costs = [r["cost_usd"] for r in records]
lats = [r["latency_ms"] for r in records]

def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]

print(f"n={len(records)}  cost mean ${statistics.mean(costs):.4f}  p50 ${pct(costs, 0.5):.4f}  p95 ${pct(costs, 0.95):.4f}")
print(f"latency p50 {pct(lats, 0.5) / 1000:.1f} s  p95 {pct(lats, 0.95) / 1000:.1f} s")
print(f"400 tickets a day at the mean: ${400 * statistics.mean(costs):.2f}")
```

```plain text
n=9  cost mean $0.0311  p50 $0.0223  p95 $0.0668
latency p50 17.0 s  p95 30.6 s
400 tickets a day at the mean: $12.42
```

A **percentile** is the value below which that share of the observations fall. p50 is the median, the typical ticket. p95 is the ticket that only one in twenty is worse than, and it is the number the support manager feels, because a queue is set by its slow tail rather than its typical member. The function above is the nearest-rank definition `judge_eval.py` uses; with nine tickets p95 is simply the second-worst one, which is why you report these on the full held-out set and say how many tickets stand behind them.

The mean, multiplied out, is finance's number. Around twelve dollars a day for the graph at Quillstack's volume of about 400 tickets, and a security ticket costs nothing because `input_guard` quarantines it before any model runs. Two things sit outside that figure. The judge from **LLM-as-Judge & Hallucination Metrics** is two calls on the stronger model per ticket, close to three cents each, so judging every ticket in production would roughly double the bill; sampling is the usual answer. And the mean hides the tail: the two expensive dev tickets, T-10009 and T-10004, were bug tickets whose reviewer sent the draft back, and each cost about three times the median.

## The controls

Three settings in `configs/graph.yaml` bound what one run can spend, and each protects against a different failure.

```yaml
max_steps: 6
revision_limit: 1
max_tokens:
  classify: 200
  specialist: 900
  review: 500
```

The **step cap** is the loop's hard stop. A specialist that has made six model turns without answering hands off with a draft that says so and confidence 0.0, which `route` sends to `escalate`. Without it, a model that keeps calling `search_kb` with slightly different queries would run until the provider cut it off, and you would pay for every turn. The revision limit does the same for the reviewer loop: one revise verdict earns a second pass, the second ends the run as an escalation, which is exactly what happened to T-10118 in the last lesson. `max_tokens` bounds each call's output, so a specialist cannot write a two-thousand-word reply, and the reviewer cannot write an essay about it.

None of those is a budget in dollars. A **token budget per run** is a fourth control, and it is a few lines you add on top of the state you already have:

```python
def over_budget(usage: dict, max_usd: float = 0.10, max_ms: int = 60_000) -> str | None:
    if usage["cost_usd"] > max_usd:
        return f"cost ${usage['cost_usd']:.4f} is over the ${max_usd:.2f} budget"
    if usage["latency_ms"] > max_ms:
        return f"latency {usage['latency_ms']} ms is over the {max_ms} ms budget"
    return None

print(over_budget(snap.values["usage"]))
print(over_budget(snap.values["usage"], max_usd=0.03))
```

```plain text
None
cost $0.0388 is over the $0.03 budget
```

Where it goes matters more than what it says. Inside the specialist loop, after `meter.add`, a breach breaks out of the loop the same way the step cap does, with a draft that says a human will follow up. In `route`, a breach is one more reason to refuse `auto_resolve`. What a budget must never do is trim the reply to fit: a truncated answer that gets auto-resolved is a worse outcome than an escalation that cost a little more. The budget's job is to make the expensive tickets visible and hand them to a person, and `escalation_reason` is where the reason is written down.

## Floors, and what CI does with them

A **floor** is the lowest value of a metric you are willing to ship. The bottom of `configs/judge.yaml` holds four:

```yaml
floors:
  routing_accuracy: 0.7
  detection_rate: 0.8
  groundedness_rate: 0.7
  cost_per_ticket_usd_max: 0.10
```

Three are minimums and one is a ceiling. The eval job in `.github/workflows/ci.yml` runs the injection suite with the guards on, then `python evals/judge_eval.py --set heldout --subset evals/ci_subset.txt --check-floors`. The subset file names ten held-out tickets, chosen once and committed, so every run scores the same tickets and a change in the number means a change in the system. Ten is enough to catch a broken guard or a routing regression, and small enough that the eval step costs well under a dollar; the suite step before it costs about as much again. The routing, groundedness, and cost floors come from that run's summary; the detection floor comes from `security/results/guards_on.json`, which the suite step wrote a minute earlier. Any miss prints `FLOORS MISSED` and exits 1, and a non-zero exit is what turns the job red.

The check itself is a comparison per line, and you can run it on the reference results without a model:

```python
from resolve.config import judge_config

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
```

```plain text
routing_accuracy         0.8889   floor 0.7    ok
detection_rate           1.0      floor 0.8    ok
groundedness_rate        0.8274   floor 0.7    ok
cost_per_ticket_usd_max  0.0311   floor 0.1    ok
```

Notice how far the floors sit below the measured values. A floor is a tripwire, set where a miss means something broke. The targets from your spec are a different thing. Setting `routing_accuracy` to 0.85 when the system measures 0.89 on nine tickets would fail on noise, and a gate that goes red for no reason gets ignored within a week. Set each floor where you would refuse to ship, write in the report why that value, and leave the targets from your spec as the numbers you aim at.

## When a number breaches a floor

The job goes red. Resist the two easy moves: rerunning until it passes, and lowering the floor. Instead, open the results file the job uploaded as an artifact and read the records, because a summary cannot tell you which ticket moved.

If routing dropped, find the ticket whose `routing_correct` flipped and read its `escalation_reason` and `review`. On the reference dev run the one miss was T-10038, where the reviewer rejected an arithmetic claim the docs did not support, which is the reviewer doing its job on a prorated-charge question, so the fix belongs in the specialist prompt and the reviewer stays as it is. If groundedness dropped, the `claims` list names the unsupported sentence. If detection dropped, the suite records which case reached a specialist. If cost rose, compare `revision_count` and the length of `tool_calls` against the last green run: more revisions means a prompt change made first drafts worse, more tool calls means a specialist has started searching in circles, and a rise with neither usually means the provider changed a price or a model, which `configs/graph.yaml` and the `judge_model` line in the results file will show.

Then fix the cause and push again. The task page asks for a red run and a green run linked from the README, and this is why: a gate that has never gone red has never been shown to gate anything. The red run is the evidence.

That closes the chapter. Everything above reads the numbers from the state and the results files on your machine; **LangSmith Observability** is where the same tokens, costs, and latencies appear per node in a trace, for every ticket, without a script.
