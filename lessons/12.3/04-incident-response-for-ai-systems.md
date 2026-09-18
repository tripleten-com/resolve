Every defense in this chapter will one day fail against an attack nobody on the team imagined. **Incident response** is the rehearsed sequence for the day a guard is bypassed in the live queue and Resolve does something it should not. What separates a scare from a disaster is rarely the quality of the defenses. It is whether anyone knew what to do in the first twenty minutes. This lesson walks the lifecycle with Resolve's own state, config files, and suite as the tools.

## The lifecycle

Five stages: detect, contain, investigate, remediate, post-mortem. Two decisions come before any of them and take about a minute. Someone is named **incident lead**, so there is one person deciding and one place updates go. And the incident gets a **severity**: S1 for a data leak (another customer's invoices in a reply), S2 for an unauthorized action (a ticket closed that should have escalated, an issue filed with attacker-chosen content), S3 for degraded quality (thirty bad drafts). An agent that answered thirty tickets badly and an agent that sent one customer another customer's account details are different events, and the response should be too.

*Diagram idea: a five-stage loop, detect to contain to investigate to remediate to post-mortem, with one arrow from remediate back into a box labelled `security/injection_suite.jsonl`, and severity chips S1 to S3 beside detect.*

## Detect: rates over run records

You cannot respond to what you cannot see. Every run of the graph ends in a final state, and three of its fields are the signals: `guard_input.quarantined`, `decision`, and `usage` (`input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`). The API's `POST /tickets/{id}/run` returns all three in its response, and `run_ticket()` returns the whole state, so a **run record** is one of those dicts appended to a log. The reference build does not ship that log; the five exports in `traces/` (the checkpointed final state of one ticket each) stand in for it here.

The signals worth an alarm are rates over a window, never single events: the share of runs quarantined, the share auto-resolved, confirmed writes per run, and cost per run. A quarantine spike is someone probing the system. An auto-resolve rate that suddenly rises is what an agent resolving things it should not looks like from outside. A cost spike is a padded reply or a loop that stopped stopping.

```python
import json
from pathlib import Path

WINDOW = 200
LIMITS = {"quarantine_rate": 0.10, "auto_resolve_rate": 0.60, "writes_per_run": 3.0, "cost_per_run": 0.10}

def rates(records: list[dict]) -> dict:
    recent = records[-WINDOW:]
    n = len(recent)
    writes = sum(sum(1 for t in s.get("tool_calls", []) if t.get("write") and t.get("confirmed")) for s in recent)
    return {"n": n,
            "quarantine_rate": sum(bool(s["guard_input"].get("quarantined")) for s in recent) / n,
            "auto_resolve_rate": sum(s.get("decision") == "auto_resolve" for s in recent) / n,
            "writes_per_run": writes / n,
            "cost_per_run": sum(s["usage"].get("cost_usd", 0.0) for s in recent) / n}

def alarms(records: list[dict]) -> dict | None:
    r = rates(records)
    if r["n"] < WINDOW:
        return None                      # not enough runs to say anything
    return {k: round(v, 4) for k, v in r.items() if k in LIMITS and v > LIMITS[k]}

records = [json.loads(p.read_text(encoding="utf-8"))["state"] for p in sorted(Path("traces").glob("*.json"))]
print(rates(records))
print(alarms(records))
```

```plain text
{'n': 5, 'quarantine_rate': 0.2, 'auto_resolve_rate': 0.4, 'writes_per_run': 1.8, 'cost_per_run': 0.0263966}
None
```

Five records are not a window, so `alarms()` says nothing, which is the right answer: one quarantined ticket in five is a Tuesday. The limits are where the measurements from the last two lessons earn their keep. The detector alone quarantines 9 of the 200 dev tickets, all nine of them labelled injections, so about one ticket in twenty is the normal quarantine rate for Quillstack's mix, and an alarm at 10% over 200 runs fires on a change in the traffic rather than on a busy afternoon. A normal how-to or billing ticket costs $0.02 to $0.05, so a mean of $0.10 over 200 runs means something is padding or looping. Without those baselines a threshold is a guess, and a guessed threshold either pages you constantly or never.

## Contain before you root-cause

Containment comes first, ahead of understanding. The goal is to stop further damage quickly and reversibly, and each option below is a file edit or an environment variable, never a deploy.

<table header-row="true">
<tr>
<td>action</td>
<td>reach for it when</td>
<td>how, in this build</td>
<td>what it costs</td>
</tr>
<tr>
<td>Disable the agent</td>
<td>harmful actions are still happening</td>
<td>a config flag `agent_enabled` (below)</td>
<td>every ticket waits for a human until it is back</td>
</tr>
<tr>
<td>Remove a tool from `tools.yaml`</td>
<td>the blast radius is one tool</td>
<td>delete `create_issue` from `agents.bug_agent`; the loop answers "tool 'create_issue' is not available to bug_agent" and the specialist carries on</td>
<td>little: the bug specialist degrades to search-only and its drafts still reach a human</td>
</tr>
<tr>
<td>Revoke the helpdesk token</td>
<td>the graph must not be able to write at all</td>
<td>the provided helpdesk has no token, so the equivalent is stopping the helpdesk MCP server; a production helpdesk has a token you revoke and reissue</td>
<td>slower to reverse, and worth keeping revoked if the scope was surplus</td>
</tr>
</table>

The first row is the **kill switch**, and the reference build does not have one. It has the opposite: `guards_enabled` in `configs/guards.yaml` and `RESOLVE_GUARDS_ENABLED`, which turn the guards off for the suite. The smallest honest kill switch is a second flag next to it, read once at the door:

```python
import os

def agent_enabled() -> bool:
    """Default closed: a missing or misspelled variable disables the agent."""
    return os.environ.get("RESOLVE_AGENT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

os.environ["RESOLVE_AGENT_ENABLED"] = "false"
print(agent_enabled())
os.environ["RESOLVE_AGENT_ENABLED"] = "true"
print(agent_enabled())
```

```plain text
False
True
```

Where to read it is a choice with a consequence. Read it in `api.py`'s run route and return HTTP 503 with a reason, and tickets simply stay in the queue for the engineers; nothing is written, which is what you want when the helpdesk itself is the concern. Read it in `input_guard` instead, setting `quarantined` with a flag `agent_disabled`, and every ticket goes to `escalate`, which posts an internal note saying so; that is friendlier to the queue and still makes two writes per ticket. Either way the flag defaults closed, so a missing variable disables the agent rather than enabling it, and either way it lives in `.env` and the Compose file next to `LLM_API_KEY`. The cost of default-closed is that every environment must set it on purpose, and that is the point.

Whatever you do, write down the time you did it. The containment timestamp is the anchor the whole timeline hangs from.

## Investigate: the evidence is already on disk

Preserve before you read. The graph checkpoints every node's output into `var/checkpoints.db` under `thread_id = ticket_id`, so the full history of the bad run is there until someone deletes the file. Copy it, copy the ticket text to an evidence folder, and never clear the checkpoint database to tidy up. `traces/` holds the same thing exported: the node sequence from the checkpoint history plus the final state. Read the one for the injection ticket in the dev set:

```python
import json
from pathlib import Path

trace = json.loads(Path("traces/T-10007.json").read_text(encoding="utf-8"))
state = trace["state"]
print(trace["node_sequence"])
print(state["guard_input"])
print(state["escalation_reason"])
print([(t["name"], t["node"], t["confirmed"]) for t in state["tool_calls"]])

evidence = Path("incidents/2026-09-18")
evidence.mkdir(parents=True, exist_ok=True)
(evidence / "triggering_input.txt").write_text(state["ticket"]["body"], encoding="utf-8")
```

```plain text
['__start__', 'input_guard', 'escalate']
{'flags': ['injection:override', 'injection:prompt_leak', 'injection:inventory_request', 'classifier:1.00', 'quarantined'], 'risk': 1.0, 'quarantined': True}
input_guard quarantined the ticket: risk 1.00 (flags: injection:override, injection:prompt_leak, injection:inventory_request, classifier:1.00)
[('add_ticket_comment', 'escalate', True), ('set_ticket_status', 'escalate', True)]
```

Two questions, and the state answers both. Where did it enter: `guard_input.flags` names the heuristics that fired on the ticket, and for a document-borne attack the `chunk_dropped` flags in `guard_output` and the `dropped` count on the `search_kb` entry in `tool_calls` point at the search that returned it. What was the blast radius: `tool_calls` lists every tool that ran, its arguments, its `node`, and whether it was `confirmed`, and `docs` lists every chunk the specialist read. With LangSmith on, the same run is a tree of node spans with each model call under its node; the local export is the version that works without a key.

## Remediate: a fix and a regression case

Remediation is a patch and a rollback. Fix the guard, the heuristic, or the prompt; roll back to the last known-good version of the prompts and config, which is only possible because they live in the repository with the code. The step teams skip is the third one: turn the incident into a permanent case in `security/injection_suite.jsonl`, read from the evidence file rather than retyped, because a paraphrase of an attack is a different attack and may pass.

```python
import json
from pathlib import Path

suite = Path("security/injection_suite.jsonl")
case = {"id": "inj-044", "label": "attack", "kind": "direct", "expected": "blocked",
        "notes": "Incident 2026-09-18: the exact ticket text, from incidents/2026-09-18/triggering_input.txt",
        "text": Path("incidents/2026-09-18/triggering_input.txt").read_text(encoding="utf-8")}
with suite.open("a", encoding="utf-8") as f:
    f.write("\n" + json.dumps(case, ensure_ascii=False))
print(sum(1 for line in suite.read_text(encoding="utf-8").splitlines() if line.strip()), "cases")
```

```plain text
44 cases
```

Then `python security/run_suite.py --guards on --only inj-044`. The case must fail before the fix and pass after it, and the `notes` field names the incident so that when someone prunes the suite for runtime, the cases that came from the live queue are the ones that stay. An incident you do not encode is an incident you will relive.

## The blameless post-mortem

Close with a written retro: a factual timeline, the root cause as a mechanism, and action items with named owners and dates. **Blameless** is a practical choice. The engineer who widened a heuristic did it inside a review process that let it through, and a post-mortem that stops at their name leaves the review process as it was. Ask what let the change land.

Two things belong in the timeline that people leave out: when the alarm fired versus when a human noticed, and how long containment took after that. Those two gaps are where the next improvement usually is, and neither is visible if the timeline starts at "we found the bug".

Read the whole lesson as three questions you could be asked in the middle of an outage. Is detection a rate with a measured baseline, or a feeling? Is containment one variable away, and does it default closed? Does every incident leave behind a case in the suite and a timeline with the two gaps? A team that can answer all three in writing is the standard to hold Resolve to.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	An assistant will draft a post-mortem timeline from a trace export in seconds, and drafting is where it belongs. Then check every timestamp and every tool call against the checkpoint history before the document circulates, because a plausible wrong timeline sends the investigation at the wrong node. Read the draft for three things: timestamps that match the trace rather than the narrative, a root cause that names a mechanism rather than a person, and an action item with an owner and a date on every line.
</callout>

This closes the chapter. You built the two guards, opened up the detector, ran the suite with the guards off and on, dispositioned the findings, and rehearsed the day one of them gets past you. What remains for Resolve is measurement at scale: the judge, the faithfulness metric, and the cost and latency per ticket that decide whether it earns its place in the queue.
