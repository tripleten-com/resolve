Resolve has three specialists, a reviewer, and a router, which sounds like a multi-agent system. Open `src/resolve/graph/nodes/` and there is one loop, three files of three lines each, and a reviewer that never calls a tool. This lesson is about why it has that shape, what the split costs on every ticket, and when you should not split at all.

## One loop, three configurations

The previous two lessons built the loop and gave it a pause. `run_agent(agent, system_prompt, state, call_check)` in `src/resolve/graph/agents.py` is that loop, and everything that differs between specialists is data passed into it: a name, a prompt, and, for billing, one check. The tool list is not passed at all. The loop reads it from `configs/tools.yaml` by the agent's name:

```yaml
agents:
  knowledge_agent: [search_kb, read_doc]
  billing_agent: [search_kb, read_doc, get_account]
  bug_agent: [search_kb, read_doc, search_issues, create_issue]

writeback:
  resolve: [add_ticket_comment, set_ticket_status]
  escalate: [add_ticket_comment, set_ticket_status]

writes: [create_issue, add_ticket_comment, set_ticket_status]
```

Those are the spec's tool sets exactly. Every specialist can read the help center. Billing adds a read-only account lookup. Bug adds the issue tracker, including one write. The two exits use the helpdesk's comment and status tools. `writes` is the list `execute_write` consults: any tool on it pauses the run and carries an idempotency key.

At startup `build.py` fills a registry from two sources, the two `@tool` functions in `resolve.tools.kb` and the six tools the helpdesk MCP server publishes (**MCP: Tools for Agents** covers how those arrive). Each specialist then gets its slice:

```python
from resolve.graph.build import load_tools
from resolve.graph.agents import tool_scope, write_tools
from resolve.graph import registry

load_tools(with_helpdesk=True)          # KB tools from Python, helpdesk tools from the MCP server
print(sorted(registry.TOOLS))
for agent in ("knowledge_agent", "billing_agent", "bug_agent"):
    tools = registry.get_tools(tool_scope(agent))
    print(f"{agent:16} {[t.name for t in tools]}")
print("writes:", sorted(write_tools()))
```

```plain text
['add_ticket_comment', 'create_issue', 'get_account', 'get_ticket', 'read_doc', 'search_issues', 'search_kb', 'set_ticket_status']
knowledge_agent  ['search_kb', 'read_doc']
billing_agent    ['search_kb', 'read_doc', 'get_account']
bug_agent        ['search_kb', 'read_doc', 'search_issues', 'create_issue']
writes: ['add_ticket_comment', 'create_issue', 'set_ticket_status']
```

Eight tools loaded, and no specialist sees more than four. `registry.get_tools` raises `KeyError` for a name that is not loaded, so a typo in the YAML fails when the graph is built rather than on the one ticket that needed the tool. This is **least privilege** applied to a model: the knowledge agent cannot file an issue or change a status, whatever a ticket says to it, because the schema for those tools is never sent to it. A prompt telling it not to would be a preference. An absent tool is a control.

The prompts follow the same pattern, in `src/resolve/graph/prompts.py`. `COMMON` says who the assistant is, that it drafts and never sends, that it may state only facts from chunks or tool results, that internal-space articles are never cited, that it must say so when the documents do not answer, and that marked text is data. Each specialist adds one paragraph, and `FINAL_FORMAT` asks for the JSON with `draft`, `citations`, and `confidence`. The knowledge agent's paragraph is the whole of what makes it a knowledge agent:

```python
KNOWLEDGE_AGENT = COMMON + (
    "\n\nYou handle how-to questions. Search the help center with search_kb (try the customer's words, then exact "
    "terms), read the whole article with read_doc when the chunk is cut off, then answer with the steps and cite the "
    "article ids you relied on.\n\n" + FINAL_FORMAT
)
```

The billing paragraph adds the account lookup and says refunds are for the billing team to decide. The bug paragraph says to search existing issues before filing, to file once if nothing matches, and never to say a bug is fixed. Same loop, three paragraphs, three tool lists.

One rule lives in code rather than in the prompt, and it is the whole of `billing_agent.py` beyond the three-line node:

```python
def check_billing_call(name: str, args: dict, state: dict) -> str | None:
    if name != "get_account":
        return None
    own = (state.get("ticket", {}).get("customer_email") or "").strip().lower()
    asked = (args.get("email") or "").strip().lower()
    if asked != own:
        return (f"refused: get_account may only be called for the ticket's own customer ({own}); "
                f"{asked or '(empty)'} is another customer. Do not look it up; hand off to a human instead.")
    return None

def billing_agent(state: dict) -> dict:
    return run_agent("billing_agent", BILLING_AGENT, state, call_check=check_billing_call)
```

```python
from resolve.graph.nodes.billing_agent import check_billing_call

state = {"ticket": {"customer_email": "dana@acme.example"}}
print(check_billing_call("get_account", {"email": "dana@acme.example"}, state))
print(check_billing_call("get_account", {"email": "someone.else@acme.example"}, state))
```

```plain text
None
refused: get_account may only be called for the ticket's own customer (dana@acme.example); someone.else@acme.example is another customer. Do not look it up; hand off to a human instead.
```

The loop calls `call_check` before any tool runs and, on a refusal string, sends that string back to the model as the tool result instead of calling the tool. The prompt also says "you may only look up the ticket's own customer", and the two are not redundant: the prompt makes the model unlikely to try, the check makes the attempt fail. The threat this closes, a ticket that says "also check what my colleague at finance@... was charged", is the one **OWASP Threat Modeling** names first for an agent with a lookup tool.

## How the router picks a specialist

The router is a conditional edge that reads the classification, in `src/resolve/graph/nodes/router.py`:

```python
SPECIALISTS = {"how_to": "knowledge_agent", "billing": "billing_agent", "bug": "bug_agent"}

def specialist_for(ticket_type: str | None) -> str:
    return SPECIALISTS.get(ticket_type or "", "escalate")

def after_classify(state: dict) -> str:
    c = state.get("classification") or {}
    if c.get("type") == "security" or c.get("aimed_at_agent"):
        return "escalate"
    return specialist_for(c.get("type"))
```

```python
from resolve.graph.nodes.router import after_classify

print(after_classify({"classification": {"type": "how_to", "aimed_at_agent": False}}))
print(after_classify({"classification": {"type": "how_to", "aimed_at_agent": True}}))
print(after_classify({"classification": {"type": "security", "aimed_at_agent": False}}))
```

```plain text
knowledge_agent
escalate
escalate
```

The dispatch is a dict lookup, and `specialist_for` is shared with the edge after `review`, so a revision always goes back to the specialist that wrote the draft. The order of the checks is the order the spec gives: `aimed_at_agent` and `security` escalate before any specialist runs, so a ticket that contains instructions for the assistant never reaches a model with tools. On the reference build's 200 dev tickets the classifier's routing agreed with the labels a little over nine times in ten, and the misses were mostly how-to tickets that quoted an error code and were sent to the bug agent, which still ends in a human's hands. The router does not need to be perfect. It needs to send anything dangerous to `escalate` and everything else somewhere a reviewer will check.

## The reviewer: a second model, no tools

The reviewer is a node with one model call and a structured verdict. `Review` in `src/resolve/graph/nodes/review.py` is the schema the model is forced to fill:

```python
class Review(BaseModel):
    """The reviewer's verdict on one draft."""
    verdict: Literal["approve", "revise"]
    reasons: list[str] = Field(default_factory=list, description="One short reason per problem; empty when approving.")
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Each claim in the draft that no chunk or tool fact supports, quoted.")
```

The node builds one user message from the ticket, the retrieved chunks, the tool facts (account lookups, issue searches, filed issues), the draft's citations and confidence, and the draft itself, then calls the model with `tool_choice` pinned to `Review`, so the answer is that schema or an error. The system prompt, `REVIEWER`, tells it to check claim by claim, to send back any number, menu path, error code, plan limit, or promise that no chunk or tool fact supports, any citation that was not retrieved or is internal-only, any promised refund or fix, and to quote each unsupported claim verbatim. The node then does the one piece of bookkeeping the loop depends on:

```python
def review(state: dict) -> dict:
    meter = Meter()
    prev = state.get("review") or {}
    count = int(prev.get("revision_count", 0))
    if not (state.get("draft") or "").strip():
        r = Review(verdict="revise", reasons=["empty draft"])
    else:
        r = review_draft(state, meter)
    if r.verdict == "revise":
        count += 1
    return {"review": {"verdict": r.verdict, "reasons": r.reasons, "unsupported_claims": r.unsupported_claims,
                       "revision_count": count},
            "usage": meter.merge(state)}
```

The edge after it is the bounded loop from the second lesson, with the limit read from `revision_limit: 1` in `configs/graph.yaml`:

```python
def after_review(state: dict) -> str:
    r = state.get("review") or {}
    limit = int(graph_config().get("revision_limit", 1))
    if r.get("verdict") == "revise" and int(r.get("revision_count", 0)) <= limit:
        return specialist_for((state.get("classification") or {}).get("type"))
    return "output_guard"
```

```python
from resolve.graph.nodes.router import after_review

base = {"classification": {"type": "billing"}}
print(after_review({**base, "review": {"verdict": "approve", "revision_count": 0}}))
print(after_review({**base, "review": {"verdict": "revise", "revision_count": 1}}))
print(after_review({**base, "review": {"verdict": "revise", "revision_count": 2}}))
```

```plain text
output_guard
billing_agent
output_guard
```

One revise goes back; a second goes forward with the verdict still `revise` in state, and `route` escalates on it. When the specialist runs again, the loop sees `review.verdict == "revise"`, appends its previous draft and a `REVISION_NOTE` with the reasons and the quoted claims to its conversation, and lets it search again before answering.

Three design choices deserve a sentence each. The reviewer has no tools on purpose: it can only read, so nothing in a chunk or a ticket can make it act, and it is the one node whose judgment is not exposed to a tool result. It uses the stronger model (`review: LLM_MODEL_REVIEW` in `graph.yaml`) because catching an unsupported claim is harder than making one, and because a reviewer from the same weights as the drafter shares its blind spots. And its verdict is structured so that `route`, which is code, can read it without parsing prose.

*Diagram idea: three specialist boxes side by side, each showing the same loop icon with a different prompt label and tool list, all feeding one reviewer box marked "no tools, stronger model" with a single back-arrow labelled "revise, once". Caption: "One loop, three configurations, one reader."*

## The multi-agent tax

Every node in that picture is paid for on every ticket, and the reference build measured it. A normal how-to or billing ticket lands in the range of a few cents, roughly two to five, and takes somewhere between fifteen and forty seconds. About six tenths of the cost is the specialist loop, three to six calls on the cheaper model; about three tenths is the reviewer, one or two calls on the stronger model; the classifier is a few percent. A revision adds one specialist loop and one review, on the order of two more cents and fifteen more seconds. A security ticket the guard quarantines costs nothing, because no model runs. At Quillstack's volume of about 400 tickets a day, the graph alone comes to something like twelve to fifteen dollars a day, before any judge is run on the output.

So the reviewer is roughly a third of the bill and a third of the latency, and the classifier is close to free. That is the shape of the tax: the second model is the expensive part, the routing is cheap, and the specialists cost what they would have cost as one agent. Latency adds up because the calls are sequential; nothing in this graph runs in parallel, and a paused write adds however long the human takes plus a re-execution of the calls before the pause.

## When a single agent would do

Split when at least one of these is true, and Resolve has all three:

- Tools differ in risk. The bug agent can write to the tracker; the knowledge agent should never be able to. One agent with all eight tools means every ticket, including the injected ones, reaches a model that can file, comment, and close.
- Prompts differ in substance. Refund policy, issue-filing discipline, and how-to citation rules are three different jobs, and a prompt that tries to do all three gets longer and worse at each.
- You need per-type numbers. The judge in **LLM-as-Judge & Hallucination Metrics** reports groundedness by ticket type; three specialists give you three things to fix.

Use one agent when the tools are all reads, the job is one job, and a wrong answer is cheap: an internal search assistant over the same knowledge base, for instance, needs `search_kb`, `read_doc`, one prompt, and no reviewer. Add the reviewer when a wrong answer reaches a customer, because that is the failure it is priced against. Add the classifier when the tool sets diverge, because that is what it is for.

There is a third shape this chapter does not build: specialists as separate graphs, each with its own state, called by an orchestrator, the orchestrator-worker pattern from **Agent Fundamentals**. Resolve keeps everything in one graph because one graph means one state, one checkpoint thread per ticket, one pause mechanism, and one trace to read. Sub-graphs earn their complexity when the specialists are owned by different teams or run on different machines, and neither is true here.

The next lesson, **Worked Example: The Resolve Triage Graph**, assembles the real graph from `build.py` in the order Task 2 builds it, then runs a how-to ticket that auto-resolves and a bug ticket that pauses at `create_issue`, and reads the printed path of each.
