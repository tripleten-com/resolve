Quillstack's support desk answers about 400 tickets a day with eight engineers, and most of those tickets are questions the help center already covers. This lesson assembles the whole of this chapter into one working agent for that majority: about sixty lines, no framework, so you can see what an agent is made of and why each piece earns its place. Build the machine by hand once and the frameworks in **LangChain & LangGraph** stop looking like magic.

## The tools

Two tools, both stubs. A **stub** returns a canned value instead of reaching a real system, which is the point of this build: the loop should be the only thing that varies. The names and signatures are Resolve's real ones, so nothing in the loop changes when the live versions arrive in **Retrieval and RAG** and **MCP: Tools for Agents**.

<table header-row="true">
<tr>
<td>Tool</td>
<td>What it does in Resolve</td>
<td>What the stub returns</td>
</tr>
<tr>
<td>`search_kb(query, k=5)`</td>
<td>Hybrid search over the help center index; returns chunks with `doc_id`, `chunk_id`, `title`, `space`, `text`, `score`</td>
<td>One canned chunk from kb-0045, "Rotate an API key", whatever the query</td>
</tr>
<tr>
<td>`create_issue(title, body, ticket_id)`</td>
<td>Files a bug issue in the tracker; a write tool, gated by a human in the real system</td>
<td>A fake id, `ISS-4711`, plus status and title</td>
</tr>
</table>

Input: one ticket, as a dict with `ticket_id`, `plan`, `subject`, and `body`. Output: a draft reply that ends with a decision line, or the fallback string.

## The architecture spec

ticket → system prompt + 2 tool schemas → loop, max 6 steps → tool dispatch → state append → draft reply ending in `DECISION: resolve` or `DECISION: escalate`, or the fallback string

One guard: a step counter set to six, plus a safe fallback for the run that exhausts it. Everything else is deliberately bare, and the gaps are the interesting part. Nothing here persists, so restart the process and the run is gone. There's no tracing beyond a `print`, no branching except what the model chooses, no reviewer, no guards on the input or output, and no pause before the write. Every one of those absences is a node in Resolve's real graph, and you should be able to feel each one by the end of the page. Watch where the ideas from this chapter enter: the loop, the tool-call contract and the system prompt, and the containment argument.

## Step 1: two stub tools, so the loop is the only moving part

```python
# 1. The tools. Stubs, so nothing depends on a live service.
CHUNK = {"doc_id": "kb-0045", "chunk_id": "kb-0045#steps", "title": "Rotate an API key",
         "space": "public", "score": 0.91,
         "text": ("Create the new key under Settings > Integrations > API keys, deploy it to "
                  "every service, then revoke the old key. Keys expire after 30 days on Free, "
                  "90 days on Team and 365 days on Business.")}

def search_kb(query, k=5):
    return [CHUNK][:k]

def create_issue(title, body, ticket_id):
    return {"issue_id": "ISS-4711", "status": "open", "title": title, "ticket_id": ticket_id}

TOOLS_IMPL = {"search_kb": search_kb, "create_issue": create_issue}

print(search_kb("rotate api key")[0]["title"])
print(create_issue("QS-1099 on /v2/pages", "every request fails", "T-10422"))
```

```plain text
Rotate an API key
{'issue_id': 'ISS-4711', 'status': 'open', 'title': 'QS-1099 on /v2/pages', 'ticket_id': 'T-10422'}
```

`TOOLS_IMPL` maps a tool name to the function behind it, and it's the only place in the program where a name the model emits turns into code that runs. The chunk carries the same six fields the real `search_kb` returns, so a draft that cites `doc_id` here will cite it the same way against OpenSearch. `k` slices a list that only ever holds one chunk, which is a known limitation of a stub and never a pattern to copy: the rule for a real tool is that the schema declares only what the function honours, because a tool that accepts a parameter and silently drops it is a lie the model has no way to detect. Notice too that `search_kb` returns the API-key article regardless of the query. That's useful, because it lets you watch what the agent does when the knowledge base answers the wrong question.

## Step 2: schemas and a prompt, the model's whole view of the world

Add this below the tools. The schemas are in the OpenAI function format from the tool-calling lesson.

```python
# 2. The schemas. This is everything the model knows about the tools.
tool_schemas = [
    {"type": "function", "function": {
        "name": "search_kb",
        "description": ("Search the Quillstack help center for how-to and troubleshooting "
                        "answers. Use it before drafting any reply."),
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "k": {"type": "integer"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_issue",
        "description": ("File a bug issue for the engineering team. Use it only when the ticket "
                        "reports a bug: an error code, a stack trace, or something that used to "
                        "work. Never use it for questions the help center answers."),
        "parameters": {"type": "object",
                       "properties": {"title": {"type": "string"},
                                      "body": {"type": "string"},
                                      "ticket_id": {"type": "string"}},
                       "required": ["title", "body", "ticket_id"]}}},
]

# 3. The prompt. Role, scope, tool policy, and the stop condition.
SYSTEM = ("You are a Quillstack support agent inside Resolve. You draft replies that a person "
          "approves before anything is sent. Search the help center before answering, and use "
          "only what the search returns; cite the doc_id. If the ticket reports a bug, file it "
          "with create_issue and tell the customer what happens next. If the help center does "
          "not cover the question, do not guess: say a support engineer will follow up. When "
          "you have enough to answer, stop calling tools and write the reply. End the reply "
          "with one line, DECISION: resolve or DECISION: escalate.")
```

Two things to read closely. `create_issue`'s description carries a restriction, "use it only when the ticket reports a bug," because without it a model under pressure files an issue for everything and the whole point of the build evaporates. And the four jobs of an agent system prompt are all present: the role and scope in the first two sentences, the tool policy in the middle three, and the stop condition in the last two. The `DECISION` line is a cheap stand-in for the `decision` field Resolve's `route` node computes in code; here the model proposes it and nothing enforces it, which is one of the gaps to feel.

## Step 3: the loop, six steps and a fallback

Add the import at the top of the file and the loop at the bottom. `chat` is the shared client from `resolve.llm`; endpoint, key, and model come from `.env`.

```python
import json
from resolve.llm import chat

# 4. The loop.
def run_agent(ticket, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Ticket {ticket['ticket_id']} ({ticket['plan']} plan)\n"
                                            f"Subject: {ticket['subject']}\n\n{ticket['body']}"}]
    for step in range(max_steps):                            # the max-steps guard
        msg = chat(messages, tools=tool_schemas).choices[0].message
        if not msg.tool_calls:
            return msg.content                               # no tool call means it is done
        messages.append(msg)                                 # record the request first
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            try:
                result = TOOLS_IMPL[call.function.name](**args)
            except Exception as e:                           # errors are results
                result = {"error": f"{call.function.name} failed: {e}"}
            print(f"[step {step}] {call.function.name}({args}) -> {str(result)[:90]}")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
    return "Max steps reached. Escalating to a support engineer."   # the safe fallback
```

`for step in range(max_steps)` is the observe-think-act cycle, with the counter welded on as a hard stop. Each turn does one thing: if the model came back with no `tool_calls`, that message is the answer, so the function returns and the loop is over. Otherwise the assistant message goes into `messages` first, then each requested tool runs and each result goes back as its own `tool` message with the matching `tool_call_id`, which is the tool-calling contract exactly. The `try/except` around the dispatch is what keeps a broken tool from ending the run: a `TypeError` from a bad argument becomes an error string the model reads on its next turn and can route around. And if the `for` finishes without ever returning, the last line hands back a safe fallback rather than hanging or crashing.

## Step 4: three runs, and what each one proves

The runs below are real, against `anthropic/claude-haiku-4.5` through OpenRouter, with the replies shortened. Your wording will differ; the shape of the trace should not.

A how-to ticket the help center covers:

```python
print(run_agent({"ticket_id": "T-10418", "plan": "Team", "subject": "API key expiring",
                 "body": "Our API key expires next week and it is wired into three services. "
                         "How do we rotate it without downtime?"}))
```

```plain text
[step 0] search_kb({'query': 'API key rotation downtime'}) -> [{'doc_id': 'kb-0045', 'chunk_id': 'kb-0045#steps', 'title': 'Rotate an API key', 'space':
[step 0] search_kb({'query': 'API key expiration renewal'}) -> [{'doc_id': 'kb-0045', 'chunk_id': 'kb-0045#steps', 'title': 'Rotate an API key', 'space':
Hi, you can rotate your API key without downtime by following these steps (from our help center):
1. Create a new key under Settings > Integrations > API keys
2. Deploy the new key to all three services
3. Revoke the old key once all services are using the new one
Since you're on a Team plan, your keys expire after 90 days, so you have a good window to plan this rotation.
See kb-0045, Rotate an API key.
DECISION: resolve
```

Two searches at step 0, both in the same turn, which is `msg.tool_calls` being a list in the wild; then a reply at step 1 with no tool call, so the loop returned. The draft cites `kb-0045` and the 90-day Team expiry it read from the chunk. `create_issue` never fired, which is the outcome that makes the whole thing worth building.

Now a bug report. The stub still hands back the API-key article, so this run tests whether the model notices that the search answered a different question:

```python
print(run_agent({"ticket_id": "T-10422", "plan": "Business", "subject": "QS-1099 on every request",
                 "body": "Since this morning every call to /v2/pages returns QS-1099 with "
                         "request_id req_8f2a. Nothing changed on our side and it worked yesterday."}))
```

```plain text
[step 0] search_kb({'query': 'QS-1099 error'}) -> [{'doc_id': 'kb-0045', 'chunk_id': 'kb-0045#steps', 'title': 'Rotate an API key', 'space':
[step 1] search_kb({'query': 'QS-1099 authentication API key expired'}) -> [{'doc_id': 'kb-0045', 'chunk_id': 'kb-0045#steps', 'title': 'Rotate an API key', 'space':
[step 1] search_kb({'query': '/v2/pages error request'}) -> [{'doc_id': 'kb-0045', 'chunk_id': 'kb-0045#steps', 'title': 'Rotate an API key', 'space':
[step 2] search_kb({'query': 'error codes QS troubleshooting API'}) -> [{'doc_id': 'kb-0045', 'chunk_id': 'kb-0045#steps', 'title': 'Rotate an API key', 'space':
[step 3] create_issue({'title': 'QS-1099 error on /v2/pages endpoint - all requests failing', 'body': 'Customer on Business plan reporting QS-1099 error on every call to /v2/pages since this morning. ...', 'ticket_id': 'T-10422'}) -> {'issue_id': 'ISS-4711', 'status': 'open', 'title': 'QS-1099 error on /v2/pages endpoint -
Thank you for reporting this issue. I've filed a bug report with our engineering team (ISS-4711) to investigate the QS-1099 errors you're experiencing on the /v2/pages endpoint.
Since this started suddenly this morning with no changes on your end, our team will look into the request ID (req_8f2a) and determine the root cause.
DECISION: escalate
```

Four searches across three steps, every one of them returning the wrong article, and then the model gave up on the help center, filed the issue with the right `ticket_id`, and escalated rather than pasting rotation instructions onto an outage. That recognition is the behavior a fixed pipeline cannot produce, because a pipeline would have stuffed the wrong chunk into the prompt and generated a confident non-answer. It also shows the cost: three wasted calls before the model changed tactics, which is why Resolve puts a classifier in front of the loop and gives the bug specialist a `search_issues` tool. Run a billing ticket about a duplicate charge the same way and the model escalates without touching `create_issue`, which is the description's restriction doing its job.

Finally, the case that tests the guard rather than the model. Swap the search stub for one that finds nothing and run the how-to ticket again:

```python
TOOLS_IMPL["search_kb"] = lambda query, k=5: []
print(run_agent({"ticket_id": "T-10418", "plan": "Team", "subject": "API key expiring",
                 "body": "Our API key expires next week and it is wired into three services. "
                         "How do we rotate it without downtime?"}))
```

```plain text
[step 0] search_kb({'query': 'API key rotation downtime'}) -> []
[step 0] search_kb({'query': 'API key expiration renewal'}) -> []
[step 1] search_kb({'query': 'API key management'}) -> []
[step 1] search_kb({'query': 'API authentication keys'}) -> []
I wasn't able to find documentation in our help center covering API key rotation procedures.
Since this is a critical operational question about managing your API keys safely, a support engineer will follow up with you to discuss the best approach for your setup.
DECISION: escalate
```

Four empty searches, then an honest escalation at step 2, well inside the limit of six. The fallback string never printed, which is what you want. The fallback exists for the run where the model won't stop on its own, so you should expect to see it rarely and be glad it's there when you do. Note what the model did not do: it did not invent rotation steps, and it did not file a bug issue for a question. Note also what nothing stopped it from doing: it offered its own advice about maintenance windows that no document supports, which is exactly the sentence Resolve's `review` node exists to catch. That trace is also your only window into a non-deterministic system, so get comfortable reading it now: **LangSmith Observability** turns exactly this into a real tool.

## Sorting the pieces by the problem each solves

Three problems, and every line of those sixty answers one of them.

Getting the model to act at all: the two tool schemas and the `TOOLS_IMPL` dispatch table. Without them a model produces a sentence and nothing else. The schemas are the vocabulary; the dispatch table is the only thing that runs.

Getting it to act sensibly: each tool's `description`, the restriction on `create_issue`, and the system prompt's stop condition. All of these are nudges on decisions made at runtime, and none of them is a guarantee.

Making sure the run ends: `max_steps=6`, the `return` on a message with no tool calls, the `try/except` that turns a tool crash into a readable result, and the fallback string. These hold when the model misbehaves, precisely because they don't ask the model for anything.

That third group is the standard to hold your own agents to. A prompt that requests good behavior and code that guarantees termination are different kinds of thing, and a system with only the first one works right up until the run where it doesn't. Resolve adds a fourth group this build lacks, code that decides what happens after the loop: `review`, `output_guard`, `route`, and the human pause before `create_issue` really writes anything.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	An agent loop is good delegation: it's pattern-based, it's short, and every mistake in it is checkable by reading. Have an assistant draft one, then review it on sight for the four failures that show up most often. (1) The assistant message with `tool_calls` isn't appended before the tool results, so the API rejects the follow-up call. (2) Only `msg.tool_calls[0]` is handled, so a model asking for two searches, as it did twice above, gets one answer and silently loses the other. (3) There's no step limit, or the limit exists with no fallback return, so an exhausted loop returns `None`. (4) A tool exception propagates instead of becoming a result. Read the loop against that list before you read it for style.
</callout>

This closes Agent Fundamentals. **LangChain & LangGraph** takes the same loop, the same two tools, and the same state list and rebuilds them as a graph with named nodes, a checkpointer, and a pause for human approval.
