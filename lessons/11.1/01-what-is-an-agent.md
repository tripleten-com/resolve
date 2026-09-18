You've already made a language model do useful work. In **Prompt Engineering** you sent one prompt and read one answer. In **Retrieval and RAG** you built a pipeline over the Quillstack help center: retrieve, stuff the context, generate. Both of those are **one pass**, meaning the sequence of steps is fixed before the model ever runs. An **agent** is what you get when you give that guarantee up.

<callout icon="📖" color="gray_bg">
	Every code block in this chapter runs from the Resolve repository with `PYTHONPATH=src`, a virtual environment built from `requirements.txt` (`pip install -r requirements.txt`; the packages this chapter touches are `openai==3.15.0`, `pydantic==2.13.5`, and `python-dotenv==1.2.3`), and a `.env` file holding `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. Setting those up is covered in **Prompt Engineering**, in the lesson on calling a model from Python. `python -m resolve.llm --check` confirms the key and model work before you start.
</callout>

## The three ingredients

An agent is a language model running in a **loop**, choosing its own next step, calling **tools** to reach outside its own text, and carrying **state** across steps until it decides it's finished. Three parts:

- The loop: the model runs repeatedly rather than once. Each turn it looks at what it knows and picks one next action.
- The tools: functions the model can ask for. Search the knowledge base, read a ticket, file a bug issue. Tools are the only way a model affects anything at all.
- The state: the accumulating record of the run, holding the original request, every tool call, and every result. State is the agent's working memory between turns.

Take away the loop and you have a pipeline. Take away the tools and you have a chatbot. An agent needs all three at once.

## One pass, and then a loop

Here's the RAG shape from **Retrieval and RAG**, stripped down to its skeleton. The client is the one every part of Resolve uses, `resolve.llm.chat`, which reads the endpoint, key, and model from `.env`; any OpenAI-compatible provider works.

```python
# One pass: you decide both steps, in this order, every time.
from resolve.llm import chat

ticket_text = "Our Team-plan API key expires next week. How do I rotate it without downtime?"

def search_kb(query, k=5):                              # stand-in for the retriever
    return [{"doc_id": "kb-0045", "title": "Rotate an API key",
             "text": "Create the new key under Settings > Integrations > API keys, "
                     "deploy it to every service, then revoke the old key."}]

chunks = search_kb(ticket_text)                          # 1. retrieve, always
prompt = f"Answer using only these passages:\n{chunks}\n\nTicket: {ticket_text}"
answer = chat([{"role": "user", "content": prompt}]).choices[0].message.content   # 2. generate, always
print(answer)
```

```plain text
Based on the available documentation, here's how to rotate your API key without downtime:

1. Create the new key under Settings > Integrations > API keys
2. Deploy it to every service that uses the key
3. Revoke the old key after deployment is complete
```

Read the two numbered comments. Both decisions are yours, written down before any input arrives: one retrieval, one generation, in that order, for every ticket that ever comes in. If a ticket needs two searches, this code cannot make two searches. If a ticket needs no search at all, it searches anyway. That rigidity is also what makes the pipeline easy to test, and the orchestration lesson later in this chapter is where you weigh the trade.

## What state actually looks like

There's no special machinery behind an agent's memory. State is a list of messages that grows by one or two entries every turn:

```python
# The state of an agent run, two turns in. It is just a list.
state = [
    {"role": "system", "content": "You are a Quillstack support agent."},
    {"role": "user", "content": "Our API key expires next week. How do I rotate it?"},
    # turn 1: the model asked for a tool instead of answering
    {"role": "assistant", "tool_calls": [
        {"id": "call_a1", "type": "function",
         "function": {"name": "search_kb",
                      "arguments": '{"query": "rotate API key"}'}}]},
    # turn 1: your code ran the tool and reported back
    {"role": "tool", "tool_call_id": "call_a1",
     "content": '[{"doc_id": "kb-0045", "title": "Rotate an API key", '
                '"text": "Create the new key, deploy it, then revoke the old one."}]'},
    # turn 2: the model had enough and stopped
    {"role": "assistant", "content": "Create a new key under Settings > Integrations > "
                                     "API keys, deploy it, then revoke the old one (kb-0045)."},
]

print(len(state), "messages,", sum(1 for m in state if m["role"] == "tool"), "tool result(s)")
```

```plain text
5 messages, 1 tool result(s)
```

Four things in that list are worth slowing down for. The `assistant` message carrying `tool_calls` and no content is the model asking rather than answering. The `tool` message right after it repeats the same `call_a1` id, which is the only link between a request and its answer. Nothing is ever removed, so on turn 2 the model re-reads the whole history, including its own earlier decision. And the last `assistant` message has content and no `tool_calls`, which is exactly the loop's exit condition: the model stopped asking for things.

In Resolve this message list lives inside a bigger record, `TicketState`, alongside fields such as `classification`, `docs`, `draft`, and `decision`. **LangChain & LangGraph** introduces that record properly. For now the message list is all the state you need.

## The agent loop

The cycle is **observe, think, act**, then observe again, repeating until something stops it:

1. Observe: read the current state, the request plus every result so far.
2. Think: the model reasons about what would help next.
3. Act: it asks for a tool, or it writes the final answer.
4. Observe: the tool result is appended to state, and the loop turns again.

A run ends when the model produces an answer with no tool call, or when a **max-steps** guard trips. Resolve caps every specialist at six steps, and the worked example at the end of this chapter wires that guard in. Never trust a model to always stop on its own.

Notice who does what in that cycle. The model owns step 2 and the choice in step 3. Your code owns everything else: it holds the state, runs the tool, appends the result, counts the steps, and decides when the loop is over. That split is the whole design of Resolve in miniature, and it is the reason an agent is something you can still test.

*Diagram idea: a wheel with four arcs labeled Observe, Think, Act, Observe feeding back into the model at the center; state drawn as a scroll beside the wheel that grows on each pass; a "max steps?" arrow exiting the wheel.*

## ReAct: reason, then act

The standard way to drive this loop is **ReAct**, short for reasoning and acting. The model interleaves a short piece of reasoning ("the ticket mentions error code QS-1099, so the API troubleshooting pages are the place to look") with one concrete action, in this case a call to the search tool. Reasoning is what makes the tool choice deliberate rather than reflexive, and it leaves you something to read when the agent goes wrong. You ask for it by name when you write agent system prompts, which the next lesson covers.

## The running example: Resolve

Quillstack's support desk is eight engineers and about 400 tickets a day, most of them routine. Resolve is the internal system this project builds to handle the first pass on every one of them. It reads the ticket, searches the help center, sometimes looks up an account or files a bug issue, and produces two things: a draft reply and a decision, `auto_resolve` or `escalate`. It is never customer-facing on its own. A person approves anything that goes out.

Now ask why that needs a loop. You don't know in advance how many searches a ticket will take, whether the help center answers the question at all, whether the ticket is a bug that needs an issue filed, or whether it's something the system should refuse to touch. Nobody can write those steps down before the ticket arrives. The path has to be chosen at runtime, by the model, which is the one job an agent does better than a pipeline.

## When the loop earns its cost

Agents aren't free, and the loop is where the bill comes from:

1. Several model calls per ticket instead of one, so more latency and more tokens. Quillstack's finance team counts cost per ticket, and an agent that burns six calls on a password reset is a loss.
2. A different path on every run, which is what makes an agent hard to test and hard to trust.
3. More surface for failure, because every tool call can time out, return nothing, or return something the model misreads.

So the question to ask about any agent is whether the dynamic loop pays for itself. If you can write the steps down in advance, it doesn't. Write a pipeline.

The next lesson, Tool Calling: Mechanics, Structured Outputs, and Prompts, gets concrete about the single move the whole loop rests on, which is how a model asks your code to run a function and how you hand the answer back so the loop can turn again.
