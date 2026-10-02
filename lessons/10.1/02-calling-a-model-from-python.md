Every lesson in this chapter calls a model, and so does most of the course after it. This lesson shows what that call is, line by line, so that no later lesson has to stop and explain it.

## What a call to a hosted model is

A **hosted model** is a large language model running on a provider's servers. You send it text over HTTPS, it sends text back, and the provider bills you for the traffic. The unit of billing is the **token**, which is the unit the model reads and writes in. A token is roughly three quarters of an English word, so "rotate an API key before it expires" is about eight tokens. Nothing runs on your laptop except the code that builds the request and reads the reply.

Almost every provider accepts the same request shape, the chat completions format that the `openai` Python package speaks. The package name misleads: Nebius Token Factory, OpenRouter, OpenAI, Anthropic's compatibility endpoint, and a local Ollama server all accept the same call, so one client covers all of them. The previous lesson set up the course route, Nebius Token Factory, and put your key in `.env`. Resolve never hardcodes a provider. Three values in `.env` decide where a call goes and which model answers it, and any OpenAI-compatible provider works.

## Three values in .env

Copy `.env.example` to `.env` and fill in the key. The model lines are the part that matters here.

```plain text
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
LLM_API_KEY=
LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
LLM_MODEL_REVIEW=google/gemma-3-27b-it
```

`LLM_BASE_URL` is the endpoint the client posts to. `LLM_API_KEY` is your key for that endpoint. `LLM_MODEL` is the model every ordinary call uses, and `LLM_MODEL_REVIEW` is a second, usually stronger, model that the reviewer and the judge use in later chapters. It may equal `LLM_MODEL`. The defaults point at Nebius Token Factory with the two open models the course uses; `.env.example` shows the equivalent lines for OpenRouter, OpenAI, Anthropic and Ollama. The outputs printed in this chapter were recorded while the course was written, against a Claude model through OpenRouter. The model name and endpoint in your own output will differ, and the exact wording of a reply may too; the shape of every reply is the same. `.env` holds a secret, so it never goes into git.

Two packages do the work, pinned to the versions the project uses.

```plain text
pip install openai==3.15.0 python-dotenv==1.2.3
```

## The first call

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env in the current folder into os.environ

client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

response = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[
        {"role": "system", "content": "You are a support engineer at Quillstack. Answer in one sentence."},
        {"role": "user", "content": (
            "A customer's API requests started failing with error QS-1042 this morning. "
            "Known fact: QS-1042 means the API key has expired. What do I tell them?"
        )},
    ],
    temperature=0,
    max_tokens=80,
)
print(response.choices[0].message.content)
```

```plain text
Ask the customer to generate a new API key in their account settings, as their current key has expired and is no longer valid for authentication.
```

Read it top to bottom. `load_dotenv()` copies the lines of `.env` into the process environment, so `os.environ` can read them like any other variable. `OpenAI(base_url=..., api_key=...)` builds a **client**, an object that knows where to send requests and how to sign them; you build one and reuse it. `client.chat.completions.create(...)` is the call itself, and everything else in the block is an argument to it.

`messages` is a list of dictionaries, each with a `role` and a `content`. The **system message** sets who the model is and how it behaves for the whole conversation. The **user message** is the request. Later, when you build agents, a third role called `assistant` carries the model's own earlier replies, and a fourth called `tool` carries results from functions the model asked you to run. For now, two roles are enough: put standing instructions in `system` and the job in `user`.

Notice the question did not rely on the model knowing Quillstack. It supplied the fact about QS-1042 in the message. A hosted model knows nothing about a company it was never trained on, and it will answer confidently anyway. Everything Resolve does with documents in **Retrieval and RAG** comes from that one observation: the facts have to be in the prompt.

## Reading the reply

The call returns a response object, and the text is only one field of it.

```python
choice = response.choices[0]
print(choice.finish_reason)
print(response.model)
print(response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens)
```

```plain text
stop
anthropic/claude-haiku-4.5
63 31 94
```

`choices` is a list because the format allows asking for several answers at once; you always want `choices[0]`. Inside it, `message.content` is the text and `finish_reason` says why the model stopped. `stop` means it finished on its own. `response.model` echoes which model actually answered, which matters when a provider routes one name to several models. `usage` is the bill: 63 tokens went in, which is the system line plus the question, and 31 came out.

## max_tokens is a ceiling

`max_tokens` caps the reply. The model does not plan around it; it writes until it is done or until the cap cuts it off.

```python
short = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[{"role": "user", "content": "Explain what an API key is to a new Quillstack customer."}],
    temperature=0,
    max_tokens=15,
)
print(repr(short.choices[0].message.content))
print(short.choices[0].finish_reason)
```

```plain text
'# What is an API Key?\n\nAn API key is a **unique'
length
```

`finish_reason` is now `length`, and the text stops mid-word. Two habits follow. Size `max_tokens` to the job: ten tokens for a label, a few hundred for a reply, more for reasoning. And check `finish_reason` whenever a truncated answer would be worse than no answer, because a JSON object cut in half looks like garbage to a parser. The reply also shows something the next lesson takes up: with no format named, the model reached for Markdown headings on its own.

## temperature

**Temperature** controls how much randomness the model uses when it picks each next token. At `0` it takes the most likely token every time. Higher values let it pick less likely ones, which reads as variety.

```python
def subject_line(temperature):
    r = client.chat.completions.create(
        model=os.environ["LLM_MODEL"],
        messages=[{"role": "user", "content": (
            "Write one short subject line for a support ticket about an expired API key. "
            "Reply with the subject line only."
        )}],
        temperature=temperature,
        max_tokens=20,
    )
    return r.choices[0].message.content.strip()

for t in (0, 0, 1.0, 1.0):
    print(f"temperature={t}: {subject_line(t)}")
```

```plain text
temperature=0: API Key Expired - Access Denied
temperature=0: API Key Expired - Access Denied
temperature=1.0: API Key Expired - Access Denied
temperature=1.0: Expired API Key - Access Denied
```

The two runs at `0` match. The two at `1.0` differ, in a small way here and in larger ways on longer outputs. Resolve pins every pipeline call to `0`, because a classifier that gives a different label on the second run cannot be tested. The one place a higher temperature earns its keep is when variety is the point, which the lesson on structured reasoning shows. Even at `0`, providers do not promise identical output across runs; you get close, and you still write the code as if it could differ.

## What a call costs

Providers price per million tokens, with input and output priced differently, and the numbers change. Read them off your provider's page for your model and keep them where the code can see them.

```python
PRICE_IN = 1.00    # dollars per million input tokens, from your provider's model page
PRICE_OUT = 5.00   # dollars per million output tokens

def cost_usd(prompt_tokens, completion_tokens):
    return prompt_tokens / 1e6 * PRICE_IN + completion_tokens / 1e6 * PRICE_OUT

u = response.usage
one_call = cost_usd(u.prompt_tokens, u.completion_tokens)
print(f"this call: ${one_call:.6f}")
print(f"400 tickets a day, 4 calls each: ${one_call * 400 * 4:.2f} per day")
```

```plain text
this call: $0.000218
400 tickets a day, 4 calls each: $0.35 per day
```

The two constants are the list price of the default model at the time of writing; substitute yours. A fraction of a cent per call looks like nothing until you multiply by Quillstack's ticket volume, and the first call was tiny. Real prompts carry worked examples, retrieved articles and the ticket itself, so a couple of thousand input tokens per call is normal, which moves the daily figure by an order of magnitude. Output tokens cost several times more than input tokens, so a reply that rambles costs more than a prompt that is long. The finance stakeholder in Resolve cares about exactly this number, and `usage` is where it comes from. Every call you make from here on should record it.

## The client Resolve uses

Repeating the client setup in every file is a mistake waiting to happen, so Resolve wraps it once in `src/resolve/llm.py`. The endpoint and model come from `.env`, and any OpenAI-compatible provider works. Check that your setup works before anything else.

```plain text
python -m resolve.llm --check
```

```plain text
anthropic/claude-haiku-4.5 via https://openrouter.ai/api/v1: 'ready' in 1501 ms (14 in / 4 out tokens)
anthropic/claude-sonnet-4.5 (review): 'ready'
```

The module exports two functions. `ask` is the simple one: a prompt in, the reply text out, with an optional `system` line and `temperature` and `max_tokens` defaulting to `0` and `800`. A first Quillstack ticket, summarised, takes five lines.

```python
from resolve.llm import ask

TICKET = """Subject: how do i archive a project?
Body: we have a project thats finished and i want to archive it but cant find the option anywhere. also can we bring it back later if we need to? im an admin on the workspace so just want to make sure i do this right. the project has alot of tasks in it maybe 4000 or so and its slowing things down when we look at the list."""

print(ask(f"Summarize this support ticket in two sentences for the engineer who will answer it.\n\n{TICKET}", max_tokens=120))
```

```plain text
# Support Ticket Summary

A workspace admin needs to archive a completed project with approximately 4,000 tasks to improve performance and wants to know where the archive option is located and whether archived projects can be restored. They're concerned about doing this correctly given the large number of tasks involved.
```

`chat` is the full version. It takes the `messages` list, so you control the roles yourself, and it returns the whole response object so you can read `usage`, `finish_reason` and, from **Agent Fundamentals** on, tool calls. `usage_of` turns the usage into a plain dictionary.

```python
from resolve.llm import chat, usage_of, MODEL

resp = chat(
    [
        {"role": "system", "content": "You are a support engineer at Quillstack. Answer in one sentence."},
        {"role": "user", "content": f"What is this customer asking for?\n\n{TICKET}"},
    ],
    max_tokens=60,
)
print(resp.choices[0].message.content)
print(usage_of(resp), MODEL)
```

```plain text
The customer is asking how to archive a completed project in Quillstack, whether archived projects can be restored later, and wants to confirm they have the right permissions as a workspace admin to do this.
{'input_tokens': 121, 'output_tokens': 44} anthropic/claude-haiku-4.5
```

`MODEL` and `MODEL_REVIEW` are the two model names read from `.env`, exported so the rest of the code can log which model produced what. From here on, every code block in this chapter starts with one of these two imports and never mentions a provider again.

Look once more at the `ask` output above. The summary is correct, and it arrived under a Markdown heading nobody asked for, in a shape nothing downstream can parse. The prompt said what to do and nothing about what the answer should look like. The next lesson, Anatomy of an Effective Prompt, takes the user message apart into the four pieces that decide whether a reply is usable.
