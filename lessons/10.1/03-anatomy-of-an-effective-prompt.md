Two requests can point at the same support ticket and come back with completely different things. Ask a model to tell you something about the ticket and you get a paragraph of hedged summary under a heading. Ask it for the ticket type and the error code and you get two strings you can write straight into a record. The model did not change between those two requests. The **prompt** did, and a prompt has an anatomy you can take apart piece by piece.

Every block below calls the model through `resolve.llm.ask`, which reads the endpoint and model from `.env`; any OpenAI-compatible provider works.

## Three shapes a prompt can take

Before the wording comes the shape. What you are asking for constrains how you have to ask for it.

<table header-row="true">
<tr>
<td>Shape</td>
<td>When you reach for it</td>
<td>Example</td>
</tr>
<tr>
<td>Open-ended</td>
<td>Ideas, options, an answer you cannot fully specify in advance</td>
<td>*"Describe three ways a support desk could route tickets before a human reads them."*</td>
</tr>
<tr>
<td>Closed-ended</td>
<td>One fact, a yes or no, a single value</td>
<td>*"Does this ticket quote an error code?"*</td>
</tr>
<tr>
<td>Task-specific</td>
<td>A job with a defined input and a defined output</td>
<td>*"Summarize this ticket in three bullet points."*</td>
</tr>
</table>

Task-specific prompts are the ones that end up inside software, and they cover four recurring jobs. Summarization compresses a long text into a short one, as in a two-sentence brief of a ticket. Classification sorts an input into a label set you defined, as in marking a ticket `how_to`, `billing`, `bug` or `security`. Extraction pulls named values out of free text, as in the error code and the plan a customer mentions. Generation produces something that was not in the input, like a draft reply. What those one-line descriptions leave out is most of what makes a prompt work twice in a row. The walkthrough below asks what separates a prompt that works from one that merely reads well; the four components under it are the answer.

<video src="https://www.youtube.com/embed/h0O8eiHZzlw"></video>

## The four components

The **instruction** says what to do, in the imperative, with nothing left to interpret. *"Can you look at this ticket?"* leaves the model to pick a length, an audience and a level of detail. *"Summarize the ticket below for the engineer who will answer it, covering only what decides the reply"* leaves it nothing to pick.

The **context** supplies the background the task depends on and that the model has no other way to reach. *"Reply to this customer about their expired key"* produces a generic reply. *"Reply to this Team-plan customer; on Team, API keys expire after 90 days and must be rotated before then"* produces one that is true for Quillstack.

The **output format** names the shape you want back, because the default shape is prose and prose is hard to parse. *"What is wrong with this ticket?"* returns paragraphs. *"Answer as two labelled lines, Problem: and Error code:"* returns something a function can split.

The **length and style** bound the response. *"Write a reply about the refund policy"* has no ceiling. *"Write a 60-word reply in plain sentences, one clause of apology at most"* has two.

None of the four is optional once a function has to read the answer. A prompt missing its format line works fine in a chat window, where you read the output yourself and forgive its shape. It breaks the first time `json.loads` gets hold of it.

## Assembling all four

Prompts are strings, so build them the way you would build any other string: named parts, one place to change each one.

```python
from resolve.llm import ask

# 1. the four components, each one editable on its own
INSTRUCTION = "Summarize the support ticket below for the engineer who will answer it."
CONTEXT = (
    "The reader has not seen the ticket. The customer's plan and role decide what "
    "they are allowed to do. Relevant product fact: a project can hold at most 5,000 "
    "open tasks; archived projects are read-only and can be restored by an Admin or "
    "Owner from Projects > Archived."
)
OUTPUT_FORMAT = (
    "Answer as three labelled lines, in this order:\n"
    "Question:\nRelevant facts:\nTicket type (how_to, billing, bug or security):"
)
STYLE = "Plain sentences. 60 words in total at most."

# 2. the payload, the only part that changes per request
TICKET = (
    "Subject: how do i archive a project?\n"
    "Plan: Team\n"
    "Body: we have a project thats finished and i want to archive it but cant find "
    "the option anywhere. also can we bring it back later if we need to? im an admin "
    "on the workspace so just want to make sure i do this right. the project has alot "
    "of tasks in it maybe 4000 or so and its slowing things down when we look at the list."
)

# 3. assembly: blank lines between parts, payload last
def build_prompt(instruction, context, output_format, style, payload):
    return "\n\n".join([instruction, context, output_format, style, payload])

prompt = build_prompt(INSTRUCTION, CONTEXT, OUTPUT_FORMAT, STYLE, TICKET)
print(ask(prompt, max_tokens=150))
```

```plain text
Question:
Customer wants to archive a completed project with ~4,000 tasks to improve performance. They're asking where the archive option is located and whether archived projects can be restored.

Relevant facts:
Customer is a workspace Admin on a Team plan. Archived projects are read-only and can be restored by an Admin or Owner via Projects > Archived. Projects can hold max 5,000 open tasks.

Ticket type:
how_to
```

Compare that with the headed paragraph the same ticket produced in the previous lesson. Same model, same ticket, and this time the three lines arrive in the order the format asked for, the facts come from the context line rather than from the model's imagination, and the type is one of the four words Resolve's classifier is allowed to return.

Read the assembly order, because it is not arbitrary. The instruction goes first so the model knows what kind of reading it is doing before it reaches the ticket. The payload goes last so that a long ticket cannot push the format line out of the model's attention. And the four components sit in constants rather than inside an f-string, so when the format changes you edit one line instead of hunting through a paragraph. That is the whole difference between a prompt and prompt engineering.

## Zero-shot: no examples at all

**Zero-shot prompting** means handing over the task with no worked examples attached and trusting the model to recognize it. That trust is usually well placed. Sentiment, summarization, aspect extraction and translation all appear constantly in the text these models are trained on, so the instruction alone is enough.

Here is a harder version of that: pull structure out of a long, mixed-feeling ticket.

```python
ZERO_SHOT = """From the support ticket below, extract two things and nothing else:
1. Product areas mentioned, as a comma-separated list
2. The customer's tone, as one word

Ticket:
"We moved our team of 25 to the Team plan in June and the pages and projects have
been great, the GitHub integration in particular saves us a lot of time. Two things
though. Our invoice INV-202609-00417 shows a charge on 3 September and then an
identical second charge the same day, which I assume is a mistake. And we started a
workspace export on Monday for our auditors and it still says pending, which is
worrying us a bit because the auditors need it by Friday. Otherwise happy customers,
just want these two sorted."
"""
print(ask(ZERO_SHOT, max_tokens=80))
```

```plain text
1. Pages, Projects, GitHub integration, Invoicing, Workspace export

2. Satisfied
```

Five areas out of a ticket that mixes praise and complaint in almost every sentence, and a single-word verdict that weighs them. No examples were needed for either half, because "product area" and "tone" mean the same thing here that they mean everywhere else. Zero-shot works exactly when your task uses the world's definitions rather than yours.

## Few-shot: showing the format instead of describing it

**Few-shot prompting** puts two to five solved examples in the prompt before the real input. It earns its tokens when the label set is yours, when the output format is fussy, or when the task sits somewhere between two things the model already knows.

Ticket classification is all three. The four types are not English categories. They are Quillstack's decision about which tickets a model may answer, which need an account lookup, which need an issue filed, and which a human must see first. Since the examples have to stay identically shaped, generate them instead of typing them.

```python
# Few-shot examples are data, not prose. Store them as pairs and render them.
EXAMPLES = [
    ("How do I rotate an API key before it expires?", "how_to"),
    ("We were charged twice on the same day for the Team plan.", "billing"),
    ("Every webhook delivery has returned QS-1061 since Tuesday, and it worked last week.", "bug"),
    ("Someone I don't know was added as an Admin on our workspace overnight.", "security"),
]

HEADER = (
    "Classify the support ticket into one type, following the examples. "
    "Answer with the type only."
)

def build_few_shot(examples, query):
    shots = [f'Ticket: "{text}"\nType: {label}' for text, label in examples]
    return "\n\n".join([HEADER, *shots, f'Ticket: "{query}"\nType:'])

prompt = build_few_shot(EXAMPLES, "Can I get an invoice with our VAT number on it?")
print(prompt)
print("--- reply ---")
print(ask(prompt, max_tokens=5))
```

```plain text
Classify the support ticket into one type, following the examples. Answer with the type only.

Ticket: "How do I rotate an API key before it expires?"
Type: how_to

Ticket: "We were charged twice on the same day for the Team plan."
Type: billing

Ticket: "Every webhook delivery has returned QS-1061 since Tuesday, and it worked last week."
Type: bug

Ticket: "Someone I don't know was added as an Admin on our workspace overnight."
Type: security

Ticket: "Can I get an invoice with our VAT number on it?"
Type:
--- reply ---
billing
```

The trailing `Type:` with nothing after it is doing real work. It makes the model's job a continuation of a pattern rather than an interpretation of an instruction, so the answer arrives as a bare label instead of a sentence explaining the label. `max_tokens=5` makes a sentence impossible anyway. Notice also what the examples teach that a definition would not: the VAT question is about invoices, so it is `billing` even though nothing is being charged or refunded, and the webhook ticket is a `bug` even though it quotes an error code a help article explains, because "it worked last week" is the signature of something that changed.

One f-string builds every example, so no shot can quietly drift out of format. That matters more than it looks: an inconsistent example set teaches the model that the format is negotiable. Three habits follow from the same logic. Cover a diverse range of inputs, so the model can **generalize** from your examples to tickets you did not think of rather than pattern-matching the four you did. Keep every example in identical shape. And when the task is genuinely hard, go to five or ten examples before you go back to rewriting the instruction.

## Which one to reach for

1. Start zero-shot. It is shorter, cheaper and faster, and for anything defined by ordinary English it is already enough.
2. Add examples when the definitions are yours. Your label set, your output format, your house style. Three examples beat three paragraphs of explanation.
3. Keep the four components either way. Examples show the model the shape of the answer. They do not tell it the audience, the length or the job.

None of that stops a prompt from failing. It fails in a handful of recognizable ways, and each one leaves a signature you can read in the output. The next lesson, Prompt Failure Modes, and Iterating Out of Them, names those signatures and turns fixing them into a loop with an exit condition.
