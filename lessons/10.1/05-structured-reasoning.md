A specific prompt is not the same thing as an easy question. Some questions need work done between the asking and the answering, and that work has to happen somewhere: inside one response, across several calls, or across several candidate answers you compare. Those three places are **chain-of-thought**, **prompt chaining** and **branching**, and they are a family rather than a ladder. Each costs something different, so the useful skill is knowing which cost you are buying.

Every call below goes through `resolve.llm.ask`, with the endpoint and model from `.env`; any OpenAI-compatible provider works.

## One call, reasoning out loud

**Chain-of-thought prompting** asks the model to produce its intermediate steps in the response, before the answer. One call, more output tokens, and the reasoning arrives where you can read it.

*Diagram idea: one box "prompt" feeding one box "response", with the response split into a "steps" region above a single labelled "answer" line, and a parser arrow pointing only at the answer line.*

Take an urgency decision. Forty people locked out, a demo in two hours. Asked flat out, the model answers flat out, and in its own shape.

```python
from resolve.llm import ask

TICKET = (
    "Subject: whole team locked out\n"
    "Body: Since about 9am nobody on our team can sign in. Everyone types their "
    "password, then the 2FA code, and gets 'account locked'. That is 40 people. We "
    "have a customer demo at 11 and the deck is in Quillstack. Nothing changed on our side."
)

print(ask(f"What is the urgency of this ticket: low, normal or high?\n\n{TICKET}", max_tokens=60))
```

```plain text
# Urgency: **HIGH**

**Reasons:**

1. **Scale**: 40 people completely unable to work
2. **Time-sensitive**: Customer demo in ~2 hours
3. **Business impact**: Demo deck is inaccessible, risking client
```

The verdict is right and the reasons are sensible, and none of it is usable: a heading, bold, a list cut off by `max_tokens`, and no line a parser could find. Ask for the steps by name, in order, and put the answer on a labelled last line.

```python
COT = """Decide the urgency of the support ticket below for the Quillstack support desk.
Work through it in this order, one short line each:
1. what the customer cannot do
2. how many people are affected and for how long
3. whether a workaround exists
Then, on a final line by itself, write:
URGENCY: <low|normal|high>

""" + TICKET
cot_reply = ask(COT, max_tokens=200)
print(cot_reply)
```

```plain text
1. Customers cannot sign in to their accounts
2. 40 people affected since 9am (2 hours); critical deadline at 11am
3. No workaround mentioned; presentation materials inaccessible

URGENCY: high
```

Read the final line separately from the three above it, because that is the whole design. The reasoning is there for you, and the `URGENCY:` line is there for your code, so a parser never has to guess where prose ends. Two cautions come with the technique. Visible reasoning is not a proof: a model can write three plausible steps and then a fourth line that does not follow from them, so read the steps rather than admiring their presence. And current models already do a good deal of this internally, so "think step by step" buys less than it once did. What still buys a lot is naming which steps, in order, the way the prompt above does.

In code the only part worth writing yourself is the parser, and it should refuse rather than guess.

```python
LEVELS = {"low", "normal", "high"}

def parse_urgency(text):
    """Return the level from the labelled last line, or None if it never arrived."""
    for line in reversed(text.splitlines()):
        if line.startswith("URGENCY:"):
            level = line.split(":", 1)[1].strip().lower()
            return level if level in LEVELS else None
    return None

print(parse_urgency(cot_reply))
print(parse_urgency("Sorry, I need more detail about the customer plan."))
```

```plain text
high
None
```

Two things make that a parser rather than a hope. It walks the response backwards, so the model can put as many reasoning lines as it likes above the labelled one at no cost. And it returns `None` when the label never arrived, or arrived with a word outside `LEVELS`, instead of a wrong level assembled out of a half-match. The second call needs no model: a string fixture is enough to test the parser, which is the payoff for naming `URGENCY:` in the prompt.

## Several calls, one pipeline

**Prompt chaining** takes the same decomposition and puts each step in its own call, feeding one step's output in as the next step's input. You give up the model's ability to see the whole problem at once, and in exchange you get a checkpoint after every step.

A morning's tickets show the shape. Step one groups them by product area, step two names the complaint in each area, step three ranks the areas by how many customers are blocked. In code the pattern is a loop over templates with one placeholder.

```python
STEPS = [
    "Group the tickets below by product area. Answer as one line per area: the area, "
    "a colon, then the ticket ids and a five-word summary. Nothing else.\n\n{input}",
    "For each product area below, write its most common complaint in one line, "
    "using only what the tickets say. Nothing else.\n\n{input}",
    "Rank the complaints below by how many customers they block. "
    "Answer as 'rank. area: reason', one line each, nothing else.\n\n{input}",
]

TICKETS = (
    "T-10201 API calls fail with QS-1042 since this morning, key was created about 90 days ago\n"
    "T-10202 webhooks stopped arriving, dashboard shows QS-1061\n"
    "T-10203 export started yesterday failed with QS-1077\n"
    "T-10204 CI pipeline gets QS-1042 on every request since the weekend\n"
)

def run_chain(steps, seed):
    state = seed
    for i, template in enumerate(steps, start=1):
        state = ask(template.format(input=state), max_tokens=200)
        print(f"--- step {i} ---")
        print(state)
    return state

run_chain(STEPS, TICKETS)
```

```plain text
--- step 1 ---
API: T-10201, T-10204 Authentication key expiration issues
Webhooks: T-10202 Webhook delivery stopped unexpectedly
Export: T-10203 Export job failed with error
--- step 2 ---
API: Authentication key expiration issues
Webhooks: Webhook delivery stopped unexpectedly
Export: Export job failed with error
--- step 3 ---
1. API: Authentication key expiration issues
2. Export: Export job failed with error
3. Webhooks: Webhook delivery stopped unexpectedly
```

The variable named `state` is the point of the whole function. Every loop pass overwrites it with the model's latest answer, which means you can print it, log it, assert on it, or throw the run away at any step instead of discovering at the end that step one misread the tickets. `ask` runs at temperature 0 by default, and that is for the same reason: with the sampling pinned, the same tickets give the same intermediate states, so when step three surprises you it is step three's fault.

The checkpoint earned its keep while this lesson was being written. The first draft of step one said "list the product areas, one per line", and it did: four bare words. Step two then received four words and no tickets, and it wrote a plausible complaint for each area out of nothing, including "undocumented breaking changes between versions" for an API whose only ticket was an expired key. The state carried the areas and dropped the evidence. The fix was to make step one carry the ticket ids and a summary forward, which is the rule for chains in general: each step's output has to contain everything the next step is allowed to rely on, because the next step sees nothing else. Step three's ranking is arguable, and that is fine; it is arguable in the open, with the two tickets behind the top line visible in step one.

## Several candidates, one winner

Chaining assumes you know the steps. **Branching**, published as Tree of Thoughts, drops that assumption: it generates several candidate answers, scores each against criteria you name, then picks. Use it when the problem is open-ended enough that there is no single right answer to walk toward.

A bug ticket where nothing is known yet is that kind of problem. A customer's 2FA code is rejected every time, the phone clock is right, the app behaves, it worked last week. Three calls, and they differ in exactly one argument.

```python
BUG = (
    "I'm trying to sign in but my 2FA code is rejected every single time. I've checked "
    "my phone clock and it's synced correctly. I'm using Google Authenticator and the "
    "code refreshes every 30 seconds like it should. This used to work fine last week."
)
CRITERIA = (
    "consistency with what the customer already reported, how cheaply the customer "
    "can test it, and how often it is the real cause"
)

# 1. generate: temperature above zero, and "distinct" asked for in words
ideas = ask(
    "Give three distinct hypotheses for why this 2FA code is rejected. "
    f"One line each, numbered, plain text with no heading.\n\n{BUG}",
    temperature=0.8, max_tokens=150,
)
# 2. score: back to zero, because a score should not wander between runs
scores = ask(
    f"Rate each hypothesis below on {CRITERIA}. One short line per criterion "
    f"per hypothesis, plain text, no headings.\n\nReport:\n{BUG}\n\nHypotheses:\n{ideas}",
    max_tokens=400,
)
# 3. select: one recommendation, in a shape a parser can find
choice = ask(
    "Based on the evaluations below, which hypothesis should the engineer ask the "
    "customer to check first, and why? Answer as 'CHOICE: <hypothesis>' on the first "
    f"line, then one sentence.\n\n{scores}",
    max_tokens=100,
)
print(ideas, scores, choice, sep="\n---\n")
```

```plain text
1. Your phone's authenticator app is out of sync with the server despite showing correct system time, possibly due to a cached or corrupted token that requires re-registering the 2FA device.

2. Your account's 2FA backup codes were compromised and someone else has already used them or changed the authentication method, locking you out.

3. The service's 2FA system experienced a backend failure or database corruption that's preventing valid codes from being accepted, even though the app is functioning correctly on your end.
---
**Hypothesis 1: Re-register 2FA device**
Consistency: High - explains why system time checks pass but codes still fail, and why it worked before.
Cost to test: Very low - customer can remove and re-add the 2FA device in account settings in minutes.
Frequency: High - token desynchronization and app cache issues are common causes of this exact symptom.

**Hypothesis 2: Backup codes compromised/2FA changed**
Consistency: Low - customer would typically receive security alerts or see the authentication method changed in their account settings.
Cost to test: Low - customer can check account security logs and 2FA settings, but requires account access they don't currently have.
Frequency: Low - this usually involves detectable account activity and wouldn't cause codes to be "rejected every single time" if the attacker changed the method.

**Hypothesis 3: Backend service failure**
Consistency: Medium - possible but customer hasn't reported widespread outages or service status alerts.
Cost to test: Very low - customer can check service status page or try signing in from a different device/browser.
Frequency: Low - backend failures typically affect many users simultaneously and are quickly detected; isolated single-user rejection is uncommon.
---
CHOICE: Hypothesis 1

Re-registering the 2FA device has the highest consistency with the symptoms, is the fastest to test, and addresses the most common cause of consistent code rejection despite correct system time.
```

The word *distinct* is carrying the branch. Drop it and three candidates come back as three flavours of one idea, which leaves the scoring call nothing to compare. The scoring call names its criteria rather than asking which idea is best, so nine judgments arrive before any verdict does. Only the third call picks, and by then it has something to pick from.

Read the `temperature` argument, because it moves between calls and the movement is the design. Generation runs at `0.8` because diversity is the mechanism: at `0` your three branches look like each other. Scoring and selection go back to `0`, so the same three hypotheses score the same way twice and a changed verdict means a changed input rather than a dice roll. The scoring output ignored "no headings" and used bold anyway; a model at temperature 0 follows format instructions most of the time, which is why the selection line, the one a parser reads, is the only line whose shape you insist on. Note also what `choice` is: a string, printed for a person to read. The model's selection is a recommendation. A human decides which hypothesis goes to the customer, and that rule outlives this technique.

## Which one earns its cost

<table header-row="true">
<tr>
<td>Technique</td>
<td>What it costs</td>
<td>What it buys</td>
</tr>
<tr>
<td>Chain-of-thought</td>
<td>One call, with more output tokens and higher latency on that call</td>
<td>Intermediate steps you can read before you trust the answer</td>
</tr>
<tr>
<td>Prompt chaining</td>
<td>One call per step, plus code that carries state between them</td>
<td>A checkpoint after every step, so you can validate, retry or abandon early</td>
</tr>
<tr>
<td>Branching</td>
<td>Several generations plus a scoring pass, so three to five times the tokens</td>
<td>Coverage of the option space when there is no single right answer</td>
</tr>
</table>

The rule that follows fits in three lines. Reach for chain-of-thought when the reasoning is what you need to inspect. Reach for chaining when a step can fail in a way you want to catch before the next one runs. Reach for branching when you do not know what a good answer looks like until you have seen a few. Anything simpler than all three, keep as one plain prompt, because each technique multiplies both cost and the number of places a run can go wrong, and the finance stakeholder counts output tokens.

## What the three have in common

All three buy the same thing with different currency: a place to look. Chain-of-thought puts it in the response, chaining puts it between calls, branching puts it in the comparison. That is why they combine cleanly, and why an agent typically runs all three at once, with a chain for the pipeline, a step inside it reasoning out loud, and a branch wherever the pipeline has to choose.

Seeing that on one page is the next lesson, Worked Example: Prompt Techniques on a Support Ticket, where four stages and one gate turn a raw Quillstack ticket into a record Resolve can route: a chain across the stages, chain-of-thought inside one of them, and branching deliberately left out, for a reason that lesson names.
