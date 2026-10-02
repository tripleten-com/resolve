Every lesson in this chapter calls a hosted language model, and so does the whole of Project 4. Those calls need an account somewhere and a key. This lesson sets both up on Nebius Token Factory, the platform the course uses, and explains what the calls cost before you make one.
## What Nebius Token Factory is
**Nebius Token Factory** is a platform that serves open-weight language models (Qwen, Llama, DeepSeek, gpt-oss, and others) through one OpenAI-compatible API. Nothing runs on your laptop. Your code sends a request over HTTPS, the platform runs the model on its own hardware, and you get the reply back. You pay per token, which is the unit the model reads and writes in: roughly three quarters of an English word.
The course code never hardcodes a provider. Three values in `.env` decide where a call goes and which model answers, so if you would rather use OpenRouter, OpenAI, Anthropic, or a local server, you can. Nebius is the route the course sets up for you, and it is the one the learning coaches can help with.
## Joining the TripleTen organization
<callout icon="✅" color="gray_bg">
	Check the inbox of the email address you use to sign in to the TripleTen platform for an invitation from Nebius Token Factory to join the **TripleTen** organization. Accept it before it expires: invitations are valid for three days. Once you've joined, skip straight to **Generating an API key** below.
	If you haven't received an invitation yet, follow the steps below to create a personal account and use the trial credit in the meantime. If your invitation has expired, ask your learning coach for a new one.
</callout>
Inside the organization you work in a project set up for the program, and you create your own key there. Nobody shares a key with you, and you never share yours. The project's usage page shows which models you've been calling and what they cost, which is also what your learning coach will look at if something seems off.
## Creating a personal account instead
If the invitation hasn't reached you, go to [tokenfactory.nebius.com](https://tokenfactory.nebius.com/) and sign up with Google or GitHub. Registration asks for a bank card.
<callout icon="❗" color="gray_bg">
	The card is needed to activate the account, but nothing is charged at signup. A new account starts with **$1 in trial credit, valid for 30 days**. At the prices of the models this course uses, that covers the Prompt Engineering chapter comfortably; the project itself needs more, which is why the organization invitation matters. Nebius bills from a prepaid balance that every request debits in real time, and the card is charged only when the balance you set runs out, so you stay in control by topping up in small amounts.
</callout>
When your invitation arrives later, accept it and move your key to the organization project. Keys are per project, so the personal one stops being used the moment you change `.env`.
## Generating an API key
In the dashboard, open your project and go to **API Keys**. Create a new key, give it a name you'll recognise later, such as `resolve-course`, and copy it straight away. The full key is shown once; after you close the dialog you can only revoke it.
Paste it into your project's `.env` along with the endpoint and the two models the course uses:
```plain text
LLM_BASE_URL=https://api.tokenfactory.nebius.com/v1/
LLM_API_KEY=your-key-here
LLM_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507
LLM_MODEL_REVIEW=openai/gpt-oss-120b
```
`LLM_MODEL` does the everyday work: classifying tickets, the specialist agents' loops, the prompts in this chapter. `LLM_MODEL_REVIEW` is the second model that reviews drafts and, later, judges them. It is from a different model family on purpose, because a judge tends to favour prose from its own family, which **LLM-as-Judge & Hallucination Metrics** explains. Both have to support two things Resolve depends on: tool calling, where the model asks your code to run a function, and JSON output that follows a schema. Not every open model does both well, so the project ships a check for exactly that (below).
## Keeping your key private
The key is the credential that spends your balance, or the program's. Treat it like a password.
- Never paste it into code or a notebook cell.
- Never commit it to a repository, even a private one. `.env` is in `.gitignore` for that reason, and the repository's `.env.example` holds the variable names with the values blank.
- Never share it in a message or a screenshot.
If a key ever leaks, revoke it in the dashboard first and generate a new one, then clean up wherever it landed. A committed key is the one finding that fails a Project 4 submission on its own.
## What a call costs
A model's price is listed per million tokens, separately for input (what you send) and output (what the model writes). The two numbers differ, and output is usually the expensive side. The current rates for every model are on the [prices page](https://tokenfactory.nebius.com/organization/prices); check there rather than trusting a number in a lesson, because prices change.
Two reference points for this course. A single prompt in this chapter sends a few hundred tokens and gets a few dozen back, so one call costs a fraction of a cent on a model the size of the ones above. One Resolve ticket, by the time it reaches Task 3, is a different animal: the classifier, a specialist loop of three to six calls that each carry the retrieved documents, and a reviewer pass add up to somewhere between thirteen and thirty thousand input tokens and about a thousand output tokens. At open-model prices that is still cents rather than dollars, and the project's cost lesson in **Running the Agent as a Service** shows where the tokens go.
## Keeping costs low
Almost all of your spend will be ordinary prompting, and ordinary prompting is cheap. The things that produce a surprising bill are different in kind:
1. A **dedicated endpoint** left running. It bills per minute of uptime whether or not you call it. Nothing in this course needs one; use the shared serverless models.
2. A **fine-tuning job**, billed as training compute. Project 3 covered fine-tuning on your own machine, and Project 4 never trains anything.
3. A **loop in your code** without a cap: a retry that never gives up, or a pass over all 200 development tickets when you meant to test on three. Every agent loop in this project has a step cap for this reason, and the task pages tell you to run a handful of tickets, not the set.
4. **Bloated prompts** that resend a whole document or the entire conversation on every call. Keep `max_tokens` small while you develop, and read what your code sends before you scale it.
You can see spend by model and by day on the **Usage** tab of your organization or project page. Look at it after your first session so you know what normal looks like.
## Check your setup
From the repository root, with `.env` filled in, run the client check that every later lesson relies on:
```bash
python -m resolve.llm --check
```
Success is one line naming your model and endpoint with the reply `ready` and the token counts. Then run the provider check, which makes five tiny calls to confirm the model does what Resolve needs:
```bash
python scripts/check_provider.py
```
It reports PASS or FAIL for a plain completion, tool calling the model chooses itself, tool calling your code forces, JSON output that follows a schema, and the review model. Five passes means the chapter and the project will work on your setup. A failure on the tool-calling or JSON lines means the model you named doesn't support them well; pick another from the catalogue and run the check again before you go on.
Next, **Calling a Model from Python** takes the client apart line by line: the messages, the system prompt, temperature, and how to read the reply and its usage.
