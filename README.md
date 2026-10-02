# Resolve

Resolve is Quillstack's internal support agent: it takes the first pass on every ticket and ends with a draft reply plus a decision, `auto_resolve` or `escalate`. It is a LangGraph workflow with fixed node names:

```
input_guard -> classify -> [router] -> knowledge_agent | billing_agent | bug_agent | escalate
specialist -> review -> (revise once -> same specialist) -> output_guard -> route -> resolve | escalate
```

The guards and the decision are code. The classifier and the reviewer are one model call each. The three specialists share one tool-calling loop (`src/resolve/graph/agents.py`) with different prompts and tool sets. Every write to the helpdesk pauses for a human and carries an idempotency key.

## Setup

You need Python 3.12, Docker with Compose, and a key for any OpenAI-compatible model endpoint. The course route is Nebius Token Factory (join the TripleTen organization by invitation, create your own key); OpenRouter, OpenAI, Anthropic or a local server work the same way. After filling in `.env`, run `python scripts/check_provider.py` to confirm the endpoint supports tool calling and JSON output.

```bash
python -m venv .venv
source .venv/Scripts/activate            # Windows Git Bash; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                     # fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_MODEL_REVIEW
```

Start the search engine and the helpdesk, seed the helpdesk, and check the model key:

```bash
docker compose up -d opensearch helpdesk
curl -s localhost:9200/_cluster/health
curl -s localhost:${HELPDESK_PORT:-8100}/health
curl -s -X POST localhost:${HELPDESK_PORT:-8100}/admin/reset     # loads data/tickets and data/accounts
export PYTHONPATH=src
python -m resolve.llm --check
```

Index the knowledge base (rerun whenever `data/kb/` changes; the count is stable across runs):

```bash
python -m resolve.kb.index            # prints "85 articles -> 558 chunks"
python -m resolve.kb.index --count
```

The injection classifier (`protectai/deberta-v3-base-prompt-injection-v2`, about 700 MB) downloads on the first run of the guards. Set `RESOLVE_INJECTION_CLASSIFIER=0` to use heuristics only.

Tracing is on when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set in `.env`, and silently off otherwise.

## Run

One ticket from the command line. Model, thresholds, and caps come from `configs/graph.yaml`; tool scopes from `configs/tools.yaml`; guard thresholds and allowed link hosts from `configs/guards.yaml`.

```bash
python -m resolve.run T-10001              # a how-to ticket: pauses before the first write
python -m resolve.approve T-10001          # shows each pending write in plain words, asks y/n, resumes
python -m resolve.run T-10004 --approval auto      # demo only: executes writes without asking
python -m resolve.run T-10004 --approval dry_run   # records writes, executes nothing
```

The helpdesk MCP server (`src/resolve/mcp_server.py`) is started by the graph as a subprocess over stdio. To run it alone: `python -m resolve.mcp_server`.

As a service, next to the search engine and the helpdesk:

```bash
docker compose up -d                       # builds the agent image the first time
curl -s localhost:8200/health              # ok only when the model endpoint, OpenSearch, the helpdesk, and the graph answer
curl -s -X POST localhost:8200/tickets/T-10004/run
curl -s -X POST localhost:8200/tickets/T-10004/approve -H 'content-type: application/json' -d '{"approved": true}'
```

Locally without Docker: `uvicorn resolve.api:app --port 8200`.

## Evaluate

Retrieval (no model calls; tune on dev, measure once on held-out):

```bash
python evals/retrieval_eval.py --set dev
python evals/retrieval_eval.py --set heldout --mode hybrid --write     # -> evals/results/retrieval.json
```

Tests, including the replay test (needs the helpdesk and a model key; skipped otherwise):

```bash
ruff check src tests evals security
pytest -q tests
```

The injection suite, guards off then on. Writes are never executed during the suite. Cases live in `security/injection_suite.jsonl`; two cases plant a chunk into the index under `kb-9901`/`kb-9902` and remove it afterwards, one files a planted issue in the helpdesk (reset the helpdesk to remove it).

```bash
python security/run_suite.py --guards off      # -> security/results/guards_off.json
python security/run_suite.py --guards on       # -> security/results/guards_on.json
```

The judge and groundedness evaluation (rubric, judge model, and CI floors in `configs/judge.yaml`):

```bash
python evals/judge_eval.py --set dev --limit 8                                  # -> evals/results/judge_dev.json
python evals/judge_eval.py --set heldout                                        # the final measurement
python evals/judge_eval.py --set heldout --subset evals/ci_subset.txt --check-floors   # what CI runs
python evals/judge_eval.py --set heldout --upload                               # LangSmith dataset, needs a key
```

CI (`.github/workflows/ci.yml`) runs lint and the tests against a helpdesk started in the job, then the suite and the ten-ticket held-out subset with floors. It needs `LLM_BASE_URL` and `LLM_API_KEY` as repository secrets.

Reference numbers from the instructor build are in `REFERENCE_NOTES.md`.
