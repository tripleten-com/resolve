# Resolve: shared design spec (Project 4)

This file is the single source of truth for every Project 4 page, lesson, and the reference build. Anyone writing a page reads it first and uses these names, fields, and numbers exactly. Nothing in here goes to students verbatim; it keeps authors consistent.

## 1. The scenario

**Company:** Quillstack, a mid-size B2B SaaS company (about 180 employees) that sells a team documentation and project-tracking product, also called Quillstack. Customers are on one of three plans: Free, Team, Business. Product surfaces: workspaces, pages, projects, an HTTP API with API keys, SSO (Business only), integrations (Slack, GitHub), data export.

**Support desk:** eight support engineers, about 400 tickets a day, most routine. Tickets arrive through a web form and email. The team uses an internal helpdesk (tickets and comments), an internal bug tracker (issues), and a documentation site with public and internal-only pages.

**Product being built:** Resolve, an internal AI system that handles the first pass on every ticket. It is not customer-facing. Its output is a draft reply plus a decision: auto-resolve (send the reply and close) or escalate (hand to a human with the research attached).

**Stakeholders (the same four as before, keep them):**
- Support manager (Priya Natarajan): backlog down, queue triaged, hard tickets reach the right person with context.
- Support engineers: a colleague, not a chatbot. File the issue, link the doc, draft the reply for approval.
- Security lead (Marcus Bell): every ticket is untrusted input; prompt injection is a certainty, not a risk.
- Finance (cost per ticket): an agent that burns tokens on a password reset is not a win.

The PM-style email in "Meet Resolve" comes from Priya. The security email comes from Marcus.

## 2. Ticket types and outcomes

Four types. The classifier returns exactly one.

| type | Looks like | Correct outcome |
|---|---|---|
| `how_to` | product question answerable from public docs ("how do I rotate an API key?") | grounded cited reply, auto-resolve if the reviewer approves; escalate if docs don't cover it |
| `billing` | charges, invoices, refunds, plan changes | grounded reply from public billing docs, read-only account lookup allowed, never another customer's data; refunds always escalate |
| `bug` | reproduction, error code, stack trace, "it used to work" | search existing issues, create an issue (write, human-approved), reply saying what happens next; never auto-resolve as "fixed" |
| `security` | vulnerability report, account compromise, or text aimed at the agent | escalate at classify time; no retrieval, no agent, no reply drafted |

Urgency: `low`, `normal`, `high`. Angry tone raises urgency, never changes type.

## 3. The workflow (LangGraph)

Node names are fixed. Use them everywhere, in code, prose, diagrams, and traces.

```
input_guard -> classify -> [router] -> knowledge_agent | billing_agent | bug_agent | escalate
specialist -> review -> (revise once -> same specialist) -> output_guard -> route -> resolve | escalate
```

- `input_guard` (code, no model): length cap, secret scrub, injection risk score. Writes `guard_input`. If `quarantined`, edge goes straight to `escalate`.
- `classify` (one model call, structured output `Classification`): `type`, `urgency`, `aimed_at_agent`, `confidence`. `security` or `aimed_at_agent` routes to `escalate`.
- `knowledge_agent`, `billing_agent`, `bug_agent`: tool-calling loops (model + tools + step cap 6). Same loop code, different system prompt and tool set. Each ends by writing `draft`, `citations`, `confidence`.
- `review` (model call, structured output `Review`, no tools): checks the draft against `docs`: every claim supported, citations present, tone, no internal-only sources. Returns `verdict` `approve` or `revise` plus `reasons`. Max one revision (`revision_count`).
- `output_guard` (code): secrets, internal-only links, schema, citation ids exist in `docs`. Writes `guard_output`.
- `route` (code): `auto_resolve` only if review approved, `confidence >= threshold` (default 0.7), no output flags, and type is `how_to` or `billing`. Everything else `escalate`. `bug` always escalates after the issue is filed (the reply is a draft for the human).
- `escalate` and `resolve`: terminal nodes that write the decision back through the helpdesk tools (a comment plus a status), gated.

Human approval: the graph uses a checkpointer and `interrupt()` before any write tool runs. Resuming with approval executes the write once; every write carries an idempotency key `f"{ticket_id}:{tool}:{sha256(args)[:8]}"`.

### State (`TicketState`, TypedDict)

```
ticket_id: str
ticket: dict            # subject, body, customer_email, plan, created_at, channel
guard_input: dict       # flags: list[str], risk: float, quarantined: bool
classification: dict    # type, urgency, aimed_at_agent, confidence
docs: list[dict]        # doc_id, chunk_id, title, space, text, score
draft: str
citations: list[str]    # doc_ids that appear in docs
confidence: float
tool_calls: list[dict]  # name, args, result_summary, confirmed: bool, idempotency_key
pending_write: dict | None
review: dict            # verdict, reasons: list[str], revision_count: int
guard_output: dict      # flags: list[str], ok: bool
decision: str           # auto_resolve | escalate
escalation_reason: str
usage: dict             # input_tokens, output_tokens, cost_usd, latency_ms
```

## 4. Tools

Retrieval tools (plain Python, in `src/resolve/kb/search.py`):
- `search_kb(query: str, k: int = 5) -> list[dict]` hybrid search over OpenSearch, returns chunks with `doc_id`, `chunk_id`, `title`, `space`, `text`, `score`.
- `read_doc(doc_id: str) -> dict` full article by id.

Helpdesk tools (exposed by the student's MCP server over the helpdesk HTTP API):

| tool | kind | what it touches |
|---|---|---|
| `get_ticket(ticket_id)` | read | ticket plus comments |
| `get_account(email)` | read | plan, invoices, last 5 charges; the agent may only pass the ticket's own `customer_email`, enforced in code by the graph's tool wrapper (`graph/agents.py`) before the call reaches the MCP server, never by the prompt alone |
| `search_issues(query)` | read | existing bug issues |
| `create_issue(title, body, ticket_id)` | write | files a bug issue, returns `issue_id` |
| `add_ticket_comment(ticket_id, body, internal: bool)` | write | the reply (internal=false) or a note (internal=true) |
| `set_ticket_status(ticket_id, status)` | write | `open`, `pending_customer`, `escalated`, `resolved` |

Tool sets: knowledge_agent = search_kb, read_doc. billing_agent = search_kb, read_doc, get_account. bug_agent = search_kb, read_doc, search_issues, create_issue. resolve/escalate nodes use add_ticket_comment and set_ticket_status. review has no tools.

## 5. Data (shipped in the student repository)

```
data/kb/public/*.md        75 public help-center articles (kb-0001..kb-0075)
data/kb/internal/*.md      10 internal-only pages (kb-0101..kb-0110) (runbooks, refund approval policy, admin tools); indexed, never cited to a customer
data/tickets/dev.jsonl     about 200 labelled tickets for building and tuning
data/tickets/heldout.jsonl about 100 labelled tickets, never used until the final evaluation
data/eval/retrieval_dev.jsonl      about 80 queries with gold doc_ids
data/eval/retrieval_heldout.jsonl  about 40 queries with gold doc_ids
data/eval/injection_seed.jsonl     about 20 attack cases and 20 benign lookalikes; students extend it
helpdesk/                  the provided helpdesk app (FastAPI + SQLite), seeded from data/tickets and an accounts table; never modified by the student
```

Issue ids in the helpdesk are `ISS-1001` upward; seeded issues are ISS-1001 to ISS-1008.

Article front matter: `id`, `title`, `space` (`public` | `internal`), `labels`, `updated`, `version`. Bodies are heading-structured procedures with tables, code blocks, and exact strings (error codes `QS-1xxx`, config keys, command flags).

Ticket record: `ticket_id` (`T-1xxxx`), `created_at`, `channel`, `customer_email`, `plan`, `subject`, `body`, plus labels `type`, `urgency`, `expected_action` (`auto_resolve` | `escalate` | `create_issue`), `gold_doc_ids`, `key_facts`, `is_injection`, `injection_kind`.

Facts that recur across articles live in `data/facts.yaml` and must agree everywhere (API keys on Team plan rotate every 90 days; refunds within 14 days of a charge; rate limit 600 requests per minute per key; SSO on Business only; export takes up to 24 hours; and so on).

## 6. Stack and conventions

- Python 3.12. Packages pinned in `requirements.txt`: openai, langchain-core, langchain-openai, langgraph, langgraph-checkpoint-sqlite, langchain-mcp-adapters, mcp, langsmith, sentence-transformers, opensearch-py, pydantic, fastapi, uvicorn, httpx, python-dotenv, pyyaml, pytest.
- Model access is the student's choice: any OpenAI-compatible endpoint. Code reads `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (agents and classifier) and `LLM_MODEL_REVIEW` (reviewer and judge; may equal `LLM_MODEL`). The course route is Nebius Token Factory (students are invited to the TripleTen organization and create their own key; a personal account with the $1 trial is the fallback), taught in the Prompt Engineering chapter's first lesson. Pages name OpenRouter, OpenAI, and Anthropic as alternatives and never mandate one. Defaults in `.env.example`: Nebius with `Qwen/Qwen3-235B-A22B-Instruct-2507` for agents and `google/gemma-3-27b-it` for the reviewer and judge (verified 2026-10-02 on Nebius: check_provider 5/5, four dev tickets through the graph, judge_eval on four tickets, 28 tests; gpt-oss-120b ignores forced tool_choice and GLM/Kimi/Qwen3.5 return empty content for structured outputs, so they are out; the recorded lesson outputs still come from OpenRouter Claude models). Never print a key; never commit `.env`.
- Chat client: the `openai` SDK in raw-loop lessons; `langchain_openai.ChatOpenAI(base_url=..., api_key=..., model=...)` in LangChain/LangGraph lessons. Tool schemas: OpenAI function format in 11.1; `@tool` / Pydantic args in 11.2 onward. `with_structured_output` for classifier and review.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, local. One model for documents and queries.
- Vector store: OpenSearch 2.x single node in Docker Compose, security plugin disabled for local use, index `resolve-kb`, `knn_vector` dim 384, HNSW, plus a `text` field for BM25. Hybrid = reciprocal rank fusion in Python.
- Tracing and evals: LangSmith cloud (free tier), env `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=resolve`.
- Judge and faithfulness: provider-neutral `ask()` helper; claim-level faithfulness with the judge model; optional local NLI (`microsoft/deberta-large-mnli`) as the no-API path. Metrics: mean faithfulness, hallucination rate, judge rubric 1 to 5 on correctness, groundedness, completeness, action safety.
- Serving: FastAPI wrapper `POST /tickets/{id}/run` and `GET /health`, run in Compose next to OpenSearch and the helpdesk, same habits as P2.
- CI: GitHub Actions runs lint, unit tests, and the eval floors on a small fixed subset.

## 7. Student repository layout

```
resolve/
├── src/resolve/
│   ├── llm.py            client from env
│   ├── kb/               index.py, search.py (hybrid), chunk.py
│   ├── graph/            state.py, nodes/, agents.py, build.py
│   ├── guards/           input.py, output.py, injection.py
│   ├── mcp_server.py     the helpdesk MCP server (Task 3)
│   └── api.py            FastAPI wrapper (Task 5)
├── helpdesk/             provided
├── data/                 provided (see above)
├── configs/              retrieval.yaml, graph.yaml, judge.yaml
├── evals/                retrieval_eval.py, judge_eval.py, results/
├── security/             THREAT_MODEL.md, injection_suite.jsonl, redteam_findings.md
├── traces/               exported LangSmith traces
├── assets/
├── tests/
├── compose.yaml
├── .github/workflows/ci.yml
├── agent_spec.md
├── REPORT.md
├── README.md
└── requirements.txt
```

## 8. Tasks (about 10 hours each)

1. **Write the spec and lock the retrieval baseline.** agent_spec.md (7 sections), repo skeleton, OpenSearch up, KB indexed with heading chunks, dev and held-out query sets, recall@5 and MRR reported, REPORT.md section 1.
2. **Build the workflow skeleton.** State, input_guard (basic), classify, router, knowledge_agent as a real tool-calling loop with search_kb and read_doc, review (basic approve/revise), route, checkpointer, LangSmith on. One dev ticket of each type reaches the outcome the spec predicts. REPORT.md section 2.
3. **Give it hands: tools through MCP.** The helpdesk MCP server, langchain-mcp-adapters into the graph, billing_agent and bug_agent, interrupt before writes, idempotency key, replay test (one issue, not two), reviewer revision loop. REPORT.md section 3.
4. **Guard it and attack it.** Full input and output guards, injection detection node with a local classifier plus heuristics, injection suite run guards-off then guards-on, threat model, red-team pass with every finding dispositioned. REPORT.md section 4.
5. **Measure it, ship it, report it.** LangSmith dataset from held-out tickets, judge and faithfulness, calibration against hand labels, cost and latency per ticket, the graph behind FastAPI in Compose, CI gate with floors, REPORT.md sections 5 to 7, defense prep.

## 9. Core chapter pages (chapter "10.00 Project 4. Resolve: build the support agent")

01 Plan your path, 02 Meet Resolve: product, stakeholders, and the brief, 03 How the project is submitted and reviewed, 04 Meet the knowledge base and the ticket stream, 05 What you need on your laptop, 06 Task 1, 07 Task 2, 08 Task 3, 09 Task 4, 10 Task 5.

## 10. Theory chapter map (chapter links point at the P4 grouped view)

P4 grouped view URL: https://app.notion.com/p/d46dbed2cbb54cf0ba78904473592794?v=3b06ed1efc9381f8b9cc000cb9074844

| chapter | lessons (new marked *) |
|---|---|
| 10.1 Prompt Engineering | 01* Calling a model from Python; 02 Anatomy of an effective prompt; 03 Failure modes; 04 Structured reasoning; 05 Worked example on a support problem |
| 10.2 Retrieval and RAG | 01 Why retrieval; 02 Lexical retrieval; 03 Sentence embeddings; 04 Vector search and chunking; 05* The knowledge base as files; 06 OpenSearch on your laptop (rewrite); 07 Hybrid search and retrieval quality |
| 10.3 RAG Warm-Up Build | 01 Worked example on the Quillstack knowledge base (rewrite) |
| 11.1 Agent Fundamentals | 01 What is an agent; 02 Tool calling; 03 Orchestration patterns; 04 Worked example: a minimal agent |
| 11.2 LangChain & LangGraph | 01 Core concepts; 02 State machines and routing; 03* The agent loop as a graph; 04 Memory, persistence and human approval; 05* Specialists and a reviewer; 06 Worked example: the triage graph (rewrite) |
| 11.3 MCP: Tools for Agents | 01 What is MCP; 02* Writing your own MCP server; 03 Connecting MCP tools to the graph (rewrite); 04 The toolbelt end to end (rewrite) |
| 11.4 Running the Agent as a Service | 01* The graph behind an endpoint; 02* The stack in Compose; 03* Cost, latency and budgets |
| 11.5 LangSmith Observability | 01 Tracing; 02 Evals; 03 Monitoring (read-only) |
| 12.1 LLM-as-Judge & Hallucination Metrics | 01 Judge patterns; 02 Hallucination metrics; 03 Worked example: eval suite |
| 12.2 OWASP Threat Modeling | 01 Top 10; 02 Threat modeling Resolve; 03 Prompt injection |
| 12.3 Defenses & Red-Teaming | 01 Guardrails; 02 Injection detection; 03 Red-teaming; 04 Incident response |

House rules for every lesson: same company (Quillstack), same node and field names, no AWS, no tasks or exercises, define each term on first use, one "Working with AI" callout at most, chapter cross-references by chapter name only, install lines and pinned versions where code first appears.
