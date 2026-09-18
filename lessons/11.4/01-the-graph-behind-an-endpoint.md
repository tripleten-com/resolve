Priya's support engineers live in the helpdesk. They are never going to open a terminal, activate a virtual environment, and type `python -m resolve.run T-10083`. In **LangChain & LangGraph** the graph ran inside one Python process on your laptop, and that is a demo: it proves the graph works and nothing else. The support team's tools need an **endpoint**, a URL that any program can call with a ticket id and get a decision back. This lesson puts the compiled graph behind three FastAPI routes, the same way Project 2 put the Pulse model behind `/predict`.

<callout icon="📖" color="gray_bg">
	Every code block in this chapter runs from the Resolve repository with `PYTHONPATH=src` and the virtual environment built from `requirements.txt` (the packages this lesson touches are `fastapi==0.141.1`, `uvicorn==0.53.0`, `httpx==0.28.1`, and `langgraph==1.2.11`). OpenSearch and the helpdesk are up from `docker compose up -d opensearch helpdesk`, the knowledge base is indexed, and `.env` holds `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_MODEL_REVIEW`, and `HELPDESK_URL`. The model client reads its endpoint and model from `.env`, and any OpenAI-compatible provider works.
</callout>

## Three routes

A **route** is a URL pattern plus an HTTP method that FastAPI maps to one Python function. Resolve needs three. `POST /tickets/{id}/run` starts a run for a ticket that already exists in the helpdesk and returns the decision, or the write the run is waiting on if it paused. `POST /tickets/{id}/approve` takes a yes or no and resumes that paused run. `GET /health` says whether the service can actually do its job right now. Everything below is `src/resolve/api.py`, about 130 lines, shown in pieces.

## Load the graph once

The Project 2 rule holds: the expensive thing loads once per process, before the first request, in the lifespan hook.

```python
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException
from resolve.graph.build import build_graph

_state: dict = {"graph": None, "error": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["graph"] = await asyncio.to_thread(build_graph)
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{exc.__class__.__name__}: {exc}"
    yield

app = FastAPI(title="Resolve", version="1.0", lifespan=lifespan)

def _graph():
    if _state["graph"] is None:
        raise HTTPException(status_code=503, detail=f"graph not loaded: {_state['error']}")
    return _state["graph"]
```

`build_graph` does more than the model load did in Project 2. It registers the retrieval tools, starts the helpdesk MCP server as a subprocess and loads its tools, opens the SQLite checkpointer, and compiles the graph with the fixed node names from **LangChain & LangGraph**. All of that is synchronous and takes a few seconds, so it runs in `asyncio.to_thread`: the lifespan is an `async` function, and a blocking call inside it would stall the event loop that serves every other request. The same helper wraps every graph call in the handlers below, for the same reason.

The `try` is a choice. In Project 2 a missing model file failed the process loudly. Here the likely failure is a dependency that is slow or absent, and a process that stays up and answers 503 with `graph not loaded: ConnectError` is easier to diagnose from inside a container than one that exited before you could read its log. `_graph()` is the guard every handler calls first.

## One thread per ticket

Recall from **LangChain & LangGraph** that a **checkpointer** saves the graph's state after every node, filed under a **thread id**, and that `interrupt()` is only possible because of it. The service does not invent a thread id. `run_config` in `src/resolve/tracing.py` uses the ticket id:

```python
def run_config(ticket_id: str, ticket_type: str | None = None, **configurable) -> dict:
    tags = [ticket_id] + ([ticket_type] if ticket_type else [])
    return {
        "configurable": {"thread_id": ticket_id, **configurable},
        "tags": tags,
        "metadata": {"ticket_id": ticket_id, "ticket_type": ticket_type or "unknown"},
        "run_name": f"resolve:{ticket_id}",
    }
```

That one line, `"thread_id": ticket_id`, is what makes the approve route possible. A **session**, in HTTP terms, is whatever the server remembers about a client between two requests. This service keeps no session table of its own. The checkpointer is the session: an approval that arrives twenty minutes later, from a different browser, finds the paused run by the ticket id alone, and because the checkpoints live in `var/checkpoints.db` on disk, a restart of the process loses nothing. The `approval` key travels in `configurable` too, so the nodes know whether to pause (`human`), execute without asking (`auto`), or record and skip (`dry_run`).

## Starting a run

```python
from resolve.graph.build import is_paused, run_ticket
from resolve.tickets import load_ticket

@app.post("/tickets/{ticket_id}/run")
async def run(ticket_id: str):
    graph = _graph()
    try:
        ticket = await asyncio.to_thread(load_ticket, ticket_id, True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    t0 = time.perf_counter()
    state = await asyncio.to_thread(run_ticket, graph, ticket_id, ticket, "human")
    out = _public(state)
    out["http_ms"] = int((time.perf_counter() - t0) * 1000)
    return out
```

`load_ticket(ticket_id, True)` asks the helpdesk first and falls back to the labelled files, so the service runs on what the support team sees, and an unknown id is a 404 with the reason. `run_ticket` invokes the graph under the ticket's thread in `human` mode, which means every write tool pauses. The clock starts after the ticket is loaded and stops when the graph returns, so `http_ms` measures the graph call and nothing else, the same discipline as `latency_ms` in the Project 2 response.

`_public` picks the fields a client may see: `ticket_id`, `paused`, `pending_write`, `decision`, `escalation_reason`, `classification`, `confidence`, `citations`, `review`, both guard results, `draft`, and `usage`. One detail is deliberate: `decision` is `None` while the run is paused, so no client can mistake a paused run for a finished one and act on a decision the graph has not made.

## Approving a write

```python
from langgraph.types import Command
from pydantic import BaseModel
from resolve.graph.build import pending_write, resume_ticket

class Approval(BaseModel):
    approved: bool

@app.post("/tickets/{ticket_id}/approve")
async def approve(ticket_id: str, body: Approval):
    graph = _graph()
    pending = await asyncio.to_thread(pending_write, graph, ticket_id)
    if pending is None:
        raise HTTPException(status_code=409, detail=f"{ticket_id} has no write waiting for approval")
    state = await asyncio.to_thread(resume_ticket, graph, ticket_id, body.approved, "human", pending)
    out = _public(state)
    out["approved_write"] = pending
    return out
```

`pending_write` reads `graph.get_state(config).interrupts` for the ticket's thread and returns the payload the node handed to `interrupt()`: the tool name, its arguments, the idempotency key, and a one-line summary. If there is none, the answer is 409, the status code for "this request conflicts with the current state": nothing is waiting, so there is nothing to approve. `resume_ticket` calls `graph.invoke(Command(resume={"approved": ..., "call": pending}), config)`. A **Command** is how langgraph resumes a paused run: the node that called `interrupt()` runs again from its start and this time `interrupt()` returns the resume value instead of pausing. The value carries the exact call the human saw, so the write that executes is the approved one with the approved key, whatever the model might propose the second time round. A how-to that auto-resolves pauses twice, once for the reply and once for the status; a bug ticket pauses three times, because the issue comes first.

## An honest health route

```python
@app.get("/health")
async def health():
    checks = {
        "llm": await _check_llm(),
        "opensearch": await _check_opensearch(),
        "helpdesk": await _check_url(helpdesk_url() + "/health"),
        "graph": None if _state["graph"] is not None else (_state["error"] or "not loaded"),
    }
    ok = all(v is None for v in checks.values())
    body = {"status": "ok" if ok else "degraded",
            "checks": {k: ("ok" if v is None else v) for k, v in checks.items()}}
    return JSONResponse(body, status_code=200 if ok else 503)
```

Each check returns `None` when it passes and a short reason when it fails. `_check_llm` calls `models.list()` on the shared OpenAI client with a ten-second timeout, which costs nothing and proves the key and the endpoint. `_check_url` fetches the helpdesk's own `/health` with `trust_env=False`, so a proxy variable on the machine cannot redirect the probe. `_check_opensearch` is the one worth reading:

```python
async def _check_opensearch() -> str | None:
    from resolve.kb.index import get_client, load_config

    def probe() -> str | None:
        client = get_client()
        status = client.cluster.health(request_timeout=5).get("status")
        if status not in ("green", "yellow"):
            return f"cluster status {status}"
        index = load_config().get("index", "resolve-kb")
        if not client.indices.exists(index=index):
            return f"index {index} missing"
        return None

    try:
        return await asyncio.to_thread(probe)
    except Exception as exc:  # noqa: BLE001
        return f"{exc.__class__.__name__}: {str(exc)[:80]}"
```

It uses the same `opensearch-py` client the search tool uses, asks for cluster health rather than an open socket, and then checks that `resolve-kb` exists. A cluster that is up with no index would pass a port check and fail every ticket. The rule from Project 2 stands: a health route that returns 200 because the process is alive is worse than none, because it tells the container runtime and the humans reading it that everything is fine.

## Everything from the environment

`api.py` never opens `.env`. `resolve.config` loads it once at import, and each module reads the variable it needs: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and `LLM_MODEL_REVIEW` in `llm.py`; `OPENSEARCH_URL` in the index module; `HELPDESK_URL` in `tickets.py` and the MCP server; `LANGSMITH_TRACING` and its key in `tracing.py`. No hostname, port, key, or model name appears in the code. That is what lets the same file run under uvicorn on your laptop now and inside a container with different hostnames in the next lesson.

## Run it

```bash
uvicorn resolve.api:app --port 8200
```

The log shows the MCP server listing its tools, then `Application startup complete`. In another terminal:

```bash
curl -s localhost:8200/health
```

```json
{"status":"ok","checks":{"llm":"ok","opensearch":"ok","helpdesk":"ok","graph":"ok"}}
```

Now a dev ticket. T-10083 is a two-line how-to, "How do I connect our Slack workspace to Quillstack?", open in the helpdesk with no comments.

```bash
curl -s -w '\nround trip: %{time_total}s\n' -X POST localhost:8200/tickets/T-10083/run
```

```json
{"ticket_id": "T-10083", "paused": true, "pending_write": {"tool": "add_ticket_comment",
  "args": {"ticket_id": "T-10083", "body": "Hi Ines,\n\nGreat question! Here's how to connect your Slack workspace to Quillstack:\n\n1. In Quillstack, go to Settings > Integrations > Slack.\n2. Click Connect to Slack. ...", "internal": false},
  "idempotency_key": "T-10083:add_ticket_comment:7e616bde", "ticket_id": "T-10083", "node": "resolve",
  "summary": "add_ticket_comment: post a public reply on T-10083: ..."},
 "decision": null,
 "classification": {"type": "how_to", "urgency": "normal", "aimed_at_agent": false, "confidence": 0.95},
 "confidence": 0.98, "citations": ["kb-0057"],
 "review": {"verdict": "approve", "reasons": [], "unsupported_claims": [], "revision_count": 1},
 "guard_output": {"flags": [], "ok": true},
 "usage": {"input_tokens": 19563, "output_tokens": 1295, "cost_usd": 0.038766, "latency_ms": 22323},
 "http_ms": 22352}
round trip: 22.746230s
```

Read it top down. The run paused in `resolve`, the exit node, on the public reply. The reviewer sent the first draft back once (`revision_count` is 1) and approved the second, which cites kb-0057, the Slack integration article. `decision` is `null` because nothing has been written yet. The whole run cost about four cents.

Approve the reply, then the status:

```bash
curl -s -X POST localhost:8200/tickets/T-10083/approve -H 'content-type: application/json' -d '{"approved": true}'
```

```json
{"paused": true, "decision": null, "approved_write": {"tool": "add_ticket_comment", "idempotency_key": "T-10083:add_ticket_comment:7e616bde", ...},
 "pending_write": {"tool": "set_ticket_status", "args": {"ticket_id": "T-10083", "status": "resolved"}, "idempotency_key": "T-10083:set_ticket_status:8d31fdc9", ...}}
```

The same call again answers `"paused": false, "decision": "auto_resolve"`, and a third call answers 409 with `T-10083 has no write waiting for approval`. The helpdesk agrees: `curl -s localhost:${HELPDESK_PORT}/tickets/T-10083` now shows status `resolved` and one public comment from `resolve`, the reply you saw in `pending_write`. Nothing changed in the helpdesk until a human said yes, and every write carried its key.

## The round trip and the latency in the state

Three clocks ran on that ticket. curl measured 22,746 ms from request to last byte. `http_ms`, started after the ticket was loaded, read 22,352 ms. `usage.latency_ms`, the sum the nodes themselves recorded, read 22,323 ms. The graph call and the graph's own accounting agree to about thirty milliseconds, which is the whole cost of the FastAPI layer: JSON in, thread hand-off, JSON out. The extra four hundred milliseconds curl saw are the helpdesk lookup before the clock started and the response on the way back.

The lesson in those numbers is where not to optimise. A support engineer waiting twenty seconds is waiting for four model calls and a reviewer, and the endpoint adds nothing you could measure without a stopwatch. The exception is the first ticket after startup: the injection classifier from **Defenses & Red-Teaming** loads lazily on the first `input_guard`, and that first run carries about ten extra seconds. A quarantined security ticket such as T-10007 shows both facts at once through this same route: `cost_usd` 0.0 and, on a warm process, well under a second.

<callout icon="🤖" color="blue_bg">
	**Working with AI**
	An assistant will happily generate this file, and the version it produces usually has two flaws: `/health` returns 200 as soon as the process is up, and the graph is built inside the handler on every request. Ask it instead to review `api.py` against three questions: what is loaded once, what each route does when nothing is pending, and what `/health` would say with OpenSearch stopped. Then stop OpenSearch and check the third answer yourself.
</callout>

The next lesson, **The Stack in Compose**, builds the image for this service, adds it to `compose.yaml` next to OpenSearch and the helpdesk, and proves the key is never in the image.
