"""The graph behind an HTTP endpoint.

    uvicorn resolve.api:app --port 8200

    POST /tickets/{id}/run       run a ticket that exists in the helpdesk; returns the decision, or the pending write
    POST /tickets/{id}/approve   {"approved": true|false} resumes a paused run
    GET  /health                 honest: model endpoint, search engine, helpdesk, and the graph, each checked

The graph loads once at startup. Every configuration value comes from the environment.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from resolve.graph.build import build_graph, is_paused, pending_write, resume_ticket, run_ticket
from resolve.tickets import helpdesk_url, load_ticket

_state: dict = {"graph": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["graph"] = await asyncio.to_thread(build_graph)
    except Exception as exc:  # noqa: BLE001
        _state["error"] = f"{exc.__class__.__name__}: {exc}"
    yield


app = FastAPI(title="Resolve", version="1.0", lifespan=lifespan)


class Approval(BaseModel):
    approved: bool


def _graph():
    if _state["graph"] is None:
        raise HTTPException(status_code=503, detail=f"graph not loaded: {_state['error']}")
    return _state["graph"]


def _public(state: dict) -> dict:
    return {
        "ticket_id": state.get("ticket_id"),
        "paused": is_paused(state),
        "pending_write": state.get("pending_write"),
        "decision": None if is_paused(state) else state.get("decision"),
        "escalation_reason": state.get("escalation_reason"),
        "classification": state.get("classification"),
        "confidence": state.get("confidence"),
        "citations": state.get("citations"),
        "review": state.get("review"),
        "guard_input": state.get("guard_input"),
        "guard_output": state.get("guard_output"),
        "draft": state.get("draft"),
        "usage": state.get("usage"),
    }


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


async def _check_llm() -> str | None:
    from resolve.llm import client
    try:
        await asyncio.to_thread(lambda: client().with_options(timeout=10).models.list())
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{exc.__class__.__name__}: {str(exc)[:120]}"


async def _check_url(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as c:
            r = await c.get(url)
        return None if r.status_code < 400 else f"HTTP {r.status_code}"
    except httpx.HTTPError as exc:
        return f"{exc.__class__.__name__}"


async def _check_opensearch() -> str | None:
    """Use the same client the index and the search tool use; the cluster must answer, not just accept a connection."""
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
