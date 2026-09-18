"""Assemble the graph: the fixed node names, the conditional edges, the checkpointer, the tools.

    from resolve.graph.build import get_graph, run_ticket, resume_ticket, pending_write
    graph = get_graph()
    state = run_ticket(graph, "T-10004")          # pauses before the first write
    state = resume_ticket(graph, "T-10004", True)  # executes it once

    input_guard -> classify -> [router] -> knowledge_agent | billing_agent | bug_agent | escalate
    specialist -> review -> (revise once -> same specialist) -> output_guard -> route -> resolve | escalate
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from resolve.config import ROOT, graph_config
from resolve.graph import registry
from resolve.graph.nodes import (
    billing_agent,
    bug_agent,
    classify,
    escalate,
    input_guard,
    knowledge_agent,
    output_guard,
    resolve,
    review,
    route,
)
from resolve.graph.nodes.router import after_classify, after_input_guard, after_review, after_route
from resolve.graph.state import TicketState, initial_state
from resolve.tickets import load_ticket
from resolve.tools.kb import KB_TOOLS
from resolve.tracing import run_config, wrap_llm_client

_graph = None


def default_checkpointer() -> SqliteSaver:
    path = ROOT / graph_config().get("checkpoint_path", "var/checkpoints.db")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def load_tools(with_helpdesk: bool = True) -> None:
    """Retrieval tools from Python, helpdesk tools from the MCP server, into one registry."""
    registry.register(KB_TOOLS)
    if with_helpdesk:
        from resolve.tools.helpdesk import load_helpdesk_tools
        registry.register(load_helpdesk_tools())


def build_graph(checkpointer=None, with_helpdesk: bool = True):
    wrap_llm_client()  # LangSmith: on when LANGSMITH_TRACING=true, silently off otherwise
    load_tools(with_helpdesk)
    g = StateGraph(TicketState)
    for name, fn in [("input_guard", input_guard), ("classify", classify), ("knowledge_agent", knowledge_agent),
                     ("billing_agent", billing_agent), ("bug_agent", bug_agent), ("review", review),
                     ("output_guard", output_guard), ("route", route), ("resolve", resolve), ("escalate", escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", after_input_guard, {"classify": "classify", "escalate": "escalate"})
    g.add_conditional_edges("classify", after_classify, {
        "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent", "bug_agent": "bug_agent",
        "escalate": "escalate"})
    for specialist in ("knowledge_agent", "billing_agent", "bug_agent"):
        g.add_edge(specialist, "review")
    g.add_conditional_edges("review", after_review, {
        "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent", "bug_agent": "bug_agent",
        "output_guard": "output_guard"})
    g.add_edge("output_guard", "route")
    g.add_conditional_edges("route", after_route, {"resolve": "resolve", "escalate": "escalate"})
    g.add_edge("resolve", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer or default_checkpointer(), name="resolve")


def get_graph():
    """The graph, built once per process."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _strip(result: dict) -> dict:
    interrupts = result.get("__interrupt__") or []
    state = {k: v for k, v in result.items() if k != "__interrupt__"}
    if interrupts:
        state["pending_write"] = interrupts[0].value
    return state


def run_ticket(graph, ticket_id: str, ticket: dict | None = None, approval: str = "human",
               ticket_type: str | None = None, prefer_helpdesk: bool = False) -> dict:
    """Run one ticket under thread_id = ticket_id. approval: human | auto | dry_run."""
    ticket = ticket or load_ticket(ticket_id, prefer_helpdesk=prefer_helpdesk)
    cfg = run_config(ticket_id, ticket_type, approval=approval)
    return _strip(graph.invoke(initial_state(ticket_id, ticket), cfg))


def pending_write(graph, ticket_id: str) -> dict | None:
    """The write the paused run is waiting on, or None."""
    snap = graph.get_state({"configurable": {"thread_id": ticket_id}})
    for intr in snap.interrupts:
        return intr.value
    return None


def resume_ticket(graph, ticket_id: str, approved: bool, approval: str = "human",
                  pending: dict | None = None, config: dict | None = None) -> dict:
    """Resume a paused run with the human's answer. The approved call is the one the human saw,
    so a replayed resume executes exactly that write with the same idempotency key."""
    pending = pending or pending_write(graph, ticket_id)
    cfg = config or run_config(ticket_id, approval=approval)
    return _strip(graph.invoke(Command(resume={"approved": bool(approved), "call": pending}), cfg))


def is_paused(state: dict) -> bool:
    return bool(state.get("pending_write"))
