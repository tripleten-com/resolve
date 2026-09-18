"""LangSmith tracing: on when LANGSMITH_TRACING=true, silently off otherwise.

LangGraph and the wrapped OpenAI client both read LANGSMITH_TRACING themselves, so this
module only adds the per-run tags and metadata and answers "is tracing on?".
"""
from __future__ import annotations

import os


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes"}


def wrap_llm_client() -> bool:
    """Wrap the shared OpenAI client so every model call inside a node shows up under the node's
    trace. A no-op unless tracing is on. Returns whether the client was wrapped."""
    if not tracing_enabled():
        return False
    from langsmith.wrappers import wrap_openai

    import resolve.llm as llm
    client = llm.client()
    if not getattr(client, "_resolve_traced", False):
        wrapped = wrap_openai(client)
        wrapped._resolve_traced = True
        llm._client = wrapped
    return True


def run_config(ticket_id: str, ticket_type: str | None = None, **configurable) -> dict:
    """The RunnableConfig for one ticket: thread_id = ticket_id, plus tags a trace can be found by."""
    tags = [ticket_id] + ([ticket_type] if ticket_type else [])
    return {
        "configurable": {"thread_id": ticket_id, **configurable},
        "tags": tags,
        "metadata": {"ticket_id": ticket_id, "ticket_type": ticket_type or "unknown"},
        "run_name": f"resolve:{ticket_id}",
    }
