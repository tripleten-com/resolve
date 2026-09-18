"""knowledge_agent: the shared loop with the how-to prompt and the two retrieval tools."""
from __future__ import annotations

from resolve.graph.agents import run_agent
from resolve.graph.prompts import KNOWLEDGE_AGENT


def knowledge_agent(state: dict) -> dict:
    return run_agent("knowledge_agent", KNOWLEDGE_AGENT, state)
