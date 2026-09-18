"""bug_agent: the shared loop with the bug prompt, search_issues, and create_issue (a write)."""
from __future__ import annotations

from resolve.graph.agents import run_agent
from resolve.graph.prompts import BUG_AGENT


def bug_agent(state: dict) -> dict:
    return run_agent("bug_agent", BUG_AGENT, state)
