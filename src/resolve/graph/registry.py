"""The tools the graph loaded at startup, by name. build.py fills it; agents.py reads it."""
from __future__ import annotations

from langchain_core.tools import BaseTool

TOOLS: dict[str, BaseTool] = {}


def register(tools: list[BaseTool]) -> None:
    for t in tools:
        TOOLS[t.name] = t


def get_tools(names: list[str]) -> list[BaseTool]:
    missing = [n for n in names if n not in TOOLS]
    if missing:
        raise KeyError(f"tools not loaded: {missing}; loaded: {sorted(TOOLS)}")
    return [TOOLS[n] for n in names]
