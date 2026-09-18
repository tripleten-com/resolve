"""Load the helpdesk MCP tools into the process and call them from synchronous graph nodes.

MultiServerMCPClient speaks to `python -m resolve.mcp_server` over stdio. MCP tools are async, and
the graph runs synchronously (SqliteSaver, interrupt), so one background event loop owns the MCP
session for the life of the process and `invoke_tool` runs a call on it and waits.
"""
from __future__ import annotations

import asyncio
import atexit
import os
import sys
import threading
from typing import Any

from langchain_core.tools import BaseTool

from resolve.config import ROOT, tools_config

_loop: asyncio.AbstractEventLoop | None = None
_session_cm = None
_tools: list[BaseTool] | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None:
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, name="resolve-mcp-loop", daemon=True).start()
        _loop = loop
    return _loop


def run_async(coro, timeout: float = 120.0) -> Any:
    """Run a coroutine on the background loop from any thread and return its result."""
    return asyncio.run_coroutine_threadsafe(coro, _ensure_loop()).result(timeout=timeout)


def server_connection() -> dict:
    cfg = tools_config().get("mcp_server", {})
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return {"transport": "stdio", "command": sys.executable,
            "args": ["-m", cfg.get("module", "resolve.mcp_server")], "env": env}


def load_helpdesk_tools() -> list[BaseTool]:
    """Start the MCP server once, open one session, and return its tools as LangChain tools."""
    global _session_cm, _tools
    with _lock:
        if _tools is not None:
            return _tools
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        name = tools_config().get("mcp_server", {}).get("name", "helpdesk")
        client = MultiServerMCPClient({name: server_connection()})

        async def _open():
            cm = client.session(name)
            session = await cm.__aenter__()
            return cm, await load_mcp_tools(session, server_name=name)

        _session_cm, _tools = run_async(_open(), timeout=60)
        atexit.register(close_helpdesk_tools)
        return _tools


def close_helpdesk_tools() -> None:
    global _session_cm, _tools
    cm, _session_cm, _tools = _session_cm, None, None
    if cm is not None and _loop is not None:
        try:
            run_async(cm.__aexit__(None, None, None), timeout=10)
        except Exception:  # noqa: BLE001
            pass


def invoke_tool(tool: BaseTool, args: dict) -> Any:
    """Call a LangChain tool from sync code: directly if it has a sync body, else on the loop."""
    if getattr(tool, "func", None) is not None:
        return tool.invoke(args)
    return run_async(tool.ainvoke(args))
