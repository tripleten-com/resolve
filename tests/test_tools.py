"""Tool tests: the MCP server's idempotent writes, tool scoping, the billing check, and the replay
test (resume an approved run twice, expect one issue in the helpdesk).

The replay test needs the helpdesk (HELPDESK_URL) and a model key; it is skipped when either is
missing so the unit tests still run offline.
"""
from __future__ import annotations

import os
import sqlite3
import uuid

import httpx
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from resolve.config import tools_config
from resolve.graph.agents import tool_scope, write_tools
from resolve.graph.nodes.billing_agent import check_billing_call
from resolve.tickets import helpdesk_url

HD = helpdesk_url()


def helpdesk_up() -> bool:
    try:
        return httpx.get(f"{HD}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


needs_helpdesk = pytest.mark.skipif(not helpdesk_up(), reason=f"helpdesk not reachable at {HD}")
needs_model = pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set")


def test_tool_scopes_match_tools_yaml():
    cfg = tools_config()
    assert tool_scope("knowledge_agent") == ["search_kb", "read_doc"]
    assert set(tool_scope("billing_agent")) == {"search_kb", "read_doc", "get_account"}
    assert set(tool_scope("bug_agent")) == {"search_kb", "read_doc", "search_issues", "create_issue"}
    assert write_tools() == {"create_issue", "add_ticket_comment", "set_ticket_status"}
    for agent, tools in cfg["agents"].items():
        assert "set_ticket_status" not in tools and "add_ticket_comment" not in tools, agent


def test_billing_lookup_limited_to_own_customer():
    state = {"ticket": {"customer_email": "Uma@QuartzPay.io"}}
    assert check_billing_call("get_account", {"email": "uma@quartzpay.io"}, state) is None
    assert check_billing_call("get_account", {"email": "dana@pinecone-health.org"}, state).startswith("refused")
    assert check_billing_call("search_kb", {"query": "x"}, state) is None


@needs_helpdesk
def test_mcp_tools_list_and_idempotent_create_issue():
    from resolve.tools.helpdesk import invoke_tool, load_helpdesk_tools
    tools = {t.name: t for t in load_helpdesk_tools()}
    assert set(tools) == {"get_ticket", "get_account", "search_issues", "create_issue", "add_ticket_comment", "set_ticket_status"}
    bad = invoke_tool(tools["set_ticket_status"], {"ticket_id": "T-10001", "status": "closed"})
    assert "error" in str(bad) and "422" in str(bad)
    key = f"T-TEST:create_issue:{uuid.uuid4().hex[:8]}"
    args = {"title": f"test issue {key}", "body": "replay check", "ticket_id": "T-TEST", "idempotency_key": key}
    first, second = invoke_tool(tools["create_issue"], args), invoke_tool(tools["create_issue"], args)
    assert '"created": true' in str(first) and '"created": false' in str(second)
    hits = httpx.get(f"{HD}/issues", params={"q": key}).json()
    assert len(hits) == 1


@needs_helpdesk
@needs_model
def test_replay_of_an_approved_run_files_one_issue():
    """Run a bug ticket, approve the issue, then resume the same checkpoint again as if the process
    had crashed after approval. One issue, not two."""
    from resolve.graph.build import build_graph, is_paused, resume_ticket, run_ticket
    graph = build_graph(checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)))
    ticket_id = f"T-REPLAY-{uuid.uuid4().hex[:6]}"
    code = f"QS-{uuid.uuid4().int % 9000 + 1000}"  # a fresh error code, so no earlier test run's issue matches
    ticket = {
        "subject": f"Export fails with {code} for our 4,000-page workspace",
        "body": f"Since Tuesday every workspace export from Settings > Export fails within a minute with {code}. "
                "It worked last week; nothing changed on our side. Team plan, about 4,000 pages, 3 GB of attachments. "
                "Retried four times, same error each time.",
        "customer_email": "ops@brightfield.io", "plan": "Team", "channel": "web", "created_at": "2026-09-17T09:00:00Z",
    }
    state = run_ticket(graph, ticket_id, ticket=ticket, approval="human", ticket_type="bug")
    assert is_paused(state), "the bug agent did not propose a write"
    pending = state["pending_write"]
    assert pending["tool"] == "create_issue", pending
    assert pending["idempotency_key"].startswith(f"{ticket_id}:create_issue:")
    cfg = {"configurable": {"thread_id": ticket_id}}
    paused_checkpoint = graph.get_state(cfg).config  # the checkpoint the human approved from

    after_first = resume_ticket(graph, ticket_id, True, pending=pending)
    calls = [c for c in after_first["tool_calls"] if c["name"] == "create_issue"]
    assert calls and calls[0]["confirmed"] and '"created": true' in calls[0]["result_summary"]

    # crash-and-restart: resume from the same checkpoint again with the same approval
    graph.invoke(Command(resume={"approved": True, "call": pending}), paused_checkpoint)

    issues = [i for i in httpx.get(f"{HD}/issues").json() if i.get("ticket_id") == ticket_id]
    assert len(issues) == 1, [i["issue_id"] for i in issues]
    assert issues[0]["idempotency_key"] == pending["idempotency_key"]
