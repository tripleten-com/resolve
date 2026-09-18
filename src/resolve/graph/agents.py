"""The tool-calling loop every specialist shares, and the approval pause in front of every write.

    run_agent("knowledge_agent", KNOWLEDGE_AGENT, state)  -> partial state update

The loop: send the conversation to the model with the tools attached; if the model asks for a
tool, run it, append the result, go round again; if the model answers, stop. Capped at
`max_steps` from configs/graph.yaml. When the model asks for a tool on the writes list, the loop
does not run it: it calls `interrupt()` with the proposed call, the checkpointer saves the state,
and the run stops until `python -m resolve.approve` (or the API) resumes it with a yes or no.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_config
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from resolve.config import graph_config, model_for, tools_config
from resolve.graph import registry
from resolve.graph.prompts import REVISION_NOTE, spotlight_ticket, spotlight_tool_result
from resolve.graph.usage import Meter
from resolve.guards import guards_enabled
from resolve.guards.injection import screen_chunks, screen_text
from resolve.llm import chat
from resolve.tools.helpdesk import invoke_tool
from resolve.tools.idempotency import idempotency_key

MAX_RESULT_CHARS = 7000
DOC_ID = re.compile(r"\bkb-\d{4}\b")


class FinalAnswer(BaseModel):
    draft: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# --- config helpers -----------------------------------------------------------

def approval_mode() -> str:
    """human (interrupt), auto (execute without asking), or dry_run (record, never execute)."""
    try:
        return get_config().get("configurable", {}).get("approval", "human")
    except Exception:  # noqa: BLE001  (outside a graph run)
        return "human"


def tool_scope(agent: str) -> list[str]:
    return list(tools_config().get("agents", {}).get(agent, []))


def write_tools() -> set[str]:
    return set(tools_config().get("writes", []))


def describe_write(name: str, args: dict) -> str:
    """The pending write in plain words, for the human."""
    tid = args.get("ticket_id", "?")
    if name == "create_issue":
        return f"create_issue: file a bug issue titled {args.get('title', '')!r} for ticket {tid}"
    if name == "add_ticket_comment":
        kind = "an internal note" if args.get("internal") else "a public reply"
        body = " ".join(str(args.get("body", "")).split())
        return f"add_ticket_comment: post {kind} on {tid}: {body[:160]!r}{'...' if len(body) > 160 else ''}"
    if name == "set_ticket_status":
        return f"set_ticket_status: set {tid} to {args.get('status')!r}"
    return f"{name}: {json.dumps(args)[:200]}"


# --- writes: propose, pause, execute -------------------------------------------

def execute_write(name: str, args: dict, state: dict, node: str) -> tuple[dict, dict]:
    """Propose a write, pause for a human (or auto/dry-run per config), execute it once if approved.

    Returns (result for the model, tool_calls entry). The resume value is either a bool or
    {"approved": bool, "call": <the write the human saw>}; when a call is given it is the one
    executed, so a replayed resume executes exactly the approved write with the same key.
    """
    tool = registry.TOOLS.get(name)
    ticket_id = state["ticket_id"]
    if tool is not None and "ticket_id" in (tool.args or {}):
        args = {**args, "ticket_id": ticket_id}  # a write may only ever touch this run's own ticket
    key = idempotency_key(ticket_id, name, args)
    proposed = {"tool": name, "args": args, "idempotency_key": key, "ticket_id": ticket_id,
                "node": node, "summary": describe_write(name, args)}
    mode = approval_mode()
    if mode == "auto":
        decision: dict = {"approved": True, "call": proposed}
    elif mode == "dry_run":
        decision = {"approved": False, "call": proposed, "dry_run": True}
    else:
        decision = interrupt(proposed)
    if isinstance(decision, bool):
        decision = {"approved": decision, "call": proposed}
    call = decision.get("call") or proposed
    entry = {"name": call["tool"], "args": call["args"], "idempotency_key": call["idempotency_key"],
             "write": True, "node": node, "confirmed": False, "attempted": True, "result_summary": ""}

    if decision.get("dry_run"):
        result = {"dry_run": True, "message": f"{name} recorded but not executed (evaluation mode).",
                  "issue_id": "ISS-DRYRUN" if name == "create_issue" else None}
        entry["result_summary"] = "dry run: not executed"
    elif decision.get("approved"):
        if tool is None:
            result = {"error": f"tool {name} is not loaded"}
        else:
            call_args = dict(call["args"])
            if "idempotency_key" in (tool.args or {}):
                call_args["idempotency_key"] = call["idempotency_key"]
            result = _parse_result(safe_invoke(tool, call_args))
        entry["confirmed"] = True
        entry["result_summary"] = summarize(result)
    else:
        result = {"declined": True, "message": "A human reviewer declined this write. Continue without it and say "
                                               "in your reply that a support engineer will handle it."}
        entry["result_summary"] = "declined by human"
    return result, entry


def safe_invoke(tool: BaseTool, args: dict):
    try:
        return invoke_tool(tool, args)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{tool.name} failed: {exc.__class__.__name__}: {str(exc)[:200]}"}


def summarize(result, n: int = 600) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _parse_result(result):
    """MCP tools return JSON text; make it a dict again when possible."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except ValueError:
            return result
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        try:
            return json.loads(result[0]["text"])
        except (ValueError, TypeError):
            return result
    return result


# --- the final answer ---------------------------------------------------------

def parse_final(content: str, docs: list[dict]) -> FinalAnswer:
    text = (content or "").strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            return FinalAnswer.model_validate_json(c)
        except (ValidationError, ValueError):
            continue
    known = {d["doc_id"] for d in docs}
    cites = [c for c in dict.fromkeys(DOC_ID.findall(text)) if c in known]
    return FinalAnswer(draft=text or "No draft was produced.", citations=cites, confidence=0.3 if text else 0.0)


# --- the loop -----------------------------------------------------------------

def run_agent(agent: str, system_prompt: str, state: dict,
              call_check: Callable[[str, dict, dict], str | None] | None = None) -> dict:
    """Run one specialist over the ticket. Returns the fields it changed."""
    cfg = graph_config()
    max_steps = int(cfg.get("max_steps", 6))
    model = model_for("specialists")
    max_tokens = int(cfg.get("max_tokens", {}).get("specialist", 900))
    tools = registry.get_tools(tool_scope(agent))
    by_name = {t.name: t for t in tools}
    schemas = [convert_to_openai_tool(t) for t in tools]
    writes = write_tools()
    meter = Meter()

    docs: list[dict] = list(state.get("docs") or [])
    seen_chunks = {d.get("chunk_id") for d in docs}
    tool_calls: list[dict] = list(state.get("tool_calls") or [])

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": spotlight_ticket(state["ticket"], state.get("ticket_id"))}]
    review = state.get("review") or {}
    if review.get("verdict") == "revise" and state.get("draft"):
        messages.append({"role": "assistant", "content": json.dumps(
            {"draft": state["draft"], "citations": state.get("citations", []), "confidence": state.get("confidence", 0)})})
        messages.append({"role": "user", "content": REVISION_NOTE.format(
            reasons="; ".join(review.get("reasons", [])) or "none given",
            claims="; ".join(review.get("unsupported_claims", [])) or "none listed")})

    final: FinalAnswer | None = None
    for _step in range(max_steps):
        resp = chat(messages, model=model, tools=schemas, max_tokens=max_tokens)
        meter.add(resp, model)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = parse_final(msg.content or "", docs)
            break
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            entry = {"name": name, "args": args, "result_summary": "", "confirmed": True,
                     "idempotency_key": None, "write": False, "node": agent}
            refusal = None
            if name not in by_name:
                refusal = f"tool {name!r} is not available to {agent}"
            elif call_check is not None:
                refusal = call_check(name, args, state)
            if refusal:
                result = {"error": refusal}
                entry["confirmed"] = False
                entry["result_summary"] = f"refused: {refusal}"
            elif name in writes:
                result, entry = execute_write(name, args, state, agent)
            else:
                result = _parse_result(safe_invoke(by_name[name], args))
                result, dropped = screen_result(name, result)
                if dropped:
                    entry["dropped"] = dropped
                if name == "search_kb" and isinstance(result, list):
                    for chunk in result:
                        if isinstance(chunk, dict) and chunk.get("chunk_id") not in seen_chunks:
                            seen_chunks.add(chunk.get("chunk_id"))
                            docs.append({k: chunk.get(k) for k in ("doc_id", "chunk_id", "title", "heading", "space", "text", "score")})
                entry["result_summary"] = summarize(result) + (f" [{dropped} result(s) dropped: injection risk]" if dropped else "")
            tool_calls.append(entry)
            text = spotlight_tool_result(name, result)
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + "\n[truncated]\n<<<END TOOL_RESULT>>>"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})

    if final is None:
        final = FinalAnswer(
            draft=f"I could not finish researching this ticket within {max_steps} steps. A support engineer will "
                  "follow up with a full answer.", citations=[], confidence=0.0)
    return {"docs": docs, "draft": final.draft, "citations": list(dict.fromkeys(final.citations)),
            "confidence": float(final.confidence), "tool_calls": tool_calls, "pending_write": None,
            "usage": meter.merge(state)}


def screen_result(name: str, result):
    """Run the injection detector over what a tool returned; drop what scores above the threshold."""
    if not guards_enabled():
        return result, 0
    if name == "search_kb" and isinstance(result, list):
        kept, dropped = screen_chunks([c for c in result if isinstance(c, dict)])
        return kept, len(dropped)
    if name == "read_doc" and isinstance(result, dict) and result.get("body"):
        if screen_text(result["body"]):
            return {"error": "article withheld: its text scored above the injection threshold"}, 1
        return result, 0
    if isinstance(result, dict) and name in ("search_issues", "get_ticket"):
        blob = json.dumps(result, ensure_ascii=False)
        if screen_text(blob):
            return {"error": f"{name} result withheld: it scored above the injection threshold"}, 1
    return result, 0
