print("=== A: add_messages + MemorySaver ===")
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # appends
    ticket_id: str                                         # overwrites

def echo(state: ChatState) -> dict:
    n = len(state["messages"])
    return {"messages": [AIMessage(f"I have seen {n} messages on {state['ticket_id']}")]}

builder = StateGraph(ChatState)
builder.add_node("echo", echo)
builder.add_edge(START, "echo")
builder.add_edge("echo", END)
graph = builder.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "T-10018"}}
out = graph.invoke({"ticket_id": "T-10018", "messages": [HumanMessage("two identical charges on August 4th")]}, cfg)
print(out["messages"][-1].content)
out = graph.invoke({"ticket_id": "T-10018", "messages": [HumanMessage("any update?")]}, cfg)
print(out["messages"][-1].content)
out = graph.invoke({"ticket_id": "T-10006", "messages": [HumanMessage("refund please")]},
                   {"configurable": {"thread_id": "T-10006"}})
print(out["messages"][-1].content)

print("=== B: SqliteSaver ===")
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

path = Path("scratch/11.2/checkpoints.db")
path.parent.mkdir(parents=True, exist_ok=True)
saver = SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
graph = builder.compile(checkpointer=saver)
graph.invoke({"ticket_id": "T-10018", "messages": [HumanMessage("two identical charges")]}, cfg)
print(graph.get_state(cfg).values["messages"][-1].content)
print(path.exists(), path.stat().st_size, "bytes")

print("=== C: interrupt + Command ===")
from typing import TypedDict
from langgraph.types import interrupt, Command

class WriteState(TypedDict, total=False):
    ticket_id: str
    tool_calls: list[dict]

def create_issue(title: str, ticket_id: str) -> dict:          # stand-in for the MCP write
    return {"issue_id": "ISS-4711", "title": title, "ticket_id": ticket_id}

runs = {"n": 0}

def bug_agent(state: WriteState) -> dict:
    runs["n"] += 1
    print(f"  bug_agent started (execution {runs['n']})")
    proposed = {"tool": "create_issue", "args": {"title": "QS-1099 on /v2/pages", "ticket_id": state["ticket_id"]}}
    decision = interrupt(proposed)                      # pause here; nothing below runs yet
    if decision["approved"]:
        result = create_issue(**decision["call"]["args"])
    else:
        result = {"declined": True}
    return {"tool_calls": [{"name": "create_issue", "result": result}]}

builder = StateGraph(WriteState)
builder.add_node("bug_agent", bug_agent)
builder.add_edge(START, "bug_agent")
builder.add_edge("bug_agent", END)
graph = builder.compile(checkpointer=MemorySaver())

cfg = {"configurable": {"thread_id": "T-10089"}}
out = graph.invoke({"ticket_id": "T-10089"}, cfg)
print(sorted(out))
print(out["__interrupt__"][0].value)

print("=== D: get_state ===")
snap = graph.get_state(cfg)
print("next:", snap.next)
print("pending:", snap.interrupts[0].value["tool"])
print("values:", snap.values)

print("=== C2: resume ===")
pending = snap.interrupts[0].value
out = graph.invoke(Command(resume={"approved": True, "call": pending}), cfg)
print(out["tool_calls"])
print("next after resume:", graph.get_state(cfg).next)

print("=== E: store ===")
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
store.put(("customers", "j.ramos@example.com"), "notes", {"reported": ["T-10004", "T-10089"]})
item = store.get(("customers", "j.ramos@example.com"), "notes")
print(item.value)
