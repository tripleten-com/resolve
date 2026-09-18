print("=== A: two-node graph on the real index ===")
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from resolve.kb.search import search_kb

load_dotenv()
model = ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                   model=os.environ["LLM_MODEL"], temperature=0, max_tokens=300)

class State(TypedDict, total=False):
    ticket_id: str
    ticket: dict            # subject, body, ...
    docs: list[dict]        # doc_id, chunk_id, title, space, text, score
    draft: str

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Resolve, Quillstack's support assistant. Answer only from the chunks. "
               "Cite the doc_id of each chunk you used."),
    ("human", "Chunks:\n{chunks}\n\nTicket subject: {subject}\n{body}"),
])

def retrieve(state: State) -> dict:
    t = state["ticket"]
    return {"docs": search_kb(f"{t['subject']} {t['body']}", k=4)}

def answer(state: State) -> dict:
    chunks = "\n\n".join(f"[{d['doc_id']}] {d['text']}" for d in state["docs"])
    msg = prompt.invoke({"chunks": chunks, **state["ticket"]})
    return {"draft": model.invoke(msg).content}

builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
graph = builder.compile()

ticket = {"subject": "how do i archive a project?",
          "body": "we have a project thats finished and i want to archive it. can we bring it back later?"}
for step in graph.stream({"ticket_id": "T-10001", "ticket": ticket}, stream_mode="values"):
    print(sorted(step))
final = graph.invoke({"ticket_id": "T-10001", "ticket": ticket})
print([d["doc_id"] for d in final["docs"]])
print(final["draft"][:400])

print("=== B: router + route, stubs ===")

class TicketState(TypedDict, total=False):
    ticket_id: str
    ticket: dict
    classification: dict    # type, urgency, aimed_at_agent, confidence
    docs: list[dict]
    draft: str
    citations: list[str]
    confidence: float
    review: dict            # verdict, reasons, revision_count
    guard_output: dict      # flags, ok
    decision: str
    escalation_reason: str

SPECIALISTS = {"how_to": "knowledge_agent", "billing": "billing_agent", "bug": "bug_agent"}

def after_classify(state: TicketState) -> str:
    c = state["classification"]
    if c["type"] == "security" or c["aimed_at_agent"]:
        return "escalate"
    return SPECIALISTS.get(c["type"], "escalate")

def classify(state: TicketState) -> dict:
    return {"classification": state["ticket"]["_label"]}   # a model call in the real build

def specialist(state: TicketState) -> dict:
    return {"draft": "Open the project, choose Archive from the menu...", "citations": ["kb-0015"], "confidence": 0.9}

def review(state: TicketState) -> dict:
    return {"review": {"verdict": "approve", "reasons": [], "revision_count": 0}}

def output_guard(state: TicketState) -> dict:
    return {"guard_output": {"flags": [], "ok": True}}

AUTO_RESOLVE_TYPES = ("how_to", "billing")
THRESHOLD = 0.7

def route(state: TicketState) -> dict:
    reasons = []
    if state["review"]["verdict"] != "approve":
        reasons.append("review did not approve")
    if state["confidence"] < THRESHOLD:
        reasons.append(f"confidence {state['confidence']} below {THRESHOLD}")
    if state["guard_output"]["flags"]:
        reasons.append("output flags")
    if state["classification"]["type"] not in AUTO_RESOLVE_TYPES:
        reasons.append(f"type {state['classification']['type']} never auto-resolves")
    if reasons:
        return {"decision": "escalate", "escalation_reason": "; ".join(reasons)}
    return {"decision": "auto_resolve", "escalation_reason": ""}

def after_route(state: TicketState) -> str:
    return "resolve" if state["decision"] == "auto_resolve" else "escalate"

def resolve(state: TicketState) -> dict:
    return {"decision": "auto_resolve"}

def escalate(state: TicketState) -> dict:
    return {"decision": "escalate"}

builder = StateGraph(TicketState)
for name, fn in [("classify", classify), ("knowledge_agent", specialist), ("billing_agent", specialist),
                 ("bug_agent", specialist), ("review", review), ("output_guard", output_guard),
                 ("route", route), ("resolve", resolve), ("escalate", escalate)]:
    builder.add_node(name, fn)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", after_classify, {
    "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent",
    "bug_agent": "bug_agent", "escalate": "escalate"})
for s in ("knowledge_agent", "billing_agent", "bug_agent"):
    builder.add_edge(s, "review")
builder.add_edge("review", "output_guard")
builder.add_edge("output_guard", "route")
builder.add_conditional_edges("route", after_route, {"resolve": "resolve", "escalate": "escalate"})
builder.add_edge("resolve", END)
builder.add_edge("escalate", END)
graph = builder.compile()

def run(label: dict) -> None:
    path = []
    final = {}
    for update in graph.stream({"ticket_id": "T-0", "ticket": {"_label": label}}, stream_mode="updates"):
        for node, delta in update.items():
            path.append(node)
            final.update(delta)
    print(" -> ".join(path), "|", final["decision"], final.get("escalation_reason", ""))

run({"type": "how_to", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.95})
run({"type": "bug", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.9})
run({"type": "how_to", "urgency": "normal", "aimed_at_agent": True, "confidence": 0.9})

print("=== C: bounded reflection loop ===")
REVISION_LIMIT = 1

def review2(state: TicketState) -> dict:
    prev = state.get("review") or {}
    count = prev.get("revision_count", 0)
    verdict = "revise" if "12 days" in state["draft"] else "approve"   # a model call in the real build
    if verdict == "revise":
        count += 1
    return {"review": {"verdict": verdict, "reasons": ["unsupported claim: 12 days"] if verdict == "revise" else [],
                       "revision_count": count}}

def after_review(state: TicketState) -> str:
    r = state["review"]
    if r["verdict"] == "revise" and r["revision_count"] <= REVISION_LIMIT:
        return SPECIALISTS[state["classification"]["type"]]
    return "output_guard"

def specialist2(state: TicketState) -> dict:
    if (state.get("review") or {}).get("verdict") == "revise":
        return {"draft": "Your next invoice is prorated; the billing team can confirm the amount.", "confidence": 0.85}
    return {"draft": "You have 12 days remaining on the current cycle.", "confidence": 0.85}

builder = StateGraph(TicketState)
for name, fn in [("classify", classify), ("knowledge_agent", specialist2), ("billing_agent", specialist2),
                 ("bug_agent", specialist2), ("review", review2), ("output_guard", output_guard),
                 ("route", route), ("resolve", resolve), ("escalate", escalate)]:
    builder.add_node(name, fn)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", after_classify, {
    "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent",
    "bug_agent": "bug_agent", "escalate": "escalate"})
for s in ("knowledge_agent", "billing_agent", "bug_agent"):
    builder.add_edge(s, "review")
builder.add_conditional_edges("review", after_review, {
    "knowledge_agent": "knowledge_agent", "billing_agent": "billing_agent",
    "bug_agent": "bug_agent", "output_guard": "output_guard"})
builder.add_edge("output_guard", "route")
builder.add_conditional_edges("route", after_route, {"resolve": "resolve", "escalate": "escalate"})
builder.add_edge("resolve", END)
builder.add_edge("escalate", END)
graph = builder.compile()

label = {"type": "billing", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.9}
path = []
for update in graph.stream({"ticket_id": "T-0", "ticket": {"_label": label}}, stream_mode="updates"):
    for node, delta in update.items():
        path.append(node)
        if node == "review":
            print("review:", delta["review"])
        if node == "route":
            print("route:", delta)
print(" -> ".join(path))
