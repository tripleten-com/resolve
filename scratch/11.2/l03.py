print("=== A: MessagesState + bind_tools ===")
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from resolve.tools.kb import search_kb, read_doc

load_dotenv()
model = ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                   model=os.environ["LLM_MODEL"], temperature=0, max_tokens=900)

TOOLS = [search_kb, read_doc]
model_with_tools = model.bind_tools(TOOLS)

class AgentState(MessagesState):
    steps: int

MAX_STEPS = 6

def agent(state: AgentState) -> dict:
    resp = model_with_tools.invoke(state["messages"])
    return {"messages": [resp], "steps": state.get("steps", 0) + 1}

print("=== B: ToolNode + tools_condition ===")
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def after_agent(state: AgentState) -> str:
    if state["steps"] >= MAX_STEPS:
        return END                       # the cap wins, whatever the model asked for
    return tools_condition(state)        # "tools" if the last message has tool calls, else END

builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile()

print("=== C: run ===")
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM = ("You are Resolve, an internal assistant for Quillstack's support team. Search the help center "
          "with search_kb before answering, read the whole article with read_doc if a chunk is cut off, "
          "state only facts that appear in what the tools returned, and end with the doc_ids you relied on.")
ticket = HumanMessage("Ticket T-10001\nSubject: how do i archive a project?\n\nwe have a project thats "
                      "finished and i want to archive it but cant find the option anywhere. also can we "
                      "bring it back later if we need to? im an admin")

out = graph.invoke({"messages": [SystemMessage(SYSTEM), ticket], "steps": 0})
for m in out["messages"]:
    if m.type == "ai" and m.tool_calls:
        print("ai  ->", [(tc["name"], tc["args"]) for tc in m.tool_calls])
    elif m.type == "tool":
        print("tool<-", m.name, str(m.content)[:80].replace("\n", " "))
print("steps:", out["steps"])
print(out["messages"][-1].content[:600])

print("=== D: the real loop ===")
from resolve.graph import registry
from resolve.graph.agents import run_agent
from resolve.graph.prompts import KNOWLEDGE_AGENT
from resolve.graph.state import initial_state
from resolve.tickets import load_ticket
from resolve.tools.kb import KB_TOOLS

registry.register(KB_TOOLS)                                   # build.py does this at startup
state = initial_state("T-10001", load_ticket("T-10001"))
update = run_agent("knowledge_agent", KNOWLEDGE_AGENT, state)
print(sorted(update))
print([t["name"] for t in update["tool_calls"]])
print(update["citations"], update["confidence"], len(update["docs"]), "chunks")
print(update["draft"][:300])
print(update["usage"])
