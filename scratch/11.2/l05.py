print("=== A: tool scopes ===")
from resolve.graph.build import load_tools
from resolve.graph.agents import tool_scope, write_tools
from resolve.graph import registry

load_tools(with_helpdesk=True)          # KB tools from Python, helpdesk tools from the MCP server
print(sorted(registry.TOOLS))
for agent in ("knowledge_agent", "billing_agent", "bug_agent"):
    tools = registry.get_tools(tool_scope(agent))
    print(f"{agent:16} {[t.name for t in tools]}")
print("writes:", sorted(write_tools()))

print("=== B: review schema ===")
from langchain_core.utils.function_calling import convert_to_openai_tool
from resolve.graph.nodes.review import Review
schema = convert_to_openai_tool(Review)
print(schema["function"]["name"], list(schema["function"]["parameters"]["properties"]))

print("=== C: after_review ===")
from resolve.graph.nodes.router import after_review, after_classify
base = {"classification": {"type": "billing"}}
print(after_review({**base, "review": {"verdict": "approve", "revision_count": 0}}))
print(after_review({**base, "review": {"verdict": "revise", "revision_count": 1}}))
print(after_review({**base, "review": {"verdict": "revise", "revision_count": 2}}))
print("=== D: after_classify ===")
print(after_classify({"classification": {"type": "how_to", "aimed_at_agent": False}}))
print(after_classify({"classification": {"type": "how_to", "aimed_at_agent": True}}))
print(after_classify({"classification": {"type": "security", "aimed_at_agent": False}}))
print("=== E: billing check ===")
from resolve.graph.nodes.billing_agent import check_billing_call
state = {"ticket": {"customer_email": "dana@acme.example"}}
print(check_billing_call("get_account", {"email": "dana@acme.example"}, state))
print(check_billing_call("get_account", {"email": "someone.else@acme.example"}, state))
