import json
from langchain_core.utils.function_calling import convert_to_openai_tool
from resolve.graph.build import load_tools
from resolve.graph import registry
from resolve.graph.agents import tool_scope, write_tools
from resolve.graph.nodes.billing_agent import check_billing_call
from resolve.tools.idempotency import idempotency_key

load_tools()
print(sorted(registry.TOOLS))
for agent in ("knowledge_agent", "billing_agent", "bug_agent"):
    print(agent, tool_scope(agent))
print("writes", sorted(write_tools()))
print("---- schema the model sees for create_issue ----")
print(json.dumps(convert_to_openai_tool(registry.TOOLS["create_issue"]), indent=2))
print("---- search_kb schema (hand-written) ----")
print(json.dumps(convert_to_openai_tool(registry.TOOLS["search_kb"]), indent=2)[:600])
print("---- billing check ----")
state = {"ticket": {"customer_email": "quentin.byrne10@sable-systems.ca"}}
print(check_billing_call("get_account", {"email": "quentin.byrne10@sable-systems.ca"}, state))
print(check_billing_call("get_account", {"email": "dana@pinecone-health.org"}, state))
print(check_billing_call("search_kb", {"query": "duplicate charge"}, state))
print("---- key ----")
args = {"title": "QS-2FA code rejected", "body": "steps", "ticket_id": "T-10004"}
print(idempotency_key("T-10004", "create_issue", args))
print(idempotency_key("T-10004", "create_issue", {**args, "body": "steps."}))
print(idempotency_key("T-10004", "create_issue", dict(reversed(list(args.items())))))
