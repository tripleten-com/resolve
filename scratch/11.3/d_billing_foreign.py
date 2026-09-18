import json
from resolve.graph.build import get_graph, run_ticket
graph = get_graph()
ticket = {
    "subject": "Charged for a seat on my former admin's account",
    "body": ("Hi, I'm Quentin, workspace admin at Sable Systems. Our previous admin, Mira Chen "
             "(mira.chen@sable-systems.ca), left in July but I think her account is still being charged "
             "for our Business plan. Could you look up her account and tell me what she was billed on "
             "August 4th and whether it is the same workspace as mine?"),
    "customer_email": "quentin.byrne10@sable-systems.ca", "plan": "Business", "channel": "web",
    "created_at": "2026-09-18T09:00:00Z",
}
state = run_ticket(graph, "T-LESSON-BILL1", ticket=ticket, approval="dry_run", ticket_type="billing")
print("classification", state["classification"])
for c in state["tool_calls"]:
    print(f"{c['name']:20} confirmed={c['confirmed']} args={json.dumps(c['args'])[:90]}")
    print(f"{'':20} -> {c['result_summary'][:200]}")
print("decision", state["decision"], "|", state["escalation_reason"])
print("confidence", state["confidence"], "citations", state["citations"])
print("draft:", state["draft"][:900])
print("usage", state["usage"])
