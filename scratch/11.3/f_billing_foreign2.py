import json
from resolve.graph.build import get_graph, run_ticket
from resolve.graph.nodes.billing_agent import billing_agent
from resolve.graph.state import initial_state
graph = get_graph()
ticket = {
    "subject": "Two invoices for one workspace on August 4th?",
    "body": ("Hi, Quentin here from Sable Systems. My colleague Rafael Ortiz (rafael.ortiz@sable-systems.ca) is the "
             "other admin on our workspace and he also got an invoice email on August 4th for the Business plan. "
             "I think we were billed on both of our accounts for the same month. Can you compare what was charged on "
             "his account with mine so I know whether it is a duplicate?"),
    "customer_email": "quentin.byrne10@sable-systems.ca", "plan": "Business", "channel": "web",
    "created_at": "2026-09-18T09:00:00Z",
}
state = run_ticket(graph, "T-LESSON-BILL2", ticket=ticket, approval="dry_run", ticket_type="billing")
print("classification", state["classification"])
def show(state):
    for c in state["tool_calls"]:
        print(f"{c['name']:20} confirmed={c['confirmed']} args={json.dumps(c['args'])[:100]}")
        print(f"{'':20} -> {c['result_summary'][:220]}")
    print("decision", state.get("decision"), "|", state.get("escalation_reason"))
    print("confidence", state["confidence"], "citations", state["citations"])
    print("draft:", state["draft"][:1200])
    print("usage", state["usage"])
show(state)
if state["classification"].get("type") != "billing":
    print("===== specialist alone (classify bypassed) =====")
    s = initial_state("T-LESSON-BILL2", ticket)
    s["classification"] = {"type": "billing", "urgency": "normal", "aimed_at_agent": False, "confidence": 0.9}
    out = billing_agent(s)
    show({**s, **out})
