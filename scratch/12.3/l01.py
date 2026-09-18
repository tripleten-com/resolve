# Lesson 01 blocks
from resolve.guards.input import scrub_secrets
text = "My key qs_live_8f3kD92mQx7 returns QS-1042 since Monday. Password: hunter2pass"
clean, flags = scrub_secrets(text)
print(clean)
print(flags)
print("-----")
from resolve.guards.input import guard_ticket
ticket = {"subject": "API key rejected", "body": "My key qs_live_8f3kD92mQx7 returns QS-1042 since Monday.",
          "customer_email": "dev@northwindlabs.io", "plan": "Team", "channel": "web"}
clean, guard = guard_ticket(ticket)
print(clean["body"])
print(guard)
print("-----")
from resolve.guards.output import strip_links
draft = ("Hi Dana, rotate the key under Settings > API keys. "
         "![status](https://collect.example.net/p.png?t=T-10042) "
         "Details: https://help.quillstack.com/api-keys and [our guide](https://rival.example/keys).")
clean, flags = strip_links(draft, ["quillstack.com", "help.quillstack.com"])
print(clean)
print(flags)
print("-----")
from resolve.guards.output import guard_draft
docs = [{"doc_id": "kb-0015", "space": "public", "title": "Archive a project"},
        {"doc_id": "kb-0104", "space": "internal", "title": "Refund approval policy"}]
draft = "Hi Dana, archive from Settings > Projects (kb-0015). Refunds follow kb-0104. Token: whsec_a1b2c3d4e5f6"
clean, guard = guard_draft(draft, ["kb-0015", "kb-0104"], docs, [{"name": "create_issue", "write": True, "node": "bug_agent"}])
print(clean)
print(guard)
print("-----")
from resolve.graph.nodes.route import decide
state = {"classification": {"type": "how_to"}, "review": {"verdict": "approve", "reasons": []},
         "confidence": 0.92, "guard_input": {"quarantined": False},
         "guard_output": {"flags": ["link_stripped:collect.example.net"], "ok": False}, "ticket": {}}
print(decide(state))
state["guard_output"] = {"flags": [], "ok": True}
print(decide(state))
