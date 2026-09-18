import json
from resolve.graph.build import get_graph
graph = get_graph()
snap = graph.get_state({"configurable": {"thread_id": "T-10004"}})
v = snap.values
print("next:", snap.next)
for c in v["tool_calls"]:
    print(f"{c['name']:20} node={c.get('node')} confirmed={c['confirmed']} write={c.get('write')} key={c.get('idempotency_key')}")
    print(f"{'':20} args={json.dumps(c['args'], ensure_ascii=False)[:160]}")
    print(f"{'':20} -> {c['result_summary'][:260]}")
print("---- docs ----")
for d in v["docs"]:
    print(d["doc_id"], d["chunk_id"], d["space"], d["title"], round(d["score"], 4))
print("---- review ----", v["review"])
print("---- guard_output ----", v["guard_output"])
print("---- decision ----", v["decision"], "|", v["escalation_reason"])
print("---- confidence/citations ----", v["confidence"], v["citations"])
print("---- draft ----")
print(v["draft"])
print("---- usage ----", v["usage"])
