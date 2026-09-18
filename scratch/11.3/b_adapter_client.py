import asyncio, json, os, sys
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "helpdesk": {"transport": "stdio", "command": sys.executable,
                 "args": ["-m", "resolve.mcp_server"],
                 "env": {**os.environ, "PYTHONPATH": "src"}},
})

async def main():
    tools = await client.get_tools()
    for t in tools:
        print(f"{t.name}: {sorted(t.args)}")
    print(type(tools[0]).__name__, tools[0].__class__.__mro__[1].__name__)
    by = {t.name: t for t in tools}
    print("---- description ----")
    print(by["search_issues"].description)
    print("---- args schema ----")
    print(json.dumps(by["create_issue"].args, indent=2))
    print("---- search_issues ----")
    r = await by["search_issues"].ainvoke({"query": "QS-"})
    print(type(r).__name__, str(r)[:600])
    print("---- bad status ----")
    r = await by["set_ticket_status"].ainvoke({"ticket_id": "T-10001", "status": "closed"})
    print(r)
    print("---- create_issue twice same key ----")
    args = {"title": "QS-1999 scratch duplicate check", "body": "lesson 02 scratch", "ticket_id": "T-SCRATCH",
            "idempotency_key": "T-SCRATCH:create_issue:deadbeef"}
    print(await by["create_issue"].ainvoke(args))
    print(await by["create_issue"].ainvoke(args))
    print("---- get_account ----")
    r = await by["get_account"].ainvoke({"email": "quentin.byrne10@sable-systems.ca"})
    print(str(r)[:700])
    r = await by["get_account"].ainvoke({"email": "nobody@example.com"})
    print(r)

asyncio.run(main())
