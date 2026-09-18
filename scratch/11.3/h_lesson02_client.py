import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(
    command=sys.executable, args=["-m", "resolve.mcp_server"],
    env={**os.environ, "PYTHONPATH": "src"})

async def call(session, name, args):
    res = await session.call_tool(name, args)
    return json.loads(res.content[0].text)

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for t in (await session.list_tools()).tools:
                print(f"{t.name:20} required={t.inputSchema['required']}")
            print(await call(session, "search_issues", {"query": "QS-1042"}))
            print(await call(session, "set_ticket_status", {"ticket_id": "T-10003", "status": "closed"}))
            args = {"title": "QS-1777 lesson duplicate check", "body": "filed twice on purpose",
                    "ticket_id": "T-LESSON-02", "idempotency_key": "T-LESSON-02:create_issue:0badc0de"}
            print(await call(session, "create_issue", args))
            print(await call(session, "create_issue", args))

asyncio.run(main())
