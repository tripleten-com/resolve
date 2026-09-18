import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(
    command=sys.executable, args=["-m", "resolve.mcp_server"],
    env={**os.environ, "PYTHONPATH": "src"})

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            for t in listed.tools:
                print(t.name)
            print("---- one entry as JSON ----")
            gt = [t for t in listed.tools if t.name == "get_ticket"][0]
            print(json.dumps({"name": gt.name, "description": gt.description, "inputSchema": gt.inputSchema}, indent=2))
            ci = [t for t in listed.tools if t.name == "create_issue"][0]
            print(json.dumps({"name": ci.name, "description": ci.description, "inputSchema": ci.inputSchema}, indent=2))
            print("---- call get_ticket ----")
            res = await session.call_tool("get_ticket", {"ticket_id": "T-10004"})
            print(res.isError, res.content[0].type)
            data = json.loads(res.content[0].text)
            print(data["subject"], data["status"], data["plan"])
            print(json.dumps(data, indent=2)[:900])
            print("structuredContent:", json.dumps(res.structuredContent)[:200] if res.structuredContent else None)
            print("---- bad ticket ----")
            res = await session.call_tool("get_ticket", {"ticket_id": "T-99999"})
            print(res.isError, res.content[0].text)

asyncio.run(main())
