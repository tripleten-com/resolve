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
            print([t.name for t in listed.tools])
            res = await session.call_tool("get_ticket", {"ticket_id": "T-10003"})
            print(res.isError, res.content[0].type)
            ticket = json.loads(res.content[0].text)
            print(ticket["subject"], "|", ticket["status"], "|", ticket["plan"], "|", len(ticket["comments"]), "comments")

asyncio.run(main())
