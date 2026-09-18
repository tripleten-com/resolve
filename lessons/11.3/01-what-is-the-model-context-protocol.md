In **Agent Fundamentals** you built an agent loop around tools you wrote by hand: a Python function, a JSON schema describing its inputs, and glue that ran the function and fed the result back to the model. That holds up while every tool lives in the same process as the loop. It stops holding up the moment a tool lives somewhere else. Resolve's helpdesk is a separate service with its own HTTP API, and the bug tracker and the account lookup sit behind that same service. Every agent framework you might use later would need its own copy of the glue for that API, with its own error handling and its own idea of what a tool description looks like. The **Model Context Protocol (MCP)** is the standard that stops you writing that glue more than once.

## One protocol instead of one integration per service

MCP is an open protocol that lets an AI application reach external tools and data through one uniform interface. Anthropic published it in November 2024, and it is now spoken by most agent clients and by a large number of servers, from filesystems to issue trackers. The official documentation compares it to USB-C: one connector, many devices. A service ships one MCP server, and every MCP-aware client can use it without knowing anything about the service's own API.

Underneath, there is little protocol to learn. Every MCP message is **JSON-RPC 2.0**: a request carrying a `method` name and `params`, and a response carrying either a `result` or an `error`, matched to its request by an `id`. That is the whole wire format, which is part of why adoption moved so fast.

## The three roles

- Host and client: your agent application is the host. It embeds an **MCP client**, and that client opens and manages one connection per server. In this course the host is the Resolve graph.
- Server: a process that exposes the capabilities of exactly one system. It advertises what it offers and executes requests. In this course the server is the one you write in the next lesson, and the system behind it is the Quillstack helpdesk.
- Transport: how the two talk. **stdio** launches the server as a local subprocess and pipes JSON-RPC over its standard input and output, one JSON object per line. **Streamable HTTP** reaches a server across the network. The messages are identical either way, so changing transport changes your connection code and nothing else. Resolve uses stdio: the graph starts the helpdesk server as a child process and talks to it through pipes.

*Diagram idea: left panel, one agent with five differently coloured cables running to five service boxes, each cable labelled "custom SDK plus auth". Right panel, the same agent with a single MCP client plugging into a row of identical server ports. Caption: "One protocol on the right."*

## What a server exposes

A server offers three kinds of primitive, and only the first one does anything.

<table header-row="true">
<tr>
<td>Primitive</td>
<td>What it is</td>
<td>Example</td>
</tr>
<tr>
<td>Tools</td>
<td>Callable actions with a name, a description, and a JSON Schema for the input</td>
<td>`get_ticket`, `create_issue`</td>
</tr>
<tr>
<td>Resources</td>
<td>Readable data addressed by a URI</td>
<td>a ticket at `helpdesk://tickets/T-10004`</td>
</tr>
<tr>
<td>Prompts</td>
<td>Reusable prompt templates the server publishes</td>
<td>a "summarize this ticket" template</td>
</tr>
</table>

Resolve's server publishes tools only. The client discovers them at runtime, so nothing about a tool is hard-coded on the client side.

## The wire, read once

Here is the exchange as it crosses the pipe. The client sends a request to list tools:

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
```

Resolve's helpdesk server answers with six entries. The first is shown in full and the rest are trimmed to their names:

```json
{"jsonrpc": "2.0", "id": 2, "result": {"tools": [
  {"name": "get_ticket",
   "description": "Read one helpdesk ticket by id (for example T-10004) with its status and the last five comments. Use it when you need the ticket's history, not for the ticket text you already have.",
   "inputSchema": {"type": "object",
                   "properties": {"ticket_id": {"type": "string", "title": "Ticket Id"}},
                   "required": ["ticket_id"],
                   "title": "get_ticketArguments"}},
  {"name": "get_account", "...": "..."},
  {"name": "search_issues", "...": "..."},
  {"name": "create_issue", "...": "..."},
  {"name": "add_ticket_comment", "...": "..."},
  {"name": "set_ticket_status", "...": "..."}]}}
```

Read that entry rather than skim it. A **tool** on the wire is a name, a human-readable description, and a JSON Schema for its arguments, and nothing in it is MCP-specific. It is the same triple you typed by hand in **Agent Fundamentals**. The description is the only thing the model reads when it decides whether to call this tool, which is why the server's author wrote it as an instruction about when to use it and when not to.

Calling a tool is a second request. The client sends the name the server advertised and an argument object that satisfies the schema:

```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "get_ticket", "arguments": {"ticket_id": "T-10003"}}}
```

```json
{"jsonrpc": "2.0", "id": 3, "result": {
  "content": [{"type": "text", "text": "{\n  \"ticket_id\": \"T-10003\",\n  \"status\": \"open\",\n  \"plan\": \"Team\",\n  \"customer_email\": \"hugo_silva@quartzpay.io\",\n  \"subject\": \"Setting up 2FA\",\n  \"body\": \"Hi, I'd like to enable two-factor authentication ...\",\n  \"comments\": []\n}"}],
  "isError": false}}
```

Two things about that response decide how every client is written. The payload is a list of **content blocks**, each with a `type`, and this server puts its JSON inside a text block, so a client parses the string before it reads a field. And a failed call does not come back as a JSON-RPC `error`; it comes back as a normal result with `isError` set to true, so the model can read the message and recover. Code that never checks the flag hands an error message to the model as if it were a ticket.

## The same handshake from Python

The `mcp` package ships the client side of this exchange. Install the pinned version:

```bash
pip install mcp==1.30.0
```

The client is asynchronous, so two definitions before the code. A **coroutine** is a function declared with `async def` that can pause at an `await` while it waits for something slow, such as a subprocess writing a line, and let other work run in the meantime. **`asyncio`** is the standard-library scheduler that runs coroutines; `asyncio.run(main())` starts one and blocks until it finishes. You do not need to write concurrent code to use the MCP client. Put `await` in front of every call to the session and `asyncio.run` around the outermost function, and that is the whole discipline.

Run this from the repository root, with `PYTHONPATH=src` and the helpdesk up:

```python
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
            print(ticket["subject"], "|", ticket["status"], "|", ticket["plan"])

asyncio.run(main())
```

```plain text
['get_ticket', 'get_account', 'search_issues', 'create_issue', 'add_ticket_comment', 'set_ticket_status']
False text
Setting up 2FA | open | Team
```

Four lines earn their place. `StdioServerParameters` is a launch recipe rather than a connection: a command, its arguments, and the environment the subprocess will see. The `env` line is the one people lose an hour to. A subprocess does not automatically see your shell; leave `env` unset and the stdio client forwards only a safe subset of variables, so the server would start and then fail to find `HELPDESK_URL` from your `.env`. `session.initialize()` is the protocol handshake, and skipping it makes every later call fail. `session.list_tools()` returns the schemas you saw on the wire, which is why you never hard-code a tool name: the server is the only authority on what it calls things. `session.call_tool` is the client side of `tools/call`, and the two prints check the two things that are easy to get wrong, the error flag and the content-block shape.

## How a discovered tool reaches the model

The schemas the server publishes are the same shape you fed a model in **Agent Fundamentals**, with one field renamed. MCP calls the argument schema `inputSchema`; the OpenAI function format calls it `parameters`:

```python
def as_openai_tools(listed):
    return [{"type": "function",
             "function": {"name": t.name,
                          "description": t.description,
                          "parameters": t.inputSchema}}
            for t in listed.tools]
```

Three keys copied and one renamed, and the model has no idea a protocol was involved. Your client forwards these schemas, the model emits a tool call, and your client turns that call into `tools/call` on whichever server owns the name. From **LangChain & LangGraph** onward you do not write this adapter yourself; a library does the same translation into LangChain tools, and **Connecting MCP Tools to a LangGraph Agent** shows it.

*Diagram idea: the agent loop from **Agent Fundamentals** drawn as a ring of four nodes, observe, think, act, observe. The arrow leaving "act" crosses out of the ring into a box labelled "MCP client", which fans out to the helpdesk server. Caption: "MCP slots into the tool-call step you already built."*

## Actions, not reads

The retriever you built in **Retrieval and RAG** only reads. Three of the six tools above take real actions: file an issue, post a comment on a ticket, change a ticket's status. That is a different threat model, so treat a server that can write as a production credential from the first connection.

- Least privilege. Give each part of the agent the narrowest set of tools that does its job. In Resolve the knowledge specialist never sees `create_issue`, and no specialist sees `set_ticket_status`; the file `configs/tools.yaml` decides who gets what, and the graph enforces it. A tool the model never sees is a tool nobody can talk it into using.
- Secrets discipline. The helpdesk URL and every key live in environment variables, never in code and never baked into an image. The client passes the environment to the server on purpose, as the `env` line above does.
- Untrusted by default. A server's tool descriptions and the text it hands back are data that an attacker may control. A customer's comment arrives through `get_ticket`; a colleague's issue body arrives through `search_issues`. That is an injection surface, and **OWASP Threat Modeling** and **Defenses & Red-Teaming** are where you harden it.

## What the protocol buys you

1. Write once, reuse everywhere. One server for the helpdesk, usable by every agent and framework you build afterwards.
2. Swap freely. Change the model, or move from the hand-rolled loop of **Agent Fundamentals** to a graph node from **LangChain & LangGraph**, without touching a tool.
3. Discovery at runtime. A server adds a tool and your agent picks it up on the next `tools/list`, with no redeploy of the agent.

The price of all three is that a tool can now change something. The next lesson, **Writing Your Own MCP Server**, builds the six-tool helpdesk server you just called, with descriptions the model reads, compact results, and errors that come back as data instead of crashing the agent's turn.

<callout icon="🤖" color="blue_bg">
	**Working with AI** Scaffolding an MCP client connection is decent delegation: mechanical, short, and cheap to verify by running it. It is also where a model's memory is least trustworthy, because method names, transport names, and SDK signatures have all changed since the first release. Review generated client code for three things on sight: that `initialize()` runs before any other call, that tool names come from `list_tools()` rather than from the model's head, and that the environment reaches the subprocess through `env` rather than by assumption. Hand over the boilerplate, then check it against a real `tools/list` response.
</callout>
