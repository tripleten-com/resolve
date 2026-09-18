import json, os, subprocess, sys

proc = subprocess.Popen([sys.executable, "-m", "resolve.mcp_server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": "src"}, text=True, encoding="utf-8")

def send(msg):
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()

def recv():
    return json.loads(proc.stdout.readline())

send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "lesson", "version": "0"}}})
init = recv()
print("INIT RESPONSE:", json.dumps(init)[:400])
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
listed = recv()
print("TOOLS/LIST names:", [t["name"] for t in listed["result"]["tools"]])
print(json.dumps(listed["result"]["tools"][0], indent=2))
print("keys per tool entry:", sorted(listed["result"]["tools"][0].keys()))
send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_ticket", "arguments": {"ticket_id": "T-10003"}}})
called = recv()
print("TOOLS/CALL:", json.dumps(called))
proc.stdin.close(); proc.wait(timeout=10)
