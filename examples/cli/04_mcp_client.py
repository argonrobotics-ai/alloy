# SPDX-License-Identifier: Apache-2.0
"""Talk to Alloy's MCP tool server the way an AI agent's client does: start it on stdio, list its tools, build a
scene, step it, read the state and close it. Standard library only (no MCP SDK needed).

To give a real agent these tools instead, register the server with its client, e.g.
``claude mcp add alloy -- alloy mcp`` or ``{"mcpServers": {"alloy": {"command": "alloy", "args": ["mcp"]}}}``.

Run: python examples/cli/04_mcp_client.py      (GPU if present, else CPU; the first build compiles kernels)
"""
import json
import subprocess
import sys

server = subprocess.Popen([sys.executable, "-m", "alloy.mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1)
next_id = 0


def request(method, params=None):
    """Send one JSON-RPC request and return its result (the server answers in order)."""
    global next_id
    next_id += 1
    server.stdin.write(json.dumps({"jsonrpc": "2.0", "id": next_id, "method": method, "params": params or {}}) + "\n")
    reply = json.loads(server.stdout.readline())
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return reply["result"]


def tool(name, **arguments):
    """Call a tool; its result is JSON text (on failure: {"error": {"message", "fix", ...}})."""
    res = request("tools/call", {"name": name, "arguments": arguments})
    body = json.loads(res["content"][0]["text"])
    if res["isError"]:
        raise RuntimeError(f"{name}: {body['error']['message']} -> {body['error']['fix']}")
    return body


info = request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "example", "version": "0"}})
server.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
print("server:", info["serverInfo"]["name"], info["serverInfo"]["version"])
print("tools:", [t["name"] for t in request("tools/list")["tools"]])

scene = {"objects": [{"add": "ground"},
                     {"add": "sphere", "name": "ball", "radius": 0.03, "pos": [0, 0, 0.3], "mass": 0.1,
                      "restitution": 0.6}]}
built = tool("build", scene=scene, id="ball")
print("built:", built["sim"], built["names"]["bodies"], "on", built["device"])
for _ in range(3):
    step = tool("step", sim="ball", steps=10, names=["ball"])
    print(f"t={step['t']:.2f} s  ball z={step['state']['ball.pos'][2]:.3f} m  health ok={step['health']['ok']}")
print("check:", tool("check", sim="ball")["ok"])
print("closed:", tool("close", sim="ball")["closed"])
server.stdin.close()
server.wait(timeout=60)
