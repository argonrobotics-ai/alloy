# SPDX-License-Identifier: Apache-2.0
"""The MCP server: handshake, tool list, tool calls (results, errors with fixes, images), JSON-RPC errors,
notifications, and a real stdio session in a subprocess."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys

import pytest

from alloy.cli.registry import OPS
from alloy.cli.session import Session
from alloy.mcp import Server, serve
from alloy.mcp.server import PROTOCOL_VERSIONS
from alloy.cli.testing import fake_build

BOX = {"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "mass": 1.0}]}


def rpc(rid, method, params=None):
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


@pytest.fixture()
def server():
    s = Server(Session(builder=fake_build))
    yield s
    s.session.close_all()


def tool_result(server, name, arguments=None):
    reply = server.handle(rpc(7, "tools/call", {"name": name, "arguments": arguments or {}}))
    assert reply["id"] == 7 and "result" in reply, reply
    res = reply["result"]
    return res, json.loads(res["content"][0]["text"])


def test_initialize_agrees_a_version(server):
    reply = server.handle(rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                "clientInfo": {"name": "test", "version": "0"}}))
    res = reply["result"]
    assert reply["id"] == 1 and res["protocolVersion"] == "2025-06-18" and res["capabilities"] == {
        "tools": {"listChanged": False}}
    assert res["serverInfo"]["name"] == "alloy" and "recipe" in res["instructions"]
    old = server.handle(rpc(2, "initialize", {"protocolVersion": "1999-01-01"}))
    assert old["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]
    assert server.handle(rpc("x", "ping")) == {"jsonrpc": "2.0", "id": "x", "result": {}}


def test_tools_list_matches_the_operations(server):
    tools = server.handle(rpc(1, "tools/list"))["result"]["tools"]
    assert [t["name"] for t in tools] == list(OPS)
    for t in tools:
        assert t["inputSchema"]["type"] == "object" and t["description"] and "Example arguments" in t["description"]
        assert set(t["annotations"]) == {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
    json.dumps(tools)


def test_tool_calls_keep_a_session(server, tmp_path):
    res, body = tool_result(server, "build", {"scene": BOX, "worlds": 2})
    assert not res["isError"] and body["sim"] == "sim1"
    res, body = tool_result(server, "step", {"steps": 10, "names": ["cube"]})
    assert body["t"] == pytest.approx(0.1) and body["state"]["cube.pos"][2] == pytest.approx(0.99)
    res, body = tool_result(server, "render", {"out": str(tmp_path / "r.png"), "width": 20, "height": 16})
    assert not res["isError"], body
    image = res["content"][1]
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"])[:4] == b"\x89PNG" and body["width"] == 20


def test_tool_errors_are_results_with_fixes(server):
    res, body = tool_result(server, "step", {"sim": "nope"})
    assert res["isError"] and body["error"]["type"] == "SessionError" and body["error"]["fix"]
    res, body = tool_result(server, "scene", {"scene": {"objects": [{"add": "box"}]}})
    assert res["isError"] and body["error"]["field"] == "objects[0].name"
    res, body = tool_result(server, "list", {"kind": "scenes", "op": "build"})      # "op" cannot redirect the call
    assert res["isError"] and "unknown argument" in body["error"]["message"]


def test_protocol_errors(server):
    assert server.handle(rpc(1, "tools/call", {"name": "nope"}))["error"]["code"] == -32602
    assert server.handle(rpc(2, "tools/call", {"arguments": {}}))["error"]["code"] == -32602
    assert server.handle(rpc(3, "tools/call", {"name": "list", "arguments": [1]}))["error"]["code"] == -32602
    assert server.handle(rpc(4, "resources/list"))["error"]["code"] == -32601
    assert server.handle({"id": 5, "method": "ping"})["error"]["code"] == -32600
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None
    batch = server.handle([rpc(1, "ping"), {"jsonrpc": "2.0", "method": "notifications/cancelled"}, rpc(2, "ping")])
    assert [r["id"] for r in batch] == [1, 2]


def test_serve_reads_lines_until_eof():
    lines = [json.dumps(rpc(1, "ping")), "", "not json", json.dumps({"jsonrpc": "2.0", "method": "x/y"}),
             json.dumps(rpc(2, "tools/call", {"name": "list", "arguments": {"kind": "scenes", "match": "peg"}}))]
    out = io.StringIO()
    assert serve(io.StringIO("\n".join(lines) + "\n"), out, Session(builder=fake_build)) == 0
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2] and replies[1]["error"]["code"] == -32700
    assert json.loads(replies[2]["result"]["content"][0]["text"])["items"] == ["demo_peg_d12"]


def test_stdio_subprocess_session():
    msgs = [rpc(1, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                  "clientInfo": {"name": "t", "version": "0"}}),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            rpc(2, "tools/list"),
            rpc(3, "tools/call", {"name": "scene", "arguments": {"scene": BOX}})]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    res = subprocess.run([sys.executable, "-m", "alloy.cli", "mcp"], input="".join(json.dumps(m) + "\n" for m in msgs),
                         capture_output=True, text=True, timeout=120, env=env)
    assert res.returncode == 0, res.stderr
    replies = [json.loads(x) for x in res.stdout.splitlines()]       # stdout carries protocol messages only
    assert [r["id"] for r in replies] == [1, 2, 3]
    assert replies[0]["result"]["protocolVersion"] == "2025-03-26"
    assert json.loads(replies[2]["result"]["content"][0]["text"])["names"] == ["ground", "cube"]
