# SPDX-License-Identifier: Apache-2.0
"""Malformed JSON-RPC never stops the MCP server: NaN or Infinity anywhere in a message is a parse error (-32700),
an id that is not a string, whole number or null and a request without a method are Invalid Request (-32600),
params that are not an object are Invalid params (-32602); the open simulations stay open."""

from __future__ import annotations

import io
import json

import pytest

from alloy.cli.session import Session
from alloy.cli.testing import fake_build
from alloy.mcp import serve

BOX = {"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "mass": 1.0}]}


def rpc(rid, method, params=None) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})


def replies(*lines: str) -> list[dict]:
    out = io.StringIO()
    serve(io.StringIO("".join(line + "\n" for line in lines)), out, Session(builder=fake_build))
    return [json.loads(line) for line in out.getvalue().splitlines()]


@pytest.mark.parametrize("bad", ['{"jsonrpc": "2.0", "id": NaN, "method": "ping"}',
                                 '{"jsonrpc": "2.0", "id": Infinity, "method": "tools/list"}',
                                 '{"jsonrpc": "2.0", "id": [NaN], "method": "ping"}',
                                 '{"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {"x": -Infinity}}'])
def test_non_finite_numbers_are_a_parse_error_and_the_sims_stay_open(bad):
    build = rpc(1, "tools/call", {"name": "build", "arguments": {"scene": BOX}})
    state = rpc(2, "tools/call", {"name": "state", "arguments": {"sim": "sim1"}})
    built, parse, after = replies(build, bad, state)
    assert not built["result"]["isError"]
    assert parse["id"] is None and parse["error"]["code"] == -32700, parse
    assert after["id"] == 2 and not after["result"]["isError"], after


@pytest.mark.parametrize(("message", "rid", "code"), [
    ({"jsonrpc": "2.0", "id": {"bad_id": 1}, "method": "tools/list"}, None, -32600),
    ({"jsonrpc": "2.0", "id": True, "method": "ping"}, None, -32600),
    ({"jsonrpc": "2.0", "id": 1.5, "method": "ping"}, None, -32600),
    ({"jsonrpc": "2.0", "id": 42, "params": {}}, 42, -32600),                  # no method: not silence
    ({"jsonrpc": "2.0", "id": 43, "method": 7}, 43, -32600),
    ({"jsonrpc": "2.0", "id": 44, "method": "tools/list", "params": []}, 44, -32602),
    ({"jsonrpc": "2.0", "id": 45, "method": "tools/list", "params": False}, 45, -32602),
    ({"jsonrpc": "2.0", "id": 46, "method": "tools/list", "params": 0}, 46, -32602),
    ({"jsonrpc": "2.0", "id": 47, "method": "tools/list", "params": ""}, 47, -32602),
])
def test_invalid_requests_get_an_error_reply(message, rid, code):
    (reply,) = replies(json.dumps(message))
    assert reply["id"] == rid and reply["error"]["code"] == code and reply["error"]["message"], reply


def test_valid_edges_still_work():
    out = replies(json.dumps({"jsonrpc": "2.0", "id": "a", "method": "ping", "params": None}),
                  json.dumps({"jsonrpc": "2.0", "id": 9, "result": {}}),          # a response: no reply
                  json.dumps({"jsonrpc": "2.0", "id": None, "method": "ping"}),
                  json.dumps({"jsonrpc": "2.0", "id": 3.0, "method": "ping"}))
    assert [(r["id"], r.get("result")) for r in out] == [("a", {}), (None, {}), (3.0, {})]
