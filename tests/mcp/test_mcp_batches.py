# SPDX-License-Identifier: Apache-2.0
"""JSON-RPC edges of the MCP server: an empty batch [] gets one Invalid Request reply (JSON-RPC 2.0 section 6), and
a huge unknown id, tool or method name comes back cut short, not echoed in full."""

from __future__ import annotations

import io
import json

from alloy.cli.session import Session
from alloy.cli.testing import fake_build
from alloy.mcp import serve


def session_lines(*messages: str) -> list[dict]:
    """Replies, in order, to raw message lines sent to one server."""
    out = io.StringIO()
    serve(io.StringIO("".join(m + "\n" for m in messages)), out, Session(builder=fake_build))
    return [json.loads(line) for line in out.getvalue().splitlines()]


def rpc(rid, method, params=None) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})


def test_an_empty_batch_gets_one_invalid_request_reply_before_the_next_one():
    replies = session_lines("[]", rpc(2, "tools/list"))
    assert len(replies) == 2
    assert replies[0]["id"] is None and replies[0]["error"]["code"] == -32600, replies[0]
    assert replies[1]["id"] == 2 and replies[1]["result"]["tools"]


def test_a_batch_of_notifications_still_gets_no_reply():
    replies = session_lines(json.dumps([{"jsonrpc": "2.0", "method": "notifications/initialized"}]),
                            rpc(3, "ping"))
    assert [r["id"] for r in replies] == [3]


def test_huge_ids_and_names_are_not_echoed_in_full():
    huge = "s" * 1_000_000
    lines = [rpc(1, "tools/call", {"name": "reset", "arguments": {"sim": huge}}),
             rpc(2, "tools/call", {"name": huge, "arguments": {}}),
             rpc(3, huge)]
    out = io.StringIO()
    serve(io.StringIO("".join(m + "\n" for m in lines)), out, Session(builder=fake_build))
    raw = out.getvalue().splitlines()
    assert len(raw) == 3 and all(len(line) < 5_000 for line in raw), [len(line) for line in raw]
    reset, tool, method = (json.loads(line) for line in raw)
    assert reset["result"]["isError"] and "SessionError" in reset["result"]["content"][0]["text"]
    assert tool["error"]["code"] == -32602 and method["error"]["code"] == -32601
