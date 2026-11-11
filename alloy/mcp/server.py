# SPDX-License-Identifier: Apache-2.0
"""A Model Context Protocol (MCP) server over stdio: JSON-RPC 2.0, one message per line, tools only.

Standard library only (no MCP SDK needed). One session per process: simulations built by one call stay open for
the next until ``close`` or the end of the connection. Example (what a client sends and gets)::

    -> {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", ...}}
    <- {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {...}}, ...}}
    -> {"jsonrpc": "2.0", "method": "notifications/initialized"}
    -> {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list", "arguments": {"kind": "scenes"}}}
    <- {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "{...}"}], "isError": false}}
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from typing import IO, Any

from alloy.cli.jsonio import json_stdout
from alloy.cli.session import Session
from alloy.mcp.tools import INSTRUCTIONS, call_tool, tool_list

log = logging.getLogger("alloy.mcp")

#: protocol versions this server speaks, newest first (tools are the same in each)
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603


class RpcError(Exception):
    """A JSON-RPC error reply. Example: ``RpcError(METHOD_NOT_FOUND, "no method 'x'")``."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


class Server:
    """Answers MCP messages for one connection. Example: ``Server().handle({"jsonrpc": "2.0", "id": 1,
    "method": "ping"})`` -> ``{"jsonrpc": "2.0", "id": 1, "result": {}}``."""

    def __init__(self, session: Session | None = None) -> None:
        self.session = session or Session()
        self.methods: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "initialize": self.initialize, "ping": lambda p: {}, "tools/list": self.tools_list,
            "tools/call": self.tools_call}

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Agree a protocol version (the client's when this server speaks it) and say what the server offers."""
        from alloy.cli.command import version  # noqa: PLC0415

        asked = params.get("protocolVersion")
        agreed = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        log.info("initialize: client %s asked %s, agreed %s", params.get("clientInfo"), asked, agreed)
        return {"protocolVersion": agreed, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "alloy", "title": "Alloy physics", "version": version()},
                "instructions": INSTRUCTIONS}

    def tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Every tool (no paging: the list is short)."""
        return {"tools": tool_list()}

    def tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run one tool; tool failures come back as results with ``isError`` and a fix, not as protocol errors."""
        name, arguments = params.get("name"), params.get("arguments")
        if not isinstance(name, str):
            raise RpcError(INVALID_PARAMS, "tools/call needs params.name (a tool name from tools/list)")
        if arguments is not None and not isinstance(arguments, dict):
            raise RpcError(INVALID_PARAMS, "tools/call params.arguments must be an object")
        try:
            return call_tool(self.session, name, arguments)
        except KeyError:
            raise RpcError(INVALID_PARAMS, f"unknown tool {name!r}; tools/list lists them") from None

    def handle(self, message: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        """The reply to one message (None for notifications); a JSON-RPC batch gets a list of replies."""
        if isinstance(message, list):
            replies = [r for r in (self._one(m) for m in message) if r is not None]
            return replies or None
        return self._one(message)

    def _one(self, msg: Any) -> dict[str, Any] | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method", ""), str):
            rid = msg.get("id") if isinstance(msg, dict) else None
            return _error(rid, INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        if "method" not in msg:
            return None                               # a response to something we never send; ignore
        rid, method, is_request = msg.get("id"), msg["method"], "id" in msg
        if not is_request:
            log.debug("notification %s", method)
            return None
        fn = self.methods.get(method)
        if fn is None:
            return _error(rid, METHOD_NOT_FOUND, f"method {method!r} is not supported (this server has tools only)")
        params = msg.get("params") or {}
        try:
            if not isinstance(params, dict):
                raise RpcError(INVALID_PARAMS, "params must be an object")
            return {"jsonrpc": "2.0", "id": rid, "result": fn(params)}
        except RpcError as exc:
            return _error(rid, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - the server must keep running
            log.exception("%s failed", method)
            return _error(rid, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")


def _error(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def serve(stdin: IO[str], stdout: IO[str], session: Session | None = None) -> int:
    """Read messages from ``stdin`` line by line and write replies to ``stdout`` until end of input.

    Example: ``serve(io.StringIO(requests), out)`` (tests); ``serve_stdio()`` for real clients."""
    server = Server(session)
    try:
        while True:
            line = stdin.readline()
            if not line:
                return 0
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                reply: Any = _error(None, PARSE_ERROR, f"not JSON: {exc.msg}")
            else:
                reply = server.handle(message)
            if reply is not None:
                stdout.write(json.dumps(reply, allow_nan=False) + "\n")
                stdout.flush()
    finally:
        server.session.close_all()


def serve_stdio() -> int:
    """Serve on the process's stdin/stdout; anything else printing to stdout goes to stderr (the protocol owns
    stdout). Example: the ``alloy mcp`` command."""
    log.info("alloy MCP server on stdio")
    with json_stdout() as out:
        return serve(sys.stdin, out)
