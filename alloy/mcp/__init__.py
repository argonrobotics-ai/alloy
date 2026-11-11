# SPDX-License-Identifier: Apache-2.0
"""Alloy as an MCP (Model Context Protocol) tool server: every ``alloy`` command-line operation is a tool, with the
same JSON arguments and results. Standard library only; stdio transport.

Start it with ``alloy mcp`` (or ``python -m alloy.mcp``). A client configuration (Claude Code, Claude Desktop,
Cursor and others use this shape)::

    {"mcpServers": {"alloy": {"command": "alloy", "args": ["mcp"]}}}

Claude Code: ``claude mcp add alloy -- alloy mcp``. Tools: list, scene, build, step, run, state, targets, reset,
close, check, render (returns the PNG too), estimate, import, export. Simulations stay open between calls, by id.

Example (in Python, without a client)::

    from alloy.mcp import Server
    server = Server()
    server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "list", "arguments": {"kind": "scenes"}}})
"""

from __future__ import annotations

from alloy.mcp.server import Server, serve, serve_stdio

__all__ = ["Server", "main", "serve", "serve_stdio"]


def main() -> int:
    """Run the server on stdio until the client disconnects. Example: ``python -m alloy.mcp``."""
    import logging  # noqa: PLC0415
    import sys  # noqa: PLC0415

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    return serve_stdio()
