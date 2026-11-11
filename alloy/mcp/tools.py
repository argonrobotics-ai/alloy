# SPDX-License-Identifier: Apache-2.0
"""MCP tool descriptors and results for the ``alloy.cli`` operations (one tool per operation, same JSON).

Example::

    from alloy.mcp.tools import tool_list, call_tool
    from alloy.cli.session import Session
    [t["name"] for t in tool_list()]                      # ['list', 'scene', 'build', 'step', ...]
    call_tool(Session(), "list", {"kind": "scenes"})       # {"content": [{"type": "text", ...}], "isError": False}
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from alloy.cli.jsonio import dumps
from alloy.cli.registry import OPS, respond_op
from alloy.cli.session import Session
from alloy.cli.spec import Op, json_schema

MAX_INLINE_IMAGE = 4 * 1024 * 1024        # bytes of PNG sent back inline from render

INSTRUCTIONS = """Alloy is a GPU physics engine for robots and objects (rigid, cloth, liquid). Units are SI \
(m, kg, s), z is up.
Typical flow: list (scenes/objects/materials) -> scene (check a description, fast) -> build (returns a sim id; the \
first build of a new scene compiles kernels, about a minute) -> step / run / state / targets -> check -> render \
(returns a PNG) -> close. vary writes n variants of a scene (objects moved, textures swapped) as Scene JSON files \
that every tool takes as scene.
A scene is a library scene name, a file path (.yaml card, .xml MJCF, .urdf, .usd, Scene .json), or a recipe: \
{"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "pos": [0, 0, 0.5], \
"mass": 1.0}]} (list kind=adds shows every add kind and its arguments).
Every error says what to change in its "fix"."""


def tool(op: Op) -> dict[str, Any]:
    """The MCP tool descriptor of an operation. Example: ``tool(OPS["step"])["inputSchema"]["properties"]``."""
    example = json.dumps({k: v for k, v in op.example.items()})
    return {"name": op.name, "title": op.summary.split(";")[0].rstrip("."),
            "description": f"{op.description}\nExample arguments: {example}",
            "inputSchema": json_schema(op),
            "annotations": {"readOnlyHint": op.read_only, "destructiveHint": False, "idempotentHint": op.read_only,
                            "openWorldHint": False}}


def tool_list() -> list[dict[str, Any]]:
    """Every operation as an MCP tool. Example: ``len(tool_list()) == len(alloy.cli.registry.OPS)``."""
    return [tool(op) for op in OPS.values()]


def call_tool(session: Session, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Run a tool and return the MCP ``CallToolResult``: the result (or error with its fix) as JSON text, plus the
    PNG an operation names in ``Op.image`` (render's picture, vary's contact sheet). Raises ``KeyError`` for an unknown tool. Example: ``call_tool(s, "scene", {...})``."""
    if name not in OPS:
        raise KeyError(name)
    reply = respond_op(session, name, arguments or {})
    body = reply["result"] if reply["ok"] else {"error": reply["error"]}
    content: list[dict[str, Any]] = [{"type": "text", "text": dumps(body)}]
    if reply["ok"] and OPS[name].image:
        content += _image(reply["result"].get(OPS[name].image))
    return {"content": content, "isError": not reply["ok"]}


def _image(path: str | None) -> list[dict[str, Any]]:
    p = Path(path) if path else None
    if p is None or not p.is_file() or p.stat().st_size > MAX_INLINE_IMAGE:
        return []
    return [{"type": "image", "data": base64.b64encode(p.read_bytes()).decode("ascii"), "mimeType": "image/png"}]
