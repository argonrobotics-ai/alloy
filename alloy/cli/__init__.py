# SPDX-License-Identifier: Apache-2.0
"""``alloy`` on the command line: build, step, run, check, render, estimate, import and export scenes, with JSON in
and JSON out, for AI agents and scripts. ``alloy mcp`` serves the same operations as an MCP tool server.

Every reply is one JSON object on stdout (logs go to stderr)::

    {"ok": true, "op": "step", "result": {"t": 0.5, "health": {"ok": true, ...}, "state": {"cube.pos": [...]}}}
    {"ok": false, "op": "step", "error": {"type": "SceneError", "message": "...", "fix": "...", "field": "size"}}

Examples::

    alloy list scenes                                       # library scenes (also: objects, materials, adds, ops)
    alloy ops step                                          # an operation's JSON schema and an example
    alloy step demo_peg_d12 --steps 50 --names peg_d12      # build, step, print state and health
    alloy check demo_peg_d12 --quick                        # cards and overlaps, no run
    alloy run '{"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1],
               "pos": [0, 0, 0.5], "mass": 1.0}]}' --seconds 1 --out cube.npz
    alloy batch requests.json                               # many calls, one process: build once, step many times

Without the console script: ``python -m alloy.cli ...``. From Python: ``alloy.cli.main(["list", "scenes"])``.
"""

from __future__ import annotations

from typing import IO, Any

__all__ = ["main", "respond"]


def main(argv: list[str] | None = None, stdout: IO[str] | None = None) -> int:
    """Run the ``alloy`` command with ``argv`` (default: ``sys.argv[1:]``); returns the exit code.

    Example: ``alloy.cli.main(["list", "scenes", "--match", "peg"])``."""
    from alloy.cli.command import main as _main  # noqa: PLC0415

    return _main(argv, stdout)


def respond(request: dict[str, Any]) -> dict[str, Any]:
    """Run one request in a fresh session and close it: ``respond({"op": "list", "kind": "scenes"})``.

    Simulations do not outlive the call; use ``alloy.cli.session.Session`` with ``alloy.cli.registry.respond`` to
    keep them open. Example: ``respond({"op": "check", "scene": "demo_peg_d12", "quick": True})["result"]["ok"]``."""
    from alloy.cli.registry import respond as _respond  # noqa: PLC0415
    from alloy.cli.session import Session  # noqa: PLC0415

    session = Session()
    try:
        return _respond(session, request)
    finally:
        session.close_all()
