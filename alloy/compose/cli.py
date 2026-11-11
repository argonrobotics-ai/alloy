# SPDX-License-Identifier: Apache-2.0
"""Command line: only JSON on stdout (engine start-up lines go to stderr). Exit code 0 = the base scene passed,
1 = built but not passing, 2 = bad input.

    python -m alloy.compose build "a mug and three cubes on a table" --variations 2 --out my_scenes
    python -m alloy.compose build --spec spec.json --check load          # spec: alloy.compose.SceneSpec as JSON
    python -m alloy.compose parse "three nuts in a tray"                 # the spec the sentence gives
    python -m alloy.compose search mug --hint ceramic -k 5               # library objects and whether they place

A first argument that is not a command is read as a sentence for ``build``.

Example::

    >>> main(["parse", "two cubes on the floor"])                          # doctest: +SKIP
    0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alloy.compose.spec import ComposeError

COMMANDS = ("build", "parse", "search")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.compose", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="sentence or spec -> checked scene cards (+ variations)")
    b.add_argument("sentence", nargs="?", help="the scene in words")
    b.add_argument("--spec", help="a SceneSpec JSON file ('-' reads stdin) instead of a sentence")
    b.add_argument("--variations", type=int, default=0)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--check", default="full", choices=("full", "load", "none"))
    b.add_argument("--device", default=None, help="cpu or cuda:0 (default: GPU if present)")
    b.add_argument("--worlds", type=int, default=None, help="worlds in the full check")
    b.add_argument("--seconds", type=float, default=None, help="seconds simulated in the full check")
    b.add_argument("--out", default=None, help="folder to save scenes/<name>.yaml and compose.json")
    b.add_argument("--root", action="append", default=[], help="your own card folder, searched first")
    b.add_argument("--no-swap", action="store_true", help="variations keep the base scene's objects")
    b.add_argument("--robot", action="append", default=[], help="a robot file (MJCF) to stand at the table edge")
    b.add_argument("--jitter", action="store_true", help="each world moves the free objects a little")
    b.add_argument("--scene-worlds", type=int, default=None, help="parallel worlds in the saved scene card")
    b.add_argument("--area", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                   help="where objects may go on the table (m)")
    pa = sub.add_parser("parse", help="sentence -> SceneSpec JSON")
    pa.add_argument("sentence")
    s = sub.add_parser("search", help="library objects matching a noun")
    s.add_argument("query")
    s.add_argument("--hint", action="append", default=[], help="material or size word, e.g. wooden, small")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--root", action="append", default=[])
    s.add_argument("--fast", action="store_true", help="names only: skip reading and measuring each hit")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code. Example: ``main(["search", "cube"])``."""
    from alloy.cli.jsonio import json_stdout  # noqa: PLC0415 - stdout carries only the JSON

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in COMMANDS and not args[0].startswith("-"):
        args = ["build", *args]
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    with json_stdout() as stream:
        try:
            ns = _parser().parse_args(args)
            out, code = _run(ns)
        except ComposeError as exc:
            out, code = exc.to_dict(), 2
        except SystemExit as exc:                   # argparse: --help (0) or a usage error (printed to stderr)
            if exc.code in (0, None):
                return 0
            out, code = {"error": "UsageError", "message": "the arguments do not parse (see stderr)",
                         "fix": "run python -m alloy.compose --help", "field": None}, 2
        stream.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    return code


def _run(ns: argparse.Namespace) -> tuple[Any, int]:
    if ns.command == "parse":
        from alloy.compose.parse import parse  # noqa: PLC0415

        return parse(ns.sentence).to_dict(), 0
    if ns.command == "search":
        from alloy.compose import search  # noqa: PLC0415

        rows = search(ns.query, tuple(ns.hint), ns.k, roots=tuple(ns.root), placeable=not ns.fast)
        return {"query": ns.query, "hints": ns.hint, "matches": rows}, 0
    return _build(ns)


def _build(ns: argparse.Namespace) -> tuple[Any, int]:
    from alloy.compose.build import compose  # noqa: PLC0415

    if (ns.sentence is None) == (ns.spec is None):
        raise ComposeError("build needs a sentence or --spec (one of them)",
                           "pass a sentence in quotes, or --spec spec.json")
    request: Any = ns.sentence if ns.spec is None else _read_spec(ns.spec)
    result = compose(request, variations=ns.variations, seed=ns.seed, check=ns.check, device=ns.device,
                     worlds=ns.worlds, seconds=ns.seconds, roots=ns.root, swap=not ns.no_swap,
                     robots=ns.robot, area=ns.area, jitter=True if ns.jitter else None,
                     scene_worlds=ns.scene_worlds)
    out = result.to_dict()
    if ns.out:
        out["saved"] = [str(p) for p in result.save(ns.out)]
    return out, 0 if result.ok else 1


def _read_spec(path: str) -> dict:
    try:
        text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, ValueError) as exc:
        raise ComposeError(f"cannot read the spec {path!r}: {exc}", "pass a JSON file written by "
                           "'python -m alloy.compose parse' or SceneSpec.to_dict()", field="spec") from exc


__all__ = ["main"]
