# SPDX-License-Identifier: Apache-2.0
"""Command line for alloy.io: JSON out on stdout, logs on stderr, exit code 0 on success and 1 on an error.

Examples::

    python -m alloy.io export --scene demo_peg_d12 --out out/peg
    python -m alloy.io export --object cube_beech_30 --out out/cube --formats urdf,usd
    python -m alloy.io import robot.urdf --out my_library            # cards from a file
    echo '{"op": "export", "scene": "demo_peg_d12", "out": "out/peg"}' | python -m alloy.io request -

Errors print ``{"ok": false, "error": "<what went wrong and what to change>"}``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("alloy.io")


def run_request(req: dict) -> dict:
    """Run one JSON request: ``{"op": "export", "scene"|"object": ..., "out": ..., "formats": [...],
    "library": ...}`` or ``{"op": "import", "file": ..., "out": ..., "name": ...}``."""
    op = req.get("op")
    if op == "export":
        from alloy.io.api import FORMATS, export_object, export_scene  # noqa: PLC0415

        kw = {"formats": tuple(req.get("formats") or FORMATS), "library": req.get("library")}
        if "out" not in req:
            raise ValueError('export needs "out": the folder to write into')
        if req.get("object"):
            return export_object(req["object"], req["out"], **kw).to_json()
        if req.get("scene"):
            return export_scene(req["scene"], req["out"], **kw).to_json()
        raise ValueError('export needs "scene" (library name or .yaml path) or "object" (library name)')
    if op == "import":
        from alloy.io.importer import import_file  # noqa: PLC0415

        if "file" not in req or "out" not in req:
            raise ValueError('import needs "file" (MJCF, URDF or USD) and "out" (a library folder to write cards to)')
        return import_file(req["file"], req["out"], name=req.get("name")).to_json()
    raise ValueError(f'unknown op {op!r}; use "export" or "import"')


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.io", description="Alloy import/export (JSON out)")
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="scene or object card -> MJCF, URDF, USD")
    ex.add_argument("--scene", help="library scene name or scene .yaml path")
    ex.add_argument("--object", help="library object name")
    ex.add_argument("--out", required=True, help="output folder")
    ex.add_argument("--formats", default="mjcf,urdf,usd", help="comma list of mjcf, urdf, usd")
    ex.add_argument("--library", help="extra card library folder, searched first")
    im = sub.add_parser("import", help="MJCF, URDF or USD file -> cards")
    im.add_argument("file")
    im.add_argument("--out", required=True, help="library folder to write cards into")
    im.add_argument("--name", help="scene name (default: the file stem)")
    rq = sub.add_parser("request", help="one JSON request from a file or - (stdin)")
    rq.add_argument("path")
    return p


def _request_of(args: argparse.Namespace) -> dict:
    if args.cmd == "request":
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text()
        return json.loads(text)
    if args.cmd == "export":
        return {"op": "export", "scene": args.scene, "object": args.object, "out": args.out,
                "formats": [f.strip() for f in args.formats.split(",") if f.strip()], "library": args.library}
    return {"op": "import", "file": args.file, "out": args.out, "name": args.name}


def _quiet_warp() -> None:
    """Warp prints its banner and kernel loads on stdout (bypassing redirects); stdout is for the JSON answer."""
    try:
        import warp as wp  # noqa: PLC0415

        wp.config.quiet = True
    except ImportError:
        pass


def json_of(stdout: str) -> dict:
    """The JSON answer in a captured stdout, skipping any lines a library printed before it."""
    start = next((i for i, line in enumerate(stdout.splitlines(keepends=True)) if line.startswith("{")), None)
    if start is None:
        raise ValueError(f"no JSON answer in the output: {stdout[:200]!r}")
    return json.loads("".join(stdout.splitlines(keepends=True)[start:]))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit code."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    _quiet_warp()
    args = _parser().parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):    # Warp prints its banner and kernel loads on stdout
            out = {"ok": True, **run_request(_request_of(args))}
        code = 0
    except Exception as exc:  # noqa: BLE001 - every failure becomes a JSON error for the caller
        log.debug("alloy.io failed", exc_info=True)
        out, code = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 1
    print(json.dumps(out, indent=2))
    return code
