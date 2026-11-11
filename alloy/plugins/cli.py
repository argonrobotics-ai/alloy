# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.plugins``: list, describe, check and scaffold plug-ins. JSON out; JSON in with ``--request``.

Examples::

    python -m alloy.plugins list --kind force
    python -m alloy.plugins describe magnet
    python -m alloy.plugins check magnet --quick
    python -m alloy.plugins new my_force --kind force       # makes ./my_force/ (project) with my_force/plugin.py
    python -m alloy.plugins check my_force                  # that folder, its package folder or its plugin.py
    python -m alloy.plugins check my_force/my_force/plugin.py --config '{"coefficient": 2.0}' --sandbox
    echo '{"command": "check", "target": "ink", "quick": true}' | python -m alloy.plugins --request -

Exit codes: 0 ok (check passed), 1 check failed, 2 bad request or plug-in error.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from alloy.plugins.errors import PluginError

log = logging.getLogger("alloy.plugins")

NO_FIX = "the message names what to change; `python -m alloy.plugins <command> --help` lists the arguments"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.plugins", description=__doc__.splitlines()[0])
    p.add_argument("--request", help="JSON request (or - for stdin): {'command': ..., <option>: ...}")
    sub = p.add_subparsers(dest="command")
    ls = sub.add_parser("list", help="registered plug-ins (built-ins + entry points)")
    ls.add_argument("--kind", choices=("force", "sensor", "object", "solver"))
    ds = sub.add_parser("describe", help="a plug-in's config fields and docs")
    ds.add_argument("target")
    ck = sub.add_parser("check", help="run the harness: laws, crash, determinism, independence, speed")
    ck.add_argument("target", help="registry name, module:Class, a plugin.py file, a package folder, or the "
                                   "project folder `new` made")
    ck.add_argument("--config", default="{}", help="JSON object of Config fields")
    ck.add_argument("--device", default="cpu")
    ck.add_argument("--quick", action="store_true", help="skip determinism, independence and speed")
    ck.add_argument("--no-speed", action="store_true")
    ck.add_argument("--sandbox", action="store_true", help="run the check in the OS sandbox (alloy.plugins.sandbox)")
    ck.add_argument("--timeout", type=float, default=1800.0, help="sandbox wall-clock limit [s]")
    ck.add_argument("--out", help="also write the JSON report here")
    nw = sub.add_parser("new", help="scaffold a plug-in package from the template")
    nw.add_argument("name", help="snake_case plug-in name")
    nw.add_argument("--kind", default="force", choices=("force", "sensor", "object", "solver"))
    nw.add_argument("--dir", default=".", help="where to create <name>/")
    return p


def _args_from_request(parser: argparse.ArgumentParser, raw: str) -> argparse.Namespace:
    text = sys.stdin.read() if raw == "-" else raw
    try:
        req = json.loads(text)
    except json.JSONDecodeError as err:
        raise PluginError(f"--request is not JSON ({err})",
                          fix='send a JSON object, e.g. {"command": "check", "target": "magnet"}')
    if not isinstance(req, dict) or "command" not in req:
        raise PluginError("--request needs a JSON object with a 'command'",
                          fix='add "command": one of list, describe, check, new')
    argv = [str(req.pop("command"))]
    for key in ("target", "name"):
        if key in req:
            argv.append(str(req.pop(key)))
    for key, value in req.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value not in (False, None):
            argv += [flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return parser.parse_args(argv)


def _cmd_list(args: argparse.Namespace) -> tuple[Any, int]:
    from alloy.plugins.registry import available  # noqa: PLC0415

    return [dataclasses.asdict(r) for r in available(args.kind)], 0


def _cmd_describe(args: argparse.Namespace) -> tuple[Any, int]:
    from alloy.plugins.registry import describe, load_target  # noqa: PLC0415

    return [describe(c) for c in load_target(args.target)], 0


def _cmd_check(args: argparse.Namespace) -> tuple[Any, int]:
    config = json.loads(args.config) if isinstance(args.config, str) else args.config
    if not isinstance(config, dict):
        raise PluginError("--config must be a JSON object of Config fields",
                          fix="pass e.g. --config '{\"coefficient\": 2.0}'; `describe <target>` lists the fields")
    if args.sandbox:
        from alloy.plugins.sandbox import sandboxed_check  # noqa: PLC0415

        report = sandboxed_check(args.target, config=config, device=args.device, quick=args.quick,
                                 speed=not args.no_speed, timeout_s=args.timeout)
    else:
        from alloy.plugins.harness.check import check  # noqa: PLC0415
        from alloy.plugins.registry import load_target  # noqa: PLC0415

        classes = load_target(args.target)
        if len(classes) != 1:
            raise PluginError(f"{args.target} holds {len(classes)} plug-ins {[c.name for c in classes]}",
                              fix="check one at a time: pass 'module:Class' or its registered name")
        report = check(classes[0](config), device=args.device, quick=args.quick, speed=not args.no_speed)
    out = report.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out, 0 if report.passed else 1


def _cmd_new(args: argparse.Namespace) -> tuple[Any, int]:
    from alloy.plugins.scaffold import new_plugin  # noqa: PLC0415

    files = new_plugin(args.name, kind=args.kind, dest=Path(args.dir))
    root = files[0].parent.parent
    return {"created": [str(f) for f in files], "plugin": str(files[0]),
            "next": f"python -m alloy.plugins check {root}", "while_editing": f"... check {root} --quick"}, 0


COMMANDS = {"list": _cmd_list, "describe": _cmd_describe, "check": _cmd_check, "new": _cmd_new}


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[Any]:
    """Send everything printed meanwhile (Warp's module messages, kernel printf) to stderr; yield the real stdout,
    so stdout carries only the JSON result."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    real, sys.stdout = sys.stdout, sys.stderr
    try:
        yield os.fdopen(os.dup(saved), "w")
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        sys.stdout = real


def main(argv: list[str] | None = None) -> int:
    """Run one command; print its JSON result (errors as ``{"error": ..., "message": ..., "fix": ...}``) and return
    the exit code."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    parser = _parser()
    with _stdout_to_stderr() as out:
        try:
            args = parser.parse_args(argv)
            if args.request:
                args = _args_from_request(parser, args.request)
            if args.command not in COMMANDS:
                parser.print_help(sys.stderr)
                return 2
            result, code = COMMANDS[args.command](args)
        except (PluginError, json.JSONDecodeError) as err:
            fix = getattr(err, "fix", "") or NO_FIX
            result, code = {"error": type(err).__name__, "message": str(err), "fix": fix}, 2
        out.write(json.dumps(result, indent=2) + "\n")
        out.close()
    return code


__all__ = ["main"]
