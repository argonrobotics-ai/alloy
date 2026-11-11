# SPDX-License-Identifier: Apache-2.0
"""Command line: ``python -m alloy.check SCENE [--json] [--worlds N] [--seconds S] [--quick]``.

SCENE is a scene YAML path, a library scene name, or ``-`` to read the scene (YAML or JSON) from stdin. With
``--json`` stdout carries only the report (schema
``alloy.check/1``); logs go to stderr. Exit code 0: no errors; 1: the scene has errors; 2: bad arguments or a scene
that cannot be read (the JSON then has ``"error"`` and ``"fix"``).

Example::

    $ python -m alloy.check library/scenes/demo_peg_d12.yaml --json --worlds 4 --seconds 0.5
    {"schema": "alloy.check/1", "scene": "demo_peg_d12", "ok": true, ...}
    $ python -m alloy.check demo_peg_d12 --quick          # cards + overlap only, a few seconds
    $ echo '{"schema": "alloy.scene/1", "robots": [], "objects": []}' | python -m alloy.check - --json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Sequence

log = logging.getLogger("alloy.check")


def parser() -> argparse.ArgumentParser:
    """The argument parser of ``python -m alloy.check``.

    Example::

        >>> parser().parse_args(["demo_peg_d12", "--quick"]).quick
        True
    """
    p = argparse.ArgumentParser(prog="alloy check", description="Physics check of an Alloy scene: card validation, "
                                "overlap at load, rest drift, penetration, health events, energy drift.")
    p.add_argument("scene", help="scene YAML path, library scene name, or - to read YAML/JSON from stdin")
    p.add_argument("--json", action="store_true", help="print the report as JSON (only JSON on stdout)")
    p.add_argument("--worlds", type=int, default=None, help="parallel worlds (default 64 on GPU, 2 on CPU)")
    p.add_argument("--seconds", type=float, default=None, help="simulated seconds (default 2 on GPU, 0.5 on CPU)")
    p.add_argument("--device", default=None, help="'cpu' or 'cuda:0' (default: cuda:0 if available)")
    p.add_argument("--quick", action="store_true", help="load checks only (cards, overlap); no simulation run")
    p.add_argument("--moving", action="store_true",
                   help="the scene moves by design (drops, pours): gross motion is reported as info, not error")
    p.add_argument("--profile", default="standard", help="integration profile: standard, fast, precision")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=float, default=25.0,
                   help="wall seconds the run may take; a slow scene runs fewer seconds (0: no limit)")
    p.add_argument("--library", action="append", default=None,
                   help="extra card library root (repeatable), searched before the default library")
    p.add_argument("--out", default=None, help="also write the JSON report to this file")
    p.add_argument("-v", "--verbose", action="store_true", help="info logs on stderr")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check from command line arguments, print the report and return the exit code (0 ok, 1 errors,
    2 usage).

    Example::

        >>> main(["demo_peg_d12", "--quick"])  # doctest: +SKIP
        0
    """
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        with contextlib.redirect_stdout(sys.stderr):      # keep stdout for the report (Warp prints at init)
            report = _run(args)
    except ValueError as exc:
        return _usage_error(args, str(exc))
    text = report.to_json() if args.json else report.summary()
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report.to_json() + "\n")
    return 0 if report.ok else 1


def _run(args: argparse.Namespace):
    import warp as wp  # noqa: PLC0415

    wp.config.quiet = True
    from alloy.check.api import check  # noqa: PLC0415

    scene = sys.stdin.read() if args.scene == "-" else args.scene
    return check(scene, args.worlds, args.seconds, device=args.device, run=not args.quick,
                 expect_rest=not args.moving, profile=args.profile, seed=args.seed, library=args.library,
                 budget_s=args.budget if args.budget > 0 else None)


def _usage_error(args: argparse.Namespace, message: str) -> int:
    fix = "pass an existing scene YAML path or a library scene name (python -m alloy.check --help)"
    if args.json:
        print(json.dumps({"schema": "alloy.check/1", "ok": False, "error": message, "fix": fix}))
    else:
        print(f"error: {message}\nfix: {fix}", file=sys.stderr)
    return 2


__all__ = ["main", "parser"]
