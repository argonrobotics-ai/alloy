# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.audit <scene>``: JSON on stdout. Exit codes: 0 no error finding, 1 errors found, 2 bad input.

    python -m alloy.audit my_scenes/scenes/scene.yaml --rounds 3 --episodes 32 --library my_scenes

Example::

    >>> main(["demo_peg_d12", "--rounds", "1", "--episodes", "4"])      # doctest: +SKIP
    0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence


EPILOG = ('output: {"schema": "alloy.audit/1", "ok", "counts": {"error", "warn", "info"}, "findings": [{"severity", '
          '"source" (check or bully), "kind", "message", "fix", "objects", "rate", "first_time_s", "replay"}], '
          '"runs", "notes", "timings_s"}')


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.audit", description=__doc__.split("\n\n")[0], epilog=EPILOG)
    p.add_argument("scene", help="a scene YAML path or library scene name")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--worlds", type=int, default=None)
    p.add_argument("--seconds", type=float, default=None, help="simulated seconds per check round and per bully "
                   "episode (defaults: alloy.check's and the bully's 2 s)")
    p.add_argument("--device", default=None)
    p.add_argument("--library", action="append", default=[], help="extra card folder, searched first")
    p.add_argument("--no-bully", action="store_true", help="alloy.check rounds only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        ns = _parser().parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            return 0
        print(json.dumps({"error": "UsageError", "message": "the arguments do not parse (see stderr)",
                          "fix": "run python -m alloy.audit --help", "field": None}))
        return 2
    from alloy.cli.jsonio import json_stdout  # noqa: PLC0415

    with json_stdout() as out:                        # Warp's start-up lines and any print go to stderr
        code, text = _run(ns)
        out.write(text + "\n")
    return code


def _run(ns: argparse.Namespace) -> tuple[int, str]:
    from alloy.audit.run import audit  # noqa: PLC0415

    try:
        report = audit(ns.scene, rounds=ns.rounds, episodes=ns.episodes, seed=ns.seed, device=ns.device,
                       worlds=ns.worlds, seconds=ns.seconds, library=ns.library, bully=not ns.no_bully)
    except ValueError as exc:
        return 2, json.dumps({"error": type(exc).__name__, "message": str(exc),
                              "fix": "change the argument it names", "field": None})
    except Exception as exc:  # noqa: BLE001 - JSON out, always; the traceback is on stderr
        from alloy.api.errors import remedy_for  # noqa: PLC0415

        logging.getLogger("alloy.audit").exception("audit failed")
        return 1, json.dumps({"error": type(exc).__name__, "message": str(exc)[:600], "fix": remedy_for(exc),
                              "field": None})
    return (0 if report.ok else 1), report.to_json()


__all__ = ["main"]
