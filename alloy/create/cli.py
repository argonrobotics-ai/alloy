# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.create``: one object card from a mesh, a photo or a text; JSON on stdout. Exit codes: 0 card
written and drop test passed, 1 drop test failed, 2 bad input (the JSON error says what to change).

    python -m alloy.create --mesh part.stl --units mm --material steel --out my_objects
    python -m alloy.create --text "a red ceramic coffee mug" --size 0.11 --category mug
    python -m alloy.create --request request.json          # a CreateRequest as JSON ('-' reads stdin)

Example::

    >>> main(["--mesh", "mug.obj", "--size", "0.11"])      # doctest: +SKIP
    0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from alloy.create.request import UNITS, UP_AXES, CreateError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.create", description=__doc__.split("\n\n")[0])
    for flag in ("mesh", "photo", "text", "name", "mass-source", "material", "category"):
        p.add_argument(f"--{flag}")
    p.add_argument("--size", type=float, help="real longest side, m")
    p.add_argument("--mass", type=float, help="kg")
    p.add_argument("--units", choices=sorted(UNITS))
    p.add_argument("--up", default="z", choices=UP_AXES)
    p.add_argument("--out", default="my_objects")
    p.add_argument("--request", help="a CreateRequest JSON file ('-': stdin) instead of the flags")
    p.add_argument("--no-check", action="store_true", help="skip the drop test")
    p.add_argument("--device", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the exit code."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    from alloy.cli.spec import dash_values  # noqa: PLC0415

    parser = _parser()
    try:                                              # --up -y: a value starting with "-" stays the flag's value
        ns = parser.parse_args(dash_values(parser, sys.argv[1:] if argv is None else argv))
    except SystemExit as exc:
        if exc.code in (0, None):
            return 0
        print(json.dumps({"error": "UsageError", "message": "the arguments do not parse (see stderr)",
                          "fix": "run python -m alloy.create --help", "field": None}))
        return 2
    from alloy.cli.jsonio import json_stdout  # noqa: PLC0415

    with json_stdout() as out:                        # Warp's start-up lines and any print go to stderr
        code, text = _run(ns)
        out.write(text + "\n")
    return code


def _run(ns: argparse.Namespace) -> tuple[int, str]:
    try:
        from alloy.create.build import create  # noqa: PLC0415

        result = create(_request(ns), check=not ns.no_check, device=ns.device)
    except CreateError as exc:
        return 2, json.dumps(exc.to_dict(), indent=2)
    except Exception as exc:  # noqa: BLE001 - JSON out, always; the traceback is on stderr
        return 1, _crash(exc)
    return (0 if result.ok else 1), result.to_json()


def _crash(exc: Exception) -> str:
    from alloy.api.errors import remedy_for  # noqa: PLC0415

    logging.getLogger("alloy.create").exception("create failed")
    return json.dumps({"error": type(exc).__name__, "message": str(exc)[:600], "fix": remedy_for(exc),
                       "field": None}, indent=2)


def _request(ns: argparse.Namespace) -> dict:
    if ns.request:
        try:
            text = sys.stdin.read() if ns.request == "-" else Path(ns.request).read_text(encoding="utf-8")
            return json.loads(text)
        except (OSError, ValueError) as exc:
            raise CreateError(f"cannot read the request {ns.request!r}: {exc}", "pass a JSON object such as "
                              '{"mesh": "part.stl", "units": "mm"}', field="request") from exc
    keys = ("mesh", "photo", "text", "name", "size", "units", "up", "mass", "mass_source", "material", "category",
            "out")
    return {k: getattr(ns, k) for k in keys if getattr(ns, k) is not None}


__all__ = ["main"]
