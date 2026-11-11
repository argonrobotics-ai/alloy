# SPDX-License-Identifier: Apache-2.0
"""Run the Gym-Anything exporters in Gym-Anything's own Python 3.12 env; prints a JSON summary on stdout.

Usage (from any directory)::

    <gym-anything-env>/bin/python tools/gymanything_export/run.py \\
        --repo <gym-anything checkout> --out <specs dir> [--sim wine_pouring --sim pencap]

Each ``ex_<sim>.py`` next to this file defines ``export(x)``; with no ``--sim`` every exporter runs. Exit code 0 if
every requested exporter ran (individual object failures are listed in the summary), 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def available() -> list[str]:
    """Sims that have an exporter (``ex_<sim>.py``)."""
    return sorted(p.stem[3:] for p in HERE.glob("ex_*.py"))


def _paths(repo: Path) -> None:
    for p in (HERE, repo / "src", repo / "benchmarks/roboworld/environments"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def run_one(sim: str, out: Path) -> dict:
    """Run ``ex_<sim>.export``; a crash of the whole exporter is reported, not raised."""
    from _common import Exporter  # noqa: PLC0415 - needs sys.path from _paths

    x = Exporter(sim, out)
    try:
        importlib.import_module(f"ex_{sim}").export(x)
    except Exception as exc:  # noqa: BLE001
        return {**x.summary(), "crashed": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
    return x.summary()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=os.environ.get("GAR_REPO"), help="Gym-Anything for Robotics checkout")
    ap.add_argument("--out", required=True, help="output folder for <sim>/<key>.json + .obj")
    ap.add_argument("--sim", action="append", help="sim to export (repeat); default: all with an exporter")
    a = ap.parse_args(argv)
    if not a.repo or not Path(a.repo, "benchmarks/roboworld/environments").is_dir():
        print(json.dumps({"error": "pass --repo <gym-anything checkout> (or set GAR_REPO)"}))
        return 1
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    _paths(Path(a.repo).resolve())
    sims = a.sim or available()
    unknown = sorted(set(sims) - set(available()))
    if unknown:
        print(json.dumps({"error": f"no exporter for {unknown}; available: {available()}"}))
        return 1
    results = [run_one(s, Path(a.out)) for s in sims]
    print(json.dumps({"results": results}, indent=1))
    return 0 if not any("crashed" in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
