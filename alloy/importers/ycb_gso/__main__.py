# SPDX-License-Identifier: Apache-2.0
"""Command line for the YCB + GSO importer; every command prints JSON.

Examples::

    python -m alloy.importers.ycb_gso list --lib ycb                 # cards, masses, hull counts
    python -m alloy.importers.ycb_gso fetch ycb_025_mug --store DIR  # binaries for one card (sha256-checked)
    python -m alloy.importers.ycb_gso jobs --out jobs --chunks 4     # batch job files (all YCB + GSO)
    python -m alloy.importers.ycb_gso build OUT [OUT ...]            # cards + manifest + queue from batch outputs
    python -m alloy.importers.ycb_gso check [NAME ...] --workers 3   # card validators + drop test
    python -m alloy.importers.ycb_gso publish OUT [OUT ...] --store gs://BUCKET

``--root`` defaults to ``<alloy-assets>/imports/ycb_gso`` (``$ALLOY_ASSETS`` or the sibling checkout); ``--store``
defaults to ``$ALLOY_ASSET_STORE``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from alloy.cards.roots import assets_root
from alloy.importers.common import manifest as mf
from alloy.importers.common.errors import ImporterError
from alloy.importers.ycb_gso import build as bd
from alloy.importers.ycb_gso import gso, jobs

UPLOAD_EXCLUDE = r".*\.(json|pbtxt)$"


def default_root() -> Path:
    """``<alloy-assets>/imports/ycb_gso``; raises with the fix when no assets checkout is found.

    Example::

        default_root() / "manifest.json"
    """
    a = assets_root()
    if a is None:
        raise ImporterError("<cli>", "root", "no alloy-assets checkout: pass --root or set $ALLOY_ASSETS")
    return a / "imports" / bd.LIB


def _cmd_list(a: argparse.Namespace) -> dict:
    man = mf.load(a.root / "manifest.json", bd.LIB)
    rows = [{"name": n, "library": e["library"], "title": e["title"], "mass_kg": e["physics"]["mass_kg"],
             "mass": e["physics"].get("mass_status", "estimate"),
             "hulls": e["physics"]["n_hulls"]} for n, e in man["objects"].items()
            if a.lib in (None, e["library"])]
    return {"count": len(rows), "objects": rows}


def _cmd_fetch(a: argparse.Namespace) -> dict:
    man = mf.load(a.root / "manifest.json", bd.LIB)
    return mf.fetch(a.root, man, a.names or None, a.store)


def _cmd_jobs(a: argparse.Namespace) -> dict:
    paths = jobs.write_chunks(jobs.ycb_jobs(), a.out, "ycb", 1)
    paths += jobs.write_chunks(jobs.gso_jobs(gso.fetch_listing()), a.out, "gso", a.chunks)
    return {"files": [str(p) for p in paths]}


def _cmd_build(a: argparse.Namespace) -> dict:
    return bd.build(a.outs, a.root)


def _cmd_check(a: argparse.Namespace) -> dict:
    from alloy.importers.ycb_gso.check import run_checks  # noqa: PLC0415 - imports Newton

    return run_checks(a.root, a.names or None, a.workers)


def _cmd_publish(a: argparse.Namespace) -> dict:
    dest = f"{mf.store_root(a.store)}/{bd.LIB}/{mf.MESH_DIR}"
    for out in a.outs:
        subprocess.run(["gcloud", "storage", "rsync", "-r", "-x", UPLOAD_EXCLUDE, str(out), dest], check=True,
                       capture_output=True)
    return {"uploaded_from": [str(o) for o in a.outs], "to": dest}


def parser() -> argparse.ArgumentParser:
    """The argument parser (one sub-command per ``COMMANDS`` entry).

    Example::

        parser().parse_args(["list", "--lib", "ycb"]).cmd   # "list"
    """
    ap = argparse.ArgumentParser(prog="python -m alloy.importers.ycb_gso", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="import library root (default: alloy-assets)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--lib", choices=("ycb", "gso"))
    p = sub.add_parser("fetch")
    p.add_argument("names", nargs="*")
    p.add_argument("--store")
    p = sub.add_parser("jobs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunks", type=int, default=4)
    p = sub.add_parser("build")
    p.add_argument("outs", type=Path, nargs="+")
    p = sub.add_parser("check")
    p.add_argument("names", nargs="*")
    p.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("publish")
    p.add_argument("outs", type=Path, nargs="+")
    p.add_argument("--store")
    return ap


COMMANDS = {"list": _cmd_list, "fetch": _cmd_fetch, "jobs": _cmd_jobs, "build": _cmd_build, "check": _cmd_check,
            "publish": _cmd_publish}


def main(argv: list[str] | None = None) -> int:
    """Run one command; prints its JSON result, or ``{"error": ...}`` and exit 1.

    Example::

        main(["list", "--lib", "ycb"])
    """
    a = parser().parse_args(argv)
    try:
        a.root = a.root or default_root()
        out = COMMANDS[a.cmd](a)
    except ImporterError as exc:
        json.dump({"error": str(exc), "step": exc.step}, sys.stdout)
        sys.stdout.write("\n")
        return 1
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
