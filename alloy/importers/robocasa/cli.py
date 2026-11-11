# SPDX-License-Identifier: Apache-2.0
"""Command line for the RoboCasa importer; every command prints one JSON object.

    python -m alloy.importers.robocasa packs
    python -m alloy.importers.robocasa list --pack objects_lightwheel [--zip PACK.zip] [--category glass_cup]
    python -m alloy.importers.robocasa import --pack objects_lightwheel --asset glass_cup/GlassCup003 --out ROOT
    python -m alloy.importers.robocasa batch --pack objects_lightwheel --out ROOT [--category C] [--jobs 8]
    python -m alloy.importers.robocasa compose --out ROOT       # cabinets from imported panels + handles
    python -m alloy.importers.robocasa post --out ROOT          # after a --no-validate/--no-thumbs batch
    python -m alloy.importers.robocasa finalize --out ROOT
    python -m alloy.importers.robocasa fetch --root ROOT --store gs://BUCKET/robocasa

``--zip`` is a downloaded pack zip (default: read the pack's URL with range requests). ``ROOT`` is the import root,
a card library root: load it with ``CardLibrary([default_root(), ROOT])``.

Example::

    >>> from alloy.importers.robocasa.cli import main
    >>> main(["packs"])                     # doctest: +SKIP
    0
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path


def _pick_gl() -> None:
    """Headless MuJoCo GL before ``import mujoco``: OSMesa if its library exists, else rendering off (``disable``)
    so MuJoCo still imports; thumbnails then wait for the ``post`` step on a machine that can render."""
    if os.environ.get("MUJOCO_GL") or os.environ.get("DISPLAY"):
        return
    os.environ["MUJOCO_GL"] = "osmesa" if ctypes.util.find_library("OSMesa") else "disable"


def _emit(obj: object) -> int:
    print(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return 0


def _packs(_: argparse.Namespace) -> int:
    from alloy.importers.robocasa.packs import PACKS  # noqa: PLC0415

    return _emit({k: {"decision": p.decision, "licence": p.licence, "kind": p.kind, "reason": p.reason, "url": p.url}
                  for k, p in PACKS.items()})


def _assets(args: argparse.Namespace) -> list[str]:
    from alloy.importers.robocasa.source import PackReader  # noqa: PLC0415

    assets = PackReader.open(args.pack, args.zip).assets()
    if getattr(args, "category", None):
        assets = [a for a in assets if a.split("/")[0] in args.category]
    if getattr(args, "limit", None):
        assets = assets[: args.limit]
    return assets


def _list(args: argparse.Namespace) -> int:
    assets = _assets(args)
    return _emit({"pack": args.pack, "count": len(assets), "assets": assets})


def _import(args: argparse.Namespace) -> int:
    from alloy.importers.robocasa.pipeline import import_asset  # noqa: PLC0415

    rec = import_asset(args.pack, args.asset, Path(args.out), args.zip, validate=not args.no_validate,
                       thumbs=not args.no_thumbs, hulls_from=args.hulls)
    _emit({k: rec.get(k) for k in ("name", "status", "mass_kg", "flags", "error", "card", "thumbnail")})
    return 0 if rec["status"] != "failed" else 1


def _batch(args: argparse.Namespace) -> int:
    from alloy.importers.robocasa.batch import run_batch  # noqa: PLC0415

    assets = _assets(args)
    status = run_batch(args.pack, assets, Path(args.out), args.zip, args.jobs, validate=not args.no_validate,
                       thumbs=not args.no_thumbs)
    return _emit({"pack": args.pack, "assets": len(assets), "status": status})


def _post(args: argparse.Namespace) -> int:
    from alloy.importers.robocasa.batch import post_pass  # noqa: PLC0415

    return _emit(post_pass(Path(args.out)))


def _compose(args: argparse.Namespace) -> int:
    from alloy.importers.robocasa.cabinets import compose_all  # noqa: PLC0415

    return _emit(compose_all(Path(args.out), validate=not args.no_validate, thumbs=not args.no_thumbs))


def _finalize(args: argparse.Namespace) -> int:
    from alloy.importers.robocasa.batch import finalize  # noqa: PLC0415

    return _emit(finalize(Path(args.out)))


def _fetch(args: argparse.Namespace) -> int:
    from alloy.importers.robocasa.store import fetch  # noqa: PLC0415

    return _emit(fetch(Path(args.root), args.store))


def _skip_flags(s: argparse.ArgumentParser) -> None:
    s.add_argument("--no-validate", action="store_true", help="skip card validation (needs the engine); run post")
    s.add_argument("--no-thumbs", action="store_true", help="skip thumbnails (needs GL); run post later")


def parser() -> argparse.ArgumentParser:
    """The argument parser (subcommands: packs, list, import, batch, compose, post, finalize, fetch)."""
    p = argparse.ArgumentParser(prog="python -m alloy.importers.robocasa", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("packs", help="RoboCasa packs and their licence decisions").set_defaults(fn=_packs)
    for name, fn in (("list", _list), ("batch", _batch)):
        s = sub.add_parser(name)
        s.add_argument("--pack", required=True)
        s.add_argument("--zip", default=None)
        s.add_argument("--category", nargs="*", default=None)
        s.add_argument("--limit", type=int, default=None)
        if name == "batch":
            s.add_argument("--out", required=True)
            s.add_argument("--jobs", type=int, default=4)
            _skip_flags(s)
        s.set_defaults(fn=fn)
    s = sub.add_parser("import", help="import one asset")
    s.add_argument("--pack", required=True)
    s.add_argument("--asset", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--zip", default=None)
    s.add_argument("--hulls", choices=("coacd", "given"), default="coacd",
                   help="collision hulls: CoACD (default) or the source model's collision meshes")
    _skip_flags(s)
    s.set_defaults(fn=_import)
    s = sub.add_parser("compose", help="build articulated cabinets from imported door panels and handles")
    s.add_argument("--out", required=True)
    _skip_flags(s)
    s.set_defaults(fn=_compose)
    s = sub.add_parser("post", help="validate cards and render thumbnails of pending records")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_post)
    s = sub.add_parser("finalize", help="write manifest, report, sourcing queue, attribution")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_finalize)
    s = sub.add_parser("fetch", help="download the binaries a manifest lists")
    s.add_argument("--root", required=True)
    s.add_argument("--store", required=True)
    s.set_defaults(fn=_fetch)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the command; errors print as JSON with what to change."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    _pick_gl()
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (KeyError, PermissionError, FileNotFoundError, ValueError, OSError) as exc:
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        return 2


__all__ = ["main", "parser"]
