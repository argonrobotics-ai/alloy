# SPDX-License-Identifier: Apache-2.0
"""Command line for the Objaverse importer; every command prints one JSON object.

    python -m alloy.importers.objaverse select --meta lvis_meta.jsonl.gz --out candidates.json
    python -m alloy.importers.objaverse survey --candidates candidates.json --out survey/ --jobs 32
    python -m alloy.importers.objaverse review --rows survey/rows.jsonl --verdicts verdicts.json --out kept.json
    python -m alloy.importers.objaverse recentre --meshes out/ --results-out results.jsonl
    python -m alloy.importers.objaverse assemble --candidates c.json --kept kept.json --results results.jsonl \
        --meshes out/ --root alloy-assets/imports/objaverse --target 2000
    python -m alloy.importers.objaverse fetch --root alloy-assets/imports/objaverse --store <asset store>

``select`` reads Objaverse annotation rows (JSON lines, optionally gzipped; fields as in
:func:`alloy.importers.objaverse.select.screen`). ``review`` applies reviewer verdicts ({uid: {keep, why, category}})
to the survey rows that passed the quality checks.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from alloy.importers.objaverse.categories import load_categories
from alloy.importers.objaverse.recentre import recentre_dir
from alloy.importers.objaverse.review import apply_verdicts
from alloy.importers.objaverse.select import select


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_select(a: argparse.Namespace) -> dict[str, Any]:
    cands, rejects = select(_read_jsonl(a.meta))
    a.out.write_text(json.dumps([c.as_row() for c in cands]))
    if a.rejects:
        a.rejects.write_text("\n".join(json.dumps(r) for r in rejects) + "\n")
    stages = collections.Counter(r["stage"] for r in rejects)
    return {"candidates": len(cands), "rejected": dict(stages), "out": str(a.out)}


def cmd_review(a: argparse.Namespace) -> dict[str, Any]:
    rows = [r for r in _read_jsonl(a.rows) if r.get("verdict") == "ok"]
    out = apply_verdicts(rows, json.loads(a.verdicts.read_text()), set(load_categories()))
    kept = [r for r in out if r["review"].startswith("ok")]
    a.out.write_text(json.dumps(kept))
    return {"surveyed_ok": len(rows), "kept": len(kept),
            "review": dict(collections.Counter(r["review"].split(":")[0].split(" (")[0] for r in out))}


def cmd_recentre(a: argparse.Namespace) -> dict[str, Any]:
    names = set(json.loads(a.names.read_text())) if a.names else None
    dirs = sorted(d for d in a.meshes.iterdir() if (d / "result.json").is_file() and (names is None or d.name in names))
    rows, moved = [], []
    for d in dirs:
        before = json.loads((d / "result.json").read_text()).get("origin_shift")
        rows.append(recentre_dir(d))
        if rows[-1].get("origin_shift") != before:
            moved.append(d.name)
    if a.results_out:
        a.results_out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return {"folders": len(dirs), "moved": len(moved), "results": str(a.results_out) if a.results_out else None}


def cmd_assemble(a: argparse.Namespace) -> dict[str, Any]:
    from alloy.importers.objaverse.assemble import pick, wall_probe, write_root  # noqa: PLC0415 - needs the engine

    results = {r["name"]: r for r in _read_jsonl(a.results)}
    exclude = frozenset(json.loads(a.exclude.read_text())) if a.exclude else frozenset()
    picked = pick(json.loads(a.candidates.read_text()), json.loads(a.kept.read_text()), results, a.target, exclude)
    dirs = {p["result"]["name"]: a.meshes / p["result"]["name"] for p in picked}
    shas = {p["result"]["name"]: p["result"]["files"]["physics.obj"] for p in picked}
    old = a.root / "manifest.json"
    cached = {n: e["physics"] for n, e in json.loads(old.read_text())["objects"].items()} if old.is_file() else {}
    walls = wall_probe(dirs, shas, a.workers, cached)
    from alloy.importers.objaverse.check import prune  # noqa: PLC0415

    stale = sorted(set(cached) - set(dirs))           # objects of an earlier run that this pick leaves out
    pruned = prune(a.root, stale) if stale else 0
    out = write_root(a.root, picked, dirs, walls) | {"pruned_stale": pruned}
    return {"picked": len(picked), **out, "root": str(a.root)}


def cmd_fetch(a: argparse.Namespace) -> dict[str, Any]:
    from alloy.importers.common import manifest as mf  # noqa: PLC0415

    return mf.fetch(a.root, mf.load(a.root / "manifest.json", "objaverse"), None, a.store)


def _add_build_commands(sub: Any) -> None:
    c = sub.add_parser("recentre", help="move each mesh folder's origin to its physics-mesh bounding-box centre")
    c.add_argument("--meshes", type=Path, required=True, help="mesh-pipeline output: <name>/result.json, ...")
    c.add_argument("--names", type=Path, help="JSON list of folder names to do (default: every folder)")
    c.add_argument("--results-out", type=Path, help="write the updated result.json rows here (JSON lines)")
    s = sub.add_parser("assemble", help="pick the best objects and write cards, manifest, attribution, queue")
    s.add_argument("--candidates", type=Path, required=True)
    s.add_argument("--kept", type=Path, required=True)
    s.add_argument("--results", type=Path, required=True, help="JSON lines of mesh-pipeline result.json rows")
    s.add_argument("--meshes", type=Path, required=True, help="folder with <name>/physics.obj, physics.hulls.obj")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--target", type=int, default=2000)
    s.add_argument("--workers", type=int, default=3, help="processes for the wall probe")
    s.add_argument("--exclude", type=Path, help="JSON list of card names to leave out (e.g. failed checks)")
    f = sub.add_parser("fetch", help="download the meshes the manifest lists (sha256-checked)")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--store", help="asset store (directory, gs:// or https:// prefix); default $ALLOY_ASSET_STORE")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.importers.objaverse", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="annotations -> ranked, capped candidates")
    s.add_argument("--meta", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--rejects", type=Path)
    sub.add_parser("survey", help="fetch + quality + previews (see worker.py)", add_help=False)
    r = sub.add_parser("review", help="apply reviewer verdicts to survey rows")
    r.add_argument("--rows", type=Path, required=True)
    r.add_argument("--verdicts", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    _add_build_commands(sub)
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["survey"]:
        from alloy.importers.objaverse.worker import main as worker_main  # noqa: PLC0415

        return worker_main(argv)
    a = ap.parse_args(argv)
    cmds = {"select": cmd_select, "review": cmd_review, "recentre": cmd_recentre, "assemble": cmd_assemble,
            "fetch": cmd_fetch}
    print(json.dumps(cmds[a.cmd](a)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
