# SPDX-License-Identifier: Apache-2.0
"""Command line for the render gates; every command prints one JSON object on stdout.

    python -m bench.render_gates spec
    python -m bench.render_gates score ROOT --gpu l4 --out report_l4.json [--engines alloy,alloy_e2e]
    python -m bench.render_gates matrix --report l4=report_l4.json --report rtx_pro_6000=report_rtx.json \
        [--rows speed.jsonl] --out matrix.json [--md MATRIX.md]
    python -m bench.render_gates fetch --dest ROOT [--scenes c3,t12] [--sizes 128x128] [--uri gs://bucket/render_gates]
    python -m bench.render_gates pack --refs REFS [--packs PACKS] --out TARS [--scenes ...] [--sizes ...] [--kinds png,exr]
    python -m bench.render_gates topng ROOT                      # EXR -> PNG through view.to_png (needs OpenEXR)
    python -m bench.render_gates nightly --work DIR --gpu l4 --render "CMD {scene} {res} {frames} {root}"

Example:
    >>> main(["spec"])  # doctest: +SKIP
    {"schema": "alloy.render_gates.matrix_spec/1", ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import matrix, nightly, refs, score
from .metrics import Scorer


def _csv(s: str | None) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=1) + "\n")


def cmd_score(a: argparse.Namespace) -> int:
    sc = Scorer(Path(a.cache) if a.cache else None)
    meta = {"renders": nightly.read_rows(Path(a.renders))} if a.renders else {}
    rep = score.score_root(Path(a.root), sc, set(_csv(a.engines)) or None, gpu=a.gpu, meta=meta)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    cells = {c: {e: {k: r[k]["status"] for k in ("parity", "equal_spp")} for e, r in blk.items() if not e.startswith("_")}
             for c, blk in rep["cells"].items()}
    _emit({"out": a.out, "gpu": a.gpu, "cells": cells})
    return 0


def cmd_matrix(a: argparse.Namespace) -> int:
    spec = matrix.load_spec(Path(a.spec) if a.spec else None)
    reports = {}
    for item in a.report:
        gpu, _, path = item.partition("=")
        if gpu not in spec["gpus"] or not path:
            raise SystemExit(f"--report {item!r}: use <gpu>=<report.json> with gpu one of {sorted(spec['gpus'])}")
        reports[gpu] = json.loads(Path(path).read_text())
    m = matrix.build_matrix(spec, reports, nightly.read_rows(Path(a.rows) if a.rows else None))
    Path(a.out).write_text(json.dumps(m, indent=1))
    if a.md:
        Path(a.md).write_text(matrix.to_markdown(m, spec))
    _emit({"out": a.out, "counts": m["counts"]})
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    entries = refs.select(refs.load_manifest(), _csv(a.scenes) or None, _csv(a.sizes) or None, a.kind)
    if not entries:
        raise SystemExit("no reference set matches; run `python -m bench.render_gates spec` for scene and size names")
    _emit({"fetched": refs.fetch(entries, Path(a.dest), a.uri), "dest": a.dest})
    return 0


def cmd_pack(a: argparse.Namespace) -> int:
    spec = matrix.load_spec()
    man = refs.load_manifest(Path(a.manifest) if a.manifest else None)
    keep = {(e["scene"], e["res"], e["kind"]): e for e in man["sets"]}
    for scene in _csv(a.scenes) or list(spec["scenes"]):
        if a.packs and (Path(a.packs) / scene).is_dir():
            keep[(scene, "*", "pack")] = refs.pack_scene(Path(a.packs) / scene, Path(a.out), scene)
        for res in _csv(a.sizes) or spec["sizes"]:
            for kind in (k for k in _csv(a.kinds) if a.refs and k != "pack"):
                e = refs.pack_set(Path(a.refs), Path(a.out), scene, res, kind)
                if e:
                    keep[(scene, res, kind)] = e
    man["sets"] = sorted(keep.values(), key=lambda e: (e["scene"], e["res"], e["kind"]))
    Path(a.manifest or refs.MANIFEST_PATH).write_text(json.dumps(man, indent=1) + "\n")
    _emit({"sets": len(man["sets"]), "manifest": str(a.manifest or refs.MANIFEST_PATH)})
    return 0


def cmd_topng(a: argparse.Namespace) -> int:
    from .view import exr_folder_to_png

    n = exr_folder_to_png(Path(a.root), _csv(a.engines) or None)
    _emit({"root": a.root, "pngs": n})
    return 0 if n else 1


def cmd_nightly(a: argparse.Namespace) -> int:
    cfg = nightly.NightlyConfig(Path(a.work), a.gpu, a.render, _csv(a.scenes), _csv(a.sizes), a.uri,
                                Path(a.rows) if a.rows else None, a.strict, a.skip_fetch)
    m = nightly.run_nightly(cfg)
    _emit({"gpu": a.gpu, "counts": m["counts"], "matrix": str(cfg.work / f"matrix_{a.gpu}.json")})
    return nightly.exit_code(m, a.strict)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bench.render_gates", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("spec", help="print the matrix spec (scenes, frames, sizes, GPUs, claims, speed bars)")
    p = sub.add_parser("score", help="score a gate root rendered on one GPU")
    p.add_argument("root")
    p.add_argument("--gpu", default="")
    p.add_argument("--out", default="report.json")
    p.add_argument("--engines", default="", help="comma list of engine dirs (default: every non-Cycles dir)")
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "alloy_render_gates"), help="'' disables")
    p.add_argument("--renders", default=None, help="JSONL of render calls (scene, res, rc, error): rc != 0 fails the cell")
    p = sub.add_parser("matrix", help="merge per-GPU reports and speed rows into the scene x size x GPU matrix")
    p.add_argument("--report", action="append", default=[], help="<gpu>=<report.json>, repeatable")
    p.add_argument("--rows", default=None, help="speed rows JSONL (gpu, scene, res, worlds, setting, images_per_s)")
    p.add_argument("--spec", default=None)
    p.add_argument("--out", default="matrix.json")
    p.add_argument("--md", default=None)
    p = sub.add_parser("fetch", help="download, verify and unpack Cycles reference sets")
    p.add_argument("--dest", required=True)
    p.add_argument("--scenes", default="")
    p.add_argument("--sizes", default="")
    p.add_argument("--kind", default="png", choices=sorted(refs.KINDS), help="png | exr | passes (albedo/normal of control renders) | pack (scene packs)")
    p.add_argument("--uri", default=None, help=f"bucket or folder of the tars (default ${refs.URI_ENV})")
    p = sub.add_parser("pack", help="tar a reference folder per scene/size/kind and update the manifest")
    p.add_argument("--refs", default=None, help="reference root (<scene>/<frame>/cycles*/...)")
    p.add_argument("--packs", default=None, help="folder of scene packs (<scene>/): adds kind 'pack' tars")
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", default="")
    p.add_argument("--sizes", default="")
    p.add_argument("--kinds", default="png,exr")
    p.add_argument("--manifest", default=None)
    p = sub.add_parser("topng", help="rebuild <scene>/<frame>/<engine>/*.png from the .exr beside them (view.json)")
    p.add_argument("root")
    p.add_argument("--engines", default="", help="comma list of engine dirs (default: every dir with EXRs)")
    p = sub.add_parser("nightly", help="fetch, render, score and write the matrix for one GPU")
    p.add_argument("--work", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--render", default=None, help="command template: {scene} {res} {frames} {root} {pack} {gpu}")
    p.add_argument("--scenes", default="")
    p.add_argument("--sizes", default="")
    p.add_argument("--uri", default=None)
    p.add_argument("--rows", default=None)
    p.add_argument("--strict", action="store_true", help="PENDING cells fail the run too")
    p.add_argument("--skip-fetch", dest="skip_fetch", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    a = parser().parse_args(argv)
    if a.cmd == "spec":
        _emit(matrix.load_spec())
        return 0
    return {"score": cmd_score, "matrix": cmd_matrix, "fetch": cmd_fetch, "pack": cmd_pack, "topng": cmd_topng,
            "nightly": cmd_nightly}[a.cmd](a)
