# SPDX-License-Identifier: Apache-2.0
"""Score a gate root: every engine against the Cycles references, per scene, frame and size, then per cell.

A *cell* is one scene at one size (all its frames). Its verdicts per engine:

- ``parity``: PASS when every frame passes parity;
- ``equal_spp``: the statistical verdict over the cell's pooled rows (:func:`checks.equal_spp_verdict`);
- ``g1``: per (spp, denoise) setting, frames passing end-to-end G1 and the worst ratio; the matrix reads the setting
  an engine claims.

Example:
    >>> from pathlib import Path
    >>> from bench.render_gates.metrics import Scorer
    >>> rep = score_root(Path("gate_root"), Scorer())  # doctest: +SKIP
    >>> rep["cells"]["c3/128x128"]["ours"]["parity"]["status"]  # doctest: +SKIP
    'PASS'
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import checks, layout
from .metrics import Scorer

log = logging.getLogger(__name__)


def _table(sc: Scorer, ref: Path, imgs: list[layout.Img], res: str) -> dict:
    return {(i.spp, i.den): {m: sc.score(ref, i.path, m) for m in ("raw", "matched")} for i in imgs if i.res == res}


def _clean(t: dict) -> dict:
    return {k: v for k, v in t.items() if "error" not in v["raw"]}


def _errors(t: dict) -> list[str]:
    return [v["raw"]["error"] for v in t.values() if "error" in v["raw"]]


def score_res(sc: Scorer, ref: Path, engines: dict[str, list[layout.Img]], res: str, only: set[str] | None) -> dict:
    """Score one scene, frame and size: control rows, and per engine parity, G1 per image and equal-spp rows."""
    wanted = [n for n in engines if n != layout.REF_ENGINE and (layout.is_control(n) or not only or n in only)]
    tables = {n: _table(sc, ref, engines[n], res) for n in wanted}
    controls = {n: _clean(t) for n, t in tables.items() if layout.is_control(n)}
    s1 = controls.get(layout.S1, {})
    out: dict = {"limit": checks.g1_limit(s1), "controls": {}, "engines": {}}
    for n, t in controls.items():
        out["controls"][n] = {"equal_spp_rows": checks.rows_for(n, t, controls)}
    for n, t in tables.items():
        if layout.is_control(n) or not t:
            continue
        eng = _clean(t)
        out["engines"][n] = {"parity": checks.check_parity(eng, s1), "g1": checks.g1_images(eng, s1),
                             "equal_spp_rows": checks.rows_for(n, eng, controls), "errors": _errors(t),
                             "images": {f"{k[0]}spp_{k[1]}": v for k, v in sorted(t.items())}}
    return out


def score_frames(root: Path, sc: Scorer, only: set[str] | None = None) -> dict:
    """``{"<scene>/<frame>": {res: score_res(...)}}`` for every frame with a reference (see layout.collect)."""
    frames: dict = {}
    for (scene, frame), engines in layout.collect(root).items():
        per = {}
        for res in sorted({i.res for i in engines[layout.REF_ENGINE] if i.spp == layout.REF_SPP and i.den == "on"}):
            ref = layout.reference_path(root, scene, frame, res)
            per[res] = score_res(sc, ref, engines, res, only)
        frames[f"{scene}/{frame}"] = per
        log.info("scored %s/%s (computed %d, cached %d)", scene, frame, sc.computed, sc.hits)
    return frames


def _pool(frames: dict) -> dict:
    """Group per-frame records by cell ``<scene>/<res>``: {cell: {"engines": {e: [rec]}, "controls": {c: [rows]}}}."""
    cells: dict = {}
    for key, per in frames.items():
        scene = key.split("/")[0]
        for res, blk in per.items():
            c = cells.setdefault(f"{scene}/{res}", {"engines": {}, "controls": {}, "frames": []})
            c["frames"].append(key)
            for e, rec in blk["engines"].items():
                c["engines"].setdefault(e, []).append({"frame": key, **rec})
            for n, rec in blk["controls"].items():
                c["controls"].setdefault(n, []).extend(rec["equal_spp_rows"])
    return cells


def _parity_cell(recs: list[dict]) -> dict:
    valid = [r["parity"] for r in recs if r["parity"]["status"] in ("PASS", "FAIL")]
    if not valid:
        return {"status": "n/a", "pass": 0, "total": 0}
    n_pass = sum(p["status"] == "PASS" for p in valid)
    return {"status": "PASS" if n_pass == len(valid) else "FAIL", "pass": n_pass, "total": len(valid),
            "worst_ratio": max(p["ratio"] for p in valid)}


def _g1_cell(recs: list[dict]) -> dict:
    acc: dict = {}
    for r in recs:
        for setting, g in r["g1"].items():
            a = acc.setdefault(setting, {"pass": 0, "total": 0, "worst_ratio": 0.0, "per_frame": {}})
            a["pass"] += int(g["pass"])
            a["total"] += 1
            a["worst_ratio"] = max(a["worst_ratio"], g["ratio"])
            a["per_frame"][r["frame"]] = round(g["ratio"], 4)
    for a in acc.values():
        a["status"] = "PASS" if a["pass"] == a["total"] else "FAIL"
    return acc


def summarize_cells(frames: dict) -> dict:
    """Per cell and engine: parity, equal_spp (statistical) and g1 per setting. ``frames`` from :func:`score_frames`."""
    out: dict = {}
    for cell, c in _pool(frames).items():
        out[cell] = {}
        for e, recs in c["engines"].items():
            rows = [row for r in recs for row in r["equal_spp_rows"]]
            out[cell][e] = {"frames": len(recs), "parity": _parity_cell(recs),
                            "equal_spp": checks.equal_spp_verdict(rows, c["controls"]), "g1": _g1_cell(recs)}
        out[cell]["_controls"] = {n: {"rows": len(r), "pass_rate": checks.pass_rate(r)} for n, r in c["controls"].items()}
    return out


def score_root(root: Path, sc: Scorer, only: set[str] | None = None, gpu: str = "", meta: dict | None = None) -> dict:
    """Full report for one gate root rendered on one GPU: thresholds, per-cell verdicts and per-frame detail."""
    frames = score_frames(root, sc, only)
    return {"schema": "alloy.render_gates.report/1", "gpu": gpu, "meta": meta or {},
            "thresholds": {"parity_factor": checks.PARITY_FACTOR, "row_factor": checks.ROW_FACTOR,
                           "row_cap": checks.ROW_CAP},
            "cells": summarize_cells(frames), "frames": frames}
