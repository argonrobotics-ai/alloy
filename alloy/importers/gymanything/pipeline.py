# SPDX-License-Identifier: Apache-2.0
"""The import run: specs -> cards in a library root -> checks -> index, as plain functions with JSON-able results.

Example::

    >>> res = import_specs(Path("specs"), Path("out"))                    # doctest: +SKIP
    >>> res["converted"], res["checked_ok"], res["failed"]                 # doctest: +SKIP
    (11, 11, {})
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alloy.importers.gymanything import citecheck, hullgen, libfiles, meshes, sourced
from alloy.importers.gymanything.checks import check_root
from alloy.importers.gymanything.convert import ConvertError, ImportRecord, convert_spec, header, object_rel
from alloy.importers.gymanything.deformable import convert_deformable
from alloy.importers.gymanything.spec import GarSpec, SpecError, find_specs, load_spec

log = logging.getLogger("alloy.importers.gymanything")

def load_all(specs_dir: Path, sims: list[str] | None = None) -> tuple[list[GarSpec], dict[str, str]]:
    """Every valid spec under ``specs_dir`` (optionally only ``sims``) and the files that failed to parse."""
    good, bad = [], {}
    for path in find_specs(specs_dir):
        if sims and path.parent.name not in sims:
            continue
        try:
            good.append(load_spec(path))
        except SpecError as exc:
            bad[str(path)] = str(exc)
    names = [s.card_name for s in good]
    for n in {n for n in names if names.count(n) > 1}:
        bad[n] = "duplicate card name (two specs with the same sim and key)"
    return [s for s in good if names.count(s.card_name) == 1], bad


def convert_all(specs: list[GarSpec], out: Path) -> tuple[list[ImportRecord], dict[str, str]]:
    """Convert every spec; failures (with what to change) are returned, not raised."""
    recs, failed = [], {}
    for s in specs:
        try:
            recs.append(convert_deformable(s, out) if s.kind in ("cloth", "cable") else convert_spec(s, out))
        except (ConvertError, meshes.MeshError, ValueError, OSError) as exc:
            failed[s.card_name] = f"{type(exc).__name__}: {exc}"
    return recs, failed


def clean_sims(out: Path, sims: list[str]) -> None:
    """Remove earlier output of ``sims`` from ``out`` (cards, materials, meshes) so a rerun leaves no stale cards."""
    for sim in sims:
        for p in (out / "objects" / sim, out / "materials" / sim, *(out / "meshes").glob(f"gar_{sim}_*")):
            shutil.rmtree(p, ignore_errors=True)


@dataclass(frozen=True)
class ImportOptions:
    """How much of the pipeline to run. ``repo``: a Gym-Anything checkout for the citation check; ``hulls``: write
    hull sidecars and pick decompose/sdf (needs the importer extra); ``hull_cache``: earlier sidecars to reuse;
    ``drop``: drop-test every rigid object (``alloy.importers.common.checks``); ``workers``: process pool size;
    ``sources``: apply the sourcing agent's records where Gym-Anything left a value open."""

    repo: Path | None = None
    hulls: bool = True
    hull_cache: Path | None = None
    drop: bool = False
    workers: int = 2
    sources: Path | None = None      # sourcing records root (``<sources>/gymanything/<card>.yaml``), see ``sourced``
    visuals: bool = True             # write visual.glb from the render parts (needs trimesh and Pillow)


def apply_hulls(out: Path, recs: list[ImportRecord], specs: dict[str, GarSpec], opt: ImportOptions) -> dict[str, Any]:
    """Hull sidecars for every rigid record, then each card's collision mode (``hullgen.collision_for``)."""
    rigid = [r for r in recs if r.collision in ("convex", "sdf")]
    stats = hullgen.hull_all(out, [r.card for r in rigid], {r.card for r in rigid if r.collision == "convex"},
                             opt.workers, opt.hull_cache)
    for r in rigid:
        st = stats.get(r.card, {})
        r.n_hulls, r.hull_volume_ratio = st.get("n_hulls"), st.get("volume_ratio")
        col = hullgen.collision_for(r.convex_ratio or 1.0, st)
        if col != r.collision:
            hullgen.set_collision(out / object_rel(specs[r.card]), col, r.n_hulls, header(specs[r.card]))
            r.collision = col
    return {k: v["error"] for k, v in stats.items() if not v.get("ok")}


def write_visuals(out: Path, recs: list[ImportRecord], specs: dict[str, GarSpec]) -> dict[str, str]:
    """``visual.glb`` for every record whose spec has render parts (needs trimesh); returns card -> failure."""
    from alloy.importers.gymanything import visualglb  # noqa: PLC0415

    failed = {}
    for r in recs:
        spec = specs[r.card]
        if not spec.visual:
            continue
        try:
            visualglb.write_visual_glb(spec, out)
            r.visual_parts = len(spec.visual)
            if spec.mesh is not None and spec.kind == "rigid":
                r.visual_bbox_ok = visualglb.visual_check(spec, out / "meshes" / r.card / "physics.obj")["bbox_ok"]
        except Exception as exc:  # noqa: BLE001 - one object's visual failure is reported, the batch goes on
            failed[r.card] = f"{type(exc).__name__}: {exc}"[:300]
    return failed


def _drop_rows(out: Path, recs: list[ImportRecord], specs: dict[str, GarSpec], workers: int) -> dict[str, dict[str, Any]]:
    """Drop tests of the dynamic rigid objects (fixtures and furniture are placed, never dropped)."""
    from alloy.importers.common.checks import check_all  # noqa: PLC0415

    names = [r.card for r in recs if r.n_hulls and specs[r.card].motion == "dynamic"]
    return {k: v["drop"] for k, v in check_all(out, names, {r.card: r.volume_m3 for r in recs}, workers).items()}


def import_specs(specs_dir: Path, out: Path, sims: list[str] | None = None,
                 opt: ImportOptions = ImportOptions()) -> dict[str, Any]:
    """Convert, hull and check; writes ``<out>/index/<sim>.json`` (one row per object, checks merged) and the shared
    library files (``libfiles``). Returns a JSON-able summary."""
    specs, bad = load_all(specs_dir, sims)
    by_card = {s.card_name: s for s in specs}
    touched = sorted({s.sim for s in specs})
    clean_sims(out, touched)
    recs, failed = convert_all(specs, out)
    failed.update(bad)
    hull_fail = apply_hulls(out, recs, by_card, opt) if opt.hulls else {}
    visual_fail = write_visuals(out, recs, by_card) if opt.visuals else {}
    applied = sourced.apply_records(out, by_card, recs, sourced.load_records(opt.sources)) if opt.sources else {}
    expected = {r.card: r.mass_kg for r in recs if {"mass", "density"} & set(r.sourced)}
    rows = {r["card"]: r for r in check_root(out, [s for s in specs if s.card_name not in failed], expected)}
    for card, d in (_drop_rows(out, recs, by_card, opt.workers) if opt.drop else {}).items():
        rows[card]["drop"] = d
        rows[card]["ok"] = rows[card].get("ok", False) and bool(d.get("ok"))
    cites = {s.card_name: citecheck.cite_rows(s, opt.repo) for s in specs} if opt.repo is not None else {}
    _write_index(out, recs, rows, cites)
    lib = libfiles.write_library_files(out, specs, recs, {r.card: r.mass_kg / r.volume_m3 for r in recs
                                                          if r.mass_kg and r.volume_m3}, touched)
    ok = sum(1 for r in rows.values() if r.get("ok"))
    summary = {"specs": len(specs), "converted": len(recs), "checked_ok": ok, "failed": failed,
               "check_failures": {k: _why(v) for k, v in rows.items() if not v.get("ok")}, "sims": touched,
               "collision": dict(Counter(r.collision for r in recs)), "hull_failures": hull_fail,
               "visual": {"objects": sum(1 for r in recs if r.visual_parts), "failures": visual_fail,
                          "bbox_off": sorted(r.card for r in recs if r.visual_bbox_ok is False)},
               "sourced": {f: sum(f in v for v in applied.values()) for f in sorted({f for v in applied.values() for f in v})},
               **lib}
    if cites:
        summary["cites"] = citecheck.summarize(cites)
    log.info("import: %d specs, %d converted, %d checks ok, %d failed", len(specs), len(recs), ok, len(failed))
    return summary


def _write_index(out: Path, recs: list[ImportRecord], rows: dict[str, dict], cites: dict[str, list]) -> None:
    by_sim: dict[str, list[dict[str, Any]]] = {}
    for rec in recs:
        row = {**rec.as_dict(), "check": rows.get(rec.card, {})}
        if rec.card in cites:
            row["cites"] = cites[rec.card]
        by_sim.setdefault(rec.sim, []).append(row)
    for sim, items in by_sim.items():
        _write_json(out / "index" / f"{sim}.json", {"sim": sim, "objects": items})


def _why(row: dict[str, Any]) -> str:
    if row.get("error"):
        return row["error"]
    if row.get("errors"):
        return "; ".join(row["errors"])[:500]
    parts = []
    if not row.get("mass_ok", True):
        parts.append(f"mass off by {row.get('mass_rel_err', 0):.1%}")
    if row.get("drop") and not row["drop"].get("ok"):
        parts.append(f"drop test: {row['drop'].get('reason')}")
    if row.get("dims_off"):
        offs = [f"{d['name']} {d['measured_m']:.4g} m vs cited {d['cited_m']:.4g} m" for d in row["dims"]
                if d["band"] == "off"]
        parts.append("dims off: " + ", ".join(offs))
    return "; ".join(parts) or "unknown"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
