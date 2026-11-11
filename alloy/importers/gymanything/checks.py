# SPDX-License-Identifier: Apache-2.0
"""Checks on imported cards: Alloy's own card validator, mass through Newton's inertia routine, cited dimensions.

Example::

    >>> rows = check_root(Path("out"), specs)            # doctest: +SKIP
    >>> [r["card"] for r in rows if not r["ok"]]          # doctest: +SKIP
    []
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from alloy.cards import geometry as geo
from alloy.cards.library import CardLibrary
from alloy.cards.mass import card_mass
from alloy.cards.paths import CardView
from alloy.cards.types import MeshSource
from alloy.core.errors import CardError
from alloy.importers.gymanything.spec import Dim, GarSpec

log = logging.getLogger("alloy.importers.gymanything")

MASS_TOL = 0.02               # relative; CARD-10's catalogue tolerance
DIM_OK, DIM_WARN = 0.05, 0.15  # relative dimension error bands


def measured_dim(d: Dim, mesh: geo.MeshData) -> float:
    """The mesh's value for a cited dimension (axis span, OBB extent or XY diameter), m."""
    v = mesh.vertices
    span = v.max(axis=0) - v.min(axis=0)
    if d.measure in ("x", "y", "z"):
        return float(span["xyz".index(d.measure)])
    if d.measure == "diameter_xy":
        c = 0.5 * (v[:, :2].max(axis=0) + v[:, :2].min(axis=0))
        return float(2.0 * np.linalg.norm(v[:, :2] - c, axis=1).max())
    ext = geo.obb_extents(v)
    return float({"max": ext[0], "mid": ext[1], "min": ext[2]}[d.measure])


def dim_rows(spec: GarSpec, mesh: geo.MeshData) -> list[dict[str, Any]]:
    """Per cited dimension: measured, ratio and band (``ok`` <= 5 %, ``warn`` <= 15 %, else ``off``)."""
    rows = []
    for d in spec.dims:
        got = measured_dim(d, mesh)
        ratio = got / d.value_m if d.value_m > 0 else float("inf")
        err = abs(ratio - 1.0)
        band = "ok" if err <= DIM_OK else "warn" if err <= DIM_WARN else "off"
        rows.append({"name": d.name, "cited_m": d.value_m, "measured_m": round(got, 6), "ratio": round(ratio, 4),
                     "band": band})
    return rows


def _mass_row(lib: CardLibrary, name: str, spec: GarSpec, want: float | None) -> dict[str, Any]:
    card = lib.object(name)
    if spec.kind == "cloth" and isinstance(card.shape, MeshSource):
        return {"newton_mass_kg": None, "cited_mass_kg": want, "mass_rel_err": None,
                "mass_ok": True, "mass_note": "sheet mesh: mass = density x area x thickness (no closed volume)"}
    got = card_mass(CardView(card, lib.material(card.material)), lambda rel: lib.asset_path(rel) or Path(rel))
    rel = abs(got - want) / want if (got is not None and want) else None
    return {"newton_mass_kg": got, "cited_mass_kg": want, "mass_rel_err": rel,
            "mass_ok": rel is None or rel <= MASS_TOL}


def check_one(lib: CardLibrary, spec: GarSpec, want: float | None = None) -> dict[str, Any]:
    """One object's check row: card issues, Newton mass vs the expected mass (``want``: a sourcing record's mass,
    else the cited mass), cited dimensions vs the mesh."""
    want = want if want is not None else (spec.mass.value if spec.mass else None)
    name = spec.card_name
    row: dict[str, Any] = {"card": name, "ok": False}
    try:
        issues = lib.object_issues(name)
    except CardError as exc:
        row["error"] = str(exc)
        return row
    row["errors"] = [f"{i.field}: {i.message}" for i in issues if i.level == "error"]
    row["warns"] = [f"{i.field}: {i.message}" for i in issues if i.level == "warn"]
    if row["errors"]:
        return row
    row.update(_mass_row(lib, name, spec, want))
    row["dims"] = dim_rows(spec, geo.read_mesh(spec.mesh)) if spec.mesh is not None else []
    row["dims_off"] = [d["name"] for d in row["dims"] if d["band"] == "off"]
    row["ok"] = row["mass_ok"] and not row["dims_off"]
    return row


def check_root(out: Path, specs: list[GarSpec], expected: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Check every converted spec against the cards in the library root ``out`` (no other roots, no assets);
    ``expected`` overrides the cited mass per card (masses applied from sourcing records)."""
    lib = CardLibrary(roots=[out], assets=None)
    rows = []
    for spec in specs:
        if not lib.has_object(spec.card_name):
            rows.append({"card": spec.card_name, "ok": False, "error": "not converted"})
            continue
        rows.append(check_one(lib, spec, (expected or {}).get(spec.card_name)))
    bad = [r["card"] for r in rows if not r["ok"]]
    log.info("checked %d cards, %d failing: %s", len(rows), len(bad), bad[:10])
    return rows
