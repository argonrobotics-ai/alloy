# SPDX-License-Identifier: Apache-2.0
"""Apply the sourcing agent's records (``alloy.sourcing``, ``sources/imports/gymanything/<card>.yaml``) to imported
material cards, where they fill what Gym-Anything left open.

Policy (the import stays faithful to Gym-Anything's own values):
- ``restitution``, ``mu_static``/``mu_dynamic``, ``density``: only where the card is still at the library default
  (their code sets no value);
- ``mass``: where Gym-Anything gives no object mass or only an ``estimate``/``authored`` one; a cited datasheet,
  measured or derived mass is kept.

Example::

    >>> recs = load_records(Path("alloy-assets/sources/imports"))                         # doctest: +SKIP
    >>> applied = apply_records(Path("gar"), specs_by_card, import_records, recs)           # doctest: +SKIP
    >>> applied["gar_filleting_tweezers"]                                                   # doctest: +SKIP
    ['mass', 'restitution']
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from alloy.cards.loader import dump_card, load_material
from alloy.importers.gymanything.convert import ImportRecord, header, material_rel
from alloy.importers.gymanything.spec import GarSpec

log = logging.getLogger("alloy.importers.gymanything")

LIBRARY = "gymanything"
OPEN_MASS = ("estimate", "authored")
DEFAULT_KEYS = {"restitution": "surface.restitution", "mu_static": "surface.mu", "mu_dynamic": "surface.mu",
                "density": "density"}


def load_records(sources: Path) -> dict:
    """``card -> ObjectRecord`` from ``<sources>/gymanything/*.yaml`` (``alloy.sourcing.store.load_records``)."""
    from alloy.sourcing.store import load_records as _load  # noqa: PLC0415

    return _load(LIBRARY, sources)


def allowed_fields(spec: GarSpec, rec: ImportRecord, values: dict) -> list[str]:
    """The record fields this policy applies to one card (see the module docstring)."""
    out = [f for f, key in DEFAULT_KEYS.items() if f in values and key in rec.defaults]
    if "mass" in values and spec.kind in ("rigid", "soft") and (spec.mass is None or spec.mass.basis in OPEN_MASS):
        out.append("mass")
    return sorted(set(out) - ({"density"} if "mass" in out else set()))


def _defaults_left(rec: ImportRecord, fields: list[str]) -> list[str]:
    gone = {DEFAULT_KEYS[f] for f in fields if f in DEFAULT_KEYS} | ({"density"} if "mass" in fields else set())
    return [d for d in rec.defaults if d not in gone]


def apply_records(out: Path, specs: dict[str, GarSpec], recs: list[ImportRecord], records: dict) -> dict[str, list[str]]:
    """Rewrite each card's material with its allowed record values; updates the import records in place
    (``sourced``, ``defaults``, ``mass_kg``, ``mass_basis``). Returns card -> fields applied."""
    from alloy.sourcing.store import apply_to_material  # noqa: PLC0415

    applied: dict[str, list[str]] = {}
    for r in recs:
        orec, spec = records.get(r.card), specs[r.card]
        fields = allowed_fields(spec, r, orec.values) if orec is not None else []
        if not fields:
            continue
        vals = {f: orec.values[f] for f in fields}
        for a, b in (("mu_static", "mu_dynamic"), ("mu_dynamic", "mu_static")):   # one Coulomb value, as their sims
            if a in vals and b not in vals:
                vals[b] = dataclasses.replace(vals[a], field=b)
        sub = dataclasses.replace(orec, values=vals)
        path = out / material_rel(spec)
        mat = apply_to_material(load_material(path), sub, volume=r.volume_m3)
        path.write_text(dump_card(mat, header=header(spec)), encoding="utf-8")
        if "mass" in fields or "density" in fields:
            r.mass_kg = mat.density.value * (r.volume_m3 or 0.0)
            v = orec.values.get("mass") or orec.values.get("density")
            r.mass_basis = f"sourcing record ({v.tag})"
        r.sourced, r.defaults = fields, _defaults_left(r, fields)
        applied[r.card] = fields
    log.info("sourcing records: %d cards updated", len(applied))
    return applied
