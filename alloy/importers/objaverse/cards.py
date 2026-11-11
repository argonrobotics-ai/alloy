# SPDX-License-Identifier: Apache-2.0
"""Card specs for imported Objaverse objects, written by the shared importer writers (``alloy.importers.common``).

Each object becomes ``objects/objaverse/<name>.yaml`` (``collision: decompose`` on ``physics.obj``) and
``materials/objaverse/<name>_eff.yaml`` (its density with the base material's surface). The density is an ESTIMATE
either way (:mod:`alloy.importers.objaverse.physical`): the base material's own density when density x volume is a
plausible mass for the category, else the density that gives the category's typical mass. Each card header carries
the CC attribution; each object queues one ``material.density`` row for the sourcing agent.

Example:
    >>> from alloy.importers.objaverse.cards import header_lines
    >>> header_lines({"uid": "abc", "name": "Mug", "user_display": "Ann", "license": "by",
    ...               "viewerUrl": "https://sketchfab.com/3d-models/abc"}, "300 ml mug", 0.11)[1][:21]
    'Attribution: "Mug" by'
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alloy.cards import types as ct
from alloy.importers.common import cards as cw
from alloy.importers.common.queue import QueueRow
from alloy.importers.objaverse.categories import Category, card_name
from alloy.importers.objaverse.errors import ImportRejected
from alloy.importers.objaverse.licence import DATASET_CREDIT, attribution
from alloy.importers.objaverse.physical import choose_mass

LIBRARY = "objaverse"
PHYSICS, HULLS = "physics.obj", "physics.hulls.obj"


@dataclass(frozen=True)
class Processed:
    """The subset of the shared pipeline's ``result.json`` a card needs."""

    files: Mapping[str, str]             # file name -> sha256
    volume: float                        # m^3 of physics.obj at real scale
    n_hulls: int
    extents: tuple[float, float, float]  # m, real scale, Z-up

    @classmethod
    def from_result(cls, r: Mapping[str, Any]) -> Processed:
        ext = (r.get("mass_props") or {}).get("extents") or r.get("extents") or (0.0, 0.0, 0.0)
        return cls(dict(r["files"]), float(r["volume"]), int(r["n_hulls"]), tuple(float(x) for x in ext))


def header_lines(meta: Mapping[str, Any], spec: str, size: float) -> list[str]:
    """Card header: what it is, CC attribution, dataset credit, the scale estimate."""
    return [f"{spec}; geometry: Objaverse uid {meta['uid']}",
            f"Attribution: {attribution(meta)['text']}",
            f"Dataset: {DATASET_CREDIT}",
            f"Scale: ESTIMATE longest side {size * 1000:.0f} mm (category prior, no source)"]


def _mass_spec(cat: Category, base: ct.MaterialCard, volume: float, uid: str) -> cw.MassSpec:
    mc = choose_mass(cat, base.density.value, volume, uid)
    if mc.effective:
        return cw.MassSpec("density", mc.density, (mc.density / 2.0, mc.density * 2.0),
                           mc.note.removeprefix("ESTIMATE "))
    note = f"library material {base.name} density; {mc.note}"
    return cw.MassSpec("density", mc.density, base.density.unc, note)


def card_spec(meta: Mapping[str, Any], cat: Category, proc: Processed, base: ct.MaterialCard,
              min_wall: float | None, min_wall_probe: float | None = None) -> tuple[cw.CardSpec, QueueRow]:
    """The shared writer's spec for one object and its sourcing row. ``min_wall`` [m] is the thinnest wall of
    ``physics.obj`` (``common.checks.wall_records``) when it is under the validator's threshold, else None;
    ``min_wall_probe`` is the validator probe's reading when it was re-measured (it sets the card's note). Raises
    :class:`ImportRejected`."""
    uid = str(meta["uid"])
    for f in (PHYSICS, HULLS):
        if f not in proc.files:
            raise ImportRejected(uid, "card", f"pipeline output lacks {f}; re-run the mesh stage for this object")
    if not 1 <= proc.n_hulls <= ct.Budget().collision_parts:
        raise ImportRejected(uid, "card", f"{proc.n_hulls} hulls outside 1..{ct.Budget().collision_parts}")
    name = card_name(cat.name, uid)
    mass = _mass_spec(cat, base, proc.volume, uid)
    header = "\n".join(header_lines(meta, f"{cat.spec} ({cat.group})", cat.size.value))
    spec = cw.CardSpec(name=name, base_material=base.name, material_note=cat.spec, mass=mass, volume=proc.volume,
                       mesh_path=f"meshes/{name}/{PHYSICS}", mesh_sha256=proc.files[PHYSICS],
                       max_parts=proc.n_hulls, header=header, min_wall=min_wall,
                       min_wall_probe=min_wall_probe)
    row = QueueRow(card=name, field="material.density", value=mass.value, unc=mass.unc, note=mass.note,
                   category=cat.name, query=f"typical mass of a {cat.spec}",
                   extra={"mass_kg": round(mass.value * proc.volume, 6), "volume_m3": proc.volume,
                          "size_m": cat.size.value, "group": cat.group})
    return spec, row
