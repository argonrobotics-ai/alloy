# SPDX-License-Identifier: Apache-2.0
"""Object and effective-material cards for imported meshes (canonical YAML via ``alloy.cards.loader.dump_card``).

Each imported object gets two cards:

- ``<name>_eff`` (``alloy.material/1``): density from the object's mass over its ``physics.obj`` volume, with the
  surface (friction, restitution) and elastic values copied from a library material named in every note. Density
  ``source`` is ``fit`` for a measured or catalogue mass (``library/SOURCES.md``: such cards are excluded from
  CARD-10) and ``model`` with a note starting ``ESTIMATE`` otherwise.
- ``<name>`` (``alloy.object/1``): ``shape.mesh`` = ``physics.obj`` with ``collision: decompose`` (hulls in the
  ``physics.hulls.obj`` sidecar), ``max_parts`` = hull count.
"""

from __future__ import annotations

import math

import dataclasses
from dataclasses import dataclass
from typing import Literal

from alloy.cards import types as ct
from alloy.cards.loader import dump_card
from alloy.core.quantity import Q

VOLUME_REL_UNC = 0.05            # scan volume vs the real object (closed scans of rigid goods)
MIN_WALL_NOTE = ("thinnest wall of the physics mesh by Alloy's inward-ray probe (alloy.cards.geometry.wall_thickness); "
                 "scan defects can make it thinner than the real part; collision uses the convex hulls")


def min_wall_note(value: float, probe: float) -> str:
    """The ``features.min_wall`` note: ``MIN_WALL_NOTE`` when the probe's reading is declared as is, else how
    ``walls.wall`` re-measured it (and what the probe read).

    Example::

        min_wall_note(1.17e-3, 2.3e-5)
    """
    from alloy.importers.common.checks import THIN  # noqa: PLC0415 - checks loads the drop-test stack

    if value == probe:
        return MIN_WALL_NOTE
    if value >= math.nextafter(THIN, 0.0):
        return (f"no wall of the physics mesh is thinner than {THIN * 1e3:.3g} mm over 10 mm^2 of faces (inward rays from "
                f"every face that ignore hits on the face's own neighbours, alloy.importers.common.walls); declared at "
                f"that threshold because Alloy's plain probe read {probe:.2g} m on remesh defects (folds, slivers, inner "
                f"shells); collision uses the convex hulls")
    why = "on remesh defects (folds, slivers, inner shells)" if value > probe else "from its 1,024 sampled faces"
    return (f"thinnest wall of the physics mesh that faces of at least 10 mm^2 agree on, by inward rays from every face "
            f"that ignore hits on the face's own neighbours (alloy.importers.common.walls); Alloy's plain probe read "
            f"{probe:.2g} m {why}; collision uses the convex hulls")
EFF_SUFFIX = "_eff"


@dataclass(frozen=True)
class MassSpec:
    """Where an object's mass comes from.

    ``kind``: ``mass`` (a measured or catalogue mass in kg; becomes a ``fit`` density) or ``density`` (an ESTIMATE
    effective density in kg/m^3; becomes a ``model`` density). ``unc`` is an absolute (lo, hi) interval in the same
    unit, ``note`` the citation or derivation.

    Example::

        MassSpec("mass", 0.118, (0.117, 0.119), "118 g, Calli et al., ICAR 2015, Table I")
    """

    kind: Literal["mass", "density"]
    value: float
    unc: tuple[float, float] | None
    note: str


@dataclass(frozen=True)
class CardSpec:
    """Everything needed to write one object's two cards.

    Example::

        CardSpec("ycb_025_mug", "pp_homopolymer", "rigid plastic kitchenware", MassSpec(...), 1.378e-4,
                 "meshes/ycb_025_mug/physics.obj", "ab12...", 32, "geometry: ...; licence: CC-BY-4.0")
    """

    name: str
    base_material: str
    material_note: str
    mass: MassSpec
    volume: float                  # m^3 of physics.obj
    mesh_path: str                 # relative to the library root that holds the card
    mesh_sha256: str
    max_parts: int
    header: str                    # comment lines (generator, geometry source, licence, files)
    min_wall: float | None = None  # m, thinnest wall of the mesh to declare (common.walls.min_wall), if declared
    min_wall_probe: float | None = None   # m, the validator probe's reading when it differs (sets the note)


def _density(spec: CardSpec) -> Q:
    m, v = spec.mass, spec.volume
    if m.kind == "density":
        return Q(m.value, "model", m.unc, f"ESTIMATE: {m.note}")
    lo, hi = m.unc if m.unc is not None else (m.value, m.value)
    unc = (lo / (v * (1 + VOLUME_REL_UNC)), hi / (v * (1 - VOLUME_REL_UNC)))
    note = (f"effective: {m.value * 1e3:.4g} g ({m.note}) over the {v * 1e6:.4g} cm^3 physics mesh volume "
            f"(scan volume +-{VOLUME_REL_UNC:.0%}); FIT")
    return Q(m.value / v, "fit", unc, note)


def _relabel(q: Q | None, base: str) -> Q | None:
    if q is None:
        return None
    note = f"library material {base}: {q.note}" if q.note else f"library material {base} (default)"
    return Q(q.value, q.source, q.unc, note) if q.source != "default" else q


def _relabel_all(obj, base: str):
    """Copy of a Surface/Elastic/ClothAniso dataclass with every Q note prefixed by its library material."""
    if obj is None:
        return None
    changes = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, Q):
            changes[f.name] = _relabel(v, base)
        elif dataclasses.is_dataclass(v):
            changes[f.name] = _relabel_all(v, base)
    return dataclasses.replace(obj, **changes)


def effective_material(spec: CardSpec, base: ct.MaterialCard) -> ct.MaterialCard:
    """``<name>_eff``: the spec's density with ``base``'s surface and elastic values (notes name ``base``).

    Example::

        mat = effective_material(spec, CardLibrary().material("pp_homopolymer"))
    """
    return ct.MaterialCard(name=spec.name + EFF_SUFFIX, density=_density(spec),
                           surface=_relabel_all(base.surface, base.name),
                           elastic=_relabel_all(base.elastic, base.name))


def object_card(spec: CardSpec) -> ct.ObjectCard:
    """The object card: ``physics.obj`` with ``collision: decompose`` and the ``_eff`` material.

    Example::

        card = object_card(spec)
        card.shape.collision   # "decompose"
    """
    shape = ct.MeshSource(path=spec.mesh_path, sha256=spec.mesh_sha256, collision="decompose",
                          max_parts=max(1, min(spec.max_parts, ct.Budget().collision_parts)))
    features = ct.Features()
    if spec.min_wall is not None:
        probe = spec.min_wall if spec.min_wall_probe is None else spec.min_wall_probe
        features = ct.Features(min_wall=Q(spec.min_wall, "model", None, min_wall_note(spec.min_wall, probe)))
    return ct.ObjectCard(name=spec.name, shape=shape, material=spec.name + EFF_SUFFIX, features=features)


def material_card(spec: CardSpec, base: ct.MaterialCard, record=None) -> ct.MaterialCard:
    """``effective_material`` with a sourcing record's values written over it (``alloy.sourcing``), if one is given.

    A record's ``mass`` becomes the density through ``spec.volume``, unless the spec already holds a measured or
    catalogue mass (``kind == "mass"``: its ``fit`` density, with the scan-volume interval, stays). The record's
    friction, restitution and elastic values replace the library material's. A record that implies an impossible
    value raises ``alloy.sourcing.SourcingError``; callers keep the estimate and report the object.

    Example::

        rec = alloy.sourcing.load_records("ycb_gso").get(spec.name)
        mat = material_card(spec, lib.material(spec.base_material), rec)
    """
    mat = effective_material(spec, base)
    if record is None:
        return mat
    from alloy.sourcing import apply_to_material  # noqa: PLC0415 - optional at import time

    out = apply_to_material(mat, record, spec.volume)
    return dataclasses.replace(out, density=mat.density) if spec.mass.kind == "mass" else out


def card_texts(spec: CardSpec, base: ct.MaterialCard, record=None) -> tuple[str, str]:
    """``(object_yaml, material_yaml)`` in canonical form with the spec's header comments.

    ``record``: an ``alloy.sourcing.ObjectRecord`` for this object (see ``material_card``), or None.

    Example::

        obj_yaml, mat_yaml = card_texts(spec, lib.material(spec.base_material))
    """
    mat_header = f"{spec.header}\nsurface and elastic: library material {base.name} ({spec.material_note})"
    if record is not None:
        mat_header += "\nsourced values: alloy-assets sources/imports (alloy.sourcing records) override the above"
    return dump_card(object_card(spec), header=spec.header), dump_card(material_card(spec, base, record),
                                                                        header=mat_header)
