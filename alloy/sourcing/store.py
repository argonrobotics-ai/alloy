# SPDX-License-Identifier: Apache-2.0
"""Per-object sourced records, ``<assets>/sources/imports/<library>/<object>.yaml``, and applying them to a card.

.. code-block:: yaml

    schema: alloy.sourcing.object/1
    library: ycb_gso
    object: gso_porcelain_mug_white
    category: ceramic_mug          # cache entry the priors came from, if any
    material: stoneware_glazed
    updated: 2026-11-06
    values:
      mass: {value: 0.35, lo: 0.28, hi: 0.45, tag: ESTIMATE, reason: "...", evidence: [...]}
      mu_static: {...}

Importers turn a record into card quantities with :func:`card_overrides` (a sourced ``mass`` becomes the material
density through the object's volume) or :func:`apply_to_material`.

Example:
    >>> from alloy.sourcing.record import SourcedValue
    >>> rec = ObjectRecord(library="demo", object="cup", values={"mass": SourcedValue(
    ...     field="mass", value=0.2, lo=0.15, hi=0.25, tag="ESTIMATE", reason="category prior")})
    >>> round(card_overrides(rec, volume=2e-4)["material.density"].value)
    1000
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from alloy.cards import types as ct
from alloy.core.quantity import Q
from alloy.sourcing.fields import SourcingError, check_bounds, spec, yaml_dump, yaml_load
from alloy.sourcing.paths import sources_root
from alloy.sourcing.record import SourcedValue, note_for, value_to_dict, values_from_dict

log = logging.getLogger("alloy.sourcing")

SCHEMA: Final[str] = "alloy.sourcing.object/1"
_KEYS: Final[frozenset[str]] = frozenset({"schema", "library", "object", "category", "material", "updated", "values"})
_RESERVED: Final[frozenset[str]] = frozenset({"cache"})


@dataclass(frozen=True)
class ObjectRecord:
    """Sourced values for one imported object (``values`` keyed by field name)."""

    library: str
    object: str
    values: dict[str, SourcedValue] = field(default_factory=dict)
    category: str = ""
    material: str = ""
    updated: str = ""

    def __post_init__(self) -> None:
        if not self.library or not self.object:
            raise SourcingError("an object record needs library and object names")
        if self.library in _RESERVED:
            raise SourcingError(f"library name {self.library!r} is reserved for the cache folder; rename it")
        for name, v in self.values.items():
            if name != v.field:
                raise SourcingError(f"{self.object}: value under {name!r} is for field {v.field!r}")


def record_to_dict(r: ObjectRecord) -> dict[str, Any]:
    """Plain dict in canonical key order."""
    out: dict[str, Any] = {"schema": SCHEMA, "library": r.library, "object": r.object}
    out.update({k: getattr(r, k) for k in ("category", "material", "updated") if getattr(r, k)})
    out["values"] = {k: value_to_dict(v) for k, v in sorted(r.values.items())}
    return out


def record_from_dict(raw: Any, label: str) -> ObjectRecord:
    """Parse a record mapping; raises ``SourcingError`` naming the bad key."""
    if not isinstance(raw, dict):
        raise SourcingError(f"{label}: a record must be a mapping")
    extra = sorted(set(raw) - _KEYS)
    if extra:
        raise SourcingError(f"{label}: unknown keys {extra}; allowed {sorted(_KEYS)}")
    if raw.get("schema") != SCHEMA:
        raise SourcingError(f"{label}: schema must be {SCHEMA!r}, got {raw.get('schema')!r}")
    return ObjectRecord(
        library=str(raw.get("library", "")), object=str(raw.get("object", "")),
        values=values_from_dict(raw.get("values"), label), category=str(raw.get("category", "")),
        material=str(raw.get("material", "")), updated=str(raw.get("updated", "")))


def record_path(library: str, obj: str, sources: Path | None = None) -> Path:
    """``<sources>/<library>/<object>.yaml``."""
    return (sources or sources_root()) / library / f"{obj}.yaml"


def write_record(rec: ObjectRecord, sources: Path | None = None) -> Path:
    """Write (or replace) one record; returns its path."""
    path = record_path(rec.library, rec.object, sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_dump(record_to_dict(rec)),
                    encoding="utf-8")
    log.info("record: wrote %s/%s (%s)", rec.library, rec.object, ", ".join(sorted(rec.values)))
    return path


def remove_record(library: str, obj: str, sources: Path | None = None) -> bool:
    """Delete one record file (when no value is left in it); True if a file was removed."""
    path = record_path(library, obj, sources)
    if not path.is_file():
        return False
    path.unlink()
    log.info("record: removed %s/%s (no values left)", library, obj)
    return True


def load_record(path: Path) -> ObjectRecord:
    """Read one record file; the stem must equal the object name."""
    rec = record_from_dict(yaml_load(path.read_text(encoding="utf-8")), str(path))
    if path.stem != rec.object:
        raise SourcingError(f"{path}: file name must be {rec.object}.yaml (the object name)")
    return rec


def load_records(library: str, sources: Path | None = None) -> dict[str, ObjectRecord]:
    """All records of one library by object name (empty if none)."""
    folder = (sources or sources_root()) / library
    return {p.stem: load_record(p) for p in sorted(folder.glob("*.yaml"))} if folder.is_dir() else {}


def _density_from_mass(v: SourcedValue, volume: float) -> Q:
    if not volume > 0:
        raise SourcingError(f"mass -> density needs the object's enclosed volume > 0 m^3, got {volume!r}")
    d, lo, hi = v.value / volume, v.lo / volume, v.hi / volume
    check_bounds("effective_density", d, f"mass {v.value:g} kg over volume {volume:.4g} m^3")
    note = f"{note_for(v)} | density = mass / volume {volume:.4g} m^3"
    # library/SOURCES.md: a density derived from a whole-object mass is "fit" (an estimated mass gives "model")
    return Q(d, "model" if v.tag == "ESTIMATE" else "fit", (lo, hi), note[:400])


def card_overrides(rec: ObjectRecord, volume: float | None = None) -> dict[str, Q]:
    """Card path -> ``Q`` for every value in the record. A ``mass`` (with ``volume``, the enclosed mesh volume in
    m^3) becomes ``material.density``; an explicit ``density``/``effective_density`` wins over it."""
    out: dict[str, Q] = {}
    mass = rec.values.get("mass")
    if mass is not None and volume is not None:
        out["material.density"] = _density_from_mass(mass, volume)
    for name in ("effective_density", "density"):
        if name in rec.values:
            out["material.density"] = rec.values[name].to_q()
    for name, v in rec.values.items():
        path = spec(name).card_path
        if path and path != "material.density":
            out[path] = v.to_q()
    return out


def apply_to_material(mat: ct.MaterialCard, rec: ObjectRecord, volume: float | None = None) -> ct.MaterialCard:
    """A copy of ``mat`` with the record's values written in (elastic values only if ``mat`` has ``elastic``).

    Example:
        >>> from alloy.cards.library import CardLibrary
        >>> mat = CardLibrary().material("abs")
        >>> rec = ObjectRecord(library="demo", object="toy", values={"mu_static": SourcedValue(
        ...     field="mu_static", value=0.45, lo=0.35, hi=0.55, tag="ESTIMATE", reason="ABS on wood prior")})
        >>> apply_to_material(mat, rec).surface.mu_static.value
        0.45
    """
    surface, elastic, density = mat.surface, mat.elastic, mat.density
    for path, q in card_overrides(rec, volume).items():
        leaf = path.split(".")[-1]
        if path == "material.density":
            density = q
        elif path.startswith("material.surface."):
            surface = dataclasses.replace(surface, **{leaf: q})
        elif path.startswith("material.elastic.") and elastic is not None:
            elastic = dataclasses.replace(elastic, **{leaf: q})
    return dataclasses.replace(mat, density=density, surface=surface, elastic=elastic)
