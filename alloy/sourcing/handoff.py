# SPDX-License-Identifier: Apache-2.0
"""Read the importers' hand-off files into per-object hints for the queue.

Two formats are read from ``<assets>/imports/<library>/``:

- ``queue.jsonl`` (the shared importer pipeline, ``alloy.importers.common.queue``): one row per estimated card value,
  ``{"card", "field", "value", "unc", "note", "category", "query", "extra"}``; ``field`` is a card path such as
  ``material.density`` (an effective density, so the value to source is the object's ``mass``);
- ``sourcing_queue.jsonl`` (this package): ``{"object", "category", "product", "url", "fields"}``; also accepted:
  ``card`` for ``object``, one ``field`` (``mass`` or a card path) instead of ``fields``, ``value`` (kg, the
  importer's own mass estimate, kept as ``extra.mass_kg``); rows with ``field: flag`` carry a plausibility note only.
  A row without ``extra.volume_m3`` whose note says the mass is "density x visual-mesh volume" of solid parts of one
  density gets the scan volume back as mass / density, unless the importer flagged the object's size.

Example:
    >>> row = {"card": "gso_x", "field": "material.density", "category": "Shoe", "query": "Reebok Zig shoe weight",
    ...        "extra": {"brand": "Reebok"}}
    >>> hint_from_pipeline_row(row)["fields"]
    ['mass']
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from alloy.sourcing.fields import PATH_TO_FIELD, SourcingError

PIPELINE_FILE: Final[str] = "queue.jsonl"
HINTS_FILE: Final[str] = "sourcing_queue.jsonl"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SourcingError(f"{path}:{n}: not JSON ({e}); one object per line") from e
        if not isinstance(row, dict):
            raise SourcingError(f"{path}:{n}: each line must be a JSON object")
        rows.append(row)
    return rows


def field_for_path(card_path: str) -> str | None:
    """Sourcing field for a card path: ``material.density`` -> ``mass`` (effective density), others by leaf.

    Example:
        >>> field_for_path("material.surface.mu_static")
        'mu_static'
    """
    if card_path == "material.density":
        return "mass"
    rel = card_path.removeprefix("material.")
    return PATH_TO_FIELD.get(rel)


def hint_from_pipeline_row(row: dict[str, Any]) -> dict[str, Any]:
    """One importer-pipeline queue row as a hint (category, product query, fields, extra)."""
    fname = field_for_path(str(row.get("field", "")))
    return {"object": str(row.get("card", "")), "category": str(row.get("category") or ""),
            "product": str(row.get("query") or ""), "url": "", "fields": [fname] if fname else [],
            "extra": row.get("extra") or {}, "note": str(row.get("note") or ""),
            "density_is_mass": fname == "mass"}


def _merge(into: dict[str, dict[str, Any]], hint: dict[str, Any]) -> None:
    name = hint["object"]
    old = into.setdefault(name, {"object": name, "category": "", "product": "", "url": "", "fields": [],
                                 "extra": {}, "note": "", "density_is_mass": False})
    for key in ("category", "product", "url", "note"):
        old[key] = old[key] or str(hint.get(key) or "")
    old["density_is_mass"] = old["density_is_mass"] or bool(hint.get("density_is_mass"))
    old["fields"] = sorted(set(old["fields"]) | set(hint.get("fields") or ()))
    flags = [f for f in (old["extra"].get("flag"), (hint.get("extra") or {}).get("flag")) if f]
    old["extra"] = {**(hint.get("extra") or {}), **old["extra"]}
    if flags:
        old["extra"]["flag"] = "; ".join(flags)


MESH_NOTE: Final[str] = "density x visual-mesh volume"
_PART: Final[re.Pattern[str]] = re.compile(r"([0-9.]+) kg/m3, (\w+)")


def volume_from_note(note: str, mass: float) -> float | None:
    """The scan volume behind an importer's "density x visual-mesh volume" mass: mass / density when every part
    listed is solid and of one density; None otherwise (shells, mixed densities, other notes).

    Example:
        >>> note = "ESTIMATE: density x visual-mesh volume; TomatoSlice001: robocasa_tomato 990 kg/m3, solid"
        >>> round(volume_from_note(note, 0.0198), 9)
        2e-05
    """
    parts = _PART.findall(note) if MESH_NOTE in note else []
    rhos = {float(rho) for rho, kind in parts}
    if not parts or any(kind != "solid" for _, kind in parts) or len(rhos) != 1:
        return None
    rho = rhos.pop()
    return mass / rho if rho > 0 and mass > 0 else None


def _add_volume(hint: dict[str, Any]) -> None:
    extra = hint["extra"]
    mass = extra.get("mass_kg")
    if "volume_m3" in extra or not isinstance(mass, (int, float)) or "size " in str(extra.get("flag", "")):
        return  # a flagged size means a mis-scaled mesh: its volume is no guide to the object's mass
    vol = volume_from_note(str(hint.get("note") or ""), float(mass))
    if vol is not None:
        extra["volume_m3"] = vol


def hint_from_queue_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """One ``sourcing_queue.jsonl`` row as a hint; None for rows that only flag (``field: flag``).

    Example:
        >>> hint_from_queue_row({"card": "rc_jar_001", "field": "mass", "value": 0.4, "category": "jar"})["extra"]
        {'mass_kg': 0.4}
    """
    name = row.get("object") or row.get("card")
    if not name:
        return None
    fields = list(row.get("fields") or ())
    one = str(row.get("field") or "")
    if one == "flag":
        return {"object": str(name), "category": str(row.get("category") or ""), "fields": [],
                "extra": {"flag": str(row.get("why") or "")}}
    if one:
        fname = one if one in ("mass", "density") or "." not in one else field_for_path(one)
        fields += [fname] if fname else []
    extra = dict(row.get("extra") or {})
    if one == "mass" and isinstance(row.get("value"), (int, float)):
        extra.setdefault("mass_kg", float(row["value"]))
    return {"object": str(name), "category": str(row.get("category") or ""), "product": str(row.get("product") or ""),
            "url": str(row.get("url") or ""), "fields": fields, "extra": extra, "note": str(row.get("note") or "")}


def read_hints(folder: Path) -> dict[str, dict[str, Any]]:
    """Hints by object name from both hand-off files (this package's file wins on conflicts).

    Example:
        >>> import tempfile
        >>> read_hints(Path(tempfile.mkdtemp()))
        {}
    """
    hints: dict[str, dict[str, Any]] = {}
    for n, row in enumerate(_rows(folder / HINTS_FILE), 1):
        hint = hint_from_queue_row(row)
        if hint is None:
            raise SourcingError(f"{folder / HINTS_FILE}:{n}: each line needs an 'object' (or 'card') key")
        _merge(hints, hint)
    for n, row in enumerate(_rows(folder / PIPELINE_FILE), 1):
        if not row.get("card"):
            raise SourcingError(f"{folder / PIPELINE_FILE}:{n}: each row needs a 'card' key")
        _merge(hints, hint_from_pipeline_row(row))
    for hint in hints.values():
        _add_volume(hint)
    return hints
