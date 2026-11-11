# SPDX-License-Identifier: Apache-2.0
"""Sourced vs ESTIMATE flags of an object, for its index row: where its mass and material values come from.

``mass_basis`` is the importer's own record (YCB/GSO ``mass_status``: sourced, measured or estimate; Gym-Anything
``mass_basis``: datasheet, authored, derived, estimate or none); objects without one (RoboCasa, Objaverse) take their
mass from a material density, recorded as ``estimate``. ``sources`` names the source kind of each material value
(``catalog``: a cited datasheet or handbook value; ``fit``: fitted to a measured mass; ``model``: a stated model or
typical value). ``estimate`` is true when the mass is an estimate or any material value is marked ESTIMATE.

Example::

    >>> provenance(Path("imports/ycb_gso"), "objects/ycb/ycb_025_mug.yaml",          # doctest: +SKIP
    ...            "materials/ycb/ycb_025_mug_eff.yaml", {"mass_status": "sourced"})
    {'mass_basis': 'sourced', 'estimate': False, 'sources': {'density': 'fit', 'mu_static': 'model', ...}}
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

QUANTITIES = {"density": ("density",), "mu_static": ("surface", "mu_static"), "mu_dynamic": ("surface", "mu_dynamic"),
              "restitution": ("surface", "restitution"), "youngs": ("elastic", "youngs"),
              "poisson": ("elastic", "poisson")}
BASIS_KEYS = ("mass_status", "mass_basis")
ESTIMATE_BASES = {"estimate", "none", ""}


def mass_is_estimate(basis: str) -> bool:
    """True when a ``mass_basis`` records an estimated mass: ``estimate``, none recorded, or a basis that says so
    (Gym-Anything's ``material density (estimate)``). Example: ``mass_is_estimate("sourced") is False``."""
    basis = basis.strip().lower()
    return basis in ESTIMATE_BASES or "estimate" in basis


def _core_materials() -> Path:
    from alloy.cards.library import default_root  # noqa: PLC0415 - the shipped library, checkout or wheel (SCR-OSS2-2)

    return default_root() / "materials"


def material_path(root: Path, card: str, material: str | None) -> Path | None:
    """The object's material card: its own file when the manifest names one, else the card's ``material:`` name in
    the root's ``materials/`` or Alloy's library materials. None when not found."""
    if material and (root / material).is_file():
        return root / material
    try:
        name = (yaml.safe_load((root / card).read_text(encoding="utf-8")) or {}).get("material")
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(name, str):
        return None
    hits = sorted((root / "materials").rglob(f"{name}.yaml")) if (root / "materials").is_dir() else []
    core = _core_materials() / f"{name}.yaml"
    return hits[0] if hits else (core if core.is_file() else None)


def _sources(doc: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key, path in QUANTITIES.items():
        node: Any = doc
        for part in path:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and node.get("source"):
            out[key] = str(node["source"])
    return out


def provenance(root: Path, card: str, material: str | None, physics: dict[str, Any]) -> dict[str, Any]:
    """``{"mass_basis", "estimate", "sources"}`` for one object. Example: see the module docstring."""
    path = material_path(Path(root), card, material)
    text = path.read_text(encoding="utf-8") if path else ""
    doc = yaml.safe_load(text) if text else {}
    basis = next((str(physics[k]).lower() for k in BASIS_KEYS if physics.get(k)), "estimate")
    sources = _sources(doc if isinstance(doc, dict) else {})
    return {"mass_basis": basis, "estimate": basis in ESTIMATE_BASES or "ESTIMATE" in text, "sources": sources}
