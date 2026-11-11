# SPDX-License-Identifier: Apache-2.0
"""Hand-authored objects as dataset specs: objects whose cards sit in a card root, with no importer manifest.

Each card's files come from following its references:
- every ``path:`` in the card;
- the files an MJCF names (``<compiler meshdir|texturedir|assetdir>``, ``file="..."``, ``<include>``);
- the object's own material cards (materials Alloy does not already ship);
- its look (``looks/objects/<card path>.json``) and its materials' looks.

Pair cards whose materials are all shipped here or with Alloy, and at least one of them here, are the library's
shared files.

Example::

    specs, shared = authored_specs(Path("assets"), ["objects/kitchen/pot.yaml"], {"pot": {"licence": "CC0-1.0"}})
    sorted(specs["pot"].files)       # ['looks/objects/kitchen/pot.json', 'mjcf/kitchen/pot.xml', 'objects/...']
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from alloy.assets.hub.build import BuildError, ObjectSpec
from alloy.assets.hub.mjcf import mjcf_refs
from alloy.assets.hub.provenance import _core_materials

LOOK = "look.json"
MATERIAL_LOOK = "material.look.json"


def _values(node: Any, key: str) -> Iterator[str]:
    """Every string value stored under ``key`` anywhere in a parsed card."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from _values(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _values(v, key)


def _rel(root: Path, path: Path) -> str:
    rel = os.path.relpath(os.path.normpath(path), root)
    if rel.startswith(".."):
        raise BuildError(f"{path} points outside the card root {root}")
    return rel.replace(os.sep, "/")


def mjcf_files(root: Path, xml_rel: str) -> list[str]:
    """The MJCF file and every file it names (meshes, textures, includes), as paths under ``root``."""
    path = root / xml_rel
    if not path.is_file():
        raise BuildError(f"{xml_rel}: MJCF not found under {root}")
    assets, includes = mjcf_refs(path)
    out = [xml_rel] + [_rel(root, a) for a in assets]
    for inc in includes:
        out += mjcf_files(root, _rel(root, inc))
    return out


def _material_file(root: Path, name: str) -> str | None:
    """The root's material card for ``name``, or None when Alloy ships it (or no card exists)."""
    if (_core_materials() / f"{name}.yaml").is_file():
        return None
    hits = sorted((root / "materials").rglob(f"{name}.yaml")) if (root / "materials").is_dir() else []
    return _rel(root, hits[0]) if hits else None


def card_spec(root: Path, card_rel: str, meta: dict[str, Any]) -> ObjectSpec:
    """One authored object: its card, the files it references, its own materials and looks."""
    doc = yaml.safe_load((root / card_rel).read_text(encoding="utf-8")) or {}
    name = doc.get("name") or Path(card_rel).stem
    files: dict[str, str | None] = {card_rel: None}
    for p in _values(doc.get("shape", doc), "path"):
        for rel in (mjcf_files(root, p) if p.endswith(".xml") else [p]):
            files[rel] = None
    rename: dict[str, str] = {}
    mats = [m for m in dict.fromkeys(_values(doc, "material")) if _material_file(root, m)]
    for m in mats:
        files[_material_file(root, m)] = None
        look = root / "looks" / "materials" / f"{m}.json"
        if look.is_file():
            files[_rel(root, look)] = None
            rename[_rel(root, look)] = MATERIAL_LOOK if len(mats) == 1 else f"{m}.look.json"
    look = root / "looks" / Path(card_rel).with_suffix(".json")            # looks/objects/<card path>.json
    if look.is_file():
        files[_rel(root, look)] = None
        rename[_rel(root, look)] = LOOK
    material = _material_file(root, mats[0]) if len(mats) == 1 else None
    category = meta.get("category") or Path(card_rel).parent.name
    m = {"category": category, "title": meta.get("title", name), "licence": meta.get("licence"),
         "attribution": meta.get("attribution", ""), "source": meta.get("source", ""),
         "showcase_ok": meta.get("showcase_ok", True), "showcase_note": meta.get("showcase_note", ""),
         "physics": {"mass_basis": meta.get("mass_basis", "authored")}, "flags": ""}
    return ObjectSpec(name, card_rel, "", files, material, None, m, rename)


def shared_pairs(root: Path, materials: set[str]) -> list[str]:
    """Pair cards between materials all shipped here or with Alloy, at least one of them here (``materials``)."""
    out = []
    for p in sorted((root / "pairs").rglob("*.yaml")) if (root / "pairs").is_dir() else []:
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        ab = [doc.get("a"), doc.get("b")]
        if any(x in materials for x in ab) and all(x in materials or (_core_materials() / f"{x}.yaml").is_file()
                                                   for x in ab):
            out.append(_rel(root, p))
    return out


def authored_specs(root: Path, cards: list[str], meta: dict[str, dict[str, Any]]) -> tuple[dict[str, ObjectSpec],
                                                                                            list[str]]:
    """Specs of the given cards (paths under ``root``) and the library's shared files (pair cards).
    ``meta[name]`` gives category, title, licence, attribution, source, showcase_ok and mass_basis per object."""
    specs = {}
    for c in cards:
        s = card_spec(Path(root), c, meta.get(Path(c).stem, {}))
        specs[s.name] = s
    shipped = {Path(r).stem for s in specs.values() for r in s.files if r.startswith("materials/")}
    return specs, shared_pairs(Path(root), shipped)
