# SPDX-License-Identifier: Apache-2.0
"""Resolve a card to its look: the object's look file, else its material's default look, else the grey fallback.

Look files are found in every root of a ``CardLibrary`` (library roots first, then the assets root), under
``looks/materials`` and ``looks/objects`` (recursive; the file stem is the key). The first root that has a key wins,
like cards. The fallback is WP20's ``plastic_matte`` and is logged with the file to add.

Example::

    >>> from alloy.looks.resolve import resolve
    >>> resolve("jar_glass_8oz").spec()["preset"]       # doctest: +SKIP
    'glass_clear'
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from alloy.cards import CardLibrary
from alloy.cards import types as ct
from alloy.cards.roots import IMPORTS_DIR
from alloy.looks.schema import META_KEYS, LookFileError, validate_look
from alloy.render.blender.presets import FALLBACK_MATERIAL

log = logging.getLogger("alloy.looks")

KINDS = ("materials", "objects")


@dataclass(frozen=True)
class Look:
    """A resolved look: ``source`` is ``object``, ``material`` or ``fallback``; ``spec(part)`` gives the WP20
    material for a part name (``assign`` regexes, else ``default``)."""

    name: str
    source: str
    materials: dict[str, dict]
    assign: tuple[dict, ...] = ()
    default: str = "body"
    path: str | None = None
    note: str = ""

    def spec(self, part: str = "") -> dict:
        """WP20 look material for ``part`` (the object's default material when no rule matches)."""
        for rule in self.assign:
            if part and re.search(rule["match"], part):
                return dict(self.materials[rule["material"]])
        return dict(self.materials[self.default])


@dataclass
class _Index:
    paths: dict[str, dict[str, Path]] = field(default_factory=lambda: {k: {} for k in KINDS})


_INDEX: dict[tuple[str, ...], _Index] = {}


def look_dirs(lib: CardLibrary) -> list[Path]:
    """``<root>/looks`` for every root of ``lib`` that has one (library roots first, then the assets root)."""
    return [r / "looks" for r in lib.search_roots if (r / "looks").is_dir()]


def _index(lib: CardLibrary) -> _Index:
    dirs = look_dirs(lib)
    key = tuple(str(d) for d in dirs)
    if key not in _INDEX:
        idx = _Index()
        for d in dirs:
            for kind in KINDS:
                for p in sorted((d / kind).rglob("*.json")):
                    idx.paths[kind].setdefault(p.stem, p)
        _INDEX[key] = idx
    return _INDEX[key]


def _read(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LookFileError(f"{path}: cannot read the look file ({exc}); fix or delete it") from exc
    issues = validate_look(d, path)
    if issues:
        raise LookFileError("; ".join(issues))
    return d


def material_look(material: str, lib: CardLibrary | None = None, name: str | None = None) -> Look | None:
    """The default look of ``material`` (None if no root has ``looks/materials/<material>.json``)."""
    lib = lib or CardLibrary()
    path = _index(lib).paths["materials"].get(material)
    if path is None:
        return None
    d = _read(path)
    spec = {k: v for k, v in d.items() if k not in META_KEYS}
    return Look(name or material, "material", {material: spec}, (), material, str(path), d.get("note", ""))


def object_look(name: str, lib: CardLibrary | None = None) -> Look | None:
    """The object's own look (None if no root has ``looks/objects/**/<name>.json``)."""
    lib = lib or CardLibrary()
    path = _index(lib).paths["objects"].get(name)
    if path is None:
        return None
    d = _read(path)
    return Look(name, "object", dict(d["materials"]), tuple(d.get("assign") or ()), d["default"], str(path),
                d.get("note", ""))


def resolve(card: str | ct.ObjectCard, lib: CardLibrary | None = None) -> Look:
    """The look to render ``card`` (a name or an ObjectCard) with: object look, else material default, else grey."""
    lib = lib or CardLibrary()
    obj = card if isinstance(card, ct.ObjectCard) else lib.peek_object(card)
    look = object_look(obj.name, lib) or material_look(obj.material, lib, obj.name)
    if look is not None:
        return look
    log.warning("%s: material %r has no default look; add looks/materials/%s.json to a card root (grey for now)",
                obj.name, obj.material, obj.material)
    return Look(obj.name, "fallback", {"fallback": {"preset": FALLBACK_MATERIAL}}, (), "fallback")


def hand_authored(lib: CardLibrary, name: str) -> bool:
    """True for an object card written by hand: in the first library root or the assets checkout, not in an imported
    library (``<assets>/imports/<source>/`` or its fetch cache)."""
    path = lib.path_of("objects", name)
    if lib.assets is not None and path.is_relative_to(lib.assets / IMPORTS_DIR):
        return False
    return path.is_relative_to(lib.roots[0]) or (lib.assets is not None and path.is_relative_to(lib.assets))


def coverage(lib: CardLibrary | None = None, skip_prefixes: tuple[str, ...] = ("dc_",)) -> dict:
    """How every hand-authored object card of ``lib`` resolves: counts per source and the cards that would render
    grey (imported libraries and ``skip_prefixes`` are left out)."""
    lib = lib or CardLibrary()
    counts = {"object": 0, "material": 0, "fallback": 0}
    grey = []
    for name in lib.names("objects"):
        if name.startswith(skip_prefixes) or not hand_authored(lib, name):
            continue
        look = resolve(name, lib)
        counts[look.source] += 1
        if look.source == "fallback":
            grey.append(name)
    return {"objects": sum(counts.values()), **counts, "grey": grey}


__all__ = ["Look", "coverage", "hand_authored", "look_dirs", "material_look", "object_look", "resolve"]
