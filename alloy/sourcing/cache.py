# SPDX-License-Identifier: Apache-2.0
"""Per-category and per-material cache of sourced values, so repeats are cheap.

One YAML file per entry under ``<alloy-assets>/sources/imports/cache/{categories,materials}/<key>.yaml``:

.. code-block:: yaml

    schema: alloy.sourcing.category/1
    key: ceramic_mug
    kind: category                  # or material
    title: Ceramic coffee mug, 10-15 fl oz
    aliases: [coffee mug, mug, porcelain mug]
    material: stoneware_glazed      # suggested Alloy material (categories only)
    updated: 2026-11-06
    values:
      mass: {value: 0.35, lo: 0.28, hi: 0.45, tag: ESTIMATE, reason: "...", evidence: [...]}

A category value applied to one object is a prior for that object, so ``alloy.sourcing.match`` re-tags it ESTIMATE
there (the cached DOC evidence stays attached). Material values (density, friction) carry their own tag.

Example:
    >>> from alloy.sourcing.cache import normalize_key
    >>> normalize_key("Threshold Porcelain Coffee-Mug (12 oz)")
    'threshold_porcelain_coffee_mug_12_oz'
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from alloy.sourcing.fields import SourcingError, yaml_dump, yaml_load
from alloy.sourcing.paths import sources_root
from alloy.sourcing.record import SourcedValue, value_to_dict, values_from_dict

log = logging.getLogger("alloy.sourcing")

SCHEMA: Final[str] = "alloy.sourcing.category/1"
Kind = Literal["category", "material"]
KIND_DIRS: Final[dict[str, str]] = {"category": "categories", "material": "materials"}
MAPS_DIR: Final[str] = "maps"
"""``cache/maps/<dataset>.json``: ``{source object name: category key}`` for datasets whose names say little."""
_KEYS: Final[frozenset[str]] = frozenset(
    {"schema", "key", "kind", "title", "aliases", "material", "updated", "notes", "values"})


def normalize_key(name: str) -> str:
    """Lower-case snake key: every run of non-alphanumerics becomes one ``_``.

    Example:
        >>> normalize_key("  Mug, ceramic ")
        'mug_ceramic'
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class CacheEntry:
    """Sourced values for one object category or one material, with aliases used for matching.

    Example:
        >>> CacheEntry(key="ceramic_mug", kind="category", title="Ceramic mug").names()
        ('ceramic_mug',)
    """

    key: str
    kind: Kind
    title: str
    values: dict[str, SourcedValue] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    material: str = ""
    updated: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.key != normalize_key(self.key) or not self.key:
            raise SourcingError(f"cache key {self.key!r} must be snake_case; use {normalize_key(self.key)!r}")
        if self.kind not in KIND_DIRS:
            raise SourcingError(f"{self.key}: kind must be one of {sorted(KIND_DIRS)}, got {self.kind!r}")
        for name, v in self.values.items():
            if name != v.field:
                raise SourcingError(f"{self.key}: value under {name!r} is for field {v.field!r}")

    def names(self) -> tuple[str, ...]:
        """The key and the normalized aliases, longest first (more specific names match first)."""
        names = {self.key, *(normalize_key(a) for a in self.aliases)}
        return tuple(sorted(names, key=lambda n: (-len(n), n)))


def entry_to_dict(e: CacheEntry) -> dict[str, Any]:
    """Plain dict in canonical key order (round-trips through :func:`entry_from_dict`)."""
    out: dict[str, Any] = {"schema": SCHEMA, "key": e.key, "kind": e.kind, "title": e.title}
    if e.aliases:
        out["aliases"] = list(e.aliases)
    out.update({k: getattr(e, k) for k in ("material", "updated", "notes") if getattr(e, k)})
    out["values"] = {k: value_to_dict(v) for k, v in sorted(e.values.items())}
    return out


def entry_from_dict(raw: Any, label: str) -> CacheEntry:
    """Parse a cache file's mapping; raises ``SourcingError`` naming the bad key.

    Example:
        >>> entry_from_dict({"schema": SCHEMA, "key": "apple", "kind": "category", "title": "Apple"}, "x").key
        'apple'
    """
    if not isinstance(raw, dict):
        raise SourcingError(f"{label}: a cache entry must be a mapping")
    extra = sorted(set(raw) - _KEYS)
    if extra:
        raise SourcingError(f"{label}: unknown keys {extra}; allowed {sorted(_KEYS)}")
    if raw.get("schema") != SCHEMA:
        raise SourcingError(f"{label}: schema must be {SCHEMA!r}, got {raw.get('schema')!r}")
    for k in ("key", "kind", "title"):
        if not raw.get(k):
            raise SourcingError(f"{label}: missing {k!r}")
    return CacheEntry(
        key=str(raw["key"]), kind=raw["kind"], title=str(raw["title"]),
        values=values_from_dict(raw.get("values"), label), aliases=tuple(str(a) for a in raw.get("aliases") or ()),
        material=str(raw.get("material", "")), updated=str(raw.get("updated", "")), notes=str(raw.get("notes", "")))


def dump_entry(e: CacheEntry) -> str:
    """Canonical YAML text for a cache entry."""
    return yaml_dump(entry_to_dict(e))


def load_entry(path: Path) -> CacheEntry:
    """Read one cache file; the file stem must equal the key."""
    entry = entry_from_dict(yaml_load(path.read_text(encoding="utf-8")), str(path))
    if path.stem != entry.key:
        raise SourcingError(f"{path}: file name must be {entry.key}.yaml (the key)")
    return entry


class Cache:
    """All cache entries under ``<sources>/cache``; ``find`` matches an object name or category to an entry.

    Example:
        >>> import tempfile
        >>> c = Cache(Path(tempfile.mkdtemp()))
        >>> _ = c.put(CacheEntry(key="ceramic_mug", kind="category", title="Ceramic mug", aliases=("coffee mug",)))
        >>> c.find("Threshold_Porcelain_Coffee_Mug").key
        'ceramic_mug'
    """

    def __init__(self, sources: Path | None = None) -> None:
        self.root = (sources or sources_root()) / "cache"
        self._entries: dict[tuple[str, str], CacheEntry] | None = None
        self._maps: dict[str, str] | None = None

    def _load(self) -> dict[tuple[str, str], CacheEntry]:
        if self._entries is None:
            self._entries = {}
            for kind, sub in KIND_DIRS.items():
                for path in sorted((self.root / sub).glob("*.yaml")):
                    entry = load_entry(path)
                    if entry.kind != kind:
                        raise SourcingError(f"{path}: kind {entry.kind!r} is in the {sub}/ folder; move it")
                    self._entries[(kind, entry.key)] = entry
        return self._entries

    def entries(self, kind: Kind | None = None) -> Iterator[CacheEntry]:
        """Entries sorted by key, optionally of one kind."""
        for (k, _), e in sorted(self._load().items()):
            if kind is None or k == kind:
                yield e

    def get(self, key: str, kind: Kind = "category") -> CacheEntry | None:
        """The entry with exactly this key, or None."""
        return self._load().get((kind, normalize_key(key)))

    def lookup(self, name: str, kind: Kind = "category") -> CacheEntry | None:
        """The entry whose key or one alias equals ``name`` (normalized); no partial matches."""
        target = normalize_key(name)
        hit = self.get(target, kind)
        if hit is not None:
            return hit
        return next((e for e in self.entries(kind) if target in e.names()), None)

    def find(self, name: str, kind: Kind = "category") -> CacheEntry | None:
        """Best entry for an object name or category: exact key/alias first, then the longest alias that appears
        as a whole-word run inside ``name`` (``bowl`` matches ``red_plastic_bowl``, not ``bowling_pin``)."""
        target = normalize_key(name)
        best: tuple[int, CacheEntry] | None = None
        for e in self.entries(kind):
            for n in e.names():
                if n == target:
                    return e
                if re.search(rf"(^|_){re.escape(n)}(_|$)", target) and (best is None or len(n) > best[0]):
                    best = (len(n), e)
        return best[1] if best else None

    def _load_maps(self) -> dict[str, str]:
        if self._maps is None:
            self._maps = {}
            for path in sorted((self.root / MAPS_DIR).glob("*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise SourcingError(f"{path}: a name map must be a JSON object {{source name: category key}}")
                self._maps.update({normalize_key(k): normalize_key(str(v)) for k, v in raw.items()})
        return self._maps

    def mapped(self, name: str) -> str | None:
        """The category key a name map gives ``name``. Up to three leading segments are dropped one at a time, so an
        importer prefix still matches (``gso_reebok_ziglite_rush`` -> ``reebok_ziglite_rush``)."""
        maps = self._load_maps()
        parts = normalize_key(name).split("_")
        for k in range(min(4, len(parts))):
            hit = maps.get("_".join(parts[k:]))
            if hit:
                return hit
        return None

    def put_map(self, dataset: str, mapping: dict[str, str]) -> Path:
        """Write ``cache/maps/<dataset>.json`` (source names as given, keys normalized)."""
        path = self.root / MAPS_DIR / f"{normalize_key(dataset)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: normalize_key(v) for k, v in sorted(mapping.items())}, indent=0) + "\n",
                        encoding="utf-8")
        self._maps = None
        return path

    def put(self, entry: CacheEntry) -> Path:
        """Write (or replace) one entry; returns its path."""
        path = self.root / KIND_DIRS[entry.kind] / f"{entry.key}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_entry(entry), encoding="utf-8")
        self._load()[(entry.kind, entry.key)] = entry
        log.info("cache: wrote %s %s (%d values)", entry.kind, entry.key, len(entry.values))
        return path
