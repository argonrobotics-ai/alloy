# SPDX-License-Identifier: Apache-2.0
"""Texture pool: CC0 texture sets as looks grouped in families, so a scene variant can swap an object's look for
another texture of the same family (wood for wood, metal for metal).

Pool files live in ``looks/textures/<name>.json`` in each card root (``library/looks/textures``; the first root that
has a name wins, like cards). Each is a WP20 look material that uses one CC0 texture set, plus ``look`` (the name,
equal to the file stem), ``family`` (one of ``FAMILIES``), ``source`` (the set's page), ``author`` and ``licence``
(``"CC0-1.0"``) and ``albedo`` (the colour map's mean linear luminance); see ``alloy.looks.schema``. The images are
not in git: both renderers read them from the render asset cache (``$ALLOY_CACHE/render`` when ``ALLOY_CACHE`` is set,
else ``~/.cache/alloy-render``; ``ALLOY_RENDER_ASSETS`` overrides both). ``fetch`` fills that cache from the library
dataset on the Hugging Face Hub (``libraries/textures``) or straight from Poly Haven and ambientCG.

Example::

    >>> import random
    >>> from alloy.looks import pool
    >>> pool.family_of({"preset": "wood_oak"})
    'wood'
    >>> pool.family_of({"preset": "glass_clear"}) is None
    True
    >>> name, spec = pool.swap({"preset": "wood_oak"}, random.Random(0))    # doctest: +SKIP
    >>> pool.fetch(family="wood")                                           # doctest: +SKIP
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from alloy.cards import CardLibrary
from alloy.looks.resolve import _read, look_dirs
from alloy.looks.schema import FAMILIES, META_KEYS

POOL_DIR = "textures"
HUB_DIR = "libraries/textures"
HUB_LISTING = "files.json"

# The family of an object look that has no pool texture, by its WP20 preset. Presets left out (glass, clear and
# translucent plastics, ceramics, food, liquids, rubber, cables, robot paint, ...) have no family: a texture swap
# would make them look like something else, so variants keep them.
PRESET_FAMILY = {
    "wood_oak": "wood", "wood_plywood": "wood", "wood_walnut_oiled": "wood",
    "aluminum_bare": "metal", "aluminum_anodized_black": "metal", "stainless_steel_brushed": "metal",
    "steel_brushed": "metal", "steel_polished": "metal", "steel_black_oxide": "metal", "copper": "metal",
    "brass": "metal", "gold_plated": "metal", "zinc_plated": "metal", "powder_coat_black": "metal",
    "abs_black": "plastic", "abs_switch_housing": "plastic", "nylon_natural": "plastic", "plastic_matte": "plastic",
    "plastic_gloss": "plastic",
    "linen": "fabric", "cotton_terry": "fabric",
    "marble_white": "stone",
    "paper": "paper", "cardboard_kraft": "paper",
}


@dataclass(frozen=True)
class PoolLook:
    """One texture of the pool: ``spec`` is the WP20 look material (preset, texture, res, scale, overrides)."""

    name: str
    family: str
    spec: dict
    source: str
    author: str
    licence: str
    path: str
    note: str = ""
    albedo: float | None = None

    @property
    def texture(self) -> str:
        """The texture set reference, e.g. ``"polyhaven:oak_veneer_01"``."""
        return self.spec["texture"]


_POOL: dict[tuple[str, ...], list[PoolLook]] = {}


def entries(lib: CardLibrary | None = None, family: str | None = None) -> list[PoolLook]:
    """Every pool texture (or one family's), sorted by name. Raises ValueError for an unknown family and
    LookFileError for a bad pool file."""
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown texture family {family!r}; use one of {list(FAMILIES)}")
    lib = lib or CardLibrary()
    dirs = look_dirs(lib)
    key = tuple(str(d) for d in dirs)
    if key not in _POOL:
        paths: dict[str, Path] = {}
        for d in dirs:
            for p in sorted((d / POOL_DIR).rglob("*.json")):
                paths.setdefault(p.stem, p)
        _POOL[key] = [_entry(paths[n]) for n in sorted(paths)]
    return [e for e in _POOL[key] if family is None or e.family == family]


def _entry(path: Path) -> PoolLook:
    d = _read(path)
    spec = {k: v for k, v in d.items() if k not in META_KEYS}
    return PoolLook(d["look"], d["family"], spec, d["source"], d["author"], d["licence"], str(path), d.get("note", ""),
                    float(d["albedo"]))


def families(lib: CardLibrary | None = None) -> dict[str, int]:
    """Number of pool textures per family (every family in ``FAMILIES``, 0 when it has none)."""
    out = dict.fromkeys(FAMILIES, 0)
    for e in entries(lib):
        out[e.family] += 1
    return out


def family_of(spec: dict, lib: CardLibrary | None = None) -> str | None:
    """The family of a look material: its pool texture's family, else its preset's (``PRESET_FAMILY``), else None.
    Example: ``family_of(resolve("cutting_board").spec())``."""
    tex = spec.get("texture")
    if tex:
        for e in entries(lib):
            if e.texture == tex:
                return e.family
    return PRESET_FAMILY.get(spec.get("preset", ""))


def swap(spec: dict, rng, lib: CardLibrary | None = None) -> tuple[str, dict] | None:
    """(pool name, look material) of another texture in ``spec``'s family, drawn with ``rng`` (``random.Random`` or
    ``numpy.random.Generator``); never the texture ``spec`` already has. None when ``spec`` has no family or its
    family has no other texture: keep the look then."""
    fam = family_of(spec, lib)
    if fam is None:
        return None
    pick = [e for e in entries(lib, fam) if e.texture != spec.get("texture")]
    if not pick:
        return None
    k = int(rng.integers(len(pick))) if hasattr(rng, "integers") else rng.randrange(len(pick))
    return pick[k].name, dict(pick[k].spec)


def fetch(family: str | None = None, source: str = "hub", lib: CardLibrary | None = None,
          root: str | Path | None = None, repo: str | None = None) -> dict:
    """Put the pool's texture sets (or one family's) into the render asset cache ``root`` (default: the renderers'
    cache, ``alloy.render.blender.assets.default_cache()``), so the path tracer, which never downloads, can render
    them. ``source="hub"``: the library dataset on the
    Hugging Face Hub (public; files checked against its sha256 listing); ``"origin"``: Poly Haven and ambientCG.
    Returns counts and the sets that could not be had."""
    from alloy.render.blender.assets import AssetCache, default_cache
    sel = entries(lib, family)
    root = Path(root).expanduser() if root else default_cache()
    if source == "origin":
        cache = AssetCache(root)
        missing = [e.name for e in sel if not cache.texture(e.texture, e.spec["res"])]
        return {"source": "origin", "sets": len(sel) - len(missing), "missing": missing, "root": str(root)}
    if source != "hub":
        raise ValueError(f"source {source!r}: use 'hub' (the library dataset) or 'origin' (Poly Haven, ambientCG)")
    return _fetch_hub({e.texture for e in sel}, root, repo)


def _fetch_hub(want: set[str], root: Path, repo: str | None) -> dict:
    """Download the listed files of the ``want`` sets that are not already in ``root`` with the right sha256."""
    from alloy.assets.hub.layout import sha256_file
    from alloy.assets.hub.source import HubError, download
    staging = root / ".hub_staging"
    try:
        listing = json.loads(download(f"{HUB_DIR}/{HUB_LISTING}", staging, repo).read_text())
        files = [f for f in listing["files"] if f.get("texture") in want]
        got = kept = 0
        for f in files:
            dest = root / f["path"]
            if dest.is_file() and sha256_file(dest) == f["sha256"]:
                kept += 1
                continue
            tmp = download(f"{HUB_DIR}/{f['file']}", staging, repo)
            if sha256_file(tmp) != f["sha256"]:
                raise HubError(f"{HUB_DIR}/{f['file']} arrived with a different sha256 than {HUB_LISTING} lists",
                               "retry; if it repeats, report it with this message")
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, dest)
            got += 1
        for folder in sorted({Path(f["path"]).parent for f in files if f["path"].startswith("ambientcg/")}):
            (root / folder / ".done").write_text("ok")  # the render cache's "set complete" marker (not in the dataset)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    missing = sorted(want - {f.get("texture") for f in listing["files"]})
    return {"source": "hub", "downloaded": got, "kept": kept, "missing": missing, "root": str(root)}


__all__ = ["FAMILIES", "PRESET_FAMILY", "PoolLook", "entries", "families", "family_of", "fetch", "swap"]
