# SPDX-License-Identifier: Apache-2.0
"""Look-file schema (``alloy.look/1``) and validation; every error names the file, the field and what to change.

Material file: ``{"schema": "alloy.look/1", "material": <name>, "preset": <WP20 preset>, <override keys>,
"note": ..., "licence": ...}``. Object file: ``{"schema": "alloy.look/1", "object": <name>, "materials": {key: spec},
"assign": [{"match": <regex over part names>, "material": key}], "default": key, "note": ...}``. Texture pool
file (``alloy.looks.pool``): ``{"schema": "alloy.look/1", "look": <name>, "family": <one of FAMILIES>, "preset": ...,
"texture": "polyhaven:<id>" | "ambientcg:<id>", "res": "1k" | "2k", <override keys>, "source": <https page>,
"author": ..., "licence": "CC0-1.0", "albedo": <0-1>, "note": ...}``; ``albedo`` is the mean linear Rec. 709 luminance
of the colour map at ``res`` (every 7th pixel; ``alloy.scene.variants.contrast.texture_albedo``), so scene variants can
compare surfaces without the images.

Example::

    >>> validate_look({"schema": "alloy.look/1", "material": "abs", "preset": "abs_black", "note": "black ABS"})
    []
"""

from __future__ import annotations

import re
from pathlib import Path

from alloy.render.blender.presets import EXTRA_KEYS, OVERRIDE_KEYS, PRESETS

SCHEMA = "alloy.look/1"
META_KEYS = ("schema", "material", "object", "look", "note", "licence", "family", "source", "author", "albedo")
FAMILIES = ("wood", "metal", "plastic", "fabric", "stone", "tile", "paper", "floor")
POOL_LICENCE = "CC0-1.0"
POOL_SOURCES = ("polyhaven:", "ambientcg:")
POOL_RES = ("1k", "2k")
SPEC_KEYS = frozenset(OVERRIDE_KEYS) | frozenset(EXTRA_KEYS)


class LookFileError(ValueError):
    """A look file that cannot be used; the message says what to change."""


def spec_issues(spec: dict, where: str) -> list[str]:
    """Problems with one WP20 look material (unknown preset or key)."""
    if not isinstance(spec, dict):
        return [f"{where}: a look material must be a mapping with a 'preset', got {type(spec).__name__}"]
    out = []
    if spec.get("preset") not in PRESETS:
        out.append(f"{where}: preset {spec.get('preset')!r} is not a WP20 preset; use one of "
                   f"alloy.render.blender.presets.PRESETS (e.g. 'plastic_matte')")
    out += [f"{where}: unknown key {k!r}; allowed: WP20 override/extra keys" for k in spec if k not in SPEC_KEYS]
    return out


def _object_issues(d: dict, where: str) -> list[str]:
    mats = d.get("materials")
    if not isinstance(mats, dict) or not mats:
        return [f"{where}: 'materials' must map part material keys to WP20 specs"]
    out = [i for k, s in mats.items() for i in spec_issues(s, f"{where} materials.{k}")]
    for i, rule in enumerate(d.get("assign") or []):
        if not isinstance(rule, dict):
            out.append(f"{where} assign[{i}]: a rule is a mapping {{'match': <regex>, 'material': <key>}}")
            continue
        try:
            re.compile(rule["match"])
        except (re.error, KeyError, TypeError) as exc:
            out.append(f"{where} assign[{i}]: needs a valid 'match' regex ({exc})")
        if rule.get("material") not in mats:
            out.append(f"{where} assign[{i}]: material {rule.get('material')!r} is not a key of 'materials'")
    if d.get("default") not in mats:
        out.append(f"{where}: 'default' must be one of the 'materials' keys {sorted(mats)}")
    return out


def _pool_issues(d: dict, where: str) -> list[str]:
    """Problems with a texture pool file's family, texture set and credit fields."""
    out = []
    if d.get("family") not in FAMILIES:
        out.append(f"{where}: 'family' must be one of {list(FAMILIES)}, got {d.get('family')!r}")
    tex = d.get("texture")
    if not (isinstance(tex, str) and tex.startswith(POOL_SOURCES) and len(tex.split(":", 1)[1]) > 0):
        out.append(f"{where}: 'texture' must be a CC0 set 'polyhaven:<id>' or 'ambientcg:<id>', got {tex!r}")
    if d.get("res") not in POOL_RES:
        out.append(f"{where}: 'res' must be one of {list(POOL_RES)} (the size of the maps kept), got {d.get('res')!r}")
    if not str(d.get("source", "")).startswith("https://"):
        out.append(f"{where}: 'source' must be the set's https page, got {d.get('source')!r}")
    if not (isinstance(d.get("author"), str) and d["author"].strip()):
        out.append(f"{where}: 'author' must name who made the set")
    alb = d.get("albedo")
    if isinstance(alb, bool) or not isinstance(alb, (int, float)) or not 0.0 <= alb <= 1.0:
        out.append(f"{where}: 'albedo' must be the colour map's mean linear luminance, a number in [0, 1], got {alb!r}")
    if d.get("licence") != POOL_LICENCE:
        out.append(f"{where}: 'licence' must be {POOL_LICENCE!r} (the pool takes CC0 sets only), got {d.get('licence')!r}")
    return out


def validate_look(d: dict, path: str | Path = "<look>") -> list[str]:
    """All problems with a look file's content (empty list = usable)."""
    where = str(path)
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        return [f"{where}: 'schema' must be {SCHEMA!r}"]
    kind = "object" if "object" in d else "look" if "look" in d else "material"
    if kind not in d:
        return [f"{where}: a look file names a 'material', an 'object' or a pool 'look'"]
    stem = Path(path).stem if isinstance(path, Path) else None
    named = [f"{where}: {kind} {d[kind]!r} must match the file name {stem!r}"] if stem and stem != d[kind] else []
    if kind == "object":
        return named + _object_issues(d, where)
    pool = _pool_issues(d, where) if kind == "look" else []
    return named + pool + spec_issues({k: v for k, v in d.items() if k not in META_KEYS}, where)


__all__ = ["FAMILIES", "META_KEYS", "POOL_LICENCE", "SCHEMA", "LookFileError", "spec_issues", "validate_look"]
