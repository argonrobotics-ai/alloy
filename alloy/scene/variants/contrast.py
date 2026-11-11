# SPDX-License-Identifier: Apache-2.0
"""Look rules for the large surfaces of a variant, so a table never melts into the floor and a white floor never
blows out a render.

- Large surfaces: every support a free object rests on at the start (a table, a bin, a shelf) and every ground item.
- Floors: every ground item, and any static part a support stands on whose footprint is at least ``FLOOR_AREA_M2``.
  They take textures of the pool's ``floor`` family only (``FLOOR_FAMILY``: wood floors, laminate, parquet, tiles,
  carpet, concrete), made for large surfaces: a 1-2 m furniture veneer tiled over a 40 m ground plane reads as dunes.
- Contrast: a support and what it stands on (the ground, a mat, another static part right under it) must differ in
  texture family or in mean albedo by at least ``gap`` (0.15). On a floor, family does not count (every floor is
  ``floor``, and a pine parquet under a pine table hides its edge just the same): the albedo must differ.
- Cap (optional): no large surface's mean albedo above ``max_albedo``.

Mean albedo is the mean linear luminance (Rec. 709) of what the renderer puts on the surface, times its colour gain:
for a textured look material its colour map's, for an untextured or ``tex_tint`` one its base colour's. A pool
texture's value is its pool file's ``albedo`` (the same on every host; variants only swap pool textures in). A texture
outside the pool (a look's own, kept with ``textures="keep"`` or when its family has no other) is measured from its
colour map in the local render asset cache (decoded as the camera does, sRGB to linear); with no images it has no
albedo (``python -m alloy.looks --fetch-textures``), a rule that needs it fails, and the variant keeps its last look
draw with a warning.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.scene.variants.looks import Texture, effective, family_of

log = logging.getLogger("alloy.scene")

ALBEDO_GAP = 0.15
LUMA = np.array([0.2126, 0.7152, 0.0722])
STACK_M = 5.0e-3       # [m] a part's bottom this close to another's top (or the ground) stands on it
FLOOR_FAMILY = "floor"
FLOOR_AREA_M2 = 1.0    # [m^2] a static part under a support with at least this footprint is a floor
PIXEL_STRIDE = 7       # every 7th pixel of a colour map is enough for its mean


@dataclass(frozen=True)
class Surfaces:
    """The scene's large surfaces, its contrast pairs (support, what it stands on) and its floors."""

    large: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    floors: tuple[str, ...] = ()


def surfaces(scene: Any, base: Any, supports: dict[str, set[str]], free: list[str]) -> Surfaces:
    """``supports``: {free object: owners it rests on} (``Placer.supports``)."""
    from alloy.check.bounds import object_aabb  # noqa: PLC0415

    grounds = [n for n, it in scene.items.items() if it.get("kind") == "ground"]
    robots = {n for n, m in base.index.motion.items() if m == "robot"}
    held_up = sorted({s for owners in supports.values() for s in owners} - set(free) - robots - set(grounds))
    still = [n for n in scene.items if n not in free and n not in robots and n in base.index.names]
    state = base.state()
    box = {n: object_aabb(base.built.model, state, base.index, n) for n in still if n not in grounds}
    pairs = []
    for s in held_up:
        if box.get(s) is None:
            continue
        lo = box[s][0]
        pairs += [(s, g) for g in grounds if abs(lo[2] - base.poses[g][2]) <= STACK_M]
        pairs += [(s, t) for t, b in box.items() if t != s and b is not None and _under(box[s], b)]
    floors = grounds + sorted({g for _, g in pairs if g not in grounds and _footprint(box[g]) >= FLOOR_AREA_M2})
    return Surfaces(tuple(held_up + grounds), tuple(pairs), tuple(floors))


def _footprint(b: tuple) -> float:
    lo, hi = b
    return float((hi[0] - lo[0]) * (hi[1] - lo[1]))


def _under(top: tuple, below: tuple) -> bool:
    """``below``'s top is at ``top``'s bottom and their footprints overlap."""
    (lo, hi), (blo, bhi) = top, below
    return abs(lo[2] - bhi[2]) <= STACK_M and bool(np.all(lo[:2] < bhi[:2]) and np.all(blo[:2] < hi[:2]))


def problem(looks: dict[str, dict], surf: Surfaces, pool: dict[str, list[Texture]], gap: float,
            max_albedo: float | None) -> str | None:
    """Why a set of variant looks breaks the rules (module docstring), or None. Items without a look are skipped."""
    spec = {n: lk["materials"][lk["default"]] for n, lk in looks.items()}
    if gap > 0.0:
        for s, g in surf.pairs:
            if s in spec and g in spec:
                why = _pair_problem(s, g, spec[s], spec[g], pool, gap, g in surf.floors)
                if why:
                    return why
    if max_albedo is not None:
        for n in (n for n in surf.large if n in spec):
            a = albedo(spec[n], pool)
            if a is None or a > max_albedo:
                return f"{n}: albedo {_text(a)} over the cap {max_albedo:.2f}"
    return None


def _pair_problem(s: str, g: str, a: dict, b: dict, pool: dict, gap: float, floor: bool) -> str | None:
    """Support ``s`` on ``g``: other families pass (not on a floor), else the albedo must differ by ``gap``."""
    fa, fb = family_of(a, pool=pool), family_of(b, pool=pool)
    if fa != fb and not floor:
        return None
    la, lb = albedo(a, pool), albedo(b, pool)
    if la is not None and lb is not None and abs(la - lb) >= gap:
        return None
    what = f"on the floor {g}" if floor else f"on {g}, both {fa or 'no family'}"
    return f"{s} {what}: albedo {_text(la)} vs {_text(lb)} (need a {gap:.2f} difference)"


def _text(a: float | None) -> str:
    return "unknown (texture not in the local asset cache)" if a is None else f"{a:.2f}"


def albedo(spec: dict, pool: dict[str, list[Texture]] | None = None) -> float | None:
    """Mean linear albedo of a WP20 look material (module docstring): ``pool``'s value for its texture, else its
    colour map's; None when neither is known."""
    eff = effective(spec)
    gain = float(eff.get("color_gain", 1.0))
    if eff.get("texture") and not eff.get("tex_tint"):
        ref = str(eff["texture"])
        known = next((x.albedo for ts in (pool or {}).values() for x in ts if x.ref == ref), None)
        t = known if known is not None else texture_albedo(ref, str(eff["res"]))
        return None if t is None else t * gain
    return float(_linear(np.asarray(eff["base_color"], np.float64)[:3]) @ LUMA) * gain


@functools.lru_cache(maxsize=512)
def texture_albedo(ref: str, res: str) -> float | None:
    """Mean linear luminance of texture ``ref``'s colour map at ``res`` from the local render asset cache."""
    from alloy.camera.errors import CameraError  # noqa: PLC0415
    from alloy.camera.looks.textures import local_texture_files  # noqa: PLC0415

    try:
        path = local_texture_files(ref, res)["color"]
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), np.float64)[::PIXEL_STRIDE, ::PIXEL_STRIDE] / 255.0
    except (CameraError, FileNotFoundError, KeyError, OSError, ImportError) as exc:
        log.debug("variants: no albedo for %s %s (%s)", ref, res, exc)
        return None
    return float((_linear(rgb.reshape(-1, 3)) @ LUMA).mean())


def _linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB 0-1 to linear."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


__all__ = ["ALBEDO_GAP", "FLOOR_AREA_M2", "FLOOR_FAMILY", "Surfaces", "albedo", "problem", "surfaces", "texture_albedo"]
