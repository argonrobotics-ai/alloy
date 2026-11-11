# SPDX-License-Identifier: Apache-2.0
"""Look variants: each object's look gets another texture of its family and a small tint.

- Pool: the CC0 texture pool (``alloy.looks.pool``: ``looks/textures``, 8 families). Each texture carries the fields
  a swap copies, ``texture``, ``scale_m`` (tile size), ``res`` (the resolution its images exist at: the camera never
  downloads) and ``bump_m``, and its mean ``albedo`` (``alloy.scene.variants.contrast``). Textures that only the card
  roots' own look files use have no pool resolution or albedo: they are never swapped in, and a look that has one is
  swapped away from it.
- Family: a look material's ``family``, else ``alloy.looks.pool``'s rule (its texture's family in the pool, else its
  preset's in ``PRESET_FAMILY``). Presets with no family (glass, clear plastics, ceramics, food, rubber, cables) keep
  their texture state: a swap would make them look like something else.
- Swap (``textures="same-family"``): a look material with a family takes another texture of that family (never its
  own while others exist), textured or not; ``"textured"`` swaps only materials that already have a texture;
  ``"keep"`` swaps nothing. The preset and every other value stay.
- Tint: base colour times (1 + u) per channel and colour gain times (1 + u), u uniform in +-``tint``. A texture
  replaces the base colour in WP20's shading (except ``tex_tint`` fabrics), so on a plain texture the tint shows as
  the gain (brightness) only.

Objects without a look file (grey fallback) and robots keep their look.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("alloy.scene")

MODES = ("same-family", "textured", "keep")
NO_LOOK_KINDS = frozenset({"robot"})
TEXTURE_FIELDS = ("texture", "scale_m", "res", "bump_m")    # what a swap copies from the pool texture


@dataclass(frozen=True)
class Texture:
    """One texture of the pool: its reference ("polyhaven:oak_veneer_01"), tile size [m], family, source look, image
    resolution, bump height [m] and mean linear albedo (the pool file's ``albedo``; None when it has none)."""

    ref: str
    scale_m: float
    family: str
    source: str
    res: str = "2k"
    bump_m: float | None = None
    albedo: float | None = None

    def fields(self) -> dict:
        """The look-material fields a swap to this texture sets (``TEXTURE_FIELDS``)."""
        out = {"texture": self.ref, "scale_m": self.scale_m, "res": self.res}
        return out if self.bump_m is None else {**out, "bump_m": self.bump_m}


def texture_pool(lib: Any) -> dict[str, list[Texture]]:
    """{family: textures} of ``alloy.looks.pool`` over ``lib``'s roots (module docstring)."""
    from alloy.looks import pool as P  # noqa: PLC0415

    out: dict[str, list[Texture]] = {}
    for e in P.entries(lib):
        out.setdefault(e.family, []).append(_texture(e.spec, e.family, e.name, getattr(e, "albedo", None)))
    return {f: sorted(ts, key=lambda t: t.ref) for f, ts in sorted(out.items())}


def _texture(spec: dict, family: str, source: str, albedo: float | None = None) -> Texture:
    """A pool Texture from a look material's effective values (its own, else its preset's, else WP20's defaults)."""
    eff = effective(spec)
    return Texture(str(eff["texture"]), float(eff["scale_m"]), family, source, str(eff["res"]),
                   None if eff.get("bump_m") is None else float(eff["bump_m"]), albedo)


def effective(spec: dict) -> dict:
    """A look material's values over its WP20 preset's over WP20's defaults."""
    from alloy.render.blender.presets import DEFAULTS, PRESETS  # noqa: PLC0415

    return {**DEFAULTS, **PRESETS.get(str(spec.get("preset")), {}), **spec}


def texture_of(spec: dict) -> tuple[str | None, float]:
    """(texture reference, tile size) of a WP20 look material: its own, else its preset's."""
    eff = effective(spec)
    return (str(eff["texture"]) if eff.get("texture") else None), float(eff["scale_m"])


def family_of(spec: dict, pool: dict[str, list[Texture]] | None = None) -> str | None:
    """The family of a look material: its ``family``, else its texture's family in ``pool``, else its preset's
    (``alloy.looks.pool.PRESET_FAMILY``), else None."""
    from alloy.looks.pool import PRESET_FAMILY  # noqa: PLC0415

    fam = spec.get("family")
    if fam:
        return str(fam)
    ref, _ = texture_of(spec)
    if ref and pool:
        fam = next((t.family for ts in pool.values() for t in ts if t.ref == ref), None)
    return fam or PRESET_FAMILY.get(str(spec.get("preset", "")))


def item_look(name: str, item: dict, lib: Any) -> Any:
    """The ``alloy.looks`` Look of a scene item (its card's object look, else its material's), or None for robots,
    liquids without a look and the grey fallback."""
    from alloy.looks.resolve import object_look, resolve  # noqa: PLC0415

    if item.get("kind") in NO_LOOK_KINDS:
        return None
    card = item.get("card_name") or name
    try:
        look = object_look(card, lib) or resolve(name, lib)
    except Exception as exc:  # noqa: BLE001 - a card or look file the variants cannot read: keep the look
        log.info("variants: %s keeps its look (%s)", name, exc)
        return None
    return None if look.source == "fallback" else look


def vary_look(look: Any, pool: dict[str, list[Texture]], rng: np.random.Generator, mode: str, tint: float,
              family: str | None = None) -> tuple[dict, dict[str, str]]:
    """(object-form look {materials, assign, default}, {material key: texture now}) for one variant; ``family``:
    swap within this family whatever the look's own (a floor's)."""
    mats, chosen = {}, {}
    for key, spec in look.materials.items():
        spec = dict(spec)
        if mode in ("same-family", "textured"):
            spec = _swap(spec, pool, rng, textured_only=mode == "textured", family=family)
        if tint > 0.0:
            spec = _tint(spec, rng, tint)
        mats[key] = spec
        ref, _ = texture_of(spec)
        if ref:
            chosen[key] = ref
    return {"materials": mats, "assign": [dict(r) for r in look.assign], "default": look.default}, chosen


def _swap(spec: dict, pool: dict[str, list[Texture]], rng: np.random.Generator, textured_only: bool,
          family: str | None = None) -> dict:
    """Another texture of ``family`` (else ``spec``'s, ``family_of``) with its pool fields; ``spec`` unchanged
    without one."""
    ref, _ = texture_of(spec)
    fam = family if family and pool.get(family) else family_of(spec, pool=pool)
    if not fam or (textured_only and not ref):
        return spec
    options = [t for t in pool.get(fam, []) if t.ref != ref]
    if not options:
        return spec
    pick = options[int(rng.integers(len(options)))]
    return {**spec, **pick.fields()}


def _tint(spec: dict, rng: np.random.Generator, tint: float) -> dict:
    from alloy.render.blender.presets import DEFAULTS, PRESETS  # noqa: PLC0415

    preset = PRESETS.get(str(spec.get("preset")), {})
    base = np.asarray(spec.get("base_color", preset.get("base_color", DEFAULTS["base_color"])), np.float64)
    color = np.clip(base[:3] * (1.0 + rng.uniform(-tint, tint, 3)), 0.0, 1.0)
    gain = float(spec.get("color_gain", 1.0)) * (1.0 + float(rng.uniform(-tint, tint)))
    return {**spec, "base_color": [round(float(c), 4) for c in color], "color_gain": round(gain, 4)}


def scene_look(scene: Any, lights: list[dict] | None = None) -> dict:
    """A camera look (``alloy.camera.camera_scene(model, look=...)``) holding every item's variant look: its
    materials as ``<item>:<key>`` and rules matching the item's shape labels (``<item>``, ``<item>/<part>``)."""
    mats: dict[str, dict] = {}
    assign: list[dict] = []
    for name, item in scene.items.items():
        look = item.get("look")
        if not look:
            continue
        esc = re.escape(name)
        for key, spec in look["materials"].items():
            mats[f"{name}:{key}"] = dict(spec)
        for rule in look.get("assign") or ():
            assign.append({"match": f"^{esc}/.*?(?:{rule['match']})", "material": f"{name}:{rule['material']}"})
        assign.append({"match": f"^{esc}($|[/#])", "material": f"{name}:{look['default']}"})
    out: dict[str, Any] = {"materials": mats, "assign": assign}
    if lights:
        out["lights"] = list(lights)
    return out


__all__ = ["MODES", "TEXTURE_FIELDS", "Texture", "effective", "family_of", "item_look", "scene_look", "texture_of",
           "texture_pool", "vary_look"]
