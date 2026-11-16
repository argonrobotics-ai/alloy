# SPDX-License-Identifier: Apache-2.0
"""A look's ``world`` section as the full-material camera draws it, key by key as the Blender renderer reads it
(docs/RENDER.md section 3): the HDRI (or the studio gradient when it names none) at ``strength``, turned
``rotation_deg`` about +z, and what camera rays see: the HDRI itself with ``show_hdri``, else the flat
``background_color`` (sRGB), or with ``background_from_hdri`` the environment's mean radiance x strength. Missing
keys take the look schema's defaults (``alloy/render/blender/look.py`` WORLD_DEFAULTS, a module that imports
Blender-side code). A key the camera cannot draw raises a CameraError naming the keys it reads, so none is dropped.

Example::

    world = look_world({"world": {"hdri": "polyhaven:studio_small_09", "rotation_deg": 150, "show_hdri": True}})
"""
from __future__ import annotations

import math

import numpy as np

from .errors import CameraError

WORLD_DEFAULTS: dict = {"hdri": None, "strength": 1.0, "rotation_deg": 0.0, "show_hdri": False,
                        "background_color": [0.02, 0.02, 0.02], "background_from_hdri": False}
WORLD_KEYS = (*WORLD_DEFAULTS, "res")       # res: the HDRI's size in the asset cache ("2k", "4k"; default "4k")
STUDIO_MEAN = (0.3, 0.3, 0.3)               # the studio gradient's mean, as the Blender build takes it


def world_section(look: dict) -> dict:
    """The look's world keys over the schema's defaults; raises CameraError for a key the camera cannot draw or a
    value of the wrong kind."""
    raw = look.get("world") or {}
    if not isinstance(raw, dict):
        raise CameraError(f"look world is {raw!r}", "give world as a mapping of " + ", ".join(WORLD_KEYS))
    for key in raw:
        if key not in WORLD_KEYS:
            what = ("the full-material camera does not draw world.fade (it needs the Blender renderer)"
                    if key == "fade" else f"look world key {key!r} is not one the full-material camera reads")
            raise CameraError(what, "use only the world keys " + ", ".join(WORLD_KEYS))
    cfg = {**WORLD_DEFAULTS, **raw}
    for key in ("strength", "rotation_deg"):
        cfg[key] = _number(cfg[key], f"world.{key}")
    color = cfg["background_color"]
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise CameraError(f"look world.background_color is {color!r}", "give 3 sRGB numbers from 0 to 1")
    cfg["background_color"] = [_number(c, "world.background_color") for c in color]
    for key in ("show_hdri", "background_from_hdri"):
        if not isinstance(cfg[key], bool):
            raise CameraError(f"look world.{key} is {cfg[key]!r}", "use true or false")
    return cfg


def _number(value, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CameraError(f"look {where} is {value!r}", "give a number") from exc


def z_turn(degrees: float) -> np.ndarray:
    """World direction -> HDRI direction for a turn of ``degrees`` about +z (Blender's world Mapping node)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def equirect_mean(image: np.ndarray) -> np.ndarray:
    """Solid-angle-weighted mean linear RGB of an equirect image (H, W, 3), as the Blender build measures an HDRI."""
    from alloy.render.blender.motion import equirect_mean as mean
    img = np.asarray(image, np.float32)[..., :3]
    h, w, _ = img.shape
    rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=2)
    return mean(rgba.ravel(), h, w)


def studio_sky(height: int = 64, width: int = 128) -> np.ndarray:
    """Alloy's studio background as an equirect image (rows top first): dark floor, grey horizon, bright top
    (colour ramp 0.35 -> 0.03, 0.55 -> 0.25, 1.0 -> 1.0 over 0.5 z + 0.5), as the Blender build's default world."""
    z = np.cos(np.pi * (np.arange(height) + 0.5) / height)  # row 0 = straight up
    x = 0.5 * z + 0.5
    v = np.interp(x, [0.35, 0.55, 1.0], [0.03, 0.25, 1.0])
    return np.repeat(np.repeat(v[:, None, None], width, axis=1), 3, axis=2).astype(np.float32)


def look_world(look: dict) -> dict:
    """The pack world of a look: ``image`` (rows top first), ``strength``, ``use_mis``, ``rotation`` (3x3, world ->
    image direction) and, unless ``show_hdri``, ``camera_color`` (linear RGB camera rays see)."""
    from alloy.render.blender.motion import srgb_to_linear
    cfg = world_section(look)
    if cfg["hdri"]:
        from alloy.render.blender.assets import AssetCache
        from .pbr.lights import load_hdr
        path = AssetCache().hdri(cfg["hdri"], cfg.get("res", "4k"))
        if path is None:
            raise CameraError(f"HDRI {cfg['hdri']!r} is not available", "check the reference or the network")
        image = load_hdr(str(path))
        mean = equirect_mean(image)
    else:
        image, mean = studio_sky(), np.asarray(STUDIO_MEAN)
    out = {"image": image, "strength": cfg["strength"], "use_mis": True,
           "rotation": z_turn(cfg["rotation_deg"]).tolist()}
    if not cfg["show_hdri"]:
        color = (np.asarray(mean, np.float64)[:3] * cfg["strength"] if cfg["background_from_hdri"]
                 else srgb_to_linear(cfg["background_color"]))
        out["camera_color"] = [float(c) for c in color]
    return out
