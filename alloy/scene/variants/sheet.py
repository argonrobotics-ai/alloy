# SPDX-License-Identifier: Apache-2.0
"""A contact sheet of the variants: one low-sample image each, from one fixed three-quarter view of the base scene,
tiled in a grid (row-major, variant 0 first) so a person can compare them at a glance.

The images come from the base assembly with each variant's moved start state, so no variant is built.
``quality="full"`` renders the variant looks (textures, tints) with ``alloy.camera``'s full-material tracer; a look it
cannot draw (a texture missing from the local asset cache, a ``tex_tint`` fabric) falls back to the fast diffuse
path for that tile, which shows shape colours only. ``quality="fast"`` uses the fast path throughout.

The base assembly holds the shapes of rigid objects only: items stepped as particles (cloth, film, soft bodies, rods on
``rod_vbd``, MPM materials, liquids), which a sim draws from its state, are not on the tiles, and :func:`not_drawn`
names each of them in the sheet's ``left_out`` (before, they were missing while ``left_out`` was empty).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("alloy.scene")

SUN = {"type": "sun", "eye": [1.0, -1.5, 2.5], "target": [0.0, 0.0, 0.0], "strength": 3.0}
GAP_PX = 4
FOV_DEG = 40.0
PARTICLE_ROUTES = ("rod_vbd", "cloth_vbd", "cloth_style3d", "cloth_film", "soft_vbd", "grain_mpm", "paste_mpm",
                   "liquid_sph", "liquid_mpm")  # drawn from a sim's state, not from the base assembly's shapes


def view(base: Any, state: Any, names: list[str] | None = None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(eye, target): a three-quarter view framing ``names`` (the objects that move; default: every object but the
    ground)."""
    from alloy.check.bounds import object_aabb  # noqa: PLC0415

    boxes = [object_aabb(base.built.model, state, base.index, n) for n in (names or base.index.names)]
    boxes = [b for b in boxes if b is not None and np.all(np.isfinite(b[0])) and np.all(np.isfinite(b[1]))
             and float(np.max(b[1] - b[0])) < 20.0]                  # a ground plane spans 50 m: not framed
    if not boxes:
        return (1.0, -1.0, 1.0), (0.0, 0.0, 0.0)
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    center, radius = 0.5 * (lo + hi), max(0.5 * float(np.linalg.norm(hi - lo)), 0.05)
    direction = np.array([1.0, -1.2, 0.9]) / np.linalg.norm([1.0, -1.2, 0.9])
    eye = center + direction * radius / math.sin(math.radians(0.5 * FOV_DEG)) * 1.1
    return tuple(float(v) for v in eye), tuple(float(v) for v in center)


def not_drawn(base: Any) -> list[str]:
    """The items no tile draws (module docstring), one ``"<name>: <route> (...)"`` line each, logged once."""
    decisions = getattr(getattr(base.built, "report", None), "decisions", None) or {}
    out = [f"{name}: {d.route} (not drawn on the contact sheet; render the variant file to see it)"
           for name, d in decisions.items() if str(d.route) in PARTICLE_ROUTES]
    if out:
        log.warning("variants contact sheet: %d item(s) stepped as particles are not drawn: %s", len(out),
                    ", ".join(o.split(":")[0] for o in out))
    return out


def drawable(look: dict, res: str = "2k") -> tuple[dict, list[str]]:
    """(``look`` the full-material camera can draw, what was left out): a texture missing from the local asset cache
    is dropped (the material keeps its preset colour) and features the camera does not mirror (``tex_tint``,
    procedural crust / crumb / scratches, decals, imperfection maps, planar mapping) are turned off. Only the contact
    sheet uses this; the variant scenes keep their looks."""
    from alloy.camera.errors import CameraError  # noqa: PLC0415
    from alloy.camera.looks.spec import UNSUPPORTED_DETAILS, resolve_spec  # noqa: PLC0415
    from alloy.camera.looks.textures import local_texture_files  # noqa: PLC0415

    out, dropped, have = {**look, "materials": {}}, [], {}
    for key, entry in look["materials"].items():
        spec, eff = dict(entry), resolve_spec(key, entry)
        ref = eff.get("texture")
        if ref and ref not in have:
            try:
                have[ref] = bool(local_texture_files(ref, str(eff.get("res", res))))
            except (CameraError, FileNotFoundError, OSError):
                have[ref] = False
        if ref and not have[ref]:
            spec["texture"] = None
            dropped.append(f"{key}: {ref} (not in the local asset cache)")
        for field, off, on in (("tex_tint", False, bool(eff.get("tex_tint"))),
                               ("detail", None, eff.get("detail") in UNSUPPORTED_DETAILS),
                               ("texture_alpha", False, bool(eff.get("texture_alpha"))),
                               ("imperfection", None, bool(eff.get("imperfection"))),
                               ("mapping", "triplanar", eff.get("mapping") == "planar")):
            if on:
                spec[field] = off
                dropped.append(f"{key}: {field}")
        out["materials"][key] = spec
    return out, dropped


def render_tile(base: Any, state: Any, look: dict | None, look_at: tuple, size: tuple[int, int], spp: int) -> tuple[
        np.ndarray, str]:
    """(image, path used: "full" or "fast") of ``state`` with ``look`` (None: fast)."""
    from alloy.camera.errors import CameraError  # noqa: PLC0415
    from alloy.camera.still import snapshot  # noqa: PLC0415

    kw = {"look_at": look_at, "width": size[0], "height": size[1], "spp": spp, "fov": FOV_DEG}
    if look is not None:
        try:
            return snapshot(base.built.model, state, look=look, **kw), "full"
        except (CameraError, FileNotFoundError) as exc:
            log.warning("variants contact sheet: full-material tile failed (%s); fast path for this tile", exc)
    return snapshot(base.built.model, state, quality="fast", **kw), "fast"


def tile(images: list[np.ndarray]) -> np.ndarray:
    """Images of one size in a near-square grid with white gaps."""
    h, w = images[0].shape[:2]
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    out = np.full((rows * h + (rows + 1) * GAP_PX, cols * w + (cols + 1) * GAP_PX, 3), 255, np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        y, x = GAP_PX + r * (h + GAP_PX), GAP_PX + c * (w + GAP_PX)
        out[y:y + h, x:x + w] = img[..., :3]
    return out


def write_sheet(path: Path, images: list[np.ndarray]) -> Path:
    from alloy.camera.view import write_png  # noqa: PLC0415

    return write_png(path, tile(images))


__all__ = ["PARTICLE_ROUTES", "SUN", "drawable", "not_drawn", "render_tile", "tile", "view", "write_sheet"]
