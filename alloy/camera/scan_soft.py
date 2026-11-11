# SPDX-License-Identifier: Apache-2.0
"""Photo textures on scanned objects that simulate as soft bodies (a soft fruit material: the YCB apple and banana).

A soft body is drawn from its particle surface (``deform.DeformObject``), not from a display mesh, so it has no UVs
of its own. :func:`soft_scan_uvs` gives each surface vertex the UV of the nearest vertex of the scan's glb, with the
glb placed at the object's build pose (the item's orientation; its position by the boxes' centres). The UVs are fixed
at the build pose: a soft body that deforms stretches its texture with it. Triangles of the particle surface that
straddle a seam of the texture's layout interpolate across the seam (a thin streak there); :func:`seam_fraction`
measures the share of such triangles.

Example::

    objs = soft_scan_uvs(sim, objs)      # surfaces of scanned soft items gain uv (V, 2) and image (H, W, 3)
"""
from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from .glb import GlbError, read_glb
from .scan_textures import decode_image, scan_sources, to_body

log = logging.getLogger(__name__)

FIT_REL, FIT_ABS = 0.05, 0.003  # box match of the particle surface against the glb: 5 % + 3 mm per axis
SEAM_UV = 0.25  # a triangle whose corner UVs span more than this (of the image) straddles a seam


def _rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _glb_texture(src) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(body-frame points (V, 3), uv (V, 2), image) of the scan's one textured primitive, or None."""
    try:
        parts = [p for p in read_glb(src.glb) if p.image is not None and p.uv is not None]
    except (GlbError, OSError, KeyError, IndexError, ValueError) as exc:
        log.warning("camera: %s is not used for the soft body (%s)", src.glb, exc)
        return None
    if len(parts) != 1:
        log.warning("camera: %s has %d textured parts; a soft body is drawn with one: flat colour", src.glb, len(parts))
        return None
    img = decode_image(parts[0].image.data)
    return None if img is None else (to_body(parts[0].points, src.scale), parts[0].uv, img[0])


def surface_uvs(rest: np.ndarray, tris: np.ndarray, rot: np.ndarray, glb_pts: np.ndarray,
                glb_uv: np.ndarray) -> np.ndarray | None:
    """UV per rest vertex: the nearest glb vertex's, the glb turned by ``rot`` and centred on the surface's box;
    None when the boxes' sizes differ (5 % + 3 mm per axis)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415 - a core dependency
    used = np.unique(tris)
    local = rest @ rot  # world -> item frame (rows: R^T x)
    lo, hi = local[used].min(0), local[used].max(0)
    glo, ghi = glb_pts.min(0), glb_pts.max(0)
    if (np.abs((hi - lo) - (ghi - glo)) > FIT_REL * np.maximum(hi - lo, ghi - glo) + FIT_ABS).any():
        return None
    shift = 0.5 * (glo + ghi) - 0.5 * (lo + hi)
    _, nearest = cKDTree(glb_pts).query(local + shift)
    return np.ascontiguousarray(glb_uv[nearest], np.float32)


def seam_fraction(uv: np.ndarray, tris: np.ndarray) -> float:
    """Share of triangles whose corner UVs span more than ``SEAM_UV`` of the image (they straddle a seam)."""
    span = np.ptp(uv[tris], axis=1).max(axis=1)
    return float((span > SEAM_UV).mean()) if len(tris) else 0.0


def soft_scan_uvs(sim, objects: list) -> list:
    """``objects`` (DeformObjects) with ``uv`` and ``image`` set on the surfaces of ``sim``'s scanned soft items."""
    from .item_looks import scene_of
    sources = {s.label.rsplit("/", 1)[0]: s for s in scan_sources(sim)}
    scene = scene_of(sim)
    out = []
    for o in objects:
        src = sources.get(o.name) if o.kind == "surface" else None
        tex = _glb_texture(src) if src is not None else None
        uv = None
        if tex is not None:
            pose = (scene.items.get(o.name) or {}).get("pose") or [0, 0, 0, 0, 0, 0, 1]
            uv = surface_uvs(o.rest, o.tris, _rotation(pose[3:7]), tex[0], tex[1])
            if uv is None:
                log.warning("camera: soft body %s does not match its scan %s (box sizes differ): flat colour",
                            o.name, src.glb)
        if uv is None:
            out.append(o)
            continue
        log.info("camera: soft body %s textured from %s (%.1f %% of its triangles straddle a seam)", o.name,
                 src.glb.name, 100.0 * seam_fraction(uv, o.tris))
        out.append(replace(o, uv=uv, image=tex[2], color=(1.0, 1.0, 1.0)))
    return out
