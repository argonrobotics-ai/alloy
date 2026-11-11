# SPDX-License-Identifier: Apache-2.0
"""World-space bounding boxes of placed objects in an assembled scene: every collision shape of the object (boxes,
spheres, capsules, cylinders, cones, ellipsoids, meshes) and its particles, in one world.

Used to rest an object on a support (``alloy.check.drop``) and to size that support.

Example::

    >>> lo, hi = object_aabb(built.model, state, index, "cube_beech_30")    # doctest: +SKIP
    >>> round(hi[2] - lo[2], 3)
    0.03
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alloy.check.motion import _qmul
from alloy.check.objects import ObjectIndex

log = logging.getLogger("alloy.check")

# newton.GeoType values (newton/_src/geometry/types.py)
SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH, CONE, CONVEX_MESH = 3, 4, 5, 6, 7, 8, 9, 10


def object_aabb(model: Any, state: Any, index: ObjectIndex, name: str, world: int = 0
                ) -> tuple[np.ndarray, np.ndarray] | None:
    """(lo, hi) world-space corners of ``name`` in ``world`` (shapes and particles), or None if it has neither.

    Example::

        >>> lo, hi = object_aabb(built.model, state, index, "cube_beech_30")  # doctest: +SKIP
        >>> hi.shape
        (3,)
    """
    k = index.names.index(name)
    los, his = [], []
    sw = model.shape_world.numpy()
    body_q = state.body_q.numpy() if model.body_count else np.zeros((0, 7))
    for s in np.flatnonzero((index.shape_obj == k) & (sw == world)):
        box = shape_aabb(model, body_q, int(s))
        if box is not None:
            los.append(box[0])
            his.append(box[1])
    parts = np.flatnonzero((index.particle_obj == k) & (model.particle_world.numpy() == world)) \
        if model.particle_count else np.zeros(0, np.int64)
    if parts.size:
        q, r = state.particle_q.numpy()[parts].astype(np.float64), model.particle_radius.numpy()[parts]
        los.append((q - r[:, None]).min(axis=0))
        his.append((q + r[:, None]).max(axis=0))
    if not los:
        return None
    return np.min(los, axis=0), np.max(his, axis=0)


def shape_aabb(model: Any, body_q: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray] | None:
    """World AABB of shape ``s`` (None for planes, height fields and shapes without a size).

    Example::

        >>> lo, hi = shape_aabb(built.model, state.body_q.numpy(), 0)  # doctest: +SKIP
        >>> lo.shape
        (3,)
    """
    xf = model.shape_transform.numpy()[s].astype(np.float64)
    b = int(model.shape_body.numpy()[s])
    p, q = xf[:3], xf[3:7]
    if b >= 0:
        bq = body_q[b].astype(np.float64)
        p, q = bq[:3] + _rot(bq[3:7]) @ p, _qmul(bq[3:7], q)
    rot = _rot(q)
    kind = int(model.shape_type.numpy()[s])
    scale = model.shape_scale.numpy()[s].astype(np.float64)
    if kind in (MESH, CONVEX_MESH):
        src = model.shape_source[s]
        verts = getattr(src, "vertices", None)
        if verts is not None and len(verts):
            pts = p + (np.asarray(verts, np.float64) * scale) @ rot.T
            return pts.min(axis=0), pts.max(axis=0)
    half = _half_extent(kind, scale, rot)
    if half is None:
        r = float(model.shape_collision_radius.numpy()[s]) if hasattr(model, "shape_collision_radius") else 0.0
        if r <= 0.0:
            return None
        half = np.full(3, r)
    return p - half, p + half


def _half_extent(kind: int, scale: np.ndarray, rot: np.ndarray) -> np.ndarray | None:
    """Half extents along world x, y, z of a primitive with Newton's size convention (axis z for capsule, cylinder,
    cone: scale = (radius, half_height))."""
    if kind == BOX:
        return np.abs(rot) @ scale[:3]
    if kind == SPHERE:
        return np.full(3, scale[0])
    if kind == ELLIPSOID:
        return np.sqrt((rot * scale[:3]) ** 2 @ np.ones(3))
    if kind in (CAPSULE, CYLINDER, CONE):
        r, hh = scale[0], scale[1]
        axis = np.abs(rot[:, 2])
        disc = r * np.sqrt(np.clip(1.0 - axis ** 2, 0.0, 1.0))
        return axis * hh + (np.full(3, r) if kind == CAPSULE else disc)
    return None


def _rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


__all__ = ["object_aabb", "shape_aabb"]
