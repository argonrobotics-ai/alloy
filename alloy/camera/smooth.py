# SPDX-License-Identifier: Apache-2.0
"""Corner normals with auto-smooth: a corner takes the area-weighted mean of the face normals around its vertex
whose angle to its own face is below a threshold (Blender's "smooth by angle", 30 degrees by default).

Example::

    import numpy as np
    from alloy.camera.smooth import corner_normals
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    n = corner_normals(v, np.array([[0, 1, 2]]))          # (1, 3, 3): every corner (0, 0, 1)
"""
from __future__ import annotations

import math

import numpy as np


def _weld(v: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Vertex id per vertex, merging vertices closer than ``tol`` (primitives repeat seam vertices)."""
    key = np.round(np.asarray(v, np.float64) / tol).astype(np.int64)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    return ids.reshape(-1)


def corner_normals(v: np.ndarray, tri: np.ndarray, max_angle_deg: float = 30.0) -> np.ndarray:
    """(T, 3, 3) float32 unit normals per triangle corner (flat where the angle to a neighbour exceeds the limit)."""
    v = np.asarray(v, np.float64)
    tri = np.asarray(tri, np.int64).reshape(-1, 3)
    if len(tri) == 0:
        return np.zeros((0, 3, 3), np.float32)
    p = v[tri]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # area weighted
    unit = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    vid = _weld(v)[tri]  # (T, 3) welded vertex of each corner
    cos_lim = math.cos(math.radians(max_angle_deg))
    out = np.repeat(unit[:, None, :], 3, axis=1)
    corners = np.argsort(vid.reshape(-1), kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(vid.reshape(-1)[corners]) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(corners)]):
        group = corners[a:b] // 3  # faces around this vertex (one entry per corner)
        if len(group) < 2:
            continue
        near = unit[group] @ unit[group].T >= cos_lim  # (k, k): which faces smooth with which
        near[~unit[group].any(axis=1)] = True  # a zero-area face (sphere poles) takes the vertex's mean normal
        acc = near.astype(np.float64) @ fn[group]
        acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)
        out[group, corners[a:b] % 3] = acc
    return out.astype(np.float32)
