# SPDX-License-Identifier: Apache-2.0
"""Checks on recorded frames of a cloth or rigid-panel take: the laundry tasks and the fold clips use them.

Shipped in the wheel (the release leaves ``alloy.clips`` out); ``alloy.clips.fold_checks`` is this module.

All functions are pure numpy on host arrays of one world:
- ``table_penetration``: depth of particles below the plywood top (z = 0) by more than their radius;
- ``hull_penetration``: depth of particles inside the robot fingertip collision hulls (halfspace form, exact inside);
- ``self_intersections``: cloth edges that cross a non-adjacent triangle of the same cloth (segment-triangle tests);
- ``take_compare``: per-particle position difference of two takes of the same world (the dt/2 check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def table_penetration(q: np.ndarray, radius: float, z_top: float = 0.0) -> float:
    """Max over particles of (radius - (z - z_top)), clipped at 0 [m]; q [N, 3]."""
    return float(np.clip(radius - (q[:, 2] - z_top), 0.0, None).max())


@dataclass
class Hull:
    """A convex hull in its body frame as halfspaces n.x <= d (n unit)."""

    normals: np.ndarray        # [F, 3]
    offsets: np.ndarray        # [F]
    centre: np.ndarray         # [3]
    radius: float              # bounding sphere


def hull_from_points(pts: np.ndarray) -> Hull:
    from scipy.spatial import ConvexHull

    h = ConvexHull(pts)
    n, d = h.equations[:, :3], -h.equations[:, 3]
    c = pts.mean(axis=0)
    return Hull(n, d, c, float(np.linalg.norm(pts - c, axis=1).max()))


def quat_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hull_penetration(q: np.ndarray, radius: float, hull: Hull, body_pose: np.ndarray) -> float:
    """Max depth [m] of particle spheres (q [N, 3], world) inside ``hull`` at ``body_pose`` (p xyz, q xyzw)."""
    R, p = quat_rot(body_pose[3:]), body_pose[:3]
    local = (q - p) @ R
    near = np.linalg.norm(local - hull.centre, axis=1) < hull.radius + radius
    if not near.any():
        return 0.0
    sd = (local[near] @ hull.normals.T - hull.offsets).max(axis=1)      # > 0 outside (lower bound), < 0 inside
    return float(np.clip(radius - sd, 0.0, None).max())


def _edges(tri: np.ndarray) -> np.ndarray:
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def self_intersections(q: np.ndarray, tri: np.ndarray, cell: float | None = None, chunk: int = 20000) -> int:
    """Number of (edge, triangle) crossings between non-adjacent elements of one cloth (q [N, 3], tri [T, 3]).

    Triangles are binned by their min corner on a grid of ``cell`` [m] (default: the largest element extent this
    frame, so every pair with overlapping boxes is in one of an edge's 27 neighbour bins); the candidate pairs are
    built and tested as arrays, ``chunk`` edges at a time (fine towel grids have 20-90 k edges)."""
    edges = _edges(tri)
    a, b = q[edges[:, 0]], q[edges[:, 1]]
    t0, t1, t2 = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
    tmin = np.minimum(np.minimum(t0, t1), t2)
    tmax = np.maximum(np.maximum(t0, t1), t2)
    emin, emax = np.minimum(a, b), np.maximum(a, b)
    if cell is None:
        cell = 1.0001 * max(float((tmax - tmin).max()), float((emax - emin).max())) + 1e-9
    kt, ke = np.floor(tmin / cell).astype(np.int64), np.floor(emin / cell).astype(np.int64)
    lo = np.minimum(kt.min(axis=0), ke.min(axis=0)) - 1
    kt, ke = kt - lo, ke - lo
    dims = np.maximum(kt.max(axis=0), ke.max(axis=0)) + 2

    def code(k: np.ndarray) -> np.ndarray:
        return (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]

    order = np.argsort(code(kt), kind="stable")
    sorted_keys = code(kt)[order]
    offs = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    count = 0
    for s0 in range(0, len(edges), chunk):
        e_ids = np.arange(s0, min(s0 + chunk, len(edges)))
        for o in offs:
            c = code(ke[e_ids] + o)
            first, last = np.searchsorted(sorted_keys, c, "left"), np.searchsorted(sorted_keys, c, "right")
            n = last - first
            if not n.any():
                continue
            ei = np.repeat(e_ids, n)
            ti = order[np.repeat(first, n) + np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)]
            ok = np.all((tmin[ti] <= emax[ei]) & (tmax[ti] >= emin[ei]), axis=1)
            ei, ti = ei[ok], ti[ok]
            near = (tri[ti] == edges[ei, :1]).any(axis=1) | (tri[ti] == edges[ei, 1:]).any(axis=1)
            ei, ti = ei[~near], ti[~near]                                # skip triangles sharing a vertex
            if len(ei):
                count += int(_seg_tri(a[ei], b[ei], t0[ti], t1[ti], t2[ti]).sum())
    return count


def _seg_tri(p: np.ndarray, q: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Moller-Trumbore segment p->q (one, or [K] paired) against triangles [K]; True where the open segment crosses
    the triangle."""
    d = np.broadcast_to(q - p, v0.shape)
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p - v0
    u = inv * np.einsum("ij,ij->i", s, h)
    qv = np.cross(s, e1)
    v = inv * np.einsum("ij,ij->i", qv, d)
    t = inv * np.einsum("ij,ij->i", e2, qv)
    eps = 1e-9
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def take_compare(qa: np.ndarray, qb: np.ndarray) -> dict[str, float]:
    """Position difference of two takes [F, N, 3] of the same world: max and p99 over particles, final frame and
    worst frame [m]."""
    n = min(len(qa), len(qb))
    d = np.linalg.norm(qa[:n] - qb[:n], axis=2)
    return {"final_max_m": float(d[-1].max()), "final_p99_m": float(np.percentile(d[-1], 99)),
            "worst_max_m": float(d.max()), "worst_frame": int(d.max(axis=1).argmax())}
