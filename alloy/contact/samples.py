# SPDX-License-Identifier: Apache-2.0
"""Surface samples for GapMonitor (host numpy, deterministic): points on a shape's real surface in its shape frame.

Primitives are sampled analytically (no tessellation error); meshes on their triangles. Spacing is approximate
(``spacing`` between neighbours); every vertex of a mesh is included so sharp features are never missed. An
optional crop box (shape frame) keeps only the precision feature.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.contact.analytic_sdf import GEO_BOX, GEO_CAPSULE, GEO_CYLINDER, GEO_SPHERE
from alloy.core.errors import PrecisionSpecError

MAX_SAMPLES_PER_SHAPE = 200_000


def _n(length: float, spacing: float) -> int:
    return max(1, int(math.ceil(length / spacing)))


def _sphere(r: float, spacing: float) -> np.ndarray:
    """Fibonacci sphere with ~spacing between points."""
    n = max(8, int(math.ceil(4.0 * math.pi * r * r / (spacing * spacing))))
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5**0.5) * i
    return r * np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def _grid(u_len: float, v_len: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    nu, nv = _n(u_len, spacing) + 1, _n(v_len, spacing) + 1
    u, v = np.meshgrid(np.linspace(-u_len / 2, u_len / 2, nu), np.linspace(-v_len / 2, v_len / 2, nv), indexing="ij")
    return u.ravel(), v.ravel()


def _box(h: np.ndarray, spacing: float) -> np.ndarray:
    pts = []
    for ax in range(3):
        a, b = [k for k in range(3) if k != ax]
        u, v = _grid(2 * h[a], 2 * h[b], spacing)
        for sign in (-1.0, 1.0):
            p = np.zeros((u.size, 3))
            p[:, ax], p[:, a], p[:, b] = sign * h[ax], u, v
            pts.append(p)
    return np.unique(np.concatenate(pts).round(12), axis=0)


def _ring_points(r: float, z: np.ndarray, spacing: float) -> np.ndarray:
    m = max(8, _n(2 * math.pi * r, spacing))
    th = np.arange(m) * 2 * math.pi / m
    zz, tt = np.meshgrid(z, th, indexing="ij")
    return np.stack([r * np.cos(tt).ravel(), r * np.sin(tt).ravel(), zz.ravel()], axis=1)


def _disk(r: float, z: float, spacing: float) -> np.ndarray:
    pts = [np.array([[0.0, 0.0, z]])]
    for k in range(1, _n(r, spacing) + 1):
        pts.append(_ring_points(r * k / _n(r, spacing), np.array([z]), spacing))
    return np.concatenate(pts)


def _cylinder(r: float, hh: float, spacing: float) -> np.ndarray:
    z = np.linspace(-hh, hh, _n(2 * hh, spacing) + 1)
    return np.concatenate([_ring_points(r, z, spacing), _disk(r, -hh, spacing), _disk(r, hh, spacing)])


def _capsule(r: float, hh: float, spacing: float) -> np.ndarray:
    z = np.linspace(-hh, hh, _n(2 * hh, spacing) + 1)
    s = _sphere(r, spacing)
    caps = np.concatenate([s[s[:, 2] >= 0] + [0, 0, hh], s[s[:, 2] < 0] - [0, 0, hh]])
    return np.concatenate([_ring_points(r, z, spacing), caps])


_R2 = (0.7548776662466927, 0.5698402909980532)       # R2 low-discrepancy sequence (plastic number)


def _edge_points(verts: np.ndarray, tris: np.ndarray, spacing: float) -> np.ndarray:
    e = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    a, b = verts[e[:, 0]], verts[e[:, 1]]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / spacing)).astype(int)
    k = np.repeat(np.arange(len(e)), n - 1)
    t = np.concatenate([np.arange(1, m) / m for m in n]) if (n > 1).any() else np.zeros(0)
    return a[k] + (b[k] - a[k]) * t[:, None]


def _area_points(verts: np.ndarray, tris: np.ndarray, spacing: float) -> np.ndarray:
    """~area/spacing² interior points per triangle (R2 sequence, square-root warp to the triangle)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.floor(area / (spacing * spacing)).astype(int)
    if n.sum() == 0:
        return np.zeros((0, 3))
    k = np.repeat(np.arange(len(tris)), n)
    j = np.concatenate([np.arange(m) for m in n]) + 0.5
    u, v = np.mod(j * _R2[0], 1.0), np.mod(j * _R2[1], 1.0)
    su = np.sqrt(u)
    w0, w1, w2 = 1.0 - su, su * (1.0 - v), su * v
    return a[k] * w0[:, None] + b[k] * w1[:, None] + c[k] * w2[:, None]


def _triangles(verts: np.ndarray, tris: np.ndarray, spacing: float) -> np.ndarray:
    """Vertices, edge points at ``spacing`` and area-proportional interior points (slivers stay cheap)."""
    pts = np.concatenate([verts, _edge_points(verts, tris, spacing), _area_points(verts, tris, spacing)])
    return np.unique(pts.round(12), axis=0)


def surface_samples(geo_type: int, scale, source, spacing: float, crop: tuple | None = None) -> np.ndarray:
    """[N, 3] float64 points on the shape surface (shape frame)."""
    s = np.asarray(scale, dtype=np.float64)
    if spacing <= 0.0:
        raise ValueError(f"sample spacing must be positive, got {spacing}")
    if geo_type == GEO_SPHERE:
        pts = _sphere(s[0], spacing)
    elif geo_type == GEO_BOX:
        pts = _box(s, spacing)
    elif geo_type == GEO_CYLINDER:
        pts = _cylinder(s[0], s[1], spacing)
    elif geo_type == GEO_CAPSULE:
        pts = _capsule(s[0], s[1], spacing)
    elif source is not None and hasattr(source, "vertices"):
        v = np.asarray(source.vertices, dtype=np.float64) * s
        t = np.asarray(source.indices, dtype=np.int64).reshape(-1, 3)
        pts = _triangles(v, t, spacing)
    else:
        raise PrecisionSpecError(f"cannot sample shape of GeoType {geo_type} (no mesh source)")
    if crop is not None:
        lo, hi = np.asarray(crop[0], float), np.asarray(crop[1], float)
        pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if len(pts) > MAX_SAMPLES_PER_SHAPE:
        raise PrecisionSpecError(f"{len(pts)} GapMonitor samples > {MAX_SAMPLES_PER_SHAPE}; raise sample_spacing or "
                                 f"crop the feature")
    return pts


def surface_area_estimate(geo_type: int, scale, source) -> float:
    """Area [m^2] used to pick the smaller side of a pair (the sampled side)."""
    s = np.asarray(scale, dtype=np.float64)
    if geo_type == GEO_SPHERE:
        return 4 * math.pi * s[0] ** 2
    if geo_type == GEO_BOX:
        return 8 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
    if geo_type in (GEO_CYLINDER, GEO_CAPSULE):
        return 2 * math.pi * s[0] * 2 * s[1] + 2 * math.pi * s[0] ** 2
    if source is not None and hasattr(source, "vertices"):
        v = np.asarray(source.vertices, dtype=np.float64) * s
        t = np.asarray(source.indices, dtype=np.int64).reshape(-1, 3)
        return 0.5 * float(np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1).sum())
    return math.inf
