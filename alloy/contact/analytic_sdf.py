# SPDX-License-Identifier: Apache-2.0
"""Analytic signed distance functions, independent of Newton's narrow phase (GapMonitor reference, §3.5).

Distances are in the shape frame, metres, positive outside. Round shapes have their axis on +Z (Newton
convention). ``scale`` follows ``Model.shape_scale``: sphere (r, -, -), box (hx, hy, hz), capsule and cylinder
(r, half_height, -), plane (width, length, -) with the surface z = 0. Each Warp function has a numpy twin used by
tests and host-side checks. Shapes without an exact SDF here (ellipsoid, cone, crowned cylinder, meshes) are
measured with the winding-number path in ``monitor.py``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy.core.errors import PrecisionSpecError

# Newton GeoType codes (newton@ac7f7830: newton.GeoType); asserted in tests/contact/test_analytic_sdf.py
GEO_PLANE = 1
GEO_SPHERE = 3
GEO_CAPSULE = 4
GEO_CYLINDER = 6
GEO_BOX = 7
ANALYTIC_TYPES = frozenset({GEO_PLANE, GEO_SPHERE, GEO_CAPSULE, GEO_CYLINDER, GEO_BOX})


@wp.func
def sd_sphere(p: wp.vec3, r: float) -> float:
    return wp.length(p) - r


@wp.func
def sd_box(p: wp.vec3, h: wp.vec3) -> float:
    q = wp.vec3(wp.abs(p[0]) - h[0], wp.abs(p[1]) - h[1], wp.abs(p[2]) - h[2])
    outside = wp.length(wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0)))
    return outside + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)


@wp.func
def sd_capsule(p: wp.vec3, r: float, hh: float) -> float:
    z = wp.clamp(p[2], -hh, hh)
    return wp.length(wp.vec3(p[0], p[1], p[2] - z)) - r


@wp.func
def sd_cylinder(p: wp.vec3, r: float, hh: float) -> float:
    dx = wp.sqrt(p[0] * p[0] + p[1] * p[1]) - r
    dz = wp.abs(p[2]) - hh
    outside = wp.sqrt(wp.max(dx, 0.0) * wp.max(dx, 0.0) + wp.max(dz, 0.0) * wp.max(dz, 0.0))
    return outside + wp.min(wp.max(dx, dz), 0.0)


@wp.func
def sd_primitive(geo_type: int, scale: wp.vec3, p: wp.vec3) -> float:
    """Signed distance to a Newton primitive; returns +inf for types without an exact SDF here."""
    if geo_type == GEO_SPHERE:
        return sd_sphere(p, scale[0])
    if geo_type == GEO_BOX:
        return sd_box(p, scale)
    if geo_type == GEO_CAPSULE:
        return sd_capsule(p, scale[0], scale[1])
    if geo_type == GEO_CYLINDER:
        return sd_cylinder(p, scale[0], scale[1])
    if geo_type == GEO_PLANE:
        return p[2]
    return wp.inf


# ---------------------------------------------------------------- numpy twins (float64) ---------------------------

def sd_sphere_np(p: np.ndarray, r: float) -> np.ndarray:
    return np.linalg.norm(p, axis=-1) - r


def sd_box_np(p: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.abs(p) - np.asarray(h, float)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def sd_capsule_np(p: np.ndarray, r: float, hh: float) -> np.ndarray:
    z = np.clip(p[..., 2], -hh, hh)
    d = p.copy()
    d[..., 2] -= z
    return np.linalg.norm(d, axis=-1) - r


def sd_cylinder_np(p: np.ndarray, r: float, hh: float) -> np.ndarray:
    dx = np.hypot(p[..., 0], p[..., 1]) - r
    dz = np.abs(p[..., 2]) - hh
    return np.hypot(np.maximum(dx, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(dx, dz), 0.0)


def sd_primitive_np(geo_type: int, scale, p: np.ndarray) -> np.ndarray:
    """numpy twin of ``sd_primitive``; raises ``PrecisionSpecError`` for types without an exact SDF."""
    p = np.asarray(p, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    table = {
        GEO_SPHERE: lambda: sd_sphere_np(p, s[0]),
        GEO_BOX: lambda: sd_box_np(p, s),
        GEO_CAPSULE: lambda: sd_capsule_np(p, s[0], s[1]),
        GEO_CYLINDER: lambda: sd_cylinder_np(p, s[0], s[1]),
        GEO_PLANE: lambda: p[..., 2].copy(),
    }
    if geo_type not in table:
        raise PrecisionSpecError(f"no analytic SDF for Newton GeoType {geo_type}; use the winding-number path")
    return table[geo_type]()


def has_exact_sdf(geo_type: int, scale) -> bool:
    """True if ``sd_primitive`` is exact for this shape (a crowned cylinder, scale[2] > 0, is not)."""
    if geo_type not in ANALYTIC_TYPES:
        return False
    return not (geo_type == GEO_CYLINDER and float(scale[2]) > 0.0)
