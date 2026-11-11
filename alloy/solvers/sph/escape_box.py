# SPDX-License-Identifier: Apache-2.0
"""The default escape box of a scene-built SolverDFSPH (SCR-OSS3-2, coordinator 2026-11-06).

A liquid scene built through the ``sph`` capability (``alloy.api``, the scene assembler) gets ``world_bounds`` from its
own content when the caller sets none: the box around the finite shapes (bounding spheres at their build pose, planes
left out) and the particles (spares too: a pour's spares sit at its nozzle), padded on every side by
``max(AUTO_PAD_MIN, largest extent)``. Liquid that leaves it (ejecta of a splash, particles sliding away on a ground
plane) retires: its mass goes to the ledger's ``out`` and the per-world count to ``SPHReport.escaped`` and the
``sph.escaped`` state array, never silently.

One box serves every world (``emitters.retire_kernel``), so it is the union over the worlds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

AUTO_PAD_MIN = 0.25
"""Smallest padding [m] around the content box."""

_PLANE_RADIUS = 1.0e5  # Newton reports a huge collision radius for planes (infinite)

Bounds = tuple[tuple[float, float, float], tuple[float, float, float]]


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (N, 3) by unit quaternions q (N, 4), xyzw."""
    u, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _shape_spheres(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """World-frame centres (N, 3) and bounding radii (N,) of the finite shapes at the build pose."""
    if not int(getattr(model, "shape_count", 0) or 0) or getattr(model, "shape_collision_radius", None) is None:
        return np.zeros((0, 3)), np.zeros(0)
    x = model.shape_transform.numpy().astype(np.float64).reshape(-1, 7)
    r = model.shape_collision_radius.numpy().astype(np.float64)
    body = model.shape_body.numpy()
    c = x[:, :3].copy()
    on_body = body >= 0
    if on_body.any() and getattr(model, "body_q", None) is not None:
        bq = model.body_q.numpy().astype(np.float64).reshape(-1, 7)[body[on_body]]
        c[on_body] = bq[:, :3] + _rotate(bq[:, 3:7], c[on_body])
    keep = np.isfinite(r) & (r < _PLANE_RADIUS) & np.isfinite(c).all(axis=1)
    return c[keep], r[keep]


def _points(model: Any) -> np.ndarray:
    """Every particle with a finite position: active ones and spares (a pour's spares sit at its nozzle)."""
    if not int(getattr(model, "particle_count", 0) or 0):
        return np.zeros((0, 3))
    q = model.particle_q.numpy().astype(np.float64).reshape(-1, 3)
    return q[np.isfinite(q).all(axis=1)]


def content_box(model: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Min and max corner of the finite shapes and the particles, over all worlds; None for an empty model."""
    c, r = _shape_spheres(model)
    p = _points(model)
    if len(c) == 0 and len(p) == 0:
        return None
    lo = np.vstack([c - r[:, None], p]).min(axis=0)
    hi = np.vstack([c + r[:, None], p]).max(axis=0)
    return lo, hi


def auto_world_bounds(model: Any, pad_min: float = AUTO_PAD_MIN) -> Bounds | None:
    """The content box padded by ``max(pad_min, largest extent)`` on every side; None for an empty model."""
    box = content_box(model)
    if box is None:
        return None
    lo, hi = box
    pad = max(float(pad_min), float((hi - lo).max()))
    return tuple(float(v) for v in lo - pad), tuple(float(v) for v in hi + pad)
