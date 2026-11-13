# SPDX-License-Identifier: Apache-2.0
"""The default escape box of a scene-built SolverDFSPH (SCR-OSS3-2, coordinator 2026-11-06).

A liquid scene built through the ``sph`` capability (``alloy.api``, the scene assembler) gets ``world_bounds`` from its
own content when the caller sets none: the box around the finite shapes (bounding spheres at their build pose, planes
left out) and the particles (spares too: a pour's spares sit at its nozzle), padded on every side by
``max(AUTO_PAD_MIN, largest extent)``. Liquid that leaves it (ejecta of a splash, particles sliding away on a ground
plane) retires: its mass goes to the ledger's ``out`` and the per-world count to ``SPHReport.escaped`` and the
``sph.escaped`` state array, never silently.

One box serves every world (``emitters.retire_kernel``), so it is the union over the worlds.

A kinematic body that ``sim.drive`` moves takes the box with it (``widen``, ``drive_reach``): the box grows to hold
every place the drive can take the body's shapes, padded as at the build. A build-time box alone retired the liquid
a driven container carried past it: a sealed box driven 0.75 m at 5 m/s lost 700 of 700 particles, frozen in mid-air
where they crossed the box, while its 20 mm walls held them. A box the caller set is kept as set.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.sph")

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


def auto_box(model: Any, pad_min: float = AUTO_PAD_MIN) -> tuple[Bounds, float] | None:
    """(The content box padded by ``max(pad_min, largest extent)`` on every side, that pad); None for an empty
    model."""
    box = content_box(model)
    if box is None:
        return None
    lo, hi = box
    pad = max(float(pad_min), float((hi - lo).max()))
    return (tuple(float(v) for v in lo - pad), tuple(float(v) for v in hi + pad)), pad


def auto_world_bounds(model: Any, pad_min: float = AUTO_PAD_MIN) -> Bounds | None:
    """The content box padded by ``max(pad_min, largest extent)`` on every side; None for an empty model."""
    box = auto_box(model, pad_min)
    return None if box is None else box[0]


def drive_reach(centres: np.ndarray, radii: np.ndarray, pivot: np.ndarray, v: np.ndarray, w: np.ndarray,
                duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Min and max corner of every place spheres (``centres`` (N, 3), ``radii`` (N,)) reach while a constant twist
    moves them for ``duration`` s (inf: without end): velocity ``v`` and angular velocity ``w`` (N, 3) about ``pivot``
    (N, 3), which moves with ``v``. A turning sphere stays within its distance from the pivot, so it is bounded by
    the pivot's path widened by that distance; a sphere that does not turn, by its own path. A drive without end
    reaches infinity along its velocity. Example: ``drive_reach(c, r, c, v, 0 * v, 0.15)``."""
    with np.errstate(invalid="ignore"):
        move = np.where(v == 0.0, 0.0, v * float(duration))          # 0 x inf is no move
    turning = np.linalg.norm(w, axis=1) > 0.0
    start = np.where(turning[:, None], pivot, centres)
    reach = np.where(turning, np.linalg.norm(centres - pivot, axis=1), 0.0) + radii
    lo = (np.minimum(start, start + move) - reach[:, None]).min(axis=0)
    hi = (np.maximum(start, start + move) + reach[:, None]).max(axis=0)
    return lo, hi


def widen(solver: Any, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Grow ``solver``'s automatic escape box (the scene factory's, which sets ``solver.escape_box_pad``) to hold
    the box ``lo``-``hi`` padded by the build's pad; a box the caller set is kept. True when the box grew (a captured
    graph then holds the old one). Example: ``widen(solver, *drive_reach(...))``."""
    pad, b = getattr(solver, "escape_box_pad", None), solver.config.world_bounds
    if pad is None or b is None:
        return False
    old_lo, old_hi = np.asarray(b[0], np.float64), np.asarray(b[1], np.float64)
    new_lo, new_hi = np.minimum(old_lo, np.asarray(lo) - pad), np.maximum(old_hi, np.asarray(hi) + pad)
    if np.array_equal(new_lo, old_lo) and np.array_equal(new_hi, old_hi):
        return False
    bounds = tuple(float(x) for x in new_lo), tuple(float(x) for x in new_hi)
    solver.config = dataclasses.replace(solver.config, world_bounds=bounds)
    solver.emitters.bounds = (wp.vec3(*bounds[0]), wp.vec3(*bounds[1]))
    if solver.leaks is not None:
        solver.leaks._bounds = (1, wp.vec3(*bounds[0]), wp.vec3(*bounds[1]))
    log.info("SPH escape box widened to %s for a driven body (sim.drive)", bounds)
    return True
