# SPDX-License-Identifier: Apache-2.0
"""Cloth stretch for health (SCR-WP5-17, WP2's finding 2026-11-08): each cloth triangle's largest principal stretch
against its rest shape, the deepest per world.

A VBD sheet that blew up (WP2: edges at 500x their rest length, 2.2 m/s, every particle finite) raised no flag:
NONFINITE needs a NaN, SPEED_FAIL 20 m/s, ENERGY_JUMP reads bodies. ``ClothStretch.update(state)`` gives the stretch
per world (0.5 = 50 % longer than rest along the triangle's most stretched direction) and keeps the triangle, for the
CLOTH_STRETCH flag (``CheckPolicy.cloth_stretch_warn`` / ``cloth_stretch_fail``).

F = [x1 - x0, x2 - x0] . tri_poses (Newton's rest inverse, the same F as VBD's StVK / Neo-Hookean triangle terms);
the stretch is sqrt(largest eigenvalue of F^T F) - 1. Example: ``ClothStretch(model).update(state)`` -> [W] float32.

Limits are the same for every cloth (50 % / 500 %): woven fabric and films stay under ~13 % (WP2's drapes 1-5 %, the
L5 drape at 10 iterations 12.7 %). No material card carries an elongation at break as a field today; an elastomer
sheet (a glove or balloon membrane: 100-300 % lawful, rubber breaks at 500-700 %) on ``add_cloth`` reads DEGRADED
past 50 %. Per-card limits (about half the elongation at break, INVALID past it) come with the first such card.
A triangle folded over (a towel's fold) or crumpled flat is not flagged: a membrane's F is 3x2 with no sign, a fold is
a rotation of the triangle, and a collapsed one is lawful buckling; WP2's blow-up stretched edges, which this reads.

``RangeChecks`` is health's per-world measure checks each check step: soft tet strain (STRAIN_RANGE, WP10's
``SoftStrain``; ``HealthMonitor.set_soft_strain``) and cloth stretch (CLOTH_STRETCH; wiring sets ``ranges.cloth``):
DEGRADED past ``cloth_stretch_warn``, INVALID past ``cloth_stretch_fail``, the run's peak and triangle kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.precision import health_capacity as hc
from alloy.precision import health_kernels as hk
from alloy.precision.types import HealthFlags


@wp.kernel
def _tri_stretch(q: wp.array[wp.vec3], tri: wp.array2d[wp.int32], poses: wp.array[wp.mat22], cloth: wp.array[wp.int32],
                 pworld: wp.array[wp.int32], keys: wp.array[wp.uint64]):
    t = cloth[wp.tid()]
    i, j, k = tri[t, 0], tri[t, 1], tri[t, 2]
    w = pworld[i]
    if w < 0:
        w = 0
    x0 = q[i]
    d1 = q[j] - x0
    d2 = q[k] - x0
    m = poses[t]
    f0 = d1 * m[0, 0] + d2 * m[1, 0]
    f1 = d1 * m[0, 1] + d2 * m[1, 1]
    a = wp.dot(f0, f0)
    b = wp.dot(f0, f1)
    c = wp.dot(f1, f1)
    h = 0.5 * (a - c)
    lam = 0.5 * (a + c) + wp.sqrt(h * h + b * b)
    s = wp.sqrt(lam) - 1.0
    if wp.isfinite(s) and s > 0.0:                  # float bits of a positive float order like the floats
        wp.atomic_max(keys, w, (wp.uint64(wp.cast(s, wp.uint32)) << wp.uint64(32)) | wp.uint64(t))


@wp.kernel
def _unpack(keys: wp.array[wp.uint64], stretch: wp.array[wp.float32], where: wp.array[wp.int32]):
    w = wp.tid()
    key = keys[w]
    keys[w] = wp.uint64(0)
    stretch[w] = wp.cast(wp.uint32(key >> wp.uint64(32)), wp.float32)
    where[w] = wp.where(key == wp.uint64(0), -1, wp.int32(key & wp.uint64(0xFFFFFFFF)))


def cloth_triangles(model: Any) -> np.ndarray:
    """The model's cloth triangles: every triangle with no soft-body (tet) particle. A soft body's surface triangles
    are its tets' faces (STRAIN_RANGE judges those; a knife cut through one stretched them 66 %, lawfully)."""
    n = int(getattr(model, "tri_count", 0) or 0)
    if not n:
        return np.zeros(0, np.int32)
    tri = model.tri_indices.numpy()
    soft = np.zeros(int(model.particle_count), bool)
    if int(getattr(model, "tet_count", 0) or 0):
        soft[model.tet_indices.numpy().ravel()] = True
    return np.flatnonzero(~soft[tri].any(axis=1)).astype(np.int32)


class ClothStretch:
    """Per world, the largest triangle stretch of ``state`` (device-only; graph-capturable)."""

    def __init__(self, model: Any, world_count: int | None = None):
        self.model = model
        self.W = int(world_count if world_count is not None else max(int(getattr(model, "world_count", 1)), 1))
        dev = model.particle_q.device
        self._cloth = wp.array(cloth_triangles(model), dtype=wp.int32, device=dev)
        self.n_tri = int(self._cloth.shape[0])
        pw = getattr(model, "particle_world", None)
        self._pworld = pw if pw is not None else wp.zeros(int(model.particle_count), dtype=wp.int32, device=dev)
        self._keys = wp.zeros(self.W, dtype=wp.uint64, device=dev)
        self.stretch = wp.zeros(self.W, dtype=wp.float32, device=dev)     # [W] the last update's deepest stretch
        self.triangle = wp.full(self.W, -1, dtype=wp.int32, device=dev)   # [W] its triangle (-1: none)

    def update(self, state: Any) -> wp.array:
        if self.n_tri:
            wp.launch(_tri_stretch, dim=self.n_tri, inputs=[state.particle_q, self.model.tri_indices,
                                                             self.model.tri_poses, self._cloth, self._pworld,
                                                             self._keys],
                      device=self._keys.device)
        wp.launch(_unpack, dim=self.W, inputs=[self._keys, self.stretch, self.triangle], device=self._keys.device)
        return self.stretch


@wp.kernel
def _cloth_apply(stretch: wp.array[wp.float32], tri: wp.array[wp.int32], warn: float, fail: float, bit: wp.uint32,
                 acc: hk.Accum, peak: wp.array[wp.float32], peak_tri: wp.array[wp.int32]):
    w = wp.tid()
    s = stretch[w]
    if s > peak[w]:
        peak[w] = s
        peak_tri[w] = tri[w]
    if peak[w] > warn:                              # sticky for the run, as SPIN: a warning that passed stays
        acc.overflow[w] = acc.overflow[w] | bit
    if s > fail:
        acc.stretch_fail[w] = 1


class RangeChecks:
    """Health's per-world measure checks (module docstring). ``soft``: ``update(state)`` -> [W] float32 tet strain;
    ``cloth``: a ``ClothStretch``. Both are set before any graph capture."""

    def __init__(self, cfg: Any, world_count: int, device: Any):
        self.soft: Any = None
        self.cloth: ClothStretch | None = None
        self.W, self.device = int(world_count), device
        self.soft_range = float(getattr(cfg, "soft_strain_range", 0.6))
        self.warn = float(getattr(cfg, "cloth_stretch_warn", 0.5))
        self.fail = float(getattr(cfg, "cloth_stretch_fail", 5.0))
        self.peak = wp.zeros(self.W, dtype=wp.float32, device=device)       # [W] the run's largest cloth stretch
        self.peak_tri = wp.full(self.W, -1, dtype=wp.int32, device=device)  # [W] its triangle

    def launch(self, state: Any, acc: hk.Accum) -> None:
        if self.soft is not None:
            wp.launch(hc.flag_if_above, dim=self.W, device=self.device,
                      inputs=[self.soft.update(state), self.soft_range, wp.uint32(HealthFlags.STRAIN_RANGE), acc])
        if self.cloth is not None:
            s = self.cloth.update(state)
            wp.launch(_cloth_apply, dim=self.W, inputs=[s, self.cloth.triangle, self.warn, self.fail,
                                                        wp.uint32(HealthFlags.CLOTH_STRETCH), acc, self.peak,
                                                        self.peak_tri], device=self.device)

    def report_fields(self) -> dict[str, Any]:
        """For ``HealthReport``: the run's largest cloth stretch per world, its triangle and first particle."""
        if self.cloth is None:
            return {}
        tri = self.peak_tri.numpy().copy()
        idx = self.cloth.model.tri_indices.numpy()
        particle = np.where(tri >= 0, idx[np.clip(tri, 0, max(len(idx) - 1, 0)), 0] if len(idx) else -1, -1)
        return {"cloth_stretch": self.peak.numpy().astype(np.float64), "cloth_tri": tri,
                "cloth_particle": particle.astype(np.int64), "cloth_stretch_warn": self.warn,
                "cloth_stretch_fail": self.fail}


def install(health: Any, notes: list[str]) -> None:
    """Wiring: health measures every cloth triangle's stretch each check step (none without triangles)."""
    model = getattr(health, "model", None)
    if model is None or not hasattr(health, "ranges") or not len(cloth_triangles(model)):
        return
    health.ranges.cloth = ClothStretch(model, health.world_count)
    notes.append("health: cloth triangles stretched past 50 % flag CLOTH_STRETCH (INVALID past 500 %)")


def peak(model: Any, state: Any) -> float:
    """Host helper: the largest triangle stretch of ``state`` over all worlds (0.0 without triangles)."""
    return float(np.max(ClothStretch(model).update(state).numpy(), initial=0.0))


__all__ = ["ClothStretch", "RangeChecks", "cloth_triangles", "install", "peak"]
