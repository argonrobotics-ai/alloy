# SPDX-License-Identifier: Apache-2.0
"""The force plug-in of quasi-static mass scaling (``alloy.api._quasi_static``; physics in ``quasi_static``).

Every exchange it adds (s - 1) m0 g upward on each scaled particle, so the scaled solid weighs what it did (probe: a
x64 block at rest loads the floor with 1.0000 of its true weight, 64.0 without the force; with a free cube on it, block
plus cube). After every exchange it writes two per-world ratios to the declared tool speed, which the health monitor
flags (QUASI_STATIC, DEGRADED; physics ruling 2026-11-11, [internal note] notes/wp7-dent/slice2/STEP2-CHECK.md):

- ``motion``: the solid's mass-weighted mean velocity |sum m v| / sum m: the solid as a whole moving (a support lost, a
  throw, a push faster than declared). One fast particle moves it by its mass share only, so particle noise does not
  reach it (a correct press: at most 0.14x).
- ``contact``: the fastest point of a rigid or kinematic body that touches MPM material within the solid's bounding box
  (a grid collider node carrying an impulse): a tool striking the s-times-heavier solid. Rigid speeds carry no particle
  noise.

Both are deterministic: the mean is a fixed-order loop per world, the contact speed an atomic max. Each exchange
overwrites them, so the health monitor judges the last exchange before each of its checks. A scene with a j2 solid
on MPM runs without CUDA graphs (``graph_budget.eager_scene_reason``), so these launches are never captured."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.base import ForcePlugin

__all__ = ["QuasiStaticWeight", "Solid"]


@dataclass(frozen=True, eq=False)
class Solid:
    """One scaled solid: its particle ``rows`` [worlds, n] (global), ``s``, the declared tool ``speed`` (m/s) and the
    MPM ``voxel`` (m: the contact box reaches this far past the solid's particles)."""

    name: str
    rows: np.ndarray
    s: float
    speed: float
    voxel: float


@wp.kernel
def add_weight(rows: wp.array(dtype=wp.int32), f: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    r = rows[i]
    out[r] = out[r] + f[i]


@wp.kernel
def motion_and_box(rows: wp.array2d(dtype=wp.int32), mass: wp.array2d(dtype=wp.float32), q: wp.array(dtype=wp.vec3),
                   qd: wp.array(dtype=wp.vec3), inv_mv: wp.array(dtype=wp.float64), pad: wp.float32,
                   motion: wp.array(dtype=wp.float32), lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3)):
    """Per world, in a fixed order: |sum m v| / (sum m * declared speed), and the particles' box grown by ``pad``."""
    w = wp.tid()
    px, py, pz = wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
    a = q[rows[w, 0]]
    b = a
    for j in range(rows.shape[1]):
        r = rows[w, j]
        m, v = wp.float64(mass[w, j]), qd[r]
        px += m * wp.float64(v[0])
        py += m * wp.float64(v[1])
        pz += m * wp.float64(v[2])
        a = wp.min(a, q[r])
        b = wp.max(b, q[r])
    motion[w] = wp.float32(wp.sqrt(px * px + py * py + pz * pz) * inv_mv[w])
    lo[w] = a - wp.vec3(pad, pad, pad)
    hi[w] = b + wp.vec3(pad, pad, pad)


@wp.kernel
def contact_speed(cid: wp.array(dtype=int), imp: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3),
                  collider_body: wp.array(dtype=int), body_world: wp.array(dtype=wp.int32),
                  body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
                  body_com: wp.array(dtype=wp.vec3), lo: wp.array(dtype=wp.vec3), hi: wp.array(dtype=wp.vec3),
                  inv_v: wp.float32, contact: wp.array(dtype=wp.float32)):
    """The point speed of each body at a loaded collider node inside its world's solid box, over the declared speed."""
    i = wp.tid()
    c = cid[i]
    if c < 0 or wp.length(imp[i]) == 0.0:
        return
    body = collider_body[c]
    if body < 0:
        return
    w = int(body_world[body])
    if w < 0 or w >= lo.shape[0]:
        return
    x = pos[i]
    a, b = lo[w], hi[w]
    if x[0] < a[0] or x[1] < a[1] or x[2] < a[2] or x[0] > b[0] or x[1] > b[1] or x[2] > b[2]:
        return
    tf = body_q[body]
    com = wp.transform_point(tf, body_com[body])
    v = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), x - com)
    wp.atomic_max(contact, w, wp.length(v) * inv_v)


class QuasiStaticWeight(ForcePlugin):
    """The true weight of each scaled solid, and its per-world motion and contact ratios (module docstring)."""

    name = "quasi_static_weight"
    summary = "quasi-static mass scaling: the scaled solid's weight kept true, its motion and contact speeds measured"

    @dataclass
    class Config:
        solids: list = field(default_factory=list)      # [Solid]

    def setup(self, ctx: Any) -> None:
        m, dev = ctx.model, ctx.device
        mass = m.particle_mass.numpy().astype(np.float64)             # already x s (alloy.api._quasi_static.apply)
        g = np.asarray(m.gravity.numpy(), np.float64).reshape(-1, 3)
        self.solids, self._mpm = list(self.config.solids), None
        self._rows, self._f, self._rows2, self._mass2, self._inv = [], [], [], [], []
        self.motion, self.contact, self._lo, self._hi = [], [], [], []
        for solid in self.solids:
            rows = np.asarray(solid.rows, np.int64)
            worlds = rows.shape[0]
            world = np.repeat(np.arange(worlds), rows.shape[1])
            flat = rows.ravel()
            f = -(1.0 - 1.0 / solid.s) * mass[flat, None] * g[np.minimum(world, len(g) - 1)]   # up: (s - 1) m0 |g|
            self._rows.append(wp.array(flat.astype(np.int32), dtype=wp.int32, device=dev))
            self._f.append(wp.array(f.astype(np.float32), dtype=wp.vec3, device=dev))
            self._rows2.append(wp.array(rows.astype(np.int32), dtype=wp.int32, device=dev))
            self._mass2.append(wp.array(mass[rows].astype(np.float32), dtype=wp.float32, device=dev))
            self._inv.append(wp.array(1.0 / (mass[rows].sum(axis=1) * solid.speed), dtype=wp.float64, device=dev))
            for out, dtype in ((self.motion, wp.float32), (self.contact, wp.float32), (self._lo, wp.vec3),
                               (self._hi, wp.vec3)):
                out.append(wp.zeros(worlds, dtype=dtype, device=dev))
        self._world = m.body_world
        self._com = m.body_com

    def attach(self, mpm_solver: Any) -> None:
        """The MPM solver that holds the solids: its collider impulses tell which bodies touch them."""
        self._mpm = mpm_solver

    def apply(self, state: Any, forces: Any, dt: float) -> None:
        for rows, f in zip(self._rows, self._f, strict=True):
            wp.launch(add_weight, dim=rows.shape[0], inputs=[rows, f, forces.particle_f], device=rows.device)

    def update(self, state_in: Any, state_out: Any, dt: float) -> None:
        for k, solid in enumerate(self.solids):
            rows2 = self._rows2[k]
            wp.launch(motion_and_box, dim=rows2.shape[0], device=rows2.device,
                      inputs=[rows2, self._mass2[k], state_out.particle_q, state_out.particle_qd, self._inv[k],
                              wp.float32(1.5 * solid.voxel), self.motion[k], self._lo[k], self._hi[k]])
            self.contact[k].zero_()
            self._contact(k, solid, state_out)

    def _impulses(self, state: Any) -> tuple:
        """The MPM grid's collider impulses (unscaled: only "nonzero" is read), node positions and collider ids: the
        arrays ``collect_collider_impulses`` reads, without the scaled copy it allocates on every call."""
        src = state if hasattr(state, "impulse_field") else getattr(self._mpm, "_scratchpad", None)
        try:
            return src.impulse_field.dof_values, src.collider_position_field.dof_values, src.collider_ids
        except AttributeError:
            return self._mpm.collect_collider_impulses(state)

    def _contact(self, k: int, solid: Solid, state: Any) -> None:
        if self._mpm is None or state.body_q is None or state.body_qd is None or self._world is None:
            return
        imp, pos, cid = self._impulses(state)
        if cid.shape[0] == 0:
            return
        wp.launch(contact_speed, dim=cid.shape[0], device=cid.device,
                  inputs=[cid, imp, pos, self._mpm.collider_body_index, self._world, state.body_q, state.body_qd,
                          self._com, self._lo[k], self._hi[k], wp.float32(1.0 / solid.speed), self.contact[k]])
