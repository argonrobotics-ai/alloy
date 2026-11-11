# SPDX-License-Identifier: Apache-2.0
"""Per-world mass and momentum ledger for topology events (TOP-1, SPEC §3.8).

Sums run in float64, one thread per world over that world's bodies and particles in index order, so they are
bitwise reproducible (no float atomics, D9). They run only on steps with events (the kernel returns at once
otherwise). Per period the ledger reports the world totals just before the period's first event and
``after = before + Σ_k (after_k − before_k)`` over that period's event steps, so mass entering through other channels
between events (SPH emitters) does not pollute the topology balance. Momentum is linear (Σ m·v) and angular about
the world origin (Σ R·I·Rᵀ·ω + x_com × m·v for bodies, Σ x × m·v for particles; the MPM affine field is excluded).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.topology.geometry import to_mat33d, to_vec3d
from alloy.topology.types import TopologyLedger

log = logging.getLogger("alloy.topology")

ACTIVE = wp.constant(int(newton.ParticleFlags.ACTIVE))


def _csr_by_world(item_world: np.ndarray, world_count: int) -> tuple[np.ndarray, np.ndarray]:
    """(start [W+1], order) of items grouped by world in index order. Global items (world −1) belong to world 0 in a
    single-world model and are left out otherwise (shared static geometry)."""
    item_world = np.where(item_world < 0, 0, item_world) if world_count <= 1 else item_world
    keep = np.flatnonzero(item_world >= 0)
    order = keep[np.argsort(item_world[keep], kind="stable")].astype(np.int32)
    counts = np.bincount(item_world[keep], minlength=world_count)[:world_count]
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return start, order


@dataclass
class WorldIndex:
    """Bodies and particles grouped by world (CSR), built once on the host."""

    world_count: int
    body_start: wp.array
    body_order: wp.array
    particle_start: wp.array
    particle_order: wp.array

    @classmethod
    def from_model(cls, model: newton.Model) -> WorldIndex:
        w = max(int(model.world_count), 1)
        dev = model.device
        bw = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
        pw = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32)
        bs, bo = _csr_by_world(bw, w)
        ps, po = _csr_by_world(pw, w)
        arr = lambda a: wp.array(a, dtype=wp.int32, device=dev)  # noqa: E731
        return cls(w, arr(bs), arr(bo), arr(ps), arr(po))


@wp.kernel
def _measure_worlds(
    step_events: wp.array[wp.int32], body_start: wp.array[wp.int32], body_order: wp.array[wp.int32],
    body_active: wp.array[wp.int8], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[float], body_com: wp.array[wp.vec3], body_inertia: wp.array[wp.mat33],
    particle_start: wp.array[wp.int32], particle_order: wp.array[wp.int32], particle_flags: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3], particle_mass: wp.array[float],
    out_mass: wp.array[wp.float64], out_mom: wp.array2d[wp.float64],
):
    w = wp.tid()
    if step_events[w] == 0:
        return
    m_tot = wp.float64(0.0)
    p = wp.vec3d()
    ang = wp.vec3d()
    for k in range(body_start[w], body_start[w + 1]):
        bi = body_order[k]
        if body_active[bi] == 0 or body_mass[bi] == 0.0:
            continue
        m = wp.float64(body_mass[bi])
        x = to_vec3d(wp.transform_point(body_q[bi], body_com[bi]))
        rot = wp.quat_to_matrix(wp.transform_get_rotation(body_q[bi]))
        v = to_vec3d(wp.spatial_top(body_qd[bi]))
        om = wp.spatial_bottom(body_qd[bi])
        i_w = to_mat33d(rot * body_inertia[bi] * wp.transpose(rot))
        m_tot += m
        p += m * v
        ang += i_w * to_vec3d(om) + wp.cross(x, m * v)
    for k in range(particle_start[w], particle_start[w + 1]):
        pi = particle_order[k]
        if (particle_flags[pi] & ACTIVE) == 0:
            continue
        m = wp.float64(particle_mass[pi])
        x = to_vec3d(particle_q[pi])
        v = to_vec3d(particle_qd[pi])
        m_tot += m
        p += m * v
        ang += wp.cross(x, m * v)
    out_mass[w] = m_tot
    for j in range(3):
        out_mom[w, j] = p[j]
        out_mom[w, 3 + j] = ang[j]


@wp.kernel
def _record_before(
    step_events: wp.array[wp.int32], events: wp.array[wp.int32], cur_mass: wp.array[wp.float64],
    cur_mom: wp.array2d[wp.float64], step_mass: wp.array[wp.float64], step_mom: wp.array2d[wp.float64],
    mass_before: wp.array[wp.float64], mom_before: wp.array2d[wp.float64], mass_after: wp.array[wp.float64],
    mom_after: wp.array2d[wp.float64],
):
    w = wp.tid()
    if step_events[w] == 0:
        return
    step_mass[w] = cur_mass[w]
    if events[w] == 0:
        mass_before[w] = cur_mass[w]
        mass_after[w] = cur_mass[w]
    for j in range(6):
        step_mom[w, j] = cur_mom[w, j]
        if events[w] == 0:
            mom_before[w, j] = cur_mom[w, j]
            mom_after[w, j] = cur_mom[w, j]


@wp.kernel
def _record_after(
    step_events: wp.array[wp.int32], cur_mass: wp.array[wp.float64], cur_mom: wp.array2d[wp.float64],
    step_mass: wp.array[wp.float64], step_mom: wp.array2d[wp.float64], mass_after: wp.array[wp.float64],
    mom_after: wp.array2d[wp.float64], events: wp.array[wp.int32], reset_required: wp.array[wp.int32],
):
    w = wp.tid()
    n = step_events[w]
    if n == 0:
        return
    mass_after[w] = mass_after[w] + (cur_mass[w] - step_mass[w])
    for j in range(6):
        mom_after[w, j] = mom_after[w, j] + (cur_mom[w, j] - step_mom[w, j])
    events[w] = events[w] + n
    reset_required[w] = 1


class LedgerState:
    """Device accumulators for one period plus the per-step event counts that drive them."""

    _FIELDS = ("step_events", "events", "overflow", "reset_required", "mass_before", "mass_after", "mom_before",
               "mom_after", "step_mass", "step_mom", "cur_mass", "cur_mom")

    def __init__(self, model: newton.Model, index: WorldIndex) -> None:
        self.model, self.index = model, index
        w, dev = index.world_count, model.device
        i32 = lambda: wp.zeros(w, dtype=wp.int32, device=dev)  # noqa: E731
        f64 = lambda: wp.zeros(w, dtype=wp.float64, device=dev)  # noqa: E731
        m6 = lambda: wp.zeros((w, 6), dtype=wp.float64, device=dev)  # noqa: E731
        self.step_events, self.events, self.overflow, self.reset_required = i32(), i32(), i32(), i32()
        self.mass_before, self.mass_after, self.step_mass, self.cur_mass = f64(), f64(), f64(), f64()
        self.mom_before, self.mom_after, self.step_mom, self.cur_mom = m6(), m6(), m6(), m6()
        dyn = (model.body_mass.numpy() > 0.0).astype(np.int8) if model.body_count else np.zeros(1, np.int8)
        self.body_active = wp.array(dyn, dtype=wp.int8, device=dev)

    def _measure(self, state: newton.State) -> None:
        m, ix = self.model, self.index
        body_q = state.body_q if state.body_q is not None else wp.zeros(1, dtype=wp.transform, device=m.device)
        body_qd = state.body_qd if state.body_qd is not None else wp.zeros(1, dtype=wp.spatial_vector, device=m.device)
        wp.launch(_measure_worlds, dim=ix.world_count, device=m.device, inputs=[
            self.step_events, ix.body_start, ix.body_order, self.body_active, body_q, body_qd,
            _or_empty(m.body_mass, float, m.device), _or_empty(m.body_com, wp.vec3, m.device),
            _or_empty(m.body_inertia, wp.mat33, m.device), ix.particle_start, ix.particle_order,
            _or_empty(m.particle_flags, wp.int32, m.device), _or_empty(state.particle_q, wp.vec3, m.device),
            _or_empty(state.particle_qd, wp.vec3, m.device), _or_empty(m.particle_mass, float, m.device),
            self.cur_mass, self.cur_mom])

    def before(self, state: newton.State) -> None:
        """Record world totals before this step's native changes (only worlds with step events)."""
        self._measure(state)
        wp.launch(_record_before, dim=self.index.world_count, device=self.model.device, inputs=[
            self.step_events, self.events, self.cur_mass, self.cur_mom, self.step_mass, self.step_mom,
            self.mass_before, self.mom_before, self.mass_after, self.mom_after])

    def after(self, state: newton.State) -> None:
        """Add this step's change to the period balance and count the events."""
        self._measure(state)
        wp.launch(_record_after, dim=self.index.world_count, device=self.model.device, inputs=[
            self.step_events, self.cur_mass, self.cur_mom, self.step_mass, self.step_mom, self.mass_after,
            self.mom_after, self.events, self.reset_required])

    def snapshot(self) -> TopologyLedger:
        """Host copy of the period so far (syncs the device)."""
        return TopologyLedger(
            events=self.events.numpy().astype(np.int64), mass_before=self.mass_before.numpy().copy(),
            mass_after=self.mass_after.numpy().copy(), momentum_before=self.mom_before.numpy().copy(),
            momentum_after=self.mom_after.numpy().copy(), overflow=self.overflow.numpy().astype(np.int64),
            reset_required=self.reset_required.numpy().astype(bool))

    def reset_period(self) -> None:
        for name in ("events", "overflow", "reset_required", "mass_before", "mass_after", "mom_before", "mom_after"):
            getattr(self, name).zero_()

    def arrays(self) -> Mapping[str, wp.array]:
        out = {f"topology.ledger.{n}": getattr(self, n) for n in self._FIELDS}
        out["topology.body_active"] = self.body_active
        return out


def _or_empty(arr: wp.array | None, dtype, device) -> wp.array:
    return arr if arr is not None else wp.zeros(1, dtype=dtype, device=device)
