# SPDX-License-Identifier: Apache-2.0
"""Emitters, sinks, the escape box and the mass ledger of SolverDFSPH.

Mass is counted in int64 fixed point (2^40 kg units, ``to_fixed`` of each particle's float32 mass), so
``active + out == initial + in`` holds exactly (LIQ-1). Escaped particles (outside ``Config.world_bounds``) are
retired like sunk ones (their mass goes to ``out``) and counted separately (LIQ-8 needs zero).

Emitters own disjoint pools of their world's spare particles (sph.phase == 0, ACTIVE cleared), assigned at
``add_emitter``: the ids it is given (a pour's own reservoir), else an even share of the spares no emitter owns. A
layer of particles on a 2r lattice over the region's cross-section perpendicular to the velocity is emitted every
``n_layer·V / rate`` seconds; pools never refill (retired particles get phase -2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from .bodies import FORCE_SCALE, WORK_SCALE, _fx
from .kernels import to_fixed
from .pressure import Particles, is_fluid

log = logging.getLogger("alloy.sph")

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@dataclass
class EmitterTable:
    """Host-side emitter and sink definitions; uploaded to the device on change."""

    world: list[int] = field(default_factory=list)
    xform: list[tuple] = field(default_factory=list)
    velocity: list[tuple[float, float, float]] = field(default_factory=list)
    interval: list[float] = field(default_factory=list)
    offsets: list[np.ndarray] = field(default_factory=list)  # local layer lattice per emitter
    pool: list[np.ndarray] = field(default_factory=list)  # spare particle ids per emitter
    own: list[bool] = field(default_factory=list)  # True: the pool was given at add_emitter (split_pools keeps it)
    sink_world: list[int] = field(default_factory=list)
    sink_xform: list[tuple] = field(default_factory=list)
    sink_half: list[tuple[float, float, float]] = field(default_factory=list)


def layer_offsets(half_extents: np.ndarray, velocity_local: np.ndarray, spacing: float) -> np.ndarray:
    """Lattice points (local frame) on the region's mid cross-section perpendicular to the velocity: whole spacings
    across each side, with 1e-6 of a cell's slack (a nozzle 0.3 m wide at 0.1 m is 3 columns, not 2: 0.3/0.1 =
    2.9999999999999996; WP12's and WP2's find)."""
    v = np.asarray(velocity_local, dtype=np.float64)
    speed = float(np.linalg.norm(v))
    axis = int(np.argmax(np.abs(v))) if speed > 0.0 else 2
    others = [a for a in range(3) if a != axis]
    grids = []
    for a in others:
        n = max(int(np.floor(2.0 * half_extents[a] / spacing + 1.0e-6)), 1)
        grids.append((np.arange(n) - 0.5 * (n - 1)) * spacing)
    ga, gb = np.meshgrid(grids[0], grids[1], indexing="ij")
    pts = np.zeros((ga.size, 3))
    pts[:, others[0]] = ga.ravel()
    pts[:, others[1]] = gb.ravel()
    return pts.astype(np.float32)


@wp.kernel(enable_backward=False)
def emit_clock_kernel(
    interval: wp.array[wp.float32], t_acc: wp.array[wp.float32], dt: float, enabled: wp.array[wp.int32],
    emit_now: wp.array[wp.int32],
):
    e = wp.tid()
    emit_now[e] = 0
    if enabled[e] == 0:
        return
    t = t_acc[e] + dt
    if t >= interval[e]:
        t -= interval[e]
        emit_now[e] = 1
    t_acc[e] = t


@wp.func
def flux_add(flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64], w: int, m: float, v: wp.vec3, x: wp.vec3,
             g: wp.vec3, sign: float):
    """Energy a particle brings into (sign +1) or takes out of (-1) world w. Its D20 energy about z_ref is
    m (v^2/2 - g.(x - z_ref e_z)) = m (v^2/2 - g.x) + m g_z z_ref, so flux_e sums the first term [J] and flux_g the
    slope m g_z [N]; ``runtime.flux_export_kernel`` combines them at the health monitor's z_ref."""
    wp.atomic_add(flux_e, w, _fx(sign * m * (0.5 * wp.dot(v, v) - wp.dot(g, x)), WORK_SCALE))
    wp.atomic_add(flux_g, w, _fx(sign * m * g[2], FORCE_SCALE))


@wp.kernel(enable_backward=False)
def emit_kernel(
    off_emitter: wp.array[wp.int32], off_rank: wp.array[wp.int32], off_local: wp.array[wp.vec3],
    emit_now: wp.array[wp.int32], cursor: wp.array[wp.int32], pool_begin: wp.array[wp.int32],
    pool_size: wp.array[wp.int32], pool_ids: wp.array[wp.int32], xform: wp.array[wp.transform],
    velocity: wp.array[wp.vec3], e_world: wp.array[wp.int32], p: Particles,
    mass_in: wp.array[wp.int64], overflow: wp.array[wp.int32], gravity: wp.array[wp.vec3],
    flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64],
):
    c = wp.tid()
    e = off_emitter[c]
    if emit_now[e] == 0:
        return
    slot = cursor[e] + off_rank[c]
    w = e_world[e]
    if slot >= pool_size[e]:
        wp.atomic_add(overflow, w, 1)
        return
    i = pool_ids[pool_begin[e] + slot]
    p.q[i] = wp.transform_point(xform[e], off_local[c])
    p.qd[i] = velocity[e]
    p.flags[i] = p.flags[i] | _ACTIVE
    p.phase[i] = wp.int8(1)
    wp.atomic_add(mass_in, w, to_fixed(p.mass[i]))
    flux_add(flux_e, flux_g, w, p.mass[i], p.qd[i], p.q[i], gravity[w], 1.0)


@wp.kernel(enable_backward=False)
def advance_cursor_kernel(emit_now: wp.array[wp.int32], n_layer: wp.array[wp.int32], pool_size: wp.array[wp.int32],
                          cursor: wp.array[wp.int32]):
    e = wp.tid()
    if emit_now[e] != 0:
        cursor[e] = wp.min(cursor[e] + n_layer[e], pool_size[e])


@wp.func
def _in_box(xform: wp.transform, half: wp.vec3, x: wp.vec3) -> bool:
    xl = wp.transform_point(wp.transform_inverse(xform), x)
    return wp.abs(xl[0]) <= half[0] and wp.abs(xl[1]) <= half[1] and wp.abs(xl[2]) <= half[2]


@wp.kernel(enable_backward=False)
def retire_kernel(
    p: Particles, n_sinks: int, s_world: wp.array[wp.int32], s_xform: wp.array[wp.transform],
    s_half: wp.array[wp.vec3], has_bounds: int, lo: wp.vec3, hi: wp.vec3,
    mass_out: wp.array[wp.int64], escaped: wp.array[wp.int32], gravity: wp.array[wp.vec3],
    flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64],
):
    """Retire particles inside a sink of their world or outside the escape box (mass to ``out``, energy out of the
    flux ledger; a non-finite particle carries none)."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    w = p.world[i]
    x = p.q[i]
    gone = int(0)
    for s in range(n_sinks):
        if s_world[s] == w and _in_box(s_xform[s], s_half[s], x):
            gone = 1
    if has_bounds != 0 and gone == 0:
        out = x[0] < lo[0] or x[1] < lo[1] or x[2] < lo[2] or x[0] > hi[0] or x[1] > hi[1] or x[2] > hi[2]
        if out or not wp.isfinite(x[0] + x[1] + x[2]):
            gone = 2
            wp.atomic_add(escaped, w, 1)
    if gone != 0:
        v = p.qd[i]
        if wp.isfinite(x[0] + x[1] + x[2] + v[0] + v[1] + v[2]):
            flux_add(flux_e, flux_g, w, p.mass[i], v, x, gravity[w], -1.0)
        p.flags[i] = p.flags[i] & ~_ACTIVE
        p.phase[i] = wp.int8(-2)  # retired: never reused by an emitter
        p.qd[i] = wp.vec3(0.0, 0.0, 0.0)
        wp.atomic_add(mass_out, w, to_fixed(p.mass[i]))


@wp.kernel(enable_backward=False)
def active_mass_kernel(p: Particles, out: wp.array[wp.int64]):
    i = wp.tid()
    if is_fluid(p, i):
        wp.atomic_add(out, p.world[i], to_fixed(p.mass[i]))


class EmitterDevice:
    """Device copy of an EmitterTable (rebuilt on host changes; never inside a step)."""

    def __init__(self, table: EmitterTable, device: wp.context.Device):
        n_e = len(table.world)
        self.n_emitters = n_e
        self.n_sinks = len(table.sink_world)
        d = device
        offs = table.offsets
        counts = np.array([len(o) for o in offs], dtype=np.int32)
        self.n_offsets = int(counts.sum())
        emitter_of = np.repeat(np.arange(n_e, dtype=np.int32), counts) if n_e else np.zeros(0, np.int32)
        rank = np.concatenate([np.arange(c, dtype=np.int32) for c in counts]) if n_e else np.zeros(0, np.int32)
        local = np.concatenate(offs).astype(np.float32) if n_e else np.zeros((0, 3), np.float32)
        sizes = np.array([len(p) for p in table.pool], dtype=np.int32)
        begins = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32) if n_e else np.zeros(0, np.int32)
        ids = np.concatenate(table.pool).astype(np.int32) if n_e and sizes.sum() else np.zeros(1, np.int32)

        def arr(x, dt):
            x = np.asarray(x)
            return wp.array(x if x.size else np.zeros((1,) + x.shape[1:], x.dtype), dtype=dt, device=d)

        self.off_emitter = arr(emitter_of, wp.int32)
        self.off_rank = arr(rank, wp.int32)
        self.off_local = arr(local, wp.vec3)
        self.n_layer = arr(counts, wp.int32)
        self.pool_begin = arr(begins, wp.int32)
        self.pool_size = arr(sizes, wp.int32)
        self.pool_ids = arr(ids, wp.int32)
        self.xform = arr(np.asarray(table.xform, np.float32).reshape(-1, 7), wp.transform)
        self.velocity = arr(np.asarray(table.velocity, np.float32).reshape(-1, 3), wp.vec3)
        self.world = arr(np.asarray(table.world, np.int32), wp.int32)
        self.interval = arr(np.asarray(table.interval, np.float32), wp.float32)
        self.enabled = arr(np.ones(n_e, np.int32), wp.int32)
        self.t_acc = arr(np.zeros(n_e, np.float32), wp.float32)
        self.cursor = arr(np.zeros(n_e, np.int32), wp.int32)
        self.emit_now = arr(np.zeros(n_e, np.int32), wp.int32)
        self.s_world = arr(np.asarray(table.sink_world, np.int32), wp.int32)
        self.s_xform = arr(np.asarray(table.sink_xform, np.float32).reshape(-1, 7), wp.transform)
        self.s_half = arr(np.asarray(table.sink_half, np.float32).reshape(-1, 3), wp.vec3)

    def carry_from(self, old: EmitterDevice | None) -> None:
        """Keep clocks and cursors of emitters that existed before a table change."""
        if old is None or old.n_emitters == 0:
            return
        n = old.n_emitters
        for name in ("t_acc", "cursor", "enabled"):
            dst = getattr(self, name)
            wp.copy(dst, getattr(old, name), count=n)


def split_pools(t: EmitterTable, world: int, spares: np.ndarray, cursors: np.ndarray) -> None:
    """Share ``world``'s ``spares`` evenly among its emitters that were given no pool (only while none of them has
    emitted; after that the last one takes what is left). An emitter given its pool keeps it, and its ids are not
    shared out: two pours in one world each pour their own reservoir (they split all the spares evenly before:
    20 ml and 60 ml pours delivered 38 ml each). Example: ``split_pools(table, 0, spare_ids, cursors)``."""
    owned = [t.pool[e] for e, w in enumerate(t.world) if w == world and t.own[e]]
    free = np.setdiff1d(spares, np.concatenate(owned)) if owned else spares
    mine = [e for e, w in enumerate(t.world) if w == world and not t.own[e]]
    if any(e < len(cursors) and cursors[e] > 0 for e in mine[:-1]):
        used = np.concatenate([t.pool[e] for e in mine[:-1]]) if len(mine) > 1 else np.zeros(0, np.int32)
        t.pool[mine[-1]] = np.setdiff1d(free, used).astype(np.int32)
        return
    for k, part in enumerate(np.array_split(free, len(mine)) if mine else ()):
        t.pool[mine[k]] = part.astype(np.int32)


def rewind(dev: EmitterDevice | None, world_mask: wp.array | None, world_count: int) -> None:
    """The box emitters of the masked worlds (all when None) back to a fresh build's: layer clock and cursor 0, so a
    pour starts over from its full reservoir at the next step. ``SolverDFSPH.reset`` calls it. ``sim.reset`` does not:
    its start snapshot holds the emitters with the particles (``alloy.api._pour.remake_start``), so a reset after
    ``sim.settle`` goes on from the settled pour. ``enabled`` stays as set. Written in place: captured graphs keep
    their pointers. Example: ``rewind(solver.emitters.dev, None, 4)``."""
    if dev is None or dev.n_emitters == 0:
        return
    mask = np.ones(world_count, bool) if world_mask is None else np.asarray(
        world_mask.numpy() if hasattr(world_mask, "numpy") else world_mask)[:world_count].astype(bool)
    picked = mask[dev.world.numpy()[: dev.n_emitters]]
    for arr in (dev.t_acc, dev.cursor):
        a = arr.numpy()
        a[: dev.n_emitters][picked] = 0
        arr.assign(a)
