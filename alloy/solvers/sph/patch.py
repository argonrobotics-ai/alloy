# SPDX-License-Identifier: Apache-2.0
"""Triggered one-shot patch emitters of SolverDFSPH (SCR-DC5-5 item 1, API as DC5 asked).

``pid = add_patch_emitter(world_mask, body, local_points)`` registers, per masked world w, N_w = len(local_points[w])
points in the frame of that world's body (one body index, or one per world; -1 = world frame). Each world's
particles are its own spares, reserved at registration (box emitters of the world re-split the rest; CapacityError
if short).

``trigger(pid, world_flags)`` arms the flagged worlds. It is one kernel launch and capture-safe. At the end of the
next step every armed world emits its N_w particles once:
- positions: the body pose of that step's end applied to the local points;
- velocity: the body's point velocity plus ``velocity`` (body frame);
- flags ACTIVE, phase liquid; mass into the int64 ledger as for box emitters.

A fired world ignores later triggers until ``reset`` re-arms it. The pending and fired flags are solver state
(``alloy_state_arrays``), so snapshots and coupling restarts keep them. ``film_offsets`` builds a 2r-lattice film on a
rectangle (patch frame +z = outward face normal) for callers that want one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from .boundary import point_velocity
from .emitters import flux_add
from .kernels import to_fixed
from .pressure import Particles

log = logging.getLogger("alloy.sph")

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


def film_offsets(half_extents: Sequence[float], count: int, spacing: float,
                 frame: wp.transform | None = None) -> np.ndarray:
    """``count`` points (float32 [count, 3]) on the 2r lattice of the rectangle |x| <= hx, |y| <= hy of ``frame``
    (+z = outward face normal; identity if None), nearest the centre first (ties by x, then y), layers at z = r,
    3r, 5r, ..., mapped into the parent (body) frame."""
    axes = []
    for half in half_extents[:2]:
        n = max(int(np.floor(2.0 * float(half) / spacing + 1.0e-9)), 1)
        axes.append((np.arange(n) - 0.5 * (n - 1)) * spacing)
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    order = np.lexsort((gy, gx, np.round(gx * gx + gy * gy, 12)))
    layer = np.stack([gx[order], gy[order]], axis=1)
    k = np.arange(count)
    out = np.zeros((count, 3))
    out[:, :2] = layer[k % len(layer)]
    out[:, 2] = 0.5 * spacing + (k // len(layer)) * spacing
    if frame is not None:
        out = np.array([wp.transform_point(frame, wp.vec3(*o)) for o in out]).reshape(count, 3)
    return out.astype(np.float32)


@wp.kernel(enable_backward=False)
def patch_trigger_kernel(flags: wp.array[wp.int32], world: wp.array[wp.int32], fired: wp.array[wp.int32],
                         pending: wp.array[wp.int32]):
    e = wp.tid()
    if flags[world[e]] != 0 and fired[e] == 0:
        pending[e] = 1


@wp.kernel(enable_backward=False)
def patch_trigger_bool_kernel(flags: wp.array[wp.bool], world: wp.array[wp.int32], fired: wp.array[wp.int32],
                              pending: wp.array[wp.int32]):
    e = wp.tid()
    if flags[world[e]] and fired[e] == 0:
        pending[e] = 1


@wp.kernel(enable_backward=False)
def patch_emit_kernel(
    world: wp.array[wp.int32], body: wp.array[wp.int32], count: wp.array[wp.int32], pending: wp.array[wp.int32],
    fired: wp.array[wp.int32], begin: wp.array[wp.int32], pool_ids: wp.array[wp.int32], points: wp.array[wp.vec3],
    vel_local: wp.vec3, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3], p: Particles, mass_in: wp.array[wp.int64], gravity: wp.array[wp.vec3],
    flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64],
):
    e, k = wp.tid()
    if pending[e] == 0 or fired[e] != 0 or k >= count[e]:
        return
    b = body[e]
    X = wp.transform_identity()
    if b >= 0:
        X = body_q[b]
    x = wp.transform_point(X, points[begin[e] + k])
    v = wp.transform_vector(X, vel_local) + point_velocity(b, x, body_q, body_qd, body_com)
    i = pool_ids[begin[e] + k]
    p.q[i] = x
    p.qd[i] = v
    p.flags[i] = p.flags[i] | _ACTIVE
    p.phase[i] = wp.int8(1)
    wp.atomic_add(mass_in, world[e], to_fixed(p.mass[i]))
    flux_add(flux_e, flux_g, world[e], p.mass[i], v, x, gravity[world[e]], 1.0)


@wp.kernel(enable_backward=False)
def patch_fire_kernel(pending: wp.array[wp.int32], fired: wp.array[wp.int32]):
    e = wp.tid()
    if pending[e] != 0 and fired[e] == 0:
        fired[e] = 1
    pending[e] = 0


@dataclass
class PatchGroup:
    """One registration: entries for the masked worlds with N_w > 0; points and pool ids share ``begin``."""

    world: wp.array
    body: wp.array
    count: wp.array
    pending: wp.array
    fired: wp.array
    begin: wp.array
    pool_ids: wp.array
    points: wp.array
    velocity: wp.vec3
    n_max: int
    reserved: dict[int, np.ndarray]  # world -> particle ids (host)

    @property
    def entries(self) -> int:
        return len(self.reserved)


class PatchEmitters:
    """Host bookkeeping and device launches of the patch groups of one solver."""

    def __init__(self, solver, spares):
        self.solver = solver
        self._spares = spares  # world -> unreserved inactive spares (EmitterManager)
        self.groups: list[PatchGroup] = []

    def reserved(self, world: int) -> np.ndarray:
        ids = [g.reserved[world] for g in self.groups if world in g.reserved]
        return np.concatenate(ids) if ids else np.zeros(0, np.int32)

    @staticmethod
    def worlds_of(world_mask, local_points) -> list[int]:
        """Masked worlds with at least one point."""
        return [w for w, on in enumerate(np.asarray(world_mask, bool)) if on and len(local_points[w])]

    def add(self, world_mask, body, local_points, velocity: Sequence[float]) -> int:
        s = self.solver
        W = s.layout.world_count
        bodies = self._check(world_mask, body, local_points, W)
        worlds = self.worlds_of(world_mask, local_points)
        pts = [np.asarray(local_points[w], np.float32).reshape(-1, 3) for w in worlds]
        reserved = {w: self._reserve(w, len(q)) for w, q in zip(worlds, pts, strict=True)}
        counts = np.array([len(q) for q in pts], np.int32)
        d = s.device

        def i32(a):
            a = np.asarray(a, np.int32)
            return wp.array(a if a.size else np.zeros(1, np.int32), dtype=wp.int32, device=d)

        begin = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(worlds) else np.zeros(0)
        flat = np.concatenate(pts) if len(worlds) else np.zeros((1, 3), np.float32)
        pool = np.concatenate([reserved[w] for w in worlds]) if len(worlds) else np.zeros(1, np.int32)
        g = PatchGroup(i32(worlds), i32(bodies[worlds] if len(worlds) else []), i32(counts), i32(np.zeros(len(worlds))),
                       i32(np.zeros(len(worlds))), i32(begin), i32(pool), wp.array(flat, dtype=wp.vec3, device=d),
                       wp.vec3(*velocity), int(counts.max()) if len(counts) else 0, reserved)
        self.groups.append(g)
        log.info("SPH patch emitter %d: %d worlds, up to %d particles each", len(self.groups) - 1, len(worlds),
                 g.n_max)
        return len(self.groups) - 1

    def _check(self, world_mask, body, local_points, W: int) -> np.ndarray:
        """Validate a registration; returns the per-world body indices."""
        s = self.solver
        mask = np.asarray(world_mask, bool)
        if mask.shape != (W,) or len(local_points) != W:
            raise ValueError(f"add_patch_emitter: world_mask and local_points need one entry per world ({W})")
        bodies = np.full(W, int(body), np.int32) if np.ndim(body) == 0 else np.asarray(body, np.int32)
        if bodies.shape != (W,):
            raise ValueError(f"add_patch_emitter: body must be one index or one per world ({W})")
        if np.any((bodies < -1) | (bodies >= int(s.model.body_count))):
            raise ValueError("add_patch_emitter: body indices must be in [-1, body_count)")
        for w in np.nonzero(mask)[0]:
            if np.asarray(local_points[w]).size and np.asarray(local_points[w]).reshape(-1, 3).shape[1] != 3:
                raise ValueError(f"add_patch_emitter: local_points[{w}] must be (N, 3)")
        return bodies

    def _reserve(self, world: int, n: int) -> np.ndarray:
        from alloy.core.errors import CapacityError

        free = self._spares(world)
        if len(free) < n:
            raise CapacityError(f"add_patch_emitter: world {world} has {len(free)} free spares, needs {n}")
        return free[:n].astype(np.int32)

    def trigger(self, pid: int, world_flags: wp.array) -> None:
        """Arm the flagged worlds of group ``pid`` (capture-safe: one launch, no host sync)."""
        if not 0 <= pid < len(self.groups):
            raise ValueError(f"trigger: no patch emitter {pid}")
        g = self.groups[pid]
        if g.entries == 0:
            return
        if world_flags.shape[0] < self.solver.layout.world_count:
            raise ValueError(f"trigger: world_flags needs >= {self.solver.layout.world_count} entries")
        kernel = patch_trigger_bool_kernel if world_flags.dtype == wp.bool else patch_trigger_kernel
        wp.launch(kernel, dim=g.entries, inputs=[world_flags, g.world, g.fired, g.pending],
                  device=self.solver.device)

    def step(self, p: Particles, body_q: wp.array, body_qd: wp.array, body_com: wp.array,
             mass_in: wp.array, flux: tuple[wp.array, wp.array, wp.array]) -> None:
        """``flux`` = (gravity, flux_e, flux_g): the emitted energy goes into the flux ledger (``emitters.flux_add``)."""
        d = self.solver.device
        for g in self.groups:
            if g.entries == 0:
                continue
            wp.launch(patch_emit_kernel, dim=(g.entries, g.n_max), device=d,
                      inputs=[g.world, g.body, g.count, g.pending, g.fired, g.begin, g.pool_ids, g.points, g.velocity,
                              body_q, body_qd, body_com, p, mass_in, *flux])
            wp.launch(patch_fire_kernel, dim=g.entries, inputs=[g.pending, g.fired], device=d)

    def reset(self, mask: np.ndarray) -> None:
        """Re-arm (clear pending and fired) the entries of the masked worlds (bool [W])."""
        for g in self.groups:
            if g.entries == 0:
                continue
            world = g.world.numpy()
            for arr in (g.pending, g.fired):
                a = arr.numpy()
                a[mask[world]] = 0
                arr.assign(a)

    def arrays(self) -> dict[str, wp.array]:
        out = {}
        for k, g in enumerate(self.groups):
            if g.entries:
                out.update({f"sph.patch_pending{k}": g.pending, f"sph.patch_fired{k}": g.fired})
        return out

    def worlds(self) -> dict[str, wp.array]:
        out = {}
        for k, g in enumerate(self.groups):
            if g.entries:
                out.update({f"sph.patch_pending{k}": g.world, f"sph.patch_fired{k}": g.world})
        return out
