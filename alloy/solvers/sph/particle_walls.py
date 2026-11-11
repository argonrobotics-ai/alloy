# SPDX-License-Identifier: Apache-2.0
"""Particle walls: another entry's particles (an MPM paste or powder) as a SolverDFSPH boundary (SCR-WP12-MIX).

Which particles. In a coupled scene MPM is the source and SPH the destination of a particle proxy mapping
(``SolverCoupledProxy.Proxy(source="mpm", destination="sph", particles=...)``). The SPH view flags the mapped
particles ``ParticleFlags.PROXY``; those with no soft face (``soft_walls.py``: no triangle of three proxies) are
particle walls. Each carries the cuboid volume Newton's MPM gives it, V = 8 r^3 (r its particle radius).

The slot. For a fluid particle at x the walls' colour is c(x) = sum_j V_j W(x - x_j) over its world's wall particles
within the support h (Akinci et al. 2012): the fraction of the kernel support the wall material covers. Deep inside
it is 1 and at a flat face it is Psi(d), the half-space volume at distance d (kernels.halfspace_volume). Inverting
Psi gives the equivalent distance phi; the outward normal is -grad c / |grad c|; the wall velocity is the
W-weighted mean of the v_j. The slot is then filled exactly as an SDF wall at phi (the fitted amplitude, offset and
taper of ``kernels.boundary_fit``), so the pressure solves, the projection, the wall viscosity and CSF read it like
any wall; it is the virtual shape ``base`` (after the soft components), with a free contact line and no seal.

Grid. The wall particles sit in SolverDFSPH's own neighbour hash grid in group -2 - world (``neighbors.py``; fluid
particles are in group world, every other particle in -1), so no second grid is built and fluid neighbour lists
never see them. The grid is built from the step's start positions, after the staggered back-step below.

Reaction. A slot's impulse J on the wall is split to the wall particles in its support by their weights
V_j W_ij / c_i, summed per particle in int64 fixed point (bodies.py's scale), the rounding remainder to the largest
share: the total is exactly J's fixed-point value (momentum conserved), deterministic (integer atomics only). The
window's sum over an exchange divided by dt_x is the proxy force the harvest returns, as for soft faces. Motion is
prescribed over the exchange as for soft faces (P1): every wall particle advances with its own velocity; after a
staggered sync the first step moves them back by v dt_x.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from .bodies import IMP_SCALE, IMP_SCALE_HOST, WORK_SCALE, _fx, _ufx
from .kernels import halfspace_volume, kernel_grad, kernel_w

log = logging.getLogger("alloy.sph")

WORK_SCALE_HOST = float(2**50)  # bodies.WORK_SCALE
C_MIN = 1.0e-4          # colour below which a particle has no particle-wall slot
BISECT = 30             # bisection steps of the colour -> distance inversion (h / 2^30)


@wp.struct
class ParticleWallArrays:
    base: int  # virtual shape id of the particle wall (soft.base + soft components)
    count: int  # 0: no particle walls, every branch is skipped
    grid: wp.uint64  # SolverDFSPH's neighbour grid (walls in group -2 - world)
    x: wp.array[wp.vec3]  # the SPH state's particle positions (view order)
    v: wp.array[wp.vec3]  # ... and velocities (prescribed over the step for wall particles)
    vol: wp.array[wp.float32]  # V_j = 8 r^3 for wall particles, 0 otherwise (view order)
    h: float
    kind: int


@wp.func
def wall_group(w: int) -> int:
    """Hash-grid group of world w's wall particles (neighbors.py)."""
    return -2 - w


@wp.func
def pw_sample(pw: ParticleWallArrays, x: wp.vec3, w: int):
    """(c, grad c, mean wall velocity) of world w's wall particles at x."""
    c = float(0.0)
    gc = wp.vec3(0.0, 0.0, 0.0)
    mv = wp.vec3(0.0, 0.0, 0.0)
    if pw.count == 0:
        return c, gc, mv
    query = wp.hash_grid_query(pw.grid, x, pw.h, wall_group(w))
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        d = x - pw.x[j]
        wj = pw.vol[j] * kernel_w(wp.length(d), pw.h, pw.kind)
        if wj > 0.0:
            c += wj
            gc += pw.vol[j] * kernel_grad(d, pw.h, pw.kind)
            mv += wj * pw.v[j]
    if c > 0.0:
        mv = mv / c
    return c, gc, mv


@wp.func
def colour_distance(c: float, h: float, kind: int) -> float:
    """The distance d at which a flat wall covers c of the support: Psi(d) = c (Psi falls from 1 at -h to 0 at h)."""
    lo = -h
    hi = h
    for _k in range(BISECT):
        mid = 0.5 * (lo + hi)
        if halfspace_volume(mid, h, 0.0, kind)[0] > c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@wp.func
def pw_candidate(pw: ParticleWallArrays, x: wp.vec3, w: int) -> float:
    """phi of the particle wall at x, or 1e30 when it covers nothing there."""
    c, gc, _mv = pw_sample(pw, x, w)
    if c < C_MIN or wp.length(gc) <= 0.0:
        return 1.0e30
    return colour_distance(wp.min(c, 1.0), pw.h, pw.kind)


@wp.func
def pw_plane(pw: ParticleWallArrays, x: wp.vec3, w: int):
    """(phi, outward n, x_b, v_b) of the particle wall's local plane at x."""
    c, gc, mv = pw_sample(pw, x, w)
    lg = wp.length(gc)
    n = wp.vec3(0.0, 0.0, 1.0)
    if lg > 0.0:
        n = -gc / lg
    phi = colour_distance(wp.clamp(c, 0.0, 1.0), pw.h, pw.kind)
    return phi, n, x - phi * n, mv


@wp.func
def pw_project(xb: wp.vec3, n: wp.vec3, vb: wp.vec3, x: wp.vec3, dt: float, eps: float):
    """(x, pushed): x out of the slot's start-of-step plane advanced by dt with the wall velocity, to eps."""
    d = wp.dot(x - (xb + dt * vb), n)
    if d < eps:
        return x + (eps - d) * n, 1
    return x, 0


@wp.kernel(enable_backward=False)
def _advance_kernel(vid: wp.array[wp.int32], dt: float, q: wp.array[wp.vec3], qd: wp.array[wp.vec3]):
    k = wp.tid()
    i = vid[k]
    q[i] = q[i] + dt * qd[i]


@wp.kernel(enable_backward=False)
def _accumulate_kernel(q: wp.array[wp.vec3], world: wp.array[wp.int32], slot_shape: wp.array2d[wp.int32],
                       slot_vb: wp.array2d[wp.vec3], slot_impulse: wp.array2d[wp.vec3], pw: ParticleWallArrays,
                       cidx: wp.array[wp.int32], one_way: int, exch: wp.array2d[wp.int64],
                       work: wp.array[wp.int64]):
    """Split each particle-wall slot's impulse to its wall particles by V_j W_ij / c_i (module doc). The shares are
    rounded to fixed point and the rounding remainder goes to the largest share, so the int64 sum is exactly the
    slot's fixed-point impulse."""
    i, k = wp.tid()
    if slot_shape[i, k] != pw.base:
        return
    J = slot_impulse[i, k]
    if J[0] == 0.0 and J[1] == 0.0 and J[2] == 0.0:
        return
    w = world[i]
    wp.atomic_add(work, w, _fx(-wp.dot(J, slot_vb[i, k]), WORK_SCALE))
    if one_way != 0:
        return
    x = q[i]
    c, _gc, _mv = pw_sample(pw, x, w)
    if c <= 0.0:
        return
    left = wp.vec3l(_fx(J[0], IMP_SCALE), _fx(J[1], IMP_SCALE), _fx(J[2], IMP_SCALE))
    best = int(-1)
    sbest = float(0.0)
    query = wp.hash_grid_query(pw.grid, x, pw.h, wall_group(w))
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        s = pw.vol[j] * kernel_w(wp.length(x - pw.x[j]), pw.h, pw.kind) / c
        if s > 0.0:
            for d in range(3):
                part = _fx(J[d] * s, IMP_SCALE)
                wp.atomic_add(exch, cidx[j], d, part)
                left[d] = left[d] - part
            if s > sbest:
                sbest = s
                best = cidx[j]
    if best >= 0:
        for d in range(3):
            wp.atomic_add(exch, best, d, left[d])


@wp.kernel(enable_backward=False)
def _harvest_kernel(vid: wp.array[wp.int32], local_to_global: wp.array[int], exch: wp.array2d[wp.int64], dt: float,
                    out: wp.array[wp.vec3]):
    k = wp.tid()
    g = local_to_global[vid[k]]
    if g < 0:
        return
    out[g] = wp.vec3(_ufx(exch[k, 0], IMP_SCALE), _ufx(exch[k, 1], IMP_SCALE), _ufx(exch[k, 2], IMP_SCALE)) / dt


def _dummy(device, dtype):
    return wp.zeros(1, dtype=dtype, device=device)


def empty_walls(base: int, h: float, kind: int, device) -> ParticleWallArrays:
    a = ParticleWallArrays()
    a.base, a.count, a.grid, a.h, a.kind = int(base), 0, wp.uint64(0), float(h), int(kind)
    a.x, a.v, a.vol = _dummy(device, wp.vec3), _dummy(device, wp.vec3), _dummy(device, wp.float32)
    return a


def wall_particles(model, soft_vid: np.ndarray | None) -> np.ndarray:
    """Host: PROXY particles of the view that are not soft-face vertices (module doc)."""
    P = int(model.particle_count)
    if P == 0:
        return np.zeros(0, np.int64)
    proxy = np.flatnonzero((model.particle_flags.numpy() & int(ParticleFlags.PROXY)) != 0)
    if soft_vid is not None and len(soft_vid):
        proxy = np.setdiff1d(proxy, np.asarray(soft_vid, dtype=np.int64))
    return proxy.astype(np.int64)


class ParticleWalls:
    """Particle walls of one SolverDFSPH: the wall set and its group mask, per-step binding, advance, reactions."""

    def __init__(self, model, base: int, h: float, kind: int, device, soft_vid=None):
        self.device = device
        self.arrays = a = empty_walls(base, h, kind, device)
        vid = wall_particles(model, soft_vid)
        self.count = int(vid.size)
        self.mask = None  # per view particle: 1 for a wall particle (neighbors.NeighborLists.set_walls)
        self._shapes = None
        self.work = wp.zeros(max(int(model.world_count), 1), dtype=wp.int64, device=device)
        if not self.count:
            return
        P = int(model.particle_count)
        r = model.particle_radius.numpy()[vid].astype(np.float64)
        vol, mask, cidx = np.zeros(P, np.float32), np.zeros(P, np.int32), np.full(P, -1, np.int32)
        vol[vid], mask[vid], cidx[vid] = (8.0 * r**3).astype(np.float32), 1, np.arange(self.count, dtype=np.int32)
        self.vid = wp.array(vid.astype(np.int32), dtype=wp.int32, device=device)
        self.cidx = wp.array(cidx, dtype=wp.int32, device=device)
        self.mask = wp.array(mask, dtype=wp.int32, device=device)
        a.count = self.count
        a.vol = wp.array(vol, dtype=wp.float32, device=device)
        self.exch = wp.zeros((self.count, 3), dtype=wp.int64, device=device)
        worlds = np.unique(model.particle_world.numpy()[vid])
        if worlds.size and worlds[0] < 0:
            raise ValueError(f"SolverDFSPH: {int((model.particle_world.numpy()[vid] < 0).sum())} particle-wall "
                             "particles are in no world (world -1); a wall needs the world of the liquid it bounds")
        log.info("SolverDFSPH: %d particle-wall particles (another entry's proxies, SCR-WP12-MIX) in %d worlds",
                 self.count, len(worlds))

    def bind(self, shapes) -> None:
        """The ShapeArrays the boundary pass reads (bound again after notify_model_changed rebuilds them)."""
        self._shapes = shapes
        shapes.pw = self.arrays

    def begin(self, state, grid_id) -> None:
        """Step start: point the walls at this step's state and the neighbour grid just built from it."""
        if not self.count:
            return
        a = self.arrays
        a.x, a.v, a.grid = state.particle_q, state.particle_qd, grid_id
        self._shapes.pw = a
        self.work.zero_()

    def advance(self, state, dt: float) -> None:
        if self.count:
            wp.launch(_advance_kernel, dim=self.count, inputs=[self.vid, float(dt), state.particle_q,
                                                               state.particle_qd], device=self.device)

    def accumulate(self, q, world, slots, slot_impulse, n: int, one_way: bool) -> None:
        if self.count and n:
            wp.launch(_accumulate_kernel, dim=(n, slots.shape.shape[1]), device=self.device,
                      inputs=[q, world, slots.shape, slots.vb, slot_impulse, self.arrays, self.cidx, int(one_way),
                              self.exch, self.work])

    def new_window(self) -> None:
        if self.count:
            self.exch.zero_()

    def harvest(self, local_to_global, out, dt: float) -> None:
        if self.count:
            wp.launch(_harvest_kernel, dim=self.count, inputs=[self.vid, local_to_global, self.exch, float(dt), out],
                      device=self.device)

    def world_work(self) -> np.ndarray:
        """Host: the last step's work [J] of the walls on the liquid, per world (soft_walls.component_work's twin)."""
        return self.work.numpy().astype(np.float64) / WORK_SCALE_HOST if self.count else np.zeros(0)

    def particle_impulse(self) -> np.ndarray:
        """Host: the exchange window's impulse [N s] per wall particle (compact order)."""
        return self.exch.numpy().astype(np.float64) / IMP_SCALE_HOST if self.count else np.zeros((0, 3))
