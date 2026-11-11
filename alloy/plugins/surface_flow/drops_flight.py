# SPDX-License-Identifier: Apache-2.0
"""Ballistic drops of the drop stage (``drops.py``; design §4.7): the pool of drops in flight, per world.

- ``spawn_kernel`` (world): granted requests take free slots in order; each records its slot (``Requests.slot``) and
  the lumped body it came from (``Pool.lump``, -1 for a cell's drop), so ``drops_kernels.commit_kernel`` can give a
  lump's drop its share of the waiting species (``Pool.sp``, kg per drop).
- ``pool_full_kernel`` (world): the full-pool and request-cap counters.
- ``advance_kernel`` (world, at the start of the update): each drop flies one step, exact under constant gravity.
  A drop whose step ends in a pool's water joins that pool's tank (``to_tanks``) with its species (DC5's
  ``pool_add_sp``); one that lands on a chart gives the cell its water (``drops_kernels.arrive``) and its species
  (``src_sp``); one below ``floor_z`` is booked floored. Per world, the species book as the water does:
  ``landed_sp``, ``floored_sp``, ``tanks_sp`` [W, K] kg, and in flight in ``sp``; a lump's own books keep them in its
  ``dropped`` (its ``tanks`` are its drips that start in a pool's water, as its water's ``lump_drip_tanks``).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import point_velocity
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.drops_kernels import Arrive, Requests, arrive
from alloy.plugins.surface_flow.drops_pools import DropPools, into_pool
from alloy.plugins.surface_flow.geometry import chart_xform, ray_hit


@wp.struct
class Pool:
    pos: wp.array2d[wp.vec3]        # [W, P]
    vel: wp.array2d[wp.vec3]
    alive: wp.array2d[wp.int32]
    landed: wp.array[wp.float64]    # [W] m³
    floored: wp.array[wp.float64]
    lump: wp.array2d[wp.int32]      # [W, P] the lumped body a drop came from (-1: a cell's)
    sp: wp.array3d[wp.float64]      # [W, P, K] kg of species a lump's drop carries
    n_sp: int
    src_sp: wp.array3d[wp.float64]  # [W, C, K] the film's pending species source [kg/m²]
    landed_sp: wp.array2d[wp.float64]   # [W, K] kg the drops gave the cells they landed on
    floored_sp: wp.array2d[wp.float64]  # [W, K] kg fallen below floor_z with them
    tanks_sp: wp.array2d[wp.float64]    # [W, K] kg they took into pools' tanks


@wp.kernel
def spawn_kernel(req: Requests, pool: Pool, granted: wp.array[wp.int32]):
    """Per world: requests into free pool slots, in order; ``granted[w]`` = how many found one."""
    w = wp.tid()
    s = int(0)
    for p in range(pool.alive.shape[1]):
        if s >= req.n[w]:
            break
        if pool.alive[w, p] == 0:
            pool.pos[w, p] = req.pos[w, s]
            pool.vel[w, p] = req.vel[w, s]
            pool.alive[w, p] = 1
            c = req.cell[w, s]
            pool.lump[w, p] = -1
            if c < 0:
                pool.lump[w, p] = -c - 1
            for k in range(pool.n_sp):
                pool.sp[w, p, k] = wp.float64(0.0)
            req.slot[w, s] = p
            s += 1
    granted[w] = s


@wp.kernel
def pool_full_kernel(req: Requests, pool: Pool, granted: wp.array[wp.int32], cap: int, full: wp.array[wp.int32],
                     need: wp.array[wp.int32], req_full: wp.array[wp.int32]):
    """Per world: an exchange whose requests did not all find a free slot counts in ``full``; ``need`` keeps the most
    drops the pool was asked to hold (in flight plus asked for); one that filled every request slot (more drips held
    back) counts in ``req_full``."""
    w = wp.tid()
    if req.n[w] >= cap:
        req_full[w] = req_full[w] + 1
    alive = int(0)
    for p in range(pool.alive.shape[1]):
        alive += pool.alive[w, p]
    n = req.n[w]
    if granted[w] < n:
        full[w] = full[w] + 1
    need[w] = wp.max(need[w], alive + n - granted[w])


@wp.func
def into_tank(pool: Pool, dp: DropPools, w: int, p: int, q: int):
    """Drop p's species join pool q's tank (booked ``tanks_sp``)."""
    for k in range(pool.n_sp):
        m = pool.sp[w, p, k]
        dp.add_sp[w, q, k] = dp.add_sp[w, q, k] + m
        pool.tanks_sp[w, k] = pool.tanks_sp[w, k] + m
        pool.sp[w, p, k] = wp.float64(0.0)


@wp.func
def onto_cell(pool: Pool, area: wp.array[wp.float64], w: int, p: int, c: int):
    """Drop p's species join cell c's pending source (booked ``landed_sp``)."""
    for k in range(pool.n_sp):
        m = pool.sp[w, p, k]
        pool.src_sp[w, c, k] = pool.src_sp[w, c, k] + m / area[c]
        pool.landed_sp[w, k] = pool.landed_sp[w, k] + m
        pool.sp[w, p, k] = wp.float64(0.0)


@wp.func
def floored(pool: Pool, w: int, p: int):
    """Drop p's species leave with it below ``floor_z`` (booked ``floored_sp``)."""
    for k in range(pool.n_sp):
        pool.floored_sp[w, k] = pool.floored_sp[w, k] + pool.sp[w, p, k]
        pool.sp[w, p, k] = wp.float64(0.0)


@wp.kernel
def advance_kernel(pool: Pool, vol: wp.float64, rho: float, gravity: wp.array[wp.vec3], floor_z: float, dt: float,
                   charts: wp.array[Chart], prof: wp.array[wp.vec3], cbody: wp.array2d[wp.int32],
                   clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                   body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], a: Arrive, dp: DropPools,
                   to_tanks: wp.array[wp.float64]):
    """Per world: each drop flies one step (exact under constant gravity) and lands on the first chart its chord hits,
    on the floor below ``floor_z`` (booked), or flies on. The impulse uses the velocity at the hit. A full landing cell
    stops the drop at the chart, moving with it, to try again next step (it does not pass through). A drop whose step
    ends in a pool's water (on a flooded chart cell too) joins that pool's tank (``to_tanks``), species and all."""
    w = wp.tid()
    g = gravity[w]
    for p in range(pool.alive.shape[1]):
        if pool.alive[w, p] == 0:
            continue
        x0 = pool.pos[w, p]
        d = dt * pool.vel[w, p] + (0.5 * dt * dt) * g          # exact for constant gravity
        v = pool.vel[w, p] + dt * g
        length = wp.length(d)
        best_t = float(length)
        best_k = int(-1)
        best_c = int(-1)
        best_n = wp.vec3(0.0, 0.0, 1.0)
        if length > 0.0:
            for k in range(charts.shape[0]):
                X = chart_xform(body_q, cbody[w, k], clocal[w, k])
                hit = ray_hit(charts[k], prof, X, x0, d / length)
                if hit.cell >= 0 and hit.t <= best_t:
                    best_t = float(hit.t)
                    best_k = k
                    best_c = hit.cell
                    best_n = hit.normal
        xe = x0 + d
        if best_k >= 0:
            xe = x0 + best_t * d / length
        pin = into_pool(dp, w, xe, body_q)
        if pin >= 0:
            dp.add[w, pin] = dp.add[w, pin] + vol
            to_tanks[w] = to_tanks[w] + vol
            into_tank(pool, dp, w, p, pin)
            pool.alive[w, p] = 0
            continue
        if best_k >= 0:
            x = x0 + best_t * d / length
            cb = cbody[w, best_k]
            v_hit = pool.vel[w, p] + (dt * best_t / length) * g      # the chord's parameter as the time fraction
            v_chart = wp.vec3(0.0, 0.0, 0.0)
            if cb >= 0:
                v_chart = point_velocity(body_qd, wp.transform_point(body_q[cb], body_com[cb]), cb, x)
            Xk = chart_xform(body_q, cb, clocal[w, best_k])
            if arrive(a, w, best_k, best_c, vol, wp.float32(wp.float64(rho) * vol), v_hit - v_chart, best_n, x, Xk):
                onto_cell(pool, a.area, w, p, best_c)
                pool.alive[w, p] = 0
                pool.landed[w] = pool.landed[w] + vol
                continue
            pool.pos[w, p] = x + 1.0e-6 * best_n                    # full cell: wait on the chart
            pool.vel[w, p] = v_chart
            continue
        x1 = x0 + d
        if x1[2] < floor_z:
            floored(pool, w, p)
            pool.alive[w, p] = 0
            pool.floored[w] = pool.floored[w] + vol
            continue
        pool.pos[w, p] = x1
        pool.vel[w, p] = v


__all__ = ["Pool", "advance_kernel", "floored", "into_tank", "onto_cell", "pool_full_kernel", "spawn_kernel"]
