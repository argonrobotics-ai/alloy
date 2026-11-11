# SPDX-License-Identifier: Apache-2.0
"""Lumped bodies in DC5's pools (design note §4.9, Inflow 2; WP9 builds, WP15 reviews): a sponge or a tomato under a
pool's level fills from it and swaps water with it (GA ``pools._sponge_flooding``).

Each exchange (``lumped.py``), per world and body, one thread per world (no atomics, capture-safe):
- ``pool_book`` (before the lump step): a body that asked last exchange gets its share of its pool's confirmed take
  (``pool_took``, its ask over its pool's asks) and the take's species (``pool_took_sp``) in the same share; and each
  body's pool this exchange (``LumpState.pool_of``): its centre of mass over a pool's footprint and at least ``DEPTH``
  under its level (``supply.level``, the free surface's world height; all or nothing), so the lump step does not
  shed an immersed body's overflow as drops under the water; a body not immersed whose lowest point lies inside a
  cavity pool's vessel (a tomato in a colander: DC5's ``in_cavity``) is ``vessel_of`` it;
- ``pool_ask`` (after the lump step): a body in a vessel adds what is over its capacity to that pool (its holes then
  drain it, DC5's ``supply_holes``); an immersed body adds what is over its capacity to its pool, then adds
  ``vx = min(water, pool)·min(1, k_x·dt)`` of its water with its species
  (``k_x = k_mix0 + k_mix1·|v|``, v its speed relative to the pool's body) and asks to take back ``vx`` plus the fill
  ``min(cap - water, k_fill·cap·dt)``, at most what the pool will hold after the adds and takes asked before it;
- the supply (``supply_lump.pool_exchange``, its after_film) adds, then takes at the tank's concentration after the
  add, and confirms each pool's take.
Ledger per body: ``POOLED`` = taken - added [m³], ``sp_net`` the same for the species [kg]. A take confirmed but not
yet booked is ``pending`` (``LumpedStage.read``).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import point_velocity
from alloy.plugins.surface_flow.lumped_drip import LumpDrip, lowest
from alloy.plugins.surface_flow.lumped_kernels import POOLED, LumpSpec, LumpState
from alloy.plugins.surface_flow.supply_cavity import Cavity, in_cavity
from alloy.plugins.surface_flow.supply_kernels import PoolSpec, pool_frame

DEPTH = wp.constant(5.0e-3)     # m: a body's centre this far under a pool's level is immersed


@wp.struct
class PoolLink:
    pools: wp.array[PoolSpec]          # [P] the supply's pools
    pbody: wp.array2d[wp.int32]        # [W, P] each pool's body (-1: the world)
    level: wp.array2d[wp.float32]      # [W, P] world height of each pool's free surface (the supply's last)
    avail: wp.array2d[wp.float64]      # [W, P] m³ each tank holds after the supply's transfers this exchange
    add: wp.array2d[wp.float64]        # [W, P] m³ the bodies add (the supply zeroes it)
    add_sp: wp.array3d[wp.float64]     # [W, P, K] kg
    take: wp.array2d[wp.float64]       # [W, P] m³ the bodies ask to take
    took: wp.array2d[wp.float64]       # [W, P] m³ the supply took (last exchange's asks)
    took_sp: wp.array3d[wp.float64]    # [W, P, K] kg
    rate: wp.array[wp.vec3d]           # [B] (k_fill, k_mix0, k_mix1) [1/s, 1/s, 1/m]
    req: wp.array2d[wp.float64]        # [W, B] m³ each body asked to take
    req_pool: wp.array2d[wp.int32]     # [W, B] the pool it asked (-1: none)
    sp_net: wp.array3d[wp.float64]     # [W, B, K] kg pooled in, net
    cavs: wp.array[Cavity]             # [P] the pools' vessels (on 0: a box pool)
    cprof: wp.array[wp.vec2]


@wp.func
def immersed(pl: PoolLink, body_q: wp.array[wp.transform], w: int, p: int, x: wp.vec3) -> int:
    """1 when world point ``x`` lies over pool ``p``'s footprint, above its floor and ``DEPTH`` under its level."""
    pool = pl.pools[p]
    if x[2] > pl.level[w, p] - DEPTH:
        return 0
    q = wp.transform_point(wp.transform_inverse(pool_frame(body_q, pl.pbody[w, p])), x) - pool.floor
    if q[2] < -1.0e-6:
        return 0
    if pool.round != 0:
        return int(q[0] * q[0] + q[1] * q[1] <= pool.half[0] * pool.half[0])
    return int(wp.abs(q[0]) <= pool.half[0] and wp.abs(q[1]) <= pool.half[1])


@wp.func
def capacity(spec: LumpSpec, st: LumpState, w: int, b: int) -> wp.float64:
    """What body ``b`` holds: a sponge's pores at its strain, else its clinging film."""
    pore = spec.pore[b]
    if pore[0] > wp.float64(0.0):
        return pore[0] * (wp.float64(1.0) - st.eps[w, b])
    return spec.hold[b] * spec.area[b]


@wp.func
def rel_speed(pl: PoolLink, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
              body_com: wp.array[wp.vec3], w: int, p: int, bi: int, x: wp.vec3) -> wp.float64:
    """|v| of body ``bi``'s centre of mass ``x`` relative to the pool's body there."""
    v = point_velocity(body_qd, x, bi, x)
    pb = pl.pbody[w, p]
    if pb >= 0:
        v = v - point_velocity(body_qd, wp.transform_point(body_q[pb], body_com[pb]), pb, x)
    return wp.float64(wp.length(v))


@wp.func
def give_pool(st: LumpState, pl: PoolLink, w: int, b: int, p: int, vol: wp.float64, n_sp: int):
    """``vol`` of body b's water and its share of its species go into pool p (``pool_add``), booked pooled."""
    frac = vol / st.water[w, b]
    for k in range(n_sp):
        m = st.sp[w, b, k] * frac
        st.sp[w, b, k] = st.sp[w, b, k] - m
        pl.add_sp[w, p, k] = pl.add_sp[w, p, k] + m
        pl.sp_net[w, b, k] = pl.sp_net[w, b, k] - m
    st.water[w, b] = st.water[w, b] - vol
    st.ledger[w, b, POOLED] = st.ledger[w, b, POOLED] - vol
    pl.add[w, p] = pl.add[w, p] + vol


@wp.func
def vessel(pl: PoolLink, body_q: wp.array[wp.transform], w: int, x: wp.vec3) -> int:
    """The first cavity pool whose vessel holds world point x (inside its wall, above its floor, below its rim; DC5's
    ``in_cavity``), or -1."""
    found = int(-1)
    for p in range(pl.pools.shape[0]):
        if found < 0 and pl.cavs[p].on != 0:
            q = wp.transform_point(wp.transform_inverse(pool_frame(body_q, pl.pbody[w, p])), x) - pl.pools[p].floor
            if in_cavity(pl.cavs[p], pl.cprof, q, wp.vec3(0.0, 0.0, 1.0)) == 1:
                found = p
    return found


@wp.kernel
def pool_book(st: LumpState, pl: PoolLink, ld: LumpDrip, body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3],
              gravity: wp.array[wp.vec3], bodies: wp.array2d[wp.int32], n_sp: int):
    """Per world, before the lump step: each body's share of its pool's confirmed take (then the asks clear), and the
    pool each body is in this exchange (``st.pool_of``: the lump step leaves an immersed body's overflow to
    ``pool_ask``)."""
    w = wp.tid()
    B = pl.req.shape[1]
    for b in range(B):
        p = pl.req_pool[w, b]
        if p < 0:
            continue
        total = wp.float64(0.0)
        for c in range(B):
            if pl.req_pool[w, c] == p:
                total = total + pl.req[w, c]
        share = pl.req[w, b] / total
        got = pl.took[w, p] * share
        st.water[w, b] = st.water[w, b] + got
        st.ledger[w, b, POOLED] = st.ledger[w, b, POOLED] + got
        for k in range(n_sp):
            m = pl.took_sp[w, p, k] * share
            st.sp[w, b, k] = st.sp[w, b, k] + m
            pl.sp_net[w, b, k] = pl.sp_net[w, b, k] + m
    for b in range(B):
        pl.req[w, b] = wp.float64(0.0)
        pl.req_pool[w, b] = -1
        bi = bodies[w, b]
        x = wp.transform_point(body_q[bi], body_com[bi])
        st.pool_of[w, b] = -1
        for c in range(pl.pools.shape[0]):
            if st.pool_of[w, b] < 0 and immersed(pl, body_q, w, c, x) != 0:
                st.pool_of[w, b] = c
        st.vessel_of[w, b] = -1                       # not immersed: its lowest point inside a vessel (a colander)
        if st.pool_of[w, b] < 0:
            st.vessel_of[w, b] = vessel(pl, body_q, w, lowest(ld, body_q, w, b, wp.normalize(gravity[w])))


@wp.kernel
def pool_ask(spec: LumpSpec, st: LumpState, pl: PoolLink, body_q: wp.array[wp.transform],
             body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], bodies: wp.array2d[wp.int32],
             n_sp: int, dt: wp.float64):
    """Per world, each immersed body: its overflow into the pool, its swap's add and its take (module docstring)."""
    w = wp.tid()
    zero = wp.float64(0.0)
    for b in range(spec.area.shape[0]):
        p = st.pool_of[w, b]
        cap = capacity(spec, st, w, b)
        if p < 0:
            v = st.vessel_of[w, b]                                   # in a vessel: only its overflow goes in
            if v >= 0 and st.water[w, b] > cap:
                give_pool(st, pl, w, b, v, st.water[w, b] - cap, n_sp)
            continue
        if st.water[w, b] > cap:                                     # what the lump step left over capacity
            give_pool(st, pl, w, b, p, st.water[w, b] - cap, n_sp)
        room = pl.avail[w, p] + pl.add[w, p] - pl.take[w, p]        # the pool after the asks before this body's
        if room <= zero:
            continue
        bi = bodies[w, b]
        x = wp.transform_point(body_q[bi], body_com[bi])
        r = pl.rate[b]
        k_x = r[1] + r[2] * rel_speed(pl, body_q, body_qd, body_com, w, p, bi, x)
        have = st.water[w, b]
        vx = wp.min(have, room) * wp.min(wp.float64(1.0), k_x * dt)
        if vx > zero:
            give_pool(st, pl, w, b, p, vx, n_sp)
        fill = wp.max(wp.min(cap - have, r[0] * cap * dt), zero)
        t = wp.min(vx + fill, room + vx)
        if t > zero:
            pl.take[w, p] = pl.take[w, p] + t
            pl.req[w, b] = t
            pl.req_pool[w, b] = p


__all__ = ["DEPTH", "PoolLink", "capacity", "give_pool", "immersed", "pool_ask", "pool_book", "vessel"]
