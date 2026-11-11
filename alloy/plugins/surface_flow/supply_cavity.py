# SPDX-License-Identifier: Apache-2.0
"""Cavity pools ([internal note] notes/dc5-sf-pools-proposal.md, part 2): a pool that is the inside of a vessel on a body,
such as a glass, cup or bowl, with its surface normal to gravity at any tilt.

The cavity is the solid of revolution r <= r(z) about the pool body's z axis through ``floor``, from the floor (the
first profile node's z) up to the rim (the last node's z). Its volume below a plane n·x = h (n the body-frame up) is
a quadrature over ``slices`` discs, each cut by the plane along a chord. Per exchange and world, ``cavity_state``:
- finds the surface height h holding the tank's volume, by bisection;
- finds the capacity at the tilt: the volume below the plane through the rim's lowest point;
- finds the trap capacity: below the plane through the rim's highest point (air above it cannot leave, so a dunked
  inverted glass stays dry);
- finds the rim's lowest point and its wetted length, and publishes the world height of the surface.

``cavity_flows`` (one thread per world, cavity pools in order):
- **Spill.** What stands above the rim's lowest point leaves over the wetted rim as a sharp-crested weir,
  Q = (2/3)·Cd·L·sqrt(2g)·H^(3/2) (Cd ≈ 0.62, Rehbock), at most the excess over the capacity per exchange. It goes
  on the pool's ``spill`` source edge (its opening is set here). A ``Nozzle(rim_of=...)`` on that edge pours it from
  the rim's lowest point, outward and level, so the network books it and the jet path carries it to charts, pools or
  ``lost``.
- **Weir.** A vessel whose rim's lowest point is in its ``weir`` pool's water exchanges water with it over the rim,
  with the same law. It is drowned by the lower side's head (Villemonte 1947: Q·(1 - (H2/H1)^(3/2))^0.385), at most
  the volume that levels them, and inflow stops at the trap capacity. While the rim is under the other pool's
  surface the weir replaces the spill.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.supply_deposit import node_conc
from alloy.plugins.surface_flow.supply_kernels import WEIR, PoolSpec, pool_frame
from alloy.plugins.surface_flow.supply_species import NetSpecies

WEIR_N = wp.constant(32)        # midpoints along half a wetted rim
PI = wp.constant(3.14159265358979)


@wp.struct
class Cavity:
    on: wp.int32
    s0: wp.int32              # slices [s0, s0 + ns): (z_a, z_b, r) relative to the pool's floor point
    ns: wp.int32
    p0: wp.int32              # profile nodes [p0, p0 + np): (r, z)
    np: wp.int32
    z0: wp.float32            # floor and rim heights, rim and largest radius
    z_rim: wp.float32
    r_rim: wp.float32
    r_max: wp.float32
    vol: wp.float64
    tol: wp.float32
    cd: wp.float32
    weir: wp.vec4i            # the pools it may exchange with over the rim, in order (-1 past n_weir)
    n_weir: wp.int32
    spill: wp.int32           # its spill source edge, or -1
    q_max: wp.float64         # that edge's |flow| at full opening
    noz: wp.int32             # its rim nozzle and that nozzle's ray, or -1
    ray: wp.int32


@wp.func
def disc_below(r: wp.float64, d: wp.float64) -> wp.float64:
    """Area of a disc of radius r on the side x <= d of a chord at signed distance d from its centre."""
    if d >= r:
        return wp.float64(PI) * r * r
    if d <= -r:
        return wp.float64(0.0)
    return wp.float64(PI) * r * r - (r * r * wp.acos(d / r) - d * wp.sqrt(r * r - d * d))


@wp.func
def disc_below_integral(r: wp.float64, t: wp.float64) -> wp.float64:
    """∫ disc_below(r, τ) dτ from -r to t: t·A(t) + (2/3)·(r² - t²)^(3/2) (0 below -r, t·πr² above r)."""
    if t <= -r:
        return wp.float64(0.0)
    if t >= r:
        return t * wp.float64(PI) * r * r
    q = r * r - t * t
    return t * disc_below(r, t) + wp.float64(2.0 / 3.0) * q * wp.sqrt(q)


@wp.func
def slab_below(za: wp.float64, zb: wp.float64, r: wp.float64, up: wp.vec3d, h: wp.float64) -> wp.float64:
    """Volume of the cylinder slab r, za <= z <= zb, below the plane up·x = h: exact for a tilted plane (the chord
    area integrated in closed form), a filled height when level."""
    s = wp.sqrt(up[0] * up[0] + up[1] * up[1])
    if s < wp.float64(1.0e-6):                         # level (upright or inverted)
        if up[2] > wp.float64(0.0):
            return wp.float64(PI) * r * r * wp.clamp(h / up[2] - za, wp.float64(0.0), zb - za)
        return wp.float64(PI) * r * r * wp.clamp(zb - h / up[2], wp.float64(0.0), zb - za)
    if wp.abs(up[2]) < wp.float64(1.0e-6):             # on its side: the same chord all along
        return disc_below(r, h / s) * (zb - za)
    ta = (h - up[2] * za) / s
    tb = (h - up[2] * zb) / s
    return s / up[2] * (disc_below_integral(r, ta) - disc_below_integral(r, tb))


@wp.func
def volume_below(cav: Cavity, slices: wp.array[wp.vec3d], up: wp.vec3d, h: wp.float64) -> wp.float64:
    """The cavity's volume below the plane up·x = h (x from the floor point, body frame)."""
    v = wp.float64(0.0)
    for k in range(cav.ns):
        sl = slices[cav.s0 + k]
        v = v + slab_below(sl[0], sl[1], sl[2], up, h)
    return v


@wp.func
def level_for(cav: Cavity, slices: wp.array[wp.vec3d], up: wp.vec3d, vol: wp.float64) -> wp.float64:
    """The plane height h whose volume below is ``vol`` (bisection, 60 halvings); over the whole cavity, the excess
    stands over its mouth (π·r_rim², the spill's head: an upright vessel filled past its rim spills)."""
    s = wp.sqrt(up[0] * up[0] + up[1] * up[1])
    z0 = wp.float64(cav.z0)
    z1 = wp.float64(cav.z_rim)
    lo = wp.min(up[2] * z0, up[2] * z1) - s * wp.float64(cav.r_max)
    hi = wp.max(up[2] * z0, up[2] * z1) + s * wp.float64(cav.r_max)
    if vol <= wp.float64(0.0):
        return lo
    if vol >= cav.vol:
        return hi + (vol - cav.vol) / (wp.float64(PI) * wp.float64(cav.r_rim) * wp.float64(cav.r_rim))
    for _ in range(60):
        mid = wp.float64(0.5) * (lo + hi)
        if volume_below(cav, slices, up, mid) < vol:
            lo = mid
        else:
            hi = mid
    return wp.float64(0.5) * (lo + hi)


@wp.func
def radius_at(cav: Cavity, prof: wp.array[wp.vec2], z: float) -> float:
    """The inner wall's radius at height z (the end nodes' radii beyond them)."""
    r = prof[cav.p0][0]
    for k in range(cav.np - 1):
        a = prof[cav.p0 + k]
        b = prof[cav.p0 + k + 1]
        if z >= a[1] and z <= b[1]:
            r = a[0] + (b[0] - a[0]) * (z - a[1]) / wp.max(b[1] - a[1], 1.0e-12)
    if z > prof[cav.p0 + cav.np - 1][1]:
        r = prof[cav.p0 + cav.np - 1][0]
    return r


@wp.func
def in_cavity(cav: Cavity, prof: wp.array[wp.vec2], q: wp.vec3, n: wp.vec3) -> int:
    """1 when a cell at q (from the floor point, body frame; unit normal n, body frame) lies inside the cavity: within
    the wall, or within ``tol`` beyond the inner wall or below the floor when it faces into the vessel (a thin
    vessel's outer wall and underside face out)."""
    if q[2] < cav.z0 - cav.tol or q[2] > cav.z_rim + cav.tol:
        return 0
    r = wp.length(wp.vec2(q[0], q[1]))
    rw = radius_at(cav, prof, q[2])
    if r > rw + cav.tol:
        return 0
    if r > rw and n[0] * q[0] + n[1] * q[1] >= 0.0:      # beyond the inner wall, facing away from the axis
        return 0
    if q[2] < cav.z0 and n[2] <= 0.0:                    # below the floor, facing down
        return 0
    return 1


@wp.func
def body_up(F: wp.transform, gravity: wp.vec3) -> wp.vec3:
    """The world's up (against gravity; the body's z without gravity) in the body frame."""
    g = wp.length(gravity)
    if g <= 0.0:
        return wp.vec3(0.0, 0.0, 1.0)
    return wp.transform_vector(wp.transform_inverse(F), -gravity / g)


@wp.func
def rim_point(cav: Cavity, up: wp.vec3, low: int) -> wp.vec3:
    """The rim's lowest (``low`` = 1) or highest point along the world's up, from the floor point, body frame."""
    s = wp.sqrt(up[0] * up[0] + up[1] * up[1])
    dx = float(1.0)
    dy = float(0.0)
    if s >= 1.0e-7:
        dx = up[0] / s
        dy = up[1] / s
    if low == 1:
        dx = -dx
        dy = -dy
    return wp.vec3(cav.r_rim * dx, cav.r_rim * dy, cav.z_rim)


@wp.func
def weir_unit(cd: float, g: float, h1: float, h2: float) -> float:
    """A sharp-crested weir per unit crest length [m²/s] at head h1, drowned by the far side's head h2 (Villemonte
    1947): (2/3)·Cd·sqrt(2g)·h1^(3/2)·(1 - (h2/h1)^(3/2))^0.385."""
    if h1 <= 0.0:
        return 0.0
    q = 2.0 / 3.0 * cd * wp.sqrt(2.0 * g) * wp.pow(h1, 1.5)
    if h2 > 0.0:
        q = q * wp.pow(wp.max(1.0 - wp.pow(wp.min(h2 / h1, 1.0), 1.5), 0.0), 0.385)
    return q


@wp.func
def rim_weir(cd: float, g: float, r: float, mid: float, amp: float, z1: float, z2: float) -> float:
    """The weir [m³/s] over a rim circle of radius r whose points stand at mid + amp·cos φ, from water at z1 to water
    at z2: the local heads z1 - z(φ) and z2 - z(φ) integrated along the wetted arc (midpoint rule, WEIR_N points a
    side; the head varies along a tilted rim)."""
    if amp < 1.0e-9:
        return 2.0 * PI * r * weir_unit(cd, g, z1 - mid, z2 - mid)
    a = wp.acos(wp.clamp((z1 - mid) / amp, -1.0, 1.0))      # wetted for φ in [a, π], and its mirror image
    d = (PI - a) / float(WEIR_N)
    q = float(0.0)
    for i in range(WEIR_N):
        z = mid + amp * wp.cos(a + (float(i) + 0.5) * d)
        q = q + weir_unit(cd, g, z1 - z, z2 - z)
    return 2.0 * r * d * q


@wp.func
def available(pool: PoolSpec, w: int, pressure: wp.array2d[wp.float64], inflow: wp.array2d[wp.float64],
              dt: wp.float64) -> float:
    """A tank's volume after what this exchange already booked into or out of it (withdrawal films, other
    cavities' weirs): what the next transfer may take."""
    v = pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0))
    return float(wp.max(v + inflow[w, pool.tank] * dt, wp.float64(0.0)))


@wp.func
def weir_species(ns: NetSpecies, n_net: int, w: int, tank: int, other: int, q: wp.float64):
    """The network's species ride a weir's flow q [m³/s] (> 0: into ``tank`` from ``other``) at the giving tank's
    concentration."""
    src = other
    if q < wp.float64(0.0):
        src = tank
    for k in range(n_net):
        m = q * node_conc(w, src, k, ns.medium, ns.bulk, ns.p_ref, ns.poly, ns.pressure, ns.volume, ns.species)
        ns.inflow[w, tank, k] = ns.inflow[w, tank, k] + m
        ns.inflow[w, other, k] = ns.inflow[w, other, k] - m


@wp.kernel
def cavity_state(body_q: wp.array[wp.transform], pools: wp.array[PoolSpec], cavs: wp.array[Cavity],
                 slices: wp.array[wp.vec3d], pbody: wp.array2d[wp.int32], pressure: wp.array2d[wp.float64],
                 gravity: wp.vec3, surf: wp.array2d[wp.vec4], cap: wp.array2d[wp.vec2d],
                 level: wp.array2d[wp.float32]):
    """Per world and cavity pool: ``surf`` = (body-frame up, h), ``cap`` = (spill capacity, trap capacity) [m³],
    and ``level`` = the world height of its surface (when empty, of a plane at or below its lowest point)."""
    w, p = wp.tid()
    cav = cavs[p]
    if cav.on == 0:
        return
    pool = pools[p]
    F = pool_frame(body_q, pbody[w, p])
    u = body_up(F, gravity)
    up = wp.vec3d(wp.float64(u[0]), wp.float64(u[1]), wp.float64(u[2]))
    up = up / wp.length(up)
    vol = pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0))
    h = level_for(cav, slices, up, vol)
    surf[w, p] = wp.vec4(u[0], u[1], u[2], float(h))
    s = wp.sqrt(up[0] * up[0] + up[1] * up[1]) * wp.float64(cav.r_rim)
    z = up[2] * wp.float64(cav.z_rim)
    cap[w, p] = wp.vec2d(volume_below(cav, slices, up, z - s), volume_below(cav, slices, up, z + s))
    level[w, p] = wp.transform_point(F, pool.floor + u * float(h))[2]


@wp.func
def box_surface(pool: PoolSpec, F: wp.transform, pressure: wp.array2d[wp.float64], w: int, g_up: wp.vec3) -> float:
    """The height along ``g_up`` of a box pool's surface (its footprint's centre at its level, along its body's z)."""
    level = float(pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0)) / pool.area)
    return wp.dot(g_up, wp.transform_point(F, pool.floor + wp.vec3(0.0, 0.0, level)))


@wp.func
def in_footprint(pool: PoolSpec, F: wp.transform, x: wp.vec3) -> int:
    q = wp.transform_point(wp.transform_inverse(F), x) - pool.floor
    return int(wp.abs(q[0]) <= pool.half[0] and wp.abs(q[1]) <= pool.half[1] and q[2] >= 0.0)


@wp.kernel
def cavity_flows(dt: wp.float64, body_q: wp.array[wp.transform], pools: wp.array[PoolSpec],
                 cavs: wp.array[Cavity], pbody: wp.array2d[wp.int32], pressure: wp.array2d[wp.float64],
                 gravity: wp.vec3, surf: wp.array2d[wp.vec4], cap: wp.array2d[wp.vec2d],
                 ns: NetSpecies, n_net: int, opening: wp.array2d[wp.float32],
                 npos: wp.array2d[wp.vec3], ray_dir: wp.array2d[wp.vec3], inflow: wp.array2d[wp.float64],
                 ledger: wp.array2d[wp.float64]):
    """Per world, the cavity pools in order: the rim pour's pose, the weir with the pool around the rim, or else the
    spill over the rim (module docstring). Each takes at most what its tank has after the transfers booked before it
    this exchange (``available``), and the weir carries the network's species."""
    w = wp.tid()
    g = wp.length(gravity)
    g_up = wp.vec3(0.0, 0.0, 1.0)                        # heights along the world's up (against gravity)
    if g > 0.0:
        g_up = -gravity / g
    for p in range(cavs.shape[0]):
        cav = cavs[p]
        if cav.on == 0:
            continue
        pool = pools[p]
        F = pool_frame(body_q, pbody[w, p])
        sv = surf[w, p]
        up = wp.vec3(sv[0], sv[1], sv[2])
        vol = float(pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0)))
        have = available(pool, w, pressure, inflow, dt)
        low = rim_point(cav, up, 1)
        x_low = wp.transform_point(F, pool.floor + low)
        x_high = wp.transform_point(F, pool.floor + rim_point(cav, up, 0))
        if cav.noz >= 0:                              # the pour leaves the rim's lowest point, outward and level
            out = wp.vec3(low[0], low[1], 0.0)
            out = out - up * wp.dot(up, out)
            npos[w, cav.noz] = pool.floor + low
            ray_dir[w, cav.ray] = wp.normalize(out + wp.vec3(1.0e-9, 0.0, 0.0))
        drowned = int(0)
        z_low = wp.dot(g_up, x_low)
        z_high = wp.dot(g_up, x_high)
        o = int(-1)                                   # the first listed pool whose water holds the rim's lowest point
        z_o = float(0.0)
        for kw in range(cav.n_weir):
            cand = cav.weir[kw]
            Fc = pool_frame(body_q, pbody[w, cand])
            zc = box_surface(pools[cand], Fc, pressure, w, g_up)
            if o < 0 and in_footprint(pools[cand], Fc, x_low) == 1 and zc > z_low:
                o = cand
                z_o = zc
        if o >= 0:
            other = pools[o]
            drowned = 1
            z_c = wp.dot(g_up, wp.transform_point(F, pool.floor + up * sv[3]))    # its own surface
            if vol <= 0.0:
                z_c = z_low
            mid = 0.5 * (z_low + z_high)
            amp = 0.5 * (z_high - z_low)
            vo = available(other, w, pressure, inflow, dt)
            a_c = PI * cav.r_rim * cav.r_rim
            share = a_c * float(other.area) / (a_c + float(other.area))
            q = float(0.0)
            if z_o > z_c:                         # in over the rim, up to the trap capacity and the levels
                q = rim_weir(cav.cd, g, cav.r_rim, mid, amp, z_o, z_c)
                q = wp.min(q, wp.min(wp.max(float(cap[w, p][1]) - have, 0.0), wp.min(vo, (z_o - z_c) * share)) /
                           float(dt))
            elif z_c > z_low:                     # back out over the rim, down to the other pool's level
                q = -rim_weir(cav.cd, g, cav.r_rim, mid, amp, z_c, z_o)
                q = wp.max(q, -wp.min(have, (z_c - z_o) * share) / float(dt))
            inflow[w, pool.tank] = inflow[w, pool.tank] + wp.float64(q)
            inflow[w, other.tank] = inflow[w, other.tank] - wp.float64(q)
            ledger[w, WEIR] = ledger[w, WEIR] + wp.float64(q) * dt
            weir_species(ns, n_net, w, pool.tank, other.tank, wp.float64(q))
        if cav.spill >= 0:
            q = float(0.0)
            excess = have - float(cap[w, p][0])
            head = sv[3] - wp.dot(up, low)
            if drowned == 0 and excess > 0.0 and head > 0.0:      # free: no water beyond the rim
                amp = cav.r_rim * wp.sqrt(up[0] * up[0] + up[1] * up[1])
                q = wp.min(rim_weir(cav.cd, g, cav.r_rim, up[2] * cav.z_rim, amp, sv[3], -1.0e30), excess / float(dt))
            opening[w, cav.spill] = float(wp.min(q / float(cav.q_max), 1.0))


# ---- host side -----------------------------------------------------------------------------------------------------

def weirs_of(pool: Any) -> tuple[str, ...]:
    """A pool's ``weir``: one tank name or several (tried in order), as a tuple."""
    if pool.weir is None:
        return ()
    return (pool.weir,) if isinstance(pool.weir, str) else tuple(pool.weir)


def check_cavities(pools: tuple[Any, ...], nozzles: tuple[Any, ...], edges: dict[str, Any]) -> None:
    """A spill is a ``source`` edge out of the cavity's tank (flow < 0); a weir names another, non-cavity pool; a rim
    nozzle sits on its cavity pool's spill edge and body."""
    by_tank = {p.tank: p for p in pools}
    for p in pools:
        if p.spill is not None:
            e = edges.get(p.spill)
            if e is None or e.kind != "source" or e.a != p.tank or not e.flow < 0:
                raise PluginError(f"pool of {p.tank!r}: spill {p.spill!r} must be a source edge out of {p.tank!r} "
                                  f"with flow < 0 (its rate at full opening)")
        ws = weirs_of(p)
        if len(ws) > 4 or len(set(ws)) < len(ws) or any(x == p.tank or x not in by_tank or
                                                         by_tank[x].cavity is not None for x in ws):
            raise PluginError(f"pool of {p.tank!r}: weir {p.weir!r} must name up to 4 other pools' tanks, none a "
                              f"cavity's")
    for n in nozzles:
        if n.rim_of is None:
            continue
        p = by_tank.get(n.rim_of)
        if p is None or p.cavity is None or p.spill != n.edge or p.body != n.body:
            raise PluginError(f"nozzle {n.name!r}: rim_of {n.rim_of!r} must name a cavity pool whose spill edge is "
                              f"the nozzle's edge, on the same body")


def cavity_tables(pools: tuple[Any, ...], nozzles: tuple[Any, ...], net: Any,
                  ray_starts: list[int]) -> tuple[list[Cavity], np.ndarray, np.ndarray]:
    """One ``Cavity`` per pool (off for box pools), the slices (z_a, z_b, r) and the profile nodes (r, z)."""
    cavs, slices, prof = [], [], []
    edges = {e.name: e for e in net.edges}
    tanks = [p.tank for p in pools]
    for p in pools:
        c = Cavity()
        c.on, c.weir, c.n_weir, c.spill, c.noz, c.ray = 0, wp.vec4i(-1, -1, -1, -1), 0, -1, -1, -1
        if p.cavity is not None:
            rz = np.asarray(p.cavity, np.float64)
            zs = np.linspace(rz[0, 1], rz[-1, 1], p.slices + 1)
            rm = np.interp(0.5 * (zs[:-1] + zs[1:]), rz[:, 1], rz[:, 0])
            c.on, c.s0, c.ns, c.p0, c.np = 1, len(slices), p.slices, len(prof), len(rz)
            slices.extend(zip(zs[:-1], zs[1:], rm, strict=True))
            prof.extend(rz.tolist())
            c.z0, c.z_rim, c.r_rim, c.r_max = rz[0, 1], rz[-1, 1], rz[-1, 0], rz[:, 0].max()
            c.vol, c.tol, c.cd = float(np.sum(np.pi * rm ** 2 * np.diff(zs))), p.cavity_tol, p.weir_cd
            ws = [tanks.index(x) for x in weirs_of(p)]
            c.weir, c.n_weir = wp.vec4i(*(ws + [-1] * (4 - len(ws)))), len(ws)
            if p.spill is not None:
                c.spill, c.q_max = net.edge_index(p.spill), -float(edges[p.spill].flow)
            for n, noz in enumerate(nozzles):
                if noz.rim_of == p.tank:
                    c.noz, c.ray = n, ray_starts[n]
        cavs.append(c)
    return cavs, np.asarray(slices or [(0.0, 0.0, 0.0)], np.float64), np.asarray(prof or [(0.0, 0.0)], np.float32)


__all__ = ["Cavity", "cavity_flows", "cavity_state", "cavity_tables", "check_cavities", "disc_below", "in_cavity",
           "level_for", "volume_below"]
