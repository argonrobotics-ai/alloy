# SPDX-License-Identifier: Apache-2.0
"""Holes in a cavity pool's vessel (WP8's colander, rinse-tomatoes): rings of holes that drain the vessel's own water
into a tank, or as drips onto whatever is below.

``Pool(cavity=..., holes=((z, cd_area), ...) or ((z, cd_area, r), ...), holes_to=tank)``: each ring sits at height z
above the cavity's floor in the body frame, radius r (default: the inner wall's radius there; a floor's holes spread
over its disc are a ring at its RMS radius r_floor/√2). A ring drains Torricelli's Cd·A·sqrt(2·g·head), the head
measured along gravity from each point of the ring to the vessel's surface and averaged round the wetted part of the
ring (midpoints, as the weir), so a tipped colander drains from its low side first and a ring above the water drains
nothing. The water goes into ``holes_to``'s tank with the network's species at the vessel's concentration, never more
than the vessel has after the transfers booked before it this exchange. Ledger column ``holes``.

**Drips** (``holes_to=None``; GA's colander drips on the board or the floor as it is carried): the holes' flow Q leaves
as ``drip_rays`` (N) jets, each carrying Q/N from the point where the rings' flow, counted ring by ring and round each
wet arc from its dry edge, reaches (i + ½)/N of Q, so they crowd where the head is largest (a tipped colander drips
from its low side). Each leaves along the wall's outward normal (the floor's: down its axis) at
sqrt(2·g·max(head, 0.1 mm)) plus the vessel's velocity there, and pushes the vessel back. An internal jet nozzle per
dripping pool casts them like any jet (charts, pools, shapes, lumps, or lost) at the next exchange, with the water and
the vessel's concentration of each network species as they left (in flight: ``drip_q``, ``ns.drip``); its discharge
is a column past the network's edges in the supply's ``nozzle_q``. Ledgers: ``holes`` out of the vessel, then
``sent`` and where it went (the cross ledger: holes = what the drips sent + what is in flight).

It runs before the weirs and spills (``supply_cavity.cavity_flows``), so they see what the holes took.

Limits (v1): the head is the vessel's own water only, so a colander standing in water (its weir pool) still drains
by its inside depth and never fills through its holes (GA's colander drains the same way); a floor's holes are one
ring at the RMS radius, which is exact for the head's spread under a tilt while the whole floor is wet, not for a
nearly empty, tipped colander whose floor is partly dry; the midpoint rule's 32 points round the wet half-ring are
within about 5e-4 of the exact mean near the waterline (the sqrt's edge). A dripping vessel's own shapes do not catch
its drips only when it is on a body (a static one's collision shapes may).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.supply_cavity import Cavity, available
from alloy.plugins.surface_flow.supply_deposit import node_conc
from alloy.plugins.surface_flow.supply_kernels import HOLES, PoolSpec
from alloy.plugins.surface_flow.supply_spec import STAGE, Nozzle
from alloy.plugins.surface_flow.supply_species import NetSpecies

RING_N = wp.constant(32)        # midpoints along half a wetted ring
PI = wp.constant(3.14159265358979)
HEAD_MIN = wp.constant(1.0e-4)  # m: a drip leaves at least at sqrt(2·g·0.1 mm)


@wp.func
def ring_outflow(g: float, cd_area: float, mid: float, amp: float, level: float) -> float:
    """Torricelli outflow [m³/s] of a ring of holes (total Cd·A) whose points stand at mid + amp·cos φ below a
    surface at ``level``: Cd·A times sqrt(2·g·head) averaged round the ring (dry points give nothing)."""
    if amp < 1.0e-9:
        return cd_area * wp.sqrt(2.0 * g * wp.max(level - mid, 0.0))
    a = wp.acos(wp.clamp((level - mid) / amp, -1.0, 1.0))      # wet for φ in [a, π], and its mirror image
    d = (PI - a) / float(RING_N)
    s = float(0.0)
    for i in range(RING_N):
        s = s + wp.sqrt(2.0 * g * wp.max(level - (mid + amp * wp.cos(a + (float(i) + 0.5) * d)), 0.0))
    return cd_area * d * s / PI


@wp.func
def wet_start(mid: float, amp: float, level: float) -> float:
    """A ring whose points stand at mid + amp·cos φ is wet for φ in [a, 2π - a]: a (0 all round, π dry)."""
    if amp < 1.0e-9:
        if level > mid:
            return 0.0
        return PI
    return wp.acos(wp.clamp((level - mid) / amp, -1.0, 1.0))


@wp.func
def ring_angle(g: float, mid: float, amp: float, level: float, frac: float) -> float:
    """The angle at which a ring's flow, counted round its wet arc [a, 2π - a] from the dry edge a, reaches the share
    ``frac`` of it: over the 2·RING_N midpoint cells of ``ring_outflow`` (its half-ring and mirror), linear within a
    cell."""
    a = wet_start(mid, amp, level)
    d = (PI - a) / float(RING_N)
    total = float(0.0)
    for j in range(2 * RING_N):
        total = total + wp.sqrt(2.0 * g * wp.max(level - (mid + amp * wp.cos(a + (float(j) + 0.5) * d)), 0.0))
    target = frac * total
    acc = float(0.0)
    phi = 2.0 * PI - a
    found = int(0)
    for j in range(2 * RING_N):
        c = wp.sqrt(2.0 * g * wp.max(level - (mid + amp * wp.cos(a + (float(j) + 0.5) * d)), 0.0))
        if found == 0 and c > 0.0 and acc + c >= target:
            phi = a + (float(j) + wp.clamp((target - acc) / c, 0.0, 1.0)) * d
            found = 1
        acc = acc + c
    return phi


@wp.func
def drip_from(w: int, ring: wp.vec3, nr: wp.vec2, g: float, sv: wp.vec4, e: wp.vec3, e2: wp.vec3, floor: wp.vec3,
              phi: float, ray: int, ray_p0: wp.array2d[wp.vec3], ray_dir: wp.array2d[wp.vec3]):
    """Drip ray ``ray`` leaves ring (z, Cd·A, r) at angle φ: its exit point and its velocity along the wall's outward
    normal ``nr`` (radial, axial) at sqrt(2·g·max(head, HEAD_MIN)) (body frame)."""
    s_up = wp.sqrt(sv[0] * sv[0] + sv[1] * sv[1])
    rad = wp.cos(phi) * e + wp.sin(phi) * e2
    head = sv[3] - (sv[2] * ring[0] + ring[2] * s_up * wp.cos(phi))
    ray_p0[w, ray] = floor + ring[2] * rad + wp.vec3(0.0, 0.0, ring[0])
    ray_dir[w, ray] = wp.normalize(nr[0] * rad + wp.vec3(0.0, 0.0, nr[1])) * wp.sqrt(2.0 * g * wp.max(head, HEAD_MIN))


@wp.func
def place_drips(w: int, ho: wp.vec3i, rings: wp.array[wp.vec3], ring_n: wp.array[wp.vec2], g: float, sv: wp.vec4,
                floor: wp.vec3, q_tot: float, ray0: int, n: int, ray_p0: wp.array2d[wp.vec3],
                ray_dir: wp.array2d[wp.vec3]):
    """The n drip rays' exit points and velocities (body frame) at equal shares of the rings' flow ``q_tot`` (a share
    rounding leaves unreached: the last wet ring's lowest point)."""
    s_up = wp.sqrt(sv[0] * sv[0] + sv[1] * sv[1])
    e = wp.vec3(1.0, 0.0, 0.0)                                # φ = 0 along the surface's tilt (the high side)
    if s_up > 1.0e-9:
        e = wp.vec3(sv[0] / s_up, sv[1] / s_up, 0.0)
    e2 = wp.vec3(-e[1], e[0], 0.0)
    for i in range(n):
        target = (float(i) + 0.5) / float(n) * q_tot
        acc = float(0.0)
        done = int(0)
        last = int(-1)
        for kh in range(ho[1]):
            ring = rings[ho[0] + kh]                          # (z, Cd·A, r)
            mid = sv[2] * ring[0]
            amp = ring[2] * s_up
            q = ring_outflow(g, ring[1], mid, amp, sv[3])
            if q > 0.0:
                last = ho[0] + kh
            if done == 0 and q > 0.0 and acc + q >= target:
                drip_from(w, ring, ring_n[ho[0] + kh], g, sv, e, e2, floor, ring_angle(g, mid, amp, sv[3],
                                                                                         (target - acc) / q),
                          ray0 + i, ray_p0, ray_dir)
                done = 1
            acc = acc + q
        if done == 0 and last >= 0:
            drip_from(w, rings[last], ring_n[last], g, sv, e, e2, floor, PI, ray0 + i, ray_p0, ray_dir)


@wp.func
def move_out(ns: NetSpecies, n_net: int, w: int, src: int, dst: int, q: wp.float64):
    """The network's species in ``q`` m³/s leaving tank ``src`` at its concentration, into ``dst`` (or away: -1)."""
    for k in range(n_net):
        m = q * node_conc(w, src, k, ns.medium, ns.bulk, ns.p_ref, ns.poly, ns.pressure, ns.volume, ns.species)
        ns.inflow[w, src, k] = ns.inflow[w, src, k] - m
        if dst >= 0:
            ns.inflow[w, dst, k] = ns.inflow[w, dst, k] + m


@wp.kernel
def cavity_holes(dt: wp.float64, pools: wp.array[PoolSpec], cavs: wp.array[Cavity], holes_of: wp.array[wp.vec3i],
                 rings: wp.array[wp.vec3], ring_n: wp.array[wp.vec2], pressure: wp.array2d[wp.float64],
                 gravity: wp.vec3, surf: wp.array2d[wp.vec4], ns: NetSpecies, n_net: int,
                 drip_ray: wp.array[wp.vec2i], inflow: wp.array2d[wp.float64], ledger: wp.array2d[wp.float64],
                 drip_q: wp.array2d[wp.float64], drip_c: wp.array3d[wp.float64], ray_p0: wp.array2d[wp.vec3],
                 ray_dir: wp.array2d[wp.vec3]):
    """Per world, the cavity pools with holes in order: each ring's outflow from the surface ``surf`` (body-frame up
    and level), at most the vessel's ``available`` volume, into its ``holes_to`` tank, or (drip d: holes_of's third
    entry -2 - d) out as drips: their flow, the vessel's concentrations and the rays' exit points and velocities for
    the next exchange (``drip_ray[d]``: the drip nozzle's first ray and count)."""
    w = wp.tid()
    g = wp.length(gravity)
    for p in range(pools.shape[0]):
        ho = holes_of[p]
        if cavs[p].on == 0 or ho[1] == 0:
            continue
        pool = pools[p]
        sv = surf[w, p]
        s_up = wp.sqrt(sv[0] * sv[0] + sv[1] * sv[1])
        q_tot = float(0.0)
        for kh in range(ho[1]):
            ring = rings[ho[0] + kh]                              # (z, Cd·A, r)
            q_tot = q_tot + ring_outflow(g, ring[1], sv[2] * ring[0], ring[2] * s_up, sv[3])
        q = wp.max(wp.min(q_tot, available(pool, w, pressure, inflow, dt) / float(dt)), 0.0)
        if ho[2] <= -2:                                           # drips: in flight until the next exchange
            d = -2 - ho[2]
            drip_q[w, d] = wp.float64(q)
            if q > 0.0:
                dr = drip_ray[d]
                place_drips(w, ho, rings, ring_n, g, sv, pool.floor, q_tot, dr[0], dr[1], ray_p0, ray_dir)
                for k in range(n_net):
                    drip_c[w, d, k] = node_conc(w, pool.tank, k, ns.medium, ns.bulk, ns.p_ref, ns.poly, ns.pressure,
                                                ns.volume, ns.species)
        if q <= 0.0:
            continue
        qd = wp.float64(q)
        inflow[w, pool.tank] = inflow[w, pool.tank] - qd
        to = int(-1)
        if ho[2] >= 0:
            to = ho[2]
            inflow[w, to] = inflow[w, to] + qd
        ledger[w, HOLES] = ledger[w, HOLES] + qd * dt
        move_out(ns, n_net, w, pool.tank, to, qd)


@wp.kernel
def drip_commit(n_vessel: int, discharge: wp.array2d[wp.float64], drip_q: wp.array2d[wp.float64],
                drip_c: wp.array3d[wp.float64], nozzle_q: wp.array2d[wp.float64], drip_conc: wp.array3d[wp.float64]):
    """End of ``after_film``: the network's discharges and this exchange's drips (out of their vessels: < 0, as the
    network's) into the nozzles' table, and the drips' concentrations in flight (the n_vessel vessels' drips; the
    stages' columns after them are ``supply_stage``'s)."""
    w = wp.tid()
    m = discharge.shape[1]
    for e in range(m):
        nozzle_q[w, e] = discharge[w, e]
    for d in range(n_vessel):
        nozzle_q[w, m + d] = -drip_q[w, d]
        for k in range(drip_c.shape[2]):
            drip_conc[w, d, k] = drip_c[w, d, k]


# ---- host side -----------------------------------------------------------------------------------------------------

def _wall_normal(rz: np.ndarray, z: float) -> tuple[float, float]:
    """The cavity wall's outward normal (radial, axial) at height z; the floor's (0, -1)."""
    if z <= rz[0, 1]:
        return 0.0, -1.0
    k = min(int(np.searchsorted(rz[:, 1], z)) - 1, len(rz) - 2)
    slope = (rz[k + 1, 0] - rz[k, 0]) / (rz[k + 1, 1] - rz[k, 1])
    n = math.hypot(1.0, slope)
    return 1.0 / n, -slope / n


def hole_tables(pools: tuple[Any, ...], net: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Per pool (start, count, holes_to node, or -2 - d for the d-th dripping pool) into the rings (z, Cd·A, r) and
    their walls' outward normals (radial, axial); and whether any pool has holes."""
    of, rings, normals, drips = [], [], [], 0
    tanks = {n.name for n in net.nodes if n.medium == "tank"}
    for p in pools:
        holes = tuple(p.holes or ())
        if holes and p.cavity is None:
            raise PluginError(f"pool of {p.tank!r}: holes need a cavity")
        if p.holes_to is not None and (p.holes_to not in tanks or p.holes_to == p.tank):
            raise PluginError(f"pool of {p.tank!r}: holes_to {p.holes_to!r} must be another tank node")
        start = len(rings)
        rz = np.asarray(p.cavity if holes else ((0.0, 0.0), (0.0, 1.0)), np.float64)
        for h in holes:
            z, cda = float(h[0]), float(h[1])
            floor = z <= rz[0, 1]
            r = float(h[2]) if len(h) > 2 else (rz[0, 0] / math.sqrt(2.0) if floor else float(np.interp(z, rz[:, 1],
                                                                                                         rz[:, 0])))
            if not rz[0, 1] <= z <= rz[-1, 1] or cda <= 0.0 or r < 0.0:
                raise PluginError(f"pool of {p.tank!r}: a hole ring is (z, cd_area[, r]) with z within the cavity, "
                                  f"cd_area > 0, r >= 0; got {h}")
            rings.append((z, cda, r))                         # the profile's z: up from the floor point
            normals.append(_wall_normal(rz, z))
        to = -1
        if holes:
            to = net.node_index(p.holes_to) if p.holes_to is not None else -2 - drips
            drips += int(p.holes_to is None)
        of.append((start, len(rings) - start, to))
    return (np.asarray(of or [(0, 0, -1)], np.int32), np.asarray(rings or [(0.0, 0.0, 0.0)], np.float32),
            np.asarray(normals or [(0.0, -1.0)], np.float32), bool(len(rings)))


def drip_nozzles(pools: tuple[Any, ...]) -> tuple[Nozzle, ...]:
    """The internal jet nozzle of each dripping pool (holes, ``holes_to`` None), on its vessel's body."""
    return tuple(Nozzle(f"{p.tank}:drips", edge="", body=p.body, rays=p.drip_rays, drip_of=p.tank, exit_area=1.0)
                 for p in pools if p.holes and p.holes_to is None)


def drip_edge(nozzles: tuple[Any, ...], n: int, m: int) -> int:
    """Nozzle n's column in the nozzles' discharge table: a network edge's, or (a drip nozzle) one past the m edges."""
    return m + sum(1 for x in nozzles[:n] if x.drip_of)


class Holing:
    """``Supply``'s holes and drips (a mixin: it uses the supply's pools, cavities, nozzles and species)."""

    def _setup_holes(self, ctx: Any, film: Any) -> None:
        """Hole tables; with drips, their state and the nozzles' discharge table (else the network's own)."""
        W, f64 = ctx.world_count, wp.float64
        of, rings, normals, self.holes = hole_tables(self.pools, self.net)
        self.holes_of, self.rings = ctx.array(of, wp.vec3i), ctx.array(rings, wp.vec3)
        self.ring_n = ctx.array(normals, wp.vec2)
        drips = [(self.ray_starts[n], noz.rays) for n, noz in enumerate(self.nozzles)
                 if noz.drip_of and noz.drip_of != STAGE]       # a vessel's (the stages' come after: supply_stage)
        self.n_drips, D, K = len(drips), max(len(drips), 1), max(self.fluid.n_species, 1)
        self.drip_ray = ctx.array(np.asarray(drips or [(0, 0)], np.int32), wp.vec2i)
        extra = sum(1 for noz in self.nozzles if noz.drip_of)  # the discharge table's columns past the edges
        if not extra:
            self.nozzle_q, self.ray_p0 = self.fluid.discharge, ctx.zeros((W, 1), wp.vec3)
            self.drip_q, self.drip_c = ctx.zeros((W, 1), f64), ctx.zeros((W, 1, 1), f64)
            return
        m = self.fluid.discharge.shape[1]
        self.ray_p0 = film.add_state("supply_ray_p0", ctx.zeros((W, self.ray_count), wp.vec3))
        self.nozzle_q = film.add_state("supply_nozzle_q", ctx.zeros((W, m + extra), f64))
        self.drip_q = film.add_state("supply_drip_q", ctx.zeros((W, D), f64))
        self.drip_c = film.add_state("supply_drip_c", ctx.zeros((W, D, K), f64))
        self.ns.drip = film.add_state("supply_drip_conc", ctx.zeros((W, D, K), f64))
        self._drip_commit()

    def _holes(self, dt: float, ns: NetSpecies) -> None:
        """In ``before_film``, before the weirs and spills: the holes' outflow (``cavity_holes``)."""
        if self.holes:
            f = self.fluid
            wp.launch(cavity_holes, dim=self.ctx.world_count,
                      inputs=[wp.float64(dt), self.pool_spec, self.cav, self.holes_of, self.rings, self.ring_n,
                              f.pressure, wp.vec3(*self.gravity), self.cav_surf, ns, f.n_species, self.drip_ray],
                      outputs=[f.inflow, self.ledger, self.drip_q, self.drip_c, self.ray_p0, self.ray_dir],
                      device=self.ctx.device)

    def _drip_commit(self) -> None:
        """End of ``after_film``: this exchange's drips in flight for the next (``drip_commit``)."""
        if self.nozzle_q is not self.fluid.discharge:
            wp.launch(drip_commit, dim=self.ctx.world_count,
                      inputs=[self.n_drips, self.fluid.discharge, self.drip_q, self.drip_c],
                      outputs=[self.nozzle_q, self.ns.drip], device=self.ctx.device)


__all__ = ["HEAD_MIN", "Holing", "cavity_holes", "drip_commit", "drip_edge", "drip_from", "drip_nozzles", "hole_tables",
           "move_out", "place_drips", "ring_angle", "ring_outflow", "wet_start"]
