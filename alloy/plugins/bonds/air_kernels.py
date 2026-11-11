# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the trapped-air pockets (``air.py`` launches them; the model is in its docstring).

Per world there are ``SLOTS`` pocket slots, each one gas node of the shared fluid core. Per-world loops (``dim =
worlds``) run serially over that world's film vertices: their sums are deterministic without float atomics.
"""

from __future__ import annotations

import warp as wp

SLOTS = 8                                     # pocket slots (gas nodes) per world
P_ATM = wp.constant(wp.float64(101325.0))     # Pa
H_FLOOR = wp.constant(wp.float64(1.0e-7))     # m: a pocket's volume is at least its area times this
H_SEAL = wp.constant(wp.float64(1.0e-6))      # m: a pocket holds air above P_atm·area·(roughness + H_SEAL)


@wp.func
def held(amount: wp.float64, area: wp.float64, rough: float) -> wp.float64:
    """How far a pocket holds air, in [0, 1]: its air at P_atm stands e above the roughness layer; up to H_SEAL it is
    roughness air (0: carried by the asperities, it holds nothing off, blocks no bond and is not trapped), then the
    pocket's pressure ramps in over the next H_SEAL."""
    if area <= wp.float64(0.0):
        return wp.float64(0.0)
    e = amount / (P_ATM * area) - wp.float64(rough)
    return wp.clamp(e / H_SEAL - wp.float64(1.0), wp.float64(0.0), wp.float64(1.0))
V_PARK = wp.constant(wp.float64(1.0e-6))      # m³: an idle slot's node volume (the core's pivot stays > 0)
G_REF = wp.constant(wp.float64(1.0))          # m³/s: the vent edges' conductance at opening 1
BIG = wp.constant(1.0e30)
BOX = wp.constant(7)                          # newton.GeoType ids (checked in air.py)
CAPSULE = wp.constant(4)
CYLINDER = wp.constant(6)


@wp.struct
class Gaps:
    """The substrate shapes each world's pockets sit on, for the vertex-to-surface gap."""
    shapes: wp.array2d[wp.int32]              # [W, S] model shape ids (-1: none)
    kind: wp.array[wp.int32]                  # model.shape_type
    scale: wp.array[wp.vec3]
    xform: wp.array[wp.transform]
    margin: wp.array[float]
    body: wp.array[wp.int32]                  # shape_body (the film view's numbering)
    radius: wp.array[float]                   # film particle (contact) radius
    world: wp.array[wp.int32]                 # particle world


@wp.struct
class Air:
    """Per vertex and per (world, slot) pocket arrays."""
    start: wp.array[wp.int32]                 # [W + 1] each world's first vertex
    rough: float                              # m: the gas layer left in the surfaces' roughness at contact
    area: wp.array[float]                     # [n] vertex area
    throat: wp.array[wp.float64]              # [n] boundary vertices: w_b/(12·mu·s_b) [1/(Pa·s)], else 0
    slot: wp.array[wp.int32]                  # [n] state: the vertex's pocket slot, -1 none
    gap: wp.array[float]                      # [n] gas gap: to the substrate plus the roughness layer (> 0)
    nrm: wp.array[wp.vec3]                    # [n] the substrate's outward normal there
    amount: wp.array2d[wp.float64]            # [W, K] state: gas amount P·V [Pa·m³]
    p: wp.array2d[wp.float64]                 # [W, K] state: pressure excess the film feels [Pa]
    root: wp.array2d[wp.int32]                # [W, K] state: world-local root vertex, -1 idle
    area_k: wp.array2d[wp.float64]            # [W, K] m²
    volume: wp.array2d[wp.float64]            # [W, K] m³ (floored)
    lam: wp.array2d[wp.float64]               # [W, K] sum throat·h³ over the boundary [m³/(Pa·s)]
    enclosed: wp.array2d[wp.int32]            # [W, K]
    vsum: wp.array2d[wp.float64]              # [W, K] scratch
    csum: wp.array2d[wp.float64]              # [W, K] scratch (the colour's lumped gas stiffness)


@wp.struct
class Books:
    """Per world: the gas ledgers [Pa·m³] and the pocket readings."""
    formed: wp.array[wp.float64]              # drawn from the atmosphere into new pocket cells
    vented: wp.array[wp.float64]              # out through the vents
    vanished: wp.array[wp.float64]            # left in a pocket whose last cell bonded
    untracked: wp.array[wp.float64]           # in pockets beyond the slots (overflow)
    trapped: wp.array[wp.float64]             # m², the largest enclosed pocket holding air
    pockets: wp.array[wp.float64]             # pocket count (every component)
    overflow: wp.array[wp.float64]            # pockets without a slot


@wp.struct
class Scratch:
    """Per-vertex accumulators of the relabel, indexed by a pocket's root vertex."""
    parent: wp.array[wp.int32]
    amount: wp.array[wp.float64]
    area: wp.array[wp.float64]
    vol: wp.array[wp.float64]
    lam: wp.array[wp.float64]
    open_: wp.array[wp.int32]
    slot: wp.array[wp.int32]
    sel_root: wp.array2d[wp.int32]            # [W, K] the K largest pockets, largest first
    sel_area: wp.array2d[wp.float64]
    new_root: wp.array2d[wp.int32]            # [W, K] the root each slot takes (-1 idle)


# --- gaps -------------------------------------------------------------------------------------------------------

@wp.func
def capsule_sdf(h: wp.vec3, p: wp.vec3) -> wp.vec4:
    c = wp.vec3(0.0, 0.0, wp.clamp(p[2], -h[1], h[1]))
    v = p - c
    d = wp.length(v)
    n = wp.vec3(0.0, 0.0, 1.0)
    if d > 0.0:
        n = v / d
    return wp.vec4(n[0], n[1], n[2], d - h[0])


@wp.func
def cylinder_sdf(h: wp.vec3, p: wp.vec3) -> wp.vec4:
    rr = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    radial = wp.vec3(1.0, 0.0, 0.0)
    if rr > 0.0:
        radial = wp.vec3(p[0] / rr, p[1] / rr, 0.0)
    axial = wp.vec3(0.0, 0.0, wp.sign(p[2]))
    dr = rr - h[0]
    dz = wp.abs(p[2]) - h[1]
    if dr <= 0.0 and dz <= 0.0:
        if dr > dz:
            return wp.vec4(radial[0], radial[1], radial[2], dr)
        return wp.vec4(axial[0], axial[1], axial[2], dz)
    a = wp.max(dr, 0.0)
    b = wp.max(dz, 0.0)
    d = wp.sqrt(a * a + b * b)
    n = (radial * a + axial * b) / d
    return wp.vec4(n[0], n[1], n[2], d)


@wp.func
def box_sdf(h: wp.vec3, p: wp.vec3) -> wp.vec4:
    q = wp.vec3(wp.abs(p[0]) - h[0], wp.abs(p[1]) - h[1], wp.abs(p[2]) - h[2])
    if q[0] <= 0.0 and q[1] <= 0.0 and q[2] <= 0.0:
        a = 0
        if q[1] > q[a]:
            a = 1
        if q[2] > q[a]:
            a = 2
        n = wp.vec3(0.0, 0.0, 0.0)
        n[a] = wp.sign(p[a])
        return wp.vec4(n[0], n[1], n[2], q[a])
    o = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    d = wp.length(o)
    m = wp.vec3(wp.sign(p[0]) * o[0], wp.sign(p[1]) * o[1], wp.sign(p[2]) * o[2]) / d
    return wp.vec4(m[0], m[1], m[2], d)


@wp.func
def local_sdf(kind: int, h: wp.vec3, p: wp.vec3) -> wp.vec4:
    """(outward normal, signed distance) of point ``p`` in a shape's frame: box (half extents), capsule or cylinder
    (radius h[0], half height h[1], along z)."""
    if kind == CAPSULE:
        return capsule_sdf(h, p)
    if kind == CYLINDER:
        return cylinder_sdf(h, p)
    return box_sdf(h, p)


@wp.func
def shape_gap(gp: Gaps, body_q: wp.array[wp.transform], s: int, x: wp.vec3) -> wp.vec4:
    """(world normal, distance beyond the shape's margin) of world point ``x`` to shape ``s``."""
    tf = gp.xform[s]
    b = gp.body[s]
    if b >= 0:
        tf = wp.transform_multiply(body_q[b], tf)
    loc = local_sdf(gp.kind[s], gp.scale[s], wp.transform_point(wp.transform_inverse(tf), x))
    n = wp.quat_rotate(wp.transform_get_rotation(tf), wp.vec3(loc[0], loc[1], loc[2]))
    return wp.vec4(n[0], n[1], n[2], loc[3] - gp.margin[s])


@wp.kernel
def vertex_gaps(gp: Gaps, rough: float, select: wp.array[wp.int32], q: wp.array[wp.vec3],
                body_q: wp.array[wp.transform], gap: wp.array[float], nrm: wp.array[wp.vec3]):
    """Gas gap of each selected vertex: from its contact surface to the nearest substrate surface (>= 0), plus the
    roughness layer, and the surface's normal."""
    i = wp.tid()
    gap[i] = 0.0
    if select[i] < 0:
        return
    w = gp.world[i]
    best = BIG
    nb = wp.vec3(0.0, 0.0, 1.0)
    for j in range(gp.shapes.shape[1]):
        s = gp.shapes[w, j]
        if s >= 0:
            g = shape_gap(gp, body_q, s, q[i])
            if g[3] < best:
                best = g[3]
                nb = wp.vec3(g[0], g[1], g[2])
    gap[i] = wp.max(best - gp.radius[i], 0.0) + rough
    nrm[i] = nb


# --- the implicit pocket pressure (VBD colour hook) ------------------------------------------------------------

@wp.kernel
def pocket_pressure(colour: int, colors: wp.array[wp.int32], air: Air):
    """Per world, before a colour's solve: each pocket's Boyle pressure at the iterate, P = a/V, and its stiffness
    lumped over the colour, kappa·s (kappa = P/V, s = the sum of the area of the colour's vertices in the pocket),
    both times ``held`` (roughness air pushes nothing)."""
    w = wp.tid()
    for k in range(SLOTS):
        air.vsum[w, k] = wp.float64(0.0)
        air.csum[w, k] = wp.float64(0.0)
    for i in range(air.start[w], air.start[w + 1]):
        k = air.slot[i]
        if k >= 0:
            a = wp.float64(air.area[i])
            air.vsum[w, k] = air.vsum[w, k] + a * wp.float64(air.gap[i])
            if colors[i] == colour:
                air.csum[w, k] = air.csum[w, k] + a
    for k in range(SLOTS):
        if air.root[w, k] >= 0:
            v = wp.max(air.vsum[w, k], air.area_k[w, k] * H_FLOOR)
            pres = air.amount[w, k] / v
            f = held(air.amount[w, k], air.area_k[w, k], air.rough)
            air.p[w, k] = (pres - P_ATM) * f
            air.csum[w, k] = pres / v * air.csum[w, k] * f


@wp.kernel
def pocket_force(colour: int, colors: wp.array[wp.int32], world: wp.array[wp.int32], air: Air,
                 forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    """VBD after_accumulate slot: the pocket's pressure on each of its vertices of ``colour``, p·A_i·n, with the gas
    spring's Hessian lumped over the colour, kappa·A_i·s·n nᵀ. The coupling is rank one (kappa·A Aᵀ); the lumped
    diagonal dominates half of it (Cauchy-Schwarz), so the colour's simultaneous step never overshoots (WP13
    must-fix 2) and a uniform motion of the colour is solved exactly."""
    i = wp.tid()
    k = air.slot[i]
    if k < 0 or colors[i] != colour:
        return
    w = world[i]
    n = air.nrm[i]
    a = air.area[i]
    forces[i] = forces[i] + float(air.p[w, k]) * a * n
    hessians[i] = hessians[i] + float(air.csum[w, k]) * a * wp.outer(n, n)


# --- formation guard ----------------------------------------------------------------------------------------------

@wp.kernel
def block_trapped(air: Air, world: wp.array[wp.int32], key: wp.array[wp.int64]):
    """An enclosed pocket holding air keeps its cells free (WP13 must-fix 3: V -> 0 with air left would be P -> inf)."""
    i = wp.tid()
    k = air.slot[i]
    if k < 0:
        return
    w = world[i]
    if air.enclosed[w, k] == 1 and held(air.amount[w, k], air.area_k[w, k], air.rough) > wp.float64(0.0):
        key[i] = wp.int64(0)


# --- vent: the fluid core's gas nodes ----------------------------------------------------------------------------

@wp.kernel
def vent_in(air: Air, pressure: wp.array2d[wp.float64], volume: wp.array2d[wp.float64],
            volume_old: wp.array2d[wp.float64], opening: wp.array2d[wp.float32]):
    """Each slot's node for the core's vent step: P = a/V at a fixed volume; an open pocket's vent is the laminar slot
    of its throat, g = P_mean·sum w·h³/(12·mu·s) (throughput); an enclosed or idle one is shut."""
    w, k = wp.tid()
    v = V_PARK
    pres = P_ATM
    u = wp.float64(0.0)
    if air.root[w, k] >= 0:
        v = air.volume[w, k]
        pres = air.amount[w, k] / v
        if air.enclosed[w, k] == 0:
            u = wp.min(wp.float64(0.5) * (pres + P_ATM) * air.lam[w, k] / G_REF, wp.float64(1.0))
    volume[w, k] = v
    volume_old[w, k] = v
    pressure[w, k] = pres
    opening[w, k] = wp.float32(u)


@wp.kernel
def vent_out(air: Air, pressure: wp.array2d[wp.float64], bk: Books):
    """The amount after the vent step, booked: vented += a_before - a_after."""
    w = wp.tid()
    for k in range(SLOTS):
        if air.root[w, k] >= 0:
            a = pressure[w, k] * air.volume[w, k]
            bk.vented[w] = bk.vented[w] + air.amount[w, k] - a
            air.amount[w, k] = a


__all__ = ["Air", "Books", "Gaps", "P_ATM", "SLOTS", "Scratch", "block_trapped", "box_sdf", "capsule_sdf",
           "cylinder_sdf", "held", "local_sdf", "pocket_force", "pocket_pressure", "vent_in", "vent_out", "vertex_gaps"]
