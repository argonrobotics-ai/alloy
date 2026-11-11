# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the surface-flow supply (interface note §4): nozzle ray casts, jet forces, drain capacities and
sums, pool levels (the deposits are in ``supply_deposit.py``).

The casts run one thread per (world, ray) and write only their own hit slot. Everything that writes shared film cells
or body forces runs one thread per world, walking nozzles, rays and footprint cells in a fixed order (no atomics), so
runs are deterministic and graph-capturable.

A jet's path is ballistic: ``segments`` straight pieces of p(t) = p0 + v0·t + g·t²/2, each cast against the charts,
the pools (a tank's free surface over its footprint) and the occluding shapes. The first hit wins; on a tie a chart
wins over a pool, and a pool over a shape (a chart lies on its shape).
"""

from __future__ import annotations

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.geometry.raycast import map_ray_to_local, ray_intersect_mesh_no_normal, ray_intersect_shape_no_normal

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.geometry import chart_xform, ray_hit

JET = wp.constant(0)          # a solid jet: enters the film at its jump radius (Bhagat et al. 2018)
FAN = wp.constant(1)          # flat fan: rays across ``angle`` in one plane, each splatted where it lands
CONE = wp.constant(2)         # full cone: rays over the cone, uniform per solid angle
SPRAY = wp.constant(3)        # aerosol: Gaussian over the cone; misses are overspray

NONE = wp.constant(0)
CHART = wp.constant(1)
SHAPE = wp.constant(2)
POOL = wp.constant(3)
LUMP = wp.constant(4)          # a jet on a shape of a lumped body (supply_lump: WP9's lumped films)

# ledger columns [W, N_LEDGER] float64, m³ since the start
SENT = wp.constant(0)         # left the nozzles (Σ Q·dt)
SWEPT = wp.constant(1)        # film moved from under jets to their jump rings
TO_TANK = wp.constant(2)      # landed in a pool (a tank's free surface)
LOST = wp.constant(3)         # a jet that hit nothing (no chart, pool or shape within reach)
OVERSPRAY = wp.constant(4)    # spray rays that hit nothing (no chart, pool or shape within reach)
AWAY = wp.constant(5)         # drained to no tank (the sewer)
DRAINED = wp.constant(6)      # drained into tanks
POOLED = wp.constant(7)       # film under a pool's water that joined the pool (flooded cells)
WITHDRAWN = wp.constant(8)    # withdrawal films from pools onto cells leaving the water
REFUSED = wp.constant(9)      # withdrawal films asked for beyond what the pool's tank held (not in the balance)
WEIR = wp.constant(10)        # moved into cavity pools over their rims from the pools around them (out: negative)
MIST = wp.constant(11)        # sprays: the (1 - efficiency) share that never reaches a surface (drift, bounce)
CAUGHT = wp.constant(12)      # caught by a shape (a stencil, a box, a hand): per occluder in supply_caught_by
LUMPED = wp.constant(13)      # jets onto lumped bodies (their lump_in)
HOLES = wp.constant(14)       # out of cavity pools through their holes (into holes_to's tank, or away)
LUMP_POOL = wp.constant(15)   # into pools' tanks from lumps (WP9's pool_add; out: negative)
STAGED = wp.constant(16)      # handed to the supply by film stages as drips (supply_stage: stores' drips, dumps)
N_LEDGER = 17
STAGE_LIFT = wp.constant(1.0e-3)   # m: a stage drip starts this far above its point (supply_stage)

MESH = wp.constant(int(newton.GeoType.MESH))
CONVEX_MESH = wp.constant(int(newton.GeoType.CONVEX_MESH))
TIE = wp.constant(1.0e-4)     # m: a hit this close behind a chart hit is the chart's own surface
FREE = wp.constant(-1.0)      # Drain.cd_area of a free drain


@wp.struct
class NozzleSpec:
    edge: wp.int32
    pattern: wp.int32
    ray_start: wp.int32
    ray_count: wp.int32
    segments: wp.int32
    exit_area: wp.float32
    efficiency: wp.float32
    reach: wp.float32
    rho: wp.float32
    nu: wp.float32
    sigma: wp.float32
    jump_coeff: wp.float32
    h_min: wp.float32         # floor of the post-jump depth in u_post = q/(2π·R_J·h)
    froude: wp.float32        # post-jump Froude number: the depth floor (q/(2π·R_J·Fr·sqrt(g)))^(2/3)
    u_max: wp.float32         # cap of u_post (below the film's u_cap)
    own_pool: wp.int32        # a rim nozzle's own cavity pool (its pour never lands back in it), or -1
    drip: wp.int32            # 1: a holed vessel's drips (each ray its own exit point and velocity, supply_holes);
                              # 2: a stage's drip (world point and velocity, supply_stage)


@wp.struct
class PoolSpec:
    tank: wp.int32
    kappa: wp.float64         # A/(ρ·g) [m³/Pa]
    p_ref: wp.float64
    area: wp.float64
    floor: wp.vec3            # floor centre in the pool body's frame (z up)
    half: wp.vec2             # footprint half extents (x, y); a round one's radius (half[0])
    round: wp.int32           # 1: a round footprint (a cavity's mouth)
    flood: wp.int32           # 1: floods the chart cells under its water (supply_pool.py)
    rho: wp.float32           # the pool liquid's density (the tank node's), kinematic viscosity, surface tension
    nu: wp.float32
    sigma: wp.float32
    h_max: wp.float32         # cap of a Landau-Levich withdrawal film [m]
    flat_cos: wp.float32      # a face whose normal is within the puddle tilt of the pool's up keeps a puddle


@wp.func
def body_xform(body_q: wp.array[wp.transform], b: wp.int32, local: wp.transform) -> wp.transform:
    if b >= 0:
        return wp.transform_multiply(body_q[b], local)
    return local


@wp.func
def pool_frame(body_q: wp.array[wp.transform], b: wp.int32) -> wp.transform:
    if b >= 0:
        return body_q[b]
    return wp.transform_identity()


@wp.func
def pool_level(pool: PoolSpec, p: wp.float64) -> float:
    """Height of the free surface above the floor [m] (the tank's volume over its area)."""
    return float(pool.kappa * (p - pool.p_ref) / pool.area)


@wp.func
def shape_t(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], shape_transform: wp.array[wp.transform],
            shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3], shape_ptr: wp.array[wp.uint64],
            s: wp.int32, o: wp.vec3, d: wp.vec3, length: float) -> float:
    """Distance along the world ray (o, d unit) to shape ``s``, or -1 (Newton's analytic and mesh casts)."""
    X = body_xform(body_q, shape_body[s], shape_transform[s])
    gt = shape_type[s]
    if gt == MESH or gt == CONVEX_MESH:
        ol, dl = map_ray_to_local(X, o, d, shape_scale[s])
        tm, _nm, _u, _v, _f = ray_intersect_mesh_no_normal(ol, dl, shape_scale[s], shape_ptr[s], False, length)
        if tm > length:
            return -1.0
        return tm
    tp, _np = ray_intersect_shape_no_normal(X, shape_scale[s], gt, o, d, False)
    if tp > length:
        return -1.0
    return tp


@wp.func
def hit_time(p0: wp.vec3, v0: wp.vec3, gravity: wp.vec3, g: float, ta: float, tb: float, length: float, t: float,
             pa: wp.vec3, d: wp.vec3) -> float:
    """Flight time to the point pa + t·d of the path's segment (ta, tb): linear without gravity, else the exact
    ballistic time from the drop along g, the root nearest the linear estimate (a rising path meets a height twice)."""
    th = ta + (tb - ta) * t / length
    if g > 0.0:
        gh = gravity / g
        drop = wp.dot(pa + t * d - p0, gh)
        vg = wp.dot(v0, gh)
        root = wp.sqrt(wp.max(vg * vg + 2.0 * g * drop, 0.0))
        t_up = (-vg - root) / g
        t_dn = (-vg + root) / g
        th_lin = th
        th = t_dn
        if t_up >= 0.0 and wp.abs(t_up - th_lin) < wp.abs(t_dn - th_lin):
            th = t_up
    return th


@wp.func
def put_hit(w: int, r: int, kind: int, idx: int, cell: int, point: wp.vec3, vel: wp.vec3, s: float,
            hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32], hit_cell: wp.array2d[wp.int32],
            hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3], hit_s: wp.array2d[wp.float32]):
    hit_kind[w, r] = kind
    hit_idx[w, r] = idx
    hit_cell[w, r] = cell
    hit_p[w, r] = point                                       # chart frame for a chart hit, else world
    hit_w[w, r] = vel                                         # the impact velocity
    hit_s[w, r] = s


@wp.kernel
def cast_rays(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32],
              shape_transform: wp.array[wp.transform], shape_type: wp.array[wp.int32],
              shape_scale: wp.array[wp.vec3], shape_ptr: wp.array[wp.uint64], occ: wp.array2d[wp.int32],
              charts: wp.array[Chart], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
              prof: wp.array[wp.vec3], pools: wp.array[PoolSpec], pbody: wp.array2d[wp.int32],
              pressure: wp.array2d[wp.float64], nozzles: wp.array[NozzleSpec], nbody: wp.array2d[wp.int32],
              ray_noz: wp.array[wp.int32], npos: wp.array2d[wp.vec3], ray_dir: wp.array2d[wp.vec3],
              ray_p0: wp.array2d[wp.vec3], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
              discharge: wp.array2d[wp.float64], gravity: wp.vec3,
              hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32], hit_cell: wp.array2d[wp.int32],
              hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3], hit_s: wp.array2d[wp.float32]):
    """One ray of one world: its ballistic path, cast segment by segment; the first hit is stored (a spray ray passes
    occluders to a chart behind them, where its footprint's cells are shaded one by one: ``supply_shadow``; it is
    caught by the first only when no chart lies behind). The exit point ``npos`` [W, nozzles] and directions
    ``ray_dir`` [W, rays] are per world (a rim pour moves with its vessel); a drip ray leaves its own point ``ray_p0``
    at ``ray_dir`` m/s (body frame) plus the vessel's velocity there."""
    w, r = wp.tid()
    noz = nozzles[ray_noz[r]]
    hit_kind[w, r] = NONE
    q = -discharge[w, noz.edge]
    if q <= wp.float64(0.0):
        return
    nb = nbody[w, ray_noz[r]]
    X_n = pool_frame(body_q, nb)
    p0 = wp.transform_point(X_n, npos[w, ray_noz[r]])
    v0 = float(q) / noz.exit_area * wp.transform_vector(X_n, ray_dir[w, r])
    if noz.drip == 1:                                      # a vessel's drip: its frame, plus its velocity
        p0 = wp.transform_point(X_n, ray_p0[w, r])
        v0 = wp.transform_vector(X_n, ray_dir[w, r])
        if nb >= 0:
            v0 = v0 + point_velocity(body_qd, com_world(body_q, body_com, nb), nb, p0)
    elif noz.drip == 2:                                    # a stage's drip: world point and velocity as given,
        p0 = ray_p0[w, r]                                  # lifted against gravity (a tool pressed into the chart)
        v0 = ray_dir[w, r]
        if wp.length(gravity) > 0.0:
            p0 = p0 - STAGE_LIFT * wp.normalize(gravity)
    speed = wp.length(v0)
    g = wp.length(gravity)
    t_end = 2.0 * noz.reach / (speed + wp.sqrt(speed * speed + 2.0 * g * noz.reach))   # straight down: reach
    s_path = float(0.0)
    spray = int(0)                                         # a spray passes occluders to a chart behind them
    if noz.pattern != JET:                                 # (supply_shadow), caught by the first one otherwise
        spray = 1
    o_idx = int(-1)
    o_point = wp.vec3()
    o_vel = wp.vec3()
    o_s = float(0.0)
    for k in range(noz.segments):
        ta = t_end * float(k) / float(noz.segments)
        tb = t_end * float(k + 1) / float(noz.segments)
        pa = p0 + v0 * ta + 0.5 * gravity * ta * ta
        pb = p0 + v0 * tb + 0.5 * gravity * tb * tb
        seg = pb - pa
        length = wp.length(seg)
        if length <= 0.0:
            continue
        d = seg / length
        best_t = float(1.0e30)
        kind = int(NONE)
        idx = int(-1)
        cell = int(-1)
        point = wp.vec3()
        for c in range(charts.shape[0]):                      # charts: the wetted side within the extent
            X = chart_xform(body_q, cbody[w, c], clocal[w, c])
            h = ray_hit(charts[c], prof, X, pa, d)
            if h.cell >= 0 and h.t <= length and h.t < best_t:
                best_t = float(h.t)
                kind = int(CHART)
                idx = int(c)
                cell = int(h.cell)
                point = wp.vec3(h.point)
        for pi in range(pools.shape[0]):                       # pools: the free surface, from above
            pool = pools[pi]
            Xp = pool_frame(body_q, pbody[w, pi])
            ol = wp.transform_point(wp.transform_inverse(Xp), pa)
            dl = wp.transform_vector(wp.transform_inverse(Xp), d)
            if dl[2] < 0.0 and pi != noz.own_pool:
                zl = pool.floor[2] + pool_level(pool, pressure[w, pool.tank])
                t = (zl - ol[2]) / dl[2]
                hp = ol + t * dl
                inside = wp.abs(hp[0] - pool.floor[0]) <= pool.half[0] and wp.abs(hp[1] - pool.floor[1]) <= pool.half[1]
                if pool.round != 0:
                    inside = wp.length(wp.vec2(hp[0] - pool.floor[0], hp[1] - pool.floor[1])) <= pool.half[0]
                if t >= 0.0 and t <= length and inside and t < best_t - TIE:
                    best_t = float(t)
                    kind = int(POOL)
                    idx = int(pi)
                    cell = int(-1)
                    point = pa + t * d
        occ_t = float(1.0e30)                                  # occluders: the nearest other shape in front
        occ_i = int(-1)
        for si in range(occ.shape[1]):
            s = occ[w, si]
            if s < 0:
                break
            if noz.drip != 0 and nb >= 0 and shape_body[s] == nb:  # a drip leaves its own vessel's wall
                continue
            t = shape_t(body_q, shape_body, shape_transform, shape_type, shape_scale, shape_ptr, s, pa, d, length)
            if t >= 0.0 and t < occ_t:
                occ_t = t
                occ_i = si                                    # its occluder slot: occ[w, si] is the shape
        if occ_i >= 0 and occ_t < best_t - TIE:
            if spray == 0:
                best_t = occ_t
                kind = int(SHAPE)
                idx = occ_i
                cell = int(-1)
                point = pa + occ_t * d
            elif o_idx < 0:
                o_idx = occ_i
                o_point = pa + occ_t * d
                o_vel = v0 + gravity * hit_time(p0, v0, gravity, g, ta, tb, length, occ_t, pa, d)
                o_s = s_path + occ_t
        if kind != NONE:
            if kind != CHART and o_idx >= 0:                  # a pool behind an occluder: caught by the occluder
                put_hit(w, r, SHAPE, o_idx, -1, o_point, o_vel, o_s, hit_kind, hit_idx, hit_cell, hit_p, hit_w, hit_s)
            else:                                             # a chart hit's point is in the chart frame
                th = hit_time(p0, v0, gravity, g, ta, tb, length, best_t, pa, d)
                put_hit(w, r, kind, idx, cell, point, v0 + gravity * th, s_path + best_t, hit_kind, hit_idx,
                        hit_cell, hit_p, hit_w, hit_s)
            return
        s_path = s_path + length
    if o_idx >= 0:                                            # a spray that met only an occluder
        put_hit(w, r, SHAPE, o_idx, -1, o_point, o_vel, o_s, hit_kind, hit_idx, hit_cell, hit_p, hit_w, hit_s)


@wp.func
def weight_sum(noz: NozzleSpec, ray_wt: wp.array[wp.float32]) -> wp.float64:
    """Σ of a nozzle's ray fractions in float64 (float32 fractions need not sum to exactly 1)."""
    s = wp.float64(0.0)
    for r in range(noz.ray_start, noz.ray_start + noz.ray_count):
        s = s + wp.float64(ray_wt[r])
    return s


@wp.func
def hit_world(body_q: wp.array[wp.transform], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
              w: wp.int32, kind: wp.int32, idx: wp.int32, p: wp.vec3) -> wp.vec3:
    if kind == CHART:
        return wp.transform_point(chart_xform(body_q, cbody[w, idx], clocal[w, idx]), p)
    return p


@wp.kernel
def jet_forces(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], shape_body: wp.array[wp.int32],
               cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], nozzles: wp.array[NozzleSpec],
               nbody: wp.array2d[wp.int32], npos: wp.array2d[wp.vec3], ray_dir: wp.array2d[wp.vec3],
               ray_wt: wp.array[wp.float32], ray_p0: wp.array2d[wp.vec3],
               discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32],
               hit_cell: wp.array2d[wp.int32], hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3],
               hit_on: wp.array2d[wp.float32], cnrm: wp.array[wp.vec3], occ: wp.array2d[wp.int32],
               body_f: wp.array[wp.spatial_vector]):
    """Momentum on the struck body at the hit point: on a chart, the normal part ρ·q·(w·n)·n plus the tangential part
    times ``hit_on`` (1 while the jump stands on the chart and its sheet's friction hands the momentum over, 0 when the
    sheet runs off; the last deposit's value); on a shape, all of ρ·q·w. The thrust -ρ·Q·v0 on the nozzle's body (a
    drip's, -ρ·q·v at each hole on its vessel)."""
    w = wp.tid()
    for n in range(nozzles.shape[0]):
        noz = nozzles[n]
        q_all = float(-discharge[w, noz.edge])
        if q_all <= 0.0:
            continue
        nb = nbody[w, n]
        X_n = pool_frame(body_q, nb)
        exit_v = q_all / noz.exit_area
        thrust = wp.vec3()
        norm = float(weight_sum(noz, ray_wt))
        for r in range(noz.ray_start, noz.ray_start + noz.ray_count):
            frac = ray_wt[r] / norm
            q = q_all * noz.efficiency * frac
            if noz.drip == 0:
                thrust = thrust - noz.rho * q_all * frac * exit_v * wp.transform_vector(X_n, ray_dir[w, r])
            elif noz.drip == 1 and nb >= 0:                    # a drip pushes its vessel back at its hole
                add_wrench(body_f, nb, wrench_at(-noz.rho * q_all * frac * wp.transform_vector(X_n, ray_dir[w, r]),
                                                 wp.transform_point(X_n, ray_p0[w, r]),
                                                 com_world(body_q, body_com, nb)))
            kind = hit_kind[w, r]
            b = int(-1)
            if kind == CHART:
                b = cbody[w, hit_idx[w, r]]
            elif kind == SHAPE or kind == LUMP:
                b = shape_body[occ[w, hit_idx[w, r]]]
            if b >= 0:
                pw = hit_world(body_q, cbody, clocal, w, kind, hit_idx[w, r], hit_p[w, r])
                mom = hit_w[w, r]
                if kind == CHART:
                    X = chart_xform(body_q, cbody[w, hit_idx[w, r]], clocal[w, hit_idx[w, r]])
                    nrm = wp.transform_vector(X, cnrm[hit_cell[w, r]])
                    wn = wp.dot(mom, nrm) * nrm
                    mom = wn + hit_on[w, r] * (mom - wn)
                add_wrench(body_f, b, wrench_at(noz.rho * q * mom, pw, com_world(body_q, body_com, b)))
        if nb >= 0:
            add_wrench(body_f, nb, wrench_at(thrust, wp.transform_point(X_n, npos[w, n]), com_world(body_q, body_com,
                                                                                                    nb)))


@wp.kernel
def drain_capacity(dt_film: wp.float64, dstart: wp.array[wp.int32], dcell: wp.array[wp.int32],
                   dkind: wp.array[wp.int32], dcd: wp.array[wp.float32], dopen: wp.array2d[wp.float32],
                   h: wp.array2d[wp.float64], area: wp.array[wp.float64], g: wp.float64,
                   drain_cap: wp.array2d[wp.float64]):
    """``drain_cap`` per drain cell: free drains take everything (part open: that share of the standing water per
    film step; ``dt_film`` = the film's stride × the exchange dt, so a strided film drains at the same rate), an
    orifice Cd·A·sqrt(2·g·h̄)·opening over the drain's cells, an open edge nothing beyond its edge flux. The opening
    input is clamped to [0, 1] (0 = a stopper)."""
    w = wp.tid()
    for d in range(dkind.shape[0]):
        u = wp.clamp(wp.float64(dopen[w, d]), wp.float64(0.0), wp.float64(1.0))
        cap = wp.float64(0.0)
        if dkind[d] == 0:
            if u >= wp.float64(1.0):
                cap = wp.float64(1.0e30)
            elif u > wp.float64(0.0):
                for m in range(dstart[d], dstart[d + 1]):
                    drain_cap[w, dcell[m]] = u * h[w, dcell[m]] / dt_film
                continue
        elif dkind[d] == 1:
            vh = wp.float64(0.0)
            a = wp.float64(0.0)
            for m in range(dstart[d], dstart[d + 1]):
                vh = vh + h[w, dcell[m]] * area[dcell[m]]
                a = a + area[dcell[m]]
            cap = u * wp.float64(dcd[d]) * wp.sqrt(wp.float64(2.0) * g * vh / a) / a
        for m in range(dstart[d], dstart[d + 1]):
            drain_cap[w, dcell[m]] = cap


@wp.kernel
def drain_sums(dt: wp.float64, dstart: wp.array[wp.int32], dcell: wp.array[wp.int32], dto: wp.array[wp.int32],
               area: wp.array[wp.float64], drained: wp.array2d[wp.float64], inflow: wp.array2d[wp.float64],
               ledger: wp.array2d[wp.float64]):
    """Each drain's volume this exchange (Σ drained·area, fixed order) into its tank's inflow, or away; zeroes it."""
    w = wp.tid()
    for d in range(dto.shape[0]):
        vol = wp.float64(0.0)
        for m in range(dstart[d], dstart[d + 1]):
            c = dcell[m]
            vol = vol + drained[w, c] * area[c]
            drained[w, c] = wp.float64(0.0)
        if dto[d] >= 0:
            inflow[w, dto[d]] = inflow[w, dto[d]] + vol / dt
            ledger[w, DRAINED] = ledger[w, DRAINED] + vol
        else:
            ledger[w, AWAY] = ledger[w, AWAY] + vol


@wp.kernel
def tank_levels(body_q: wp.array[wp.transform], pools: wp.array[PoolSpec], pbody: wp.array2d[wp.int32],
                pressure: wp.array2d[wp.float64], level: wp.array2d[wp.float32]):
    """World height of each pool's free surface (interface A11)."""
    w = wp.tid()
    for k in range(pools.shape[0]):
        pool = pools[k]
        top = pool.floor + wp.vec3(0.0, 0.0, pool_level(pool, pressure[w, pool.tank]))
        level[w, k] = wp.transform_point(pool_frame(body_q, pbody[w, k]), top)[2]


@wp.kernel
def copy_openings(src: wp.array2d[wp.float32], opening: wp.array2d[wp.float32]):
    """The per-edge inputs (rows of ``src`` [m, W]) into the network's ``opening`` [W, m]."""
    w = wp.tid()
    for e in range(src.shape[0]):
        opening[w, e] = src[e, w]
