# SPDX-License-Identifier: Apache-2.0
"""Film-pool exchange ([internal note] notes/dc5-sf-pools-proposal.md, agreed with WP15 2026-11-06): the chart cells under a
flooding pool's water are ``flooded``, the film on them joins the pool, and a cell leaving the water keeps a
withdrawal film of the pool's liquid.

Each exchange, in ``before_film``:
1. ``flood_boxes`` [W, K]: per chart, the cell box that may hold any flooding pool's region (the footprint ± half,
   from the floor up to the level, in the pool body's frame), and its union with last exchange's box, so that cells
   flooded then are reset;
2. ``flood_cells`` [W, K, rows]: each cell in the union box takes the first pool whose region holds its centre, or
   -1 (a flooded cell holding film is marked ``cwet``, so the dry-tile skip steps it). A cell that was flooded and
   is not now asks its pool for a withdrawal film. The film is Landau-Levich's h = 0.94·l_c·Ca^(2/3), with
   l_c = sqrt(σ/(ρg)) and Ca = μU/σ (Landau & Levich 1942), at the speed U the surface is drawn along itself
   relative to the pool (the velocity's component in the cell's tangent plane). It is bounded by the gravity-viscous
   thickness sqrt(μU/(ρg)) (Derjaguin 1943) and the pool's ``withdraw_max``;
3. ``withdraw_grant`` [W]: per pool, the requests summed in row order and granted up to what the tank holds, taken
   from the tank as a negative inflow and booked as ``withdrawn`` (the rest as ``refused``);
4. ``withdraw_enter`` [W, K, rows]: the granted films go into ``src_acc`` (cells marked ``cwet``).

After the film (``after_film``), ``pool_sums`` [W, K, rows] and ``pool_into_tanks`` [W] move the film the flooded
cells gave up (the film's ``pooled``, m) into the pools' tanks, booked as ``pooled``.

Every element has one writer, sums run in row order, and launch sizes are fixed, so runs are deterministic and
graph-capturable.

v1 limits:
- A cell's flooding is decided at its centre.
- A vessel's inside floods with its outside; the cavity pools of part 2 fix that.
- U leaves out the level's own motion.
- A face that leaves the water level (within ``puddle_tilt``, an ESTIMATE: hysteresis is not modelled) keeps the
  static puddle of its chart's contact angle, at most the deepest water it had over it. Without a contact angle it
  comes out dry.
"""

from __future__ import annotations

import math
from typing import Any

from alloy._vendor import newton
import warp as wp

from alloy.plugins.errors import PluginError

from alloy.plugins.funcs import com_world, point_velocity
from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.geometry import chart_xform
from alloy.plugins.surface_flow.supply_cavity import Cavity, in_cavity
from alloy.plugins.surface_flow.supply_deposit import EMPTY_BOX, cell_of, union_box
from alloy.plugins.surface_flow.supply_kernels import POOLED, REFUSED, WITHDRAWN, PoolSpec, pool_frame, pool_level
from alloy.plugins.surface_flow.supply_species import SP_WITHDRAWN, NetSpecies, into_tank, refresh, tank_conc

BIG = wp.constant(1.0e30)
START_MARGIN = wp.constant(1.01)   # an empty vessel's floor film starts its pool past this x what a drained face keeps


@wp.func
def in_region(pool: PoolSpec, F: wp.transform, level: float, x: wp.vec3) -> int:
    """1 when world point ``x`` lies in the pool's water: over its footprint, from the floor up to the level (an
    empty pool holds none)."""
    if level <= 0.0:
        return 0
    q = wp.transform_point(wp.transform_inverse(F), x) - pool.floor
    if wp.abs(q[0]) > pool.half[0] or wp.abs(q[1]) > pool.half[1]:
        return 0
    return int(q[2] >= -1.0e-6 and q[2] <= level)


@wp.func
def region_box(chart: Chart, prof: wp.array[wp.vec3], mn: wp.vec3, mx: wp.vec3) -> wp.vec4i:
    """Cells of ``chart`` that may lie in the chart-frame box [mn, mx]: plane rows and columns, or a revolution
    chart's rows whose profile segment meets the box (a whole turn)."""
    if chart.kind == PLANE:
        h = chart.half
        if mn[2] > 1.0e-6 or mx[2] < -1.0e-6 or mx[0] < -h[0] or mn[0] > h[0] or mx[1] < -h[1] or mn[1] > h[1]:
            return EMPTY_BOX
        i0 = wp.clamp(int(wp.floor((mn[0] + h[0]) / chart.step[0])), 0, chart.nx - 1)
        i1 = wp.clamp(int(wp.floor((mx[0] + h[0]) / chart.step[0])), 0, chart.nx - 1)
        j0 = wp.clamp(int(wp.floor((mn[1] + h[1]) / chart.step[1])), 0, chart.ny - 1)
        j1 = wp.clamp(int(wp.floor((mx[1] + h[1]) / chart.step[1])), 0, chart.ny - 1)
        return wp.vec4i(i0, i1, j0, j1)
    cx = wp.clamp(0.0, mn[0], mx[0])
    cy = wp.clamp(0.0, mn[1], mx[1])
    r_lo = wp.sqrt(cx * cx + cy * cy)
    r_hi = wp.max(wp.max(wp.length(wp.vec2(mn[0], mn[1])), wp.length(wp.vec2(mn[0], mx[1]))),
                  wp.max(wp.length(wp.vec2(mx[0], mn[1])), wp.length(wp.vec2(mx[0], mx[1]))))
    i0 = int(1 << 30)
    i1 = int(-1)
    for i in range(chart.nx):
        a = prof[chart.prof_start + i]
        b = prof[chart.prof_start + i + 1]
        if wp.max(a[0], b[0]) >= r_lo and wp.min(a[0], b[0]) <= r_hi and wp.max(a[1], b[1]) >= mn[2] and \
                wp.min(a[1], b[1]) <= mx[2]:
            i0 = wp.min(i0, i)
            i1 = wp.max(i1, i)
    if i1 < 0:
        return EMPTY_BOX
    return wp.vec4i(i0, i1, 0, chart.ny - 1)


@wp.func
def withdrawal(pool: PoolSpec, u: float) -> float:
    """The film [m] a surface keeps leaving the pool's liquid at speed ``u``: Landau-Levich, bounded by the
    gravity-viscous thickness and ``h_max``."""
    if u <= 0.0:
        return 0.0
    mu_u = pool.rho * pool.nu * u
    rho_g = float(pool.area / pool.kappa)                  # κ = A/(ρg)
    h_ll = 0.94 * wp.pow(mu_u, 2.0 / 3.0) / (wp.pow(pool.sigma, 1.0 / 6.0) * wp.sqrt(rho_g))
    return wp.min(wp.min(h_ll, wp.sqrt(mu_u / rho_g)), pool.h_max)


@wp.func
def puddle(pool: PoolSpec, s: float) -> float:
    """The static puddle a level face keeps, 2·l_c·sin(θ/2) (``s`` = sin(θ/2); de Gennes, Brochard-Wyart & Quéré,
    *Capillarity and Wetting Phenomena* §2.1)."""
    return 2.0 * wp.sqrt(pool.sigma / float(pool.area / pool.kappa)) * s


@wp.kernel
def flood_boxes(charts: wp.array[Chart], prof: wp.array[wp.vec3], cbody: wp.array2d[wp.int32],
                clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform], pools: wp.array[PoolSpec],
                pbody: wp.array2d[wp.int32], pressure: wp.array2d[wp.float64], cavs: wp.array[Cavity],
                prev: wp.array2d[wp.vec4i], cur: wp.array2d[wp.vec4i], proc: wp.array2d[wp.vec4i]):
    """Chart k's box under any flooding pool's water (``cur``; a cavity pool's whole inside, empty or not: its floor
    film can start it, ``cavity_start``), and its union with last exchange's (``proc``)."""
    w, k = wp.tid()
    chart = charts[k]
    Xi = wp.transform_inverse(chart_xform(body_q, cbody[w, k], clocal[w, k]))
    b = EMPTY_BOX
    for p in range(pools.shape[0]):
        pool = pools[p]
        level = pool_level(pool, pressure[w, pool.tank])
        cav = cavs[p]
        if pool.flood == 0 or (level <= 0.0 and cav.on == 0):
            continue
        F = pool_frame(body_q, pbody[w, p])
        lo = wp.vec3(-pool.half[0], -pool.half[1], 0.0)
        hi = wp.vec3(pool.half[0], pool.half[1], level)
        if cav.on != 0:
            reach = cav.r_max + cav.tol
            lo = wp.vec3(-reach, -reach, cav.z0 - cav.tol)
            hi = wp.vec3(reach, reach, cav.z_rim + cav.tol)
        mn = wp.vec3(BIG, BIG, BIG)
        mx = wp.vec3(-BIG, -BIG, -BIG)
        for corner in range(8):
            c = lo
            if (corner & 1) == 1:
                c = wp.vec3(hi[0], c[1], c[2])
            if (corner & 2) == 2:
                c = wp.vec3(c[0], hi[1], c[2])
            if (corner & 4) == 4:
                c = wp.vec3(c[0], c[1], hi[2])
            lc = pool.floor + c
            q = wp.transform_point(Xi, wp.transform_point(F, lc))
            mn = wp.min(mn, q)
            mx = wp.max(mx, q)
        b = union_box(chart, b, region_box(chart, prof, mn, mx))
    cur[w, k] = b
    proc[w, k] = union_box(chart, prev[w, k], b)


@wp.func
def in_any_cavity(w: int, x: wp.vec3, n: wp.vec3, body_q: wp.array[wp.transform], pools: wp.array[PoolSpec],
                  pbody: wp.array2d[wp.int32], cavs: wp.array[Cavity], cprof: wp.array[wp.vec2]) -> int:
    """1 when the cell at world point ``x`` (world normal ``n``) is inside any cavity pool's vessel (only its own
    water floods it)."""
    inside = int(0)
    for p in range(pools.shape[0]):
        if inside == 0 and cavs[p].on != 0:
            Fi = wp.transform_inverse(pool_frame(body_q, pbody[w, p]))
            q = wp.transform_point(Fi, x) - pools[p].floor
            inside = in_cavity(cavs[p], cprof, q, wp.transform_vector(Fi, n))
    return inside


@wp.func
def cell_flood(w: int, x: wp.vec3, n: wp.vec3, body_q: wp.array[wp.transform], pools: wp.array[PoolSpec],
               pbody: wp.array2d[wp.int32], pressure: wp.array2d[wp.float64], cavs: wp.array[Cavity],
               cprof: wp.array[wp.vec2], surf: wp.array2d[wp.vec4]) -> int:
    """The first flooding pool whose water holds world point ``x``, or -1: a cavity pool's below its surface plane,
    a box pool's outside every vessel."""
    now = int(-1)
    vessel = in_any_cavity(w, x, n, body_q, pools, pbody, cavs, cprof)
    for p in range(pools.shape[0]):
        pool = pools[p]
        if now < 0 and pool.flood != 0:
            F = pool_frame(body_q, pbody[w, p])
            level = pool_level(pool, pressure[w, pool.tank])
            if cavs[p].on != 0:
                q = wp.transform_point(wp.transform_inverse(F), x) - pool.floor
                sv = surf[w, p]
                nq = wp.transform_vector(wp.transform_inverse(F), n)
                if level > 0.0 and in_cavity(cavs[p], cprof, q, nq) != 0 and \
                        wp.dot(wp.vec3(sv[0], sv[1], sv[2]), q) <= sv[3]:
                    now = p
            elif vessel == 0 and in_region(pool, F, level, x) != 0:
                now = p
    return now


@wp.func
def lowest_inside(cav: Cavity, cprof: wp.array[wp.vec2], up: wp.vec3) -> float:
    """The lowest point of a cavity's inside along ``up`` (body frame, from the floor point): min over its profile's
    nodes of up_z·z - s·r (exact for straight segments; an empty vessel's surface plane only bounds it, WP15)."""
    s = wp.sqrt(up[0] * up[0] + up[1] * up[1])
    lo = float(1.0e30)
    for k in range(cav.np):
        nd = cprof[cav.p0 + k]
        lo = wp.min(lo, up[2] * nd[1] - s * nd[0])
    return lo


@wp.func
def cavity_start(w: int, x: wp.vec3, n: wp.vec3, h: float, s_pud: float, body_q: wp.array[wp.transform],
                 pools: wp.array[PoolSpec], pbody: wp.array2d[wp.int32], cavs: wp.array[Cavity],
                 cprof: wp.array[wp.vec2], surf: wp.array2d[wp.vec4]) -> int:
    """The first flooding cavity pool whose surface is no higher than ``cav.tol`` over its lowest point (empty, or a
    rounding's femtolitres: WP15) with world point ``x`` inside it in that band and film ``h`` over ``START_MARGIN``
    x what a cell leaving its water keeps (a level face's puddle, ``s_pud`` = sin(θ/2), else ``withdraw_max``)."""
    for p in range(pools.shape[0]):
        pool = pools[p]
        cav = cavs[p]
        if pool.flood == 0 or cav.on == 0:
            continue
        sv = surf[w, p]
        up = wp.vec3(sv[0], sv[1], sv[2])
        band = lowest_inside(cav, cprof, up) + cav.tol      # its inside's lowest point (``lowest_inside``) + tol
        F = pool_frame(body_q, pbody[w, p])
        Fi = wp.transform_inverse(F)
        q = wp.transform_point(Fi, x) - pool.floor
        if sv[3] > band or in_cavity(cav, cprof, q, wp.transform_vector(Fi, n)) == 0 or wp.dot(up, q) > band:
            continue
        keep = pool.h_max
        if s_pud > 0.0 and wp.dot(n, wp.transform_vector(F, up)) >= pool.flat_cos:
            keep = wp.max(keep, puddle(pool, s_pud))
        if h > START_MARGIN * keep:
            return p
    return -1


@wp.func
def exit_speed(x: wp.vec3, n: wp.vec3, cb: int, pb: int, body_q: wp.array[wp.transform],
               body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], v_surf: wp.vec3) -> float:
    """The speed world point ``x`` (unit normal ``n``) on the chart's body ``cb`` is drawn along its own surface,
    relative to the pool's water (the pool body ``pb``, and ``v_surf``, its surface's own motion on that body): the
    relative velocity's tangential component."""
    v = -v_surf
    if cb >= 0:
        v = v + point_velocity(body_qd, com_world(body_q, body_com, cb), cb, x)
    if pb >= 0:
        v = v - point_velocity(body_qd, com_world(body_q, body_com, pb), pb, x)
    return wp.length(v - wp.dot(v, n) * n)


@wp.func
def surface_motion(dt: wp.float64, F: wp.transform, q: wp.vec3, now: wp.vec4, prev: wp.vec4) -> wp.vec3:
    """A cavity's surface's world velocity past body-frame point ``q`` (from the floor): the change of its height
    above q over the exchange (it drains, fills or tips), along its up (0 on the first exchange)."""
    up_p = wp.vec3(prev[0], prev[1], prev[2])
    if wp.length(up_p) < 0.5:
        return wp.vec3(0.0, 0.0, 0.0)
    up = wp.vec3(now[0], now[1], now[2])
    rate = ((now[3] - wp.dot(up, q)) - (prev[3] - wp.dot(up_p, q))) / float(dt)
    return wp.transform_vector(F, up) * rate


@wp.kernel
def flood_cells(dt: wp.float64, charts: wp.array[Chart], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                cpos: wp.array[wp.vec3], cnrm: wp.array[wp.vec3], area: wp.array[wp.float64],
                pools: wp.array[PoolSpec], pbody: wp.array2d[wp.int32], pressure: wp.array2d[wp.float64],
                cavs: wp.array[Cavity], cprof: wp.array[wp.vec2], surf: wp.array2d[wp.vec4],
                surf_prev: wp.array2d[wp.vec4], proc: wp.array2d[wp.vec4i], h: wp.array2d[wp.float64],
                cpud: wp.array[wp.float32],
                flooded: wp.array2d[wp.int32], cwet: wp.array2d[wp.int32], depth: wp.array2d[wp.float32],
                req: wp.array2d[wp.float64], req_pool: wp.array2d[wp.int32], part: wp.array4d[wp.float64]):
    """Row ``ii`` of chart k's box: each cell's pool now (or -1) and the deepest water it has had over it; a cell
    leaving the water requests its withdrawal film from the pool it left (``req`` m³, the row's sums per pool in
    ``part``): Landau-Levich, or on a level face the puddle of its chart's contact angle, at most that depth."""
    w, k, ii = wp.tid()
    b = proc[w, k]
    i = b[0] + ii
    if i > b[1]:
        return
    for p in range(pools.shape[0]):
        part[w, k, ii, p] = wp.float64(0.0)
    chart = charts[k]
    cb = cbody[w, k]
    X = chart_xform(body_q, cb, clocal[w, k])
    for j in range(b[2], b[3] + 1):
        c = cell_of(chart, i, j)
        x = wp.transform_point(X, cpos[c])
        nrm = wp.transform_vector(X, cnrm[c])
        now = cell_flood(w, x, nrm, body_q, pools, pbody, pressure, cavs, cprof, surf)
        if now < 0:                                          # an empty vessel's floor film starts its pool
            now = cavity_start(w, x, nrm, float(h[w, c]), cpud[k], body_q, pools, pbody, cavs, cprof, surf)
        was = flooded[w, c]
        flooded[w, c] = now
        if now >= 0 and h[w, c] > wp.float64(0.0):
            cwet[w, c] = 1                                   # its film goes to the pool at this film step
        if now >= 0:
            pn = pools[now]
            qn = wp.transform_point(wp.transform_inverse(pool_frame(body_q, pbody[w, now])), x) - pn.floor
            dn = pool_level(pn, pressure[w, pn.tank]) - qn[2]
            if cavs[now].on != 0:                            # a cavity's surface is normal to gravity
                sn = surf[w, now]
                dn = sn[3] - wp.dot(wp.vec3(sn[0], sn[1], sn[2]), qn)
            depth[w, c] = wp.max(depth[w, c], dn)
        dv = wp.float64(0.0)
        if was >= 0 and now < 0:
            pool = pools[was]
            Fw = pool_frame(body_q, pbody[w, was])
            up = wp.transform_vector(Fw, wp.vec3(0.0, 0.0, 1.0))
            v_surf = wp.vec3(0.0, 0.0, 0.0)
            if cavs[was].on != 0:                            # a vessel's water moves on its own body
                sw = surf[w, was]
                up = wp.transform_vector(Fw, wp.vec3(sw[0], sw[1], sw[2]))
                qw = wp.transform_point(wp.transform_inverse(Fw), x) - pool.floor
                v_surf = surface_motion(dt, Fw, qw, sw, surf_prev[w, was])
            hw = withdrawal(pool, exit_speed(x, nrm, cb, pbody[w, was], body_q, body_qd, body_com, v_surf))
            if cpud[k] > 0.0 and wp.dot(nrm, up) >= pool.flat_cos:
                hw = wp.max(hw, wp.min(puddle(pool, cpud[k]), depth[w, c]))
            dv = wp.float64(hw) * area[c]
            part[w, k, ii, was] = part[w, k, ii, was] + dv
        if now < 0:
            depth[w, c] = 0.0
        req[w, c] = dv
        req_pool[w, c] = was


@wp.kernel
def withdraw_grant(dt: wp.float64, pools: wp.array[PoolSpec], proc: wp.array2d[wp.vec4i],
                   part: wp.array4d[wp.float64], pressure: wp.array2d[wp.float64], ns: NetSpecies,
                   ratio: wp.array2d[wp.float64], conc: wp.array2d[wp.vec4d], inflow: wp.array2d[wp.float64],
                   ledger: wp.array2d[wp.float64]):
    """Per pool, the withdrawal requests (rows in order) granted up to the tank's volume, out of the tank, with its
    species at its concentration (``conc``, per film species)."""
    w = wp.tid()
    for p in range(pools.shape[0]):
        pool = pools[p]
        total = wp.float64(0.0)
        for k in range(proc.shape[1]):
            b = proc[w, k]
            for ii in range(b[1] - b[0] + 1):
                total = total + part[w, k, ii, p]
        r = wp.float64(1.0)
        cs = wp.vec4d()
        if total > wp.float64(0.0):
            have = pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0))
            r = wp.min(wp.float64(1.0), have / total)
            got = total * r
            inflow[w, pool.tank] = inflow[w, pool.tank] - got / dt
            ledger[w, WITHDRAWN] = ledger[w, WITHDRAWN] + got
            ledger[w, REFUSED] = ledger[w, REFUSED] + (total - got)
            for ks in range(ns.n_film):
                cs[ks] = tank_conc(ns, w, pool.tank, ks)
                kn = ns.map[ks]
                if kn >= 0:
                    ns.inflow[w, pool.tank, kn] = ns.inflow[w, pool.tank, kn] - got * cs[ks] / dt
                    ns.led[w, SP_WITHDRAWN, ks] = ns.led[w, SP_WITHDRAWN, ks] + got * cs[ks]
        ratio[w, p] = r
        conc[w, p] = cs


@wp.kernel
def withdraw_enter(charts: wp.array[Chart], area: wp.array[wp.float64], proc: wp.array2d[wp.vec4i],
                   cur: wp.array2d[wp.vec4i], req: wp.array2d[wp.float64], req_pool: wp.array2d[wp.int32],
                   ratio: wp.array2d[wp.float64], conc: wp.array2d[wp.vec4d], ns: NetSpecies,
                   src_acc: wp.array2d[wp.float64], cwet: wp.array2d[wp.int32], prev: wp.array2d[wp.vec4i]):
    """Row ``ii``: the granted withdrawal films (and their species) into the pending volume; this exchange's box
    becomes ``prev``."""
    w, k, ii = wp.tid()
    if ii == 0:
        prev[w, k] = cur[w, k]
    b = proc[w, k]
    i = b[0] + ii
    if i > b[1]:
        return
    chart = charts[k]
    for j in range(b[2], b[3] + 1):
        c = cell_of(chart, i, j)
        dv = req[w, c]
        if dv > wp.float64(0.0):
            p = req_pool[w, c]
            h = dv * ratio[w, p] / area[c]
            src_acc[w, c] = src_acc[w, c] + h
            for ks in range(ns.n_film):
                ns.src[w, c, ks] = ns.src[w, c, ks] + h * conc[w, p][ks]
            cwet[w, c] = 1


@wp.kernel
def pool_sums(charts: wp.array[Chart], area: wp.array[wp.float64], box: wp.array2d[wp.vec4i],
              flooded: wp.array2d[wp.int32], pooled: wp.array2d[wp.float64], ns: NetSpecies,
              part: wp.array4d[wp.float64], part_sp: wp.array4d[wp.vec4d]):
    """Row ``ii`` of this exchange's box: the film (m³) and species (kg) its flooded cells gave up, per pool;
    ``pooled`` and their ``drained_species`` are zeroed."""
    w, k, ii = wp.tid()
    b = box[w, k]
    i = b[0] + ii
    if i > b[1]:
        return
    for p in range(part.shape[3]):
        part[w, k, ii, p] = wp.float64(0.0)
        part_sp[w, k, ii, p] = wp.vec4d()
    chart = charts[k]
    for j in range(b[2], b[3] + 1):
        c = cell_of(chart, i, j)
        p = flooded[w, c]
        if p >= 0:
            part[w, k, ii, p] = part[w, k, ii, p] + pooled[w, c] * area[c]
            v = part_sp[w, k, ii, p]
            for ks in range(ns.n_film):
                v[ks] = v[ks] + ns.drained[w, c, ks] * area[c]
                ns.drained[w, c, ks] = wp.float64(0.0)
            part_sp[w, k, ii, p] = v
        pooled[w, c] = wp.float64(0.0)


@wp.kernel
def pool_into_tanks(dt: wp.float64, pools: wp.array[PoolSpec], box: wp.array2d[wp.vec4i],
                    part: wp.array4d[wp.float64], part_sp: wp.array4d[wp.vec4d], ns: NetSpecies,
                    inflow: wp.array2d[wp.float64], ledger: wp.array2d[wp.float64]):
    """Per pool, the film its water took this exchange (rows in order) into its tank's inflow, its species into the
    tank's species (or away)."""
    w = wp.tid()
    for p in range(pools.shape[0]):
        total = wp.float64(0.0)
        sm = wp.vec4d()
        for k in range(box.shape[1]):
            b = box[w, k]
            for ii in range(b[1] - b[0] + 1):
                total = total + part[w, k, ii, p]
                sm = sm + part_sp[w, k, ii, p]
        tank = pools[p].tank
        inflow[w, tank] = inflow[w, tank] + total / dt
        ledger[w, POOLED] = ledger[w, POOLED] + total
        for ks in range(ns.n_film):
            into_tank(ns, w, tank, ks, sm[ks], dt)


class Flooding:
    """``Supply``'s film-pool exchange: set-up and launches (the kernels above). Mixed into ``Supply``."""

    def _setup_flooding(self, ctx: Any, film: Any) -> None:
        """State and scratch of the film-pool exchange (``supply_pool``), when a pool floods."""
        self.flooding = any(p.flood for p in self.pools)
        if not self.flooding:
            return
        missing = [n for n in ("flooded", "pooled") if not hasattr(film, n)]
        if missing:
            raise PluginError(f"a flooding pool needs the film's {missing} arrays (the surface-flow film's)")
        W, Kc, P, C = ctx.world_count, len(self.charts.shapes), len(self.pools), self.charts.cell_count
        empty = wp.vec4i(1 << 30, -1, 0, -1)
        self.flood_prev = film.add_state("flood_box", ctx.full((W, Kc), empty, wp.vec4i))   # cells set last exchange
        self.flood_cur, self.flood_proc = ctx.full((W, Kc), empty, wp.vec4i), ctx.full((W, Kc), empty, wp.vec4i)
        self.wreq, self.wreq_pool = ctx.zeros((W, C), wp.float64), ctx.zeros((W, C), wp.int32)
        self.wratio, self.wconc = ctx.zeros((W, P), wp.float64), ctx.zeros((W, P), wp.vec4d)
        self.fpart = wp.zeros((W, Kc, self.rows, P), dtype=wp.float64, device=ctx.device)
        self.fpart_sp = wp.zeros((W, Kc, self.rows, P), dtype=wp.vec4d, device=ctx.device)
        self.wdepth = film.add_state("flood_depth", ctx.zeros((W, C), wp.float32))   # deepest water while flooded
        mat = getattr(film, "mat", None)                      # the chart's contact angle, else the material's
        angles = [c.contact_angle if getattr(c, "contact_angle", None) is not None else getattr(mat, "contact_angle",
                  None) for c in film.config.charts]
        self.cpud = ctx.array([0.0 if a is None else math.sin(math.radians(a) / 2.0) for a in angles], wp.float32)

    def _flood(self, state: newton.State, dt: float) -> None:
        """Flood the cells under the pools' water; cells leaving it get their withdrawal films (``supply_pool``)."""
        W, dev, f, film, c = self.ctx.world_count, self.ctx.device, self.fluid, self.film, self.charts
        cells = (W, self.flood_prev.shape[1], self.rows)
        wp.launch(flood_boxes, dim=self.flood_prev.shape,
                  inputs=[c.charts, c.prof, c.cbody, c.clocal, state.body_q, self.pool_spec, self.pbody, f.pressure,
                          self.cav, self.flood_prev], outputs=[self.flood_cur, self.flood_proc], device=dev)
        wp.launch(flood_cells, dim=cells,
                  inputs=[wp.float64(dt), c.charts, c.cbody, c.clocal, state.body_q, state.body_qd,
                          self.ctx.model.body_com, c.cpos, c.cnrm, c.area, self.pool_spec, self.pbody, f.pressure,
                          self.cav, self.cav_prof, self.cav_surf, self.cav_prev, self.flood_proc, film.h, self.cpud],
                  outputs=[film.flooded, film.cwet, self.wdepth, self.wreq, self.wreq_pool, self.fpart], device=dev)
        ns = refresh(self.ns, f)
        wp.launch(withdraw_grant, dim=W,
                  inputs=[wp.float64(dt), self.pool_spec, self.flood_proc, self.fpart, f.pressure, ns],
                  outputs=[self.wratio, self.wconc, f.inflow, self.ledger], device=dev)
        wp.launch(withdraw_enter, dim=cells,
                  inputs=[c.charts, c.area, self.flood_proc, self.flood_cur, self.wreq, self.wreq_pool, self.wratio,
                          self.wconc, ns], outputs=[film.src_acc, film.cwet, self.flood_prev], device=dev)

    def _pool_sums(self, dt: float) -> None:
        """After the film: the film and species the flooded cells gave up, into the pools' tanks."""
        W, dev, f = self.ctx.world_count, self.ctx.device, self.fluid
        ns = refresh(self.ns, f)
        wp.launch(pool_sums, dim=(W, self.flood_prev.shape[1], self.rows),
                  inputs=[self.charts.charts, self.charts.area, self.flood_prev, self.film.flooded],
                  outputs=[self.film.pooled, ns, self.fpart, self.fpart_sp], device=dev)
        wp.launch(pool_into_tanks, dim=W, inputs=[wp.float64(dt), self.pool_spec, self.flood_prev, self.fpart,
                                                  self.fpart_sp, ns], outputs=[f.inflow, self.ledger], device=dev)


__all__ = ["Flooding", "START_MARGIN", "cavity_start", "flood_boxes", "flood_cells", "in_region", "lowest_inside",
           "pool_into_tanks", "pool_sums", "region_box", "withdraw_enter", "withdraw_grant", "withdrawal"]
