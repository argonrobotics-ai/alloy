# SPDX-License-Identifier: Apache-2.0
"""Lumped bodies' drips (lumped.py's shed water, through the drop stage; design note §4.9, WP15's reviews).

Each exchange, per world and lumped body with water waiting (``LumpDrip.pending``, its species ``pending_sp``):
- its lowest point x along its world's gravity: the furthest support point of its collision shapes at their world
  pose (spheres, boxes, capsules, cylinders (barrels too), ellipsoids and cones exactly, along their local z; meshes by
  their convex hull's vertices; other kinds by their bounding sphere);
- if its first drop would start in a pool's water (a drop radius below x; ``drops_pools.into_pool``), all of it and
  its species join that pool's tank (DC5's ``pool_add``, booked ``tank``, ``tank_sp``);
- else the chart under it: a ray along gravity from a drop radius above x, as ``drops_kernels.advance_kernel`` lands
  drops; the nearest wetted-side hit is the cell under it (a surface more than r above x is not under it, nor one
  above x on the body itself: a thin-walled beaker's own floor). Its film top along the ray is ``(base + h)/cos``
  past the hit. If x is within a drop's diameter (2r) of that film top, or below it (a body resting in a puddle), the
  body gives its water and species to that cell, as much as the cell holds below ``max_depth`` (``lump_wet``,
  ``wet_sp``); the rest waits;
- otherwise whole drops leave 2r apart below x, only as many as stay a radius above that film top (all of them over no
  chart); the rest wait (a lump's request has cell ``-(b + 1)``; ``commit`` drains its water and species, booking the
  species in ``drop_sp``). At the request cap the other bodies may still rest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.funcs import point_velocity
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.drops_kernels import Arrive, Requests
from alloy.plugins.surface_flow.drops_pools import DropPools, into_pool
from alloy.plugins.surface_flow.geometry import chart_xform, ray_hit

SPHERE = wp.constant(int(newton.GeoType.SPHERE))
BOX = wp.constant(int(newton.GeoType.BOX))
CAPSULE = wp.constant(int(newton.GeoType.CAPSULE))
CYLINDER = wp.constant(int(newton.GeoType.CYLINDER))
ELLIPSOID = wp.constant(int(newton.GeoType.ELLIPSOID))
CONE = wp.constant(int(newton.GeoType.CONE))
FAR = wp.constant(1.0e30)


@wp.struct
class LumpDrip:
    """The lumped bodies' shapes and drips (lumped.py builds it, the drop stage launches ``lump_drip_kernel``)."""
    body: wp.array2d[wp.int32]       # [W, B]
    slots: wp.array3d[wp.int32]      # [W, B, M] each body's collision shapes (model ids; -1 pads)
    pts: wp.array[wp.vec3]           # hull vertices of mesh shapes, shape frame, scaled
    span: wp.array[wp.vec2i]         # [S] (start, count) into pts of a mesh shape, else (0, 0)
    radius: wp.array[wp.float32]     # [S] bounding radius of a shape of another kind
    shape_body: wp.array[wp.int32]
    shape_xf: wp.array[wp.transform]
    shape_type: wp.array[wp.int32]
    shape_scale: wp.array[wp.vec3]
    pending: wp.array2d[wp.float64]  # [W, B] m³ shed and waiting to drip
    pending_sp: wp.array3d[wp.float64]   # [W, B, K] kg of species with it
    wet_sp: wp.array3d[wp.float64]   # [W, B, K] kg given to the cells under resting bodies
    drop_sp: wp.array3d[wp.float64]  # [W, B, K] kg that left with drops
    tank: wp.array2d[wp.float64]     # [W, B] m³ whose drips started in a pool's water
    tank_sp: wp.array3d[wp.float64]  # [W, B, K] kg with it
    count: int
    n_sp: int


@wp.func
def _sign(x: float) -> float:
    if x < 0.0:
        return -1.0
    return 1.0


@wp.func
def _round_support(dl: wp.vec3, sc: wp.vec3, t: int) -> wp.vec3:
    """Shape-frame support of a cylinder (a barrel when ``sc[2] > 0``) or a cone (base at -h, apex at +h), axis z."""
    r = wp.vec3(dl[0], dl[1], 0.0)
    s = wp.length(r)
    n = wp.vec3(0.0)
    if s > 1.0e-9:
        n = r / s
    hz = wp.vec3(0.0, 0.0, sc[1])
    if t == CONE:
        rim = sc[0] * n - hz
        if wp.dot(hz, dl) >= wp.dot(rim, dl):
            return hz
        return rim
    if sc[2] > 0.0 and s > 1.0e-9:                     # a barrel: the arc of radius sc[2] through both rims
        zs = sc[2] * dl[2]
        if wp.abs(zs) <= sc[1]:
            c = sc[0] - wp.sqrt(sc[2] * sc[2] - sc[1] * sc[1])
            return (c + sc[2] * s) * n + wp.vec3(0.0, 0.0, zs)
    return sc[0] * n + _sign(dl[2]) * hz


@wp.func
def support(ld: LumpDrip, body_q: wp.array[wp.transform], s: int, d: wp.vec3) -> wp.vec3:
    """World point of shape ``s`` furthest along unit ``d``."""
    X = chart_xform(body_q, ld.shape_body[s], ld.shape_xf[s])
    dl = wp.quat_rotate_inv(wp.transform_get_rotation(X), d)
    sc = ld.shape_scale[s]
    t = ld.shape_type[s]
    p = ld.radius[s] * dl
    if t == SPHERE:
        p = sc[0] * dl
    elif t == BOX:
        p = wp.vec3(_sign(dl[0]) * sc[0], _sign(dl[1]) * sc[1], _sign(dl[2]) * sc[2])
    elif t == CAPSULE:
        p = wp.vec3(0.0, 0.0, _sign(dl[2]) * sc[1]) + sc[0] * dl
    elif t == CYLINDER or t == CONE:
        p = _round_support(dl, sc, t)
    elif t == ELLIPSOID:
        a = wp.vec3(sc[0] * dl[0], sc[1] * dl[1], sc[2] * dl[2])
        p = wp.vec3(sc[0] * a[0], sc[1] * a[1], sc[2] * a[2]) / wp.length(a)
    else:
        span = ld.span[s]
        if span[1] > 0:
            best = float(-1.0e30)
            for i in range(span[0], span[0] + span[1]):
                v = wp.dot(ld.pts[i], dl)
                if v > best:
                    best = v
                    p = ld.pts[i]
    return wp.transform_point(X, p)


@wp.func
def lowest(ld: LumpDrip, body_q: wp.array[wp.transform], w: int, b: int, down: wp.vec3) -> wp.vec3:
    """Body b's lowest point along ``down`` over its shapes."""
    x = wp.vec3(0.0)
    best = float(-1.0e30)
    for m in range(ld.slots.shape[2]):
        s = ld.slots[w, b, m]
        if s >= 0:
            p = support(ld, body_q, s, down)
            if wp.dot(p, down) > best:
                best = wp.dot(p, down)
                x = p
    return x


@wp.func
def give(ld: LumpDrip, a: Arrive, src_sp: wp.array3d[wp.float64], w: int, b: int, k: int, c: int, vol: wp.float64,
         m: float, v_rel: wp.vec3, n: wp.vec3, x: wp.vec3, X: wp.transform):
    """``vol`` of body b's pending water (and its share of the pending species) joins cell c of chart k, as
    ``drops_kernels.arrive`` books a drop (the cell's room was checked)."""
    frac = vol / ld.pending[w, b]
    a.src_acc[w, c] = a.src_acc[w, c] + vol / a.area[c]
    a.cwet[w, c] = 1
    J = (m * wp.dot(v_rel, n)) * n
    a.src_mom[w, c] = a.src_mom[w, c] + wp.transform_vector(wp.transform_inverse(X), m * v_rel - J) / wp.float32(
        a.area[c])
    a.impulse[w, k] = a.impulse[w, k] + J
    a.moment[w, k] = a.moment[w, k] + wp.cross(x, J)
    a.total[w] = a.total[w] + vol
    for q in range(ld.n_sp):
        dm = ld.pending_sp[w, b, q] * frac
        src_sp[w, c, q] = src_sp[w, c, q] + dm / a.area[c]
        ld.pending_sp[w, b, q] = ld.pending_sp[w, b, q] - dm
        ld.wet_sp[w, b, q] = ld.wet_sp[w, b, q] + dm
    ld.pending[w, b] = ld.pending[w, b] - vol


@wp.func
def to_tank(ld: LumpDrip, dp: DropPools, w: int, b: int, p: int):
    """All of body b's pending water and species join pool p's tank (``pool_add``, ``pool_add_sp``)."""
    vol = ld.pending[w, b]
    dp.add[w, p] = dp.add[w, p] + vol
    ld.tank[w, b] = ld.tank[w, b] + vol
    ld.pending[w, b] = wp.float64(0.0)
    for q in range(ld.n_sp):
        dm = ld.pending_sp[w, b, q]
        dp.add_sp[w, p, q] = dp.add_sp[w, p, q] + dm
        ld.tank_sp[w, b, q] = ld.tank_sp[w, b, q] + dm
        ld.pending_sp[w, b, q] = wp.float64(0.0)


@wp.struct
class Under:
    """The chart cell under a lowest point along gravity: ``clear`` from the point down to its film top [m]."""
    chart: wp.int32
    cell: wp.int32
    clear: wp.float32
    normal: wp.vec3


@wp.func
def under(x: wp.vec3, down: wp.vec3, radius: float, w: int, k: int, charts: wp.array[Chart], prof: wp.array[wp.vec3],
          cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
          base: wp.array2d[wp.float32], h: wp.array2d[wp.float64]) -> Under:
    """The nearest wetted-side chart hit of the ray along ``down`` from ``radius`` above x (chart -1: none). A hit
    above x on a chart riding body k itself (a thin-walled beaker's own floor) is not under it."""
    u = Under()
    u.chart = -1
    u.cell = -1
    u.clear = float(FAR)
    o = x - radius * down
    best_t = float(FAR)
    for c in range(charts.shape[0]):
        hit = ray_hit(charts[c], prof, chart_xform(body_q, cbody[w, c], clocal[w, c]), o, down)
        if hit.cell >= 0 and hit.t < best_t and not (cbody[w, c] == k and hit.t < radius):
            best_t = float(hit.t)
            cos = wp.max(-wp.dot(hit.normal, down), 1.0e-3)
            u.chart = c
            u.cell = hit.cell
            u.normal = hit.normal
            u.clear = (best_t - radius) - (base[w, hit.cell] + wp.float32(h[w, hit.cell])) / cos
    return u


@wp.kernel
def lump_drip_kernel(ld: LumpDrip, body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                     body_com: wp.array[wp.vec3], gravity: wp.array[wp.vec3], vol: wp.float64, radius: float,
                     rho: float, cap: int, charts: wp.array[Chart], prof: wp.array[wp.vec3],
                     cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], base: wp.array2d[wp.float32],
                     a: Arrive, src_sp: wp.array3d[wp.float64], req: Requests, dp: DropPools):
    """Per world, after the cells' requests (module docstring)."""
    w = wp.tid()
    down = wp.normalize(gravity[w])
    for b in range(ld.count):
        pend = ld.pending[w, b]
        if pend <= wp.float64(0.0):
            continue
        x = lowest(ld, body_q, w, b, down)
        p = into_pool(dp, w, x + radius * down, body_q)
        if p >= 0:                                                   # its drips would start in a pool's water
            to_tank(ld, dp, w, b, p)
            continue
        k = ld.body[w, b]
        com = wp.transform_point(body_q[k], body_com[k])
        u = under(x, down, radius, w, k, charts, prof, cbody, clocal, body_q, base, a.h)
        if u.chart >= 0 and u.clear <= 2.0 * radius:                 # resting: the cell under it takes what it holds
            room = (wp.float64(a.max_depth) - a.h[w, u.cell] - a.src_acc[w, u.cell]) * a.area[u.cell]
            put = wp.min(pend, wp.max(room, wp.float64(0.0)))
            if put > wp.float64(0.0):
                cb = cbody[w, u.chart]
                v_rel = point_velocity(body_qd, com, k, x)
                if cb >= 0:
                    v_rel = v_rel - point_velocity(body_qd, wp.transform_point(body_q[cb], body_com[cb]), cb, x)
                give(ld, a, src_sp, w, b, u.chart, u.cell, put, rho * float(put), v_rel, u.normal, x,
                     chart_xform(body_q, cb, clocal[w, u.chart]))
            continue
        n_drops = int(wp.floor(pend / vol))
        if u.chart >= 0:                                             # only those a radius over the film below
            n_drops = wp.min(n_drops, int(wp.floor((u.clear - 2.0 * radius) / (2.0 * radius))) + 1)
        for q in range(n_drops):
            s = req.n[w]
            if s >= cap:
                break
            xq = x + (radius + 2.0 * radius * float(q)) * down
            req.pos[w, s] = xq
            req.vel[w, s] = point_velocity(body_qd, com, k, xq)
            req.cell[w, s] = -(b + 1)
            req.n[w] = s + 1


def hull_points(model: newton.Model, shape: int) -> np.ndarray:
    """A mesh shape's convex hull vertices in its frame, scaled (all vertices if the hull fails)."""
    src = model.shape_source[shape]
    v = np.asarray(src.vertices, np.float64) * model.shape_scale.numpy()[shape].astype(np.float64)
    try:
        from scipy.spatial import ConvexHull  # noqa: PLC0415
        return v[ConvexHull(v).vertices]
    except Exception:                                           # flat or degenerate: every vertex
        return v


def build_drip(ctx: Any, bodies: np.ndarray, shapes: list[list[list[int]]], state: Any, n_sp: int) -> LumpDrip:
    """``LumpDrip`` for bodies ``bodies`` [W, B] whose collision shapes are ``shapes[w][b]`` (model ids); ``state``
    the lumps' ``LumpState`` (its drip and tank books)."""
    m = ctx.model
    W, B = bodies.shape
    M = max(1, max(len(ids) for row in shapes for ids in row))
    slots = np.full((W, B, M), -1, np.int32)
    for w in range(W):
        for b in range(B):
            slots[w, b, :len(shapes[w][b])] = shapes[w][b]
    S = int(m.shape_count)
    kind, scale = m.shape_type.numpy(), m.shape_scale.numpy()
    span, rad, pts = np.zeros((S, 2), np.int32), np.zeros(S, np.float32), []
    meshes = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    for s in sorted(set(slots[slots >= 0].tolist())):
        if int(kind[s]) in meshes and getattr(m, "shape_source", None) is not None and m.shape_source[s] is not None:
            hp = hull_points(m, s)
            span[s] = (sum(len(x) for x in pts), len(hp))
            pts.append(hp)
        else:
            rad[s] = float(np.max(scale[s][:3]))
    ld = LumpDrip()
    ld.body, ld.slots = ctx.array(bodies.astype(np.int32), wp.int32), ctx.array(slots, wp.int32)
    allp = np.concatenate(pts).astype(np.float32) if pts else np.zeros((1, 3), np.float32)
    ld.pts, ld.span = ctx.array(allp, wp.vec3), ctx.array(span, wp.vec2i)
    ld.radius = ctx.array(rad, wp.float32)
    ld.shape_body, ld.shape_xf, ld.shape_type, ld.shape_scale = m.shape_body, m.shape_transform, m.shape_type, \
        m.shape_scale
    ld.pending, ld.pending_sp, ld.wet_sp, ld.count, ld.n_sp = state.drip, state.drip_sp, state.sp_wet, B, n_sp
    ld.drop_sp, ld.tank, ld.tank_sp = state.sp_drop, state.tank, state.sp_tank
    return ld


__all__ = ["LumpDrip", "Under", "build_drip", "give", "hull_points", "lowest", "lump_drip_kernel", "support", "to_tank",
           "under"]
