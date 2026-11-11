# SPDX-License-Identifier: Apache-2.0
"""Device geometry of charts (interface note §3): the chart frame in world, ray hits and cell windows.

A ray hits a chart only from its wetted side (the outward normal faces the ray). ``Hit.cell`` is the global cell
index (``chart.cell_start + i·ny + j``), -1 when the ray misses; ``Hit.point`` is in the chart frame and
``Hit.normal`` (outward) in world.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.charts import PLANE, Chart

MISS_T = wp.constant(1.0e30)
TWO_PI = wp.constant(6.283185307179586)


@wp.struct
class Hit:
    t: wp.float32
    cell: wp.int32
    point: wp.vec3
    normal: wp.vec3


@wp.func
def chart_xform(body_q: wp.array[wp.transform], body: wp.int32, local: wp.transform) -> wp.transform:
    """The chart frame in world: ``local`` on body ``body`` (-1: ``local`` is already in world)."""
    if body >= 0:
        return wp.transform_multiply(body_q[body], local)
    return local


@wp.func
def miss() -> Hit:
    h = Hit()
    h.t = MISS_T
    h.cell = -1
    return h


@wp.func
def plane_hit(chart: Chart, o: wp.vec3, d: wp.vec3) -> Hit:
    """Chart-frame ray on a plane chart (surface z = 0, wetted side +z); the normal is left in the chart frame."""
    h = miss()
    if d[2] >= 0.0 or o[2] <= 0.0:
        return h
    t = -o[2] / d[2]
    p = o + t * d
    if wp.abs(p[0]) > chart.half[0] or wp.abs(p[1]) > chart.half[1]:
        return h
    i = wp.clamp(int(wp.floor((p[0] + chart.half[0]) / chart.step[0])), 0, chart.nx - 1)
    j = wp.clamp(int(wp.floor((p[1] + chart.half[1]) / chart.step[1])), 0, chart.ny - 1)
    h.t = t
    h.cell = chart.cell_start + i * chart.ny + j
    h.point = p
    h.normal = wp.vec3(0.0, 0.0, 1.0)
    return h


@wp.func
def _accept(chart: Chart, k: int, a: wp.vec3, b: wp.vec3, o: wp.vec3, d: wp.vec3, t: float, best: Hit,
            r_in: float) -> Hit:
    """Keep the hit at ``t`` on profile segment ``k`` (nodes a, b as (r, z, s)) if it is on the segment, on the wetted
    side and nearer than ``best``. A flat segment's inner radius is ``r_in`` (0 for the uncharted centre of a floor:
    its hits go to the innermost ring)."""
    if t <= 1.0e-7 or t >= best.t:
        return best
    p = o + t * d
    z_lo = wp.min(a[1], b[1])
    z_hi = wp.max(a[1], b[1])
    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    seg_r = b[0] - a[0]
    seg_z = b[1] - a[1]
    length = wp.sqrt(seg_r * seg_r + seg_z * seg_z)
    if wp.abs(seg_z) > 1.0e-9 * length:
        if p[2] < z_lo - 1.0e-9 or p[2] > z_hi + 1.0e-9:
            return best
    else:
        if r < r_in - 1.0e-9 or r > wp.max(a[0], b[0]) + 1.0e-9:
            return best
    theta = wp.atan2(p[1], p[0])
    if theta < 0.0:
        theta = theta + TWO_PI
    n_r = -seg_z / length
    n_z = seg_r / length
    n = wp.vec3(n_r * wp.cos(theta), n_r * wp.sin(theta), n_z)
    if wp.dot(n, d) >= 0.0:
        return best
    j = int(wp.floor(theta / chart.step[1])) % chart.ny
    out = Hit()
    out.t = t
    out.cell = chart.cell_start + k * chart.ny + j
    out.point = p
    out.normal = n
    return out


@wp.func
def revolution_hit(chart: Chart, prof: wp.array[wp.vec3], o: wp.vec3, d: wp.vec3) -> Hit:
    """Chart-frame ray on a revolution chart: the nearest wetted-side hit over the profile's cone frusta."""
    best = miss()
    for k in range(chart.nx):
        a = prof[chart.prof_start + k]
        b = prof[chart.prof_start + k + 1]
        seg_r = b[0] - a[0]
        seg_z = b[1] - a[1]
        length = wp.sqrt(seg_r * seg_r + seg_z * seg_z)
        if wp.abs(seg_z) <= 1.0e-9 * length:                    # flat annulus at z = a.z
            r_in = wp.min(a[0], b[0])
            if (k == 0 and a[0] < b[0]) or (k == chart.nx - 1 and b[0] < a[0]):
                r_in = 0.0                                      # the floor's uncharted centre: the innermost ring
            if wp.abs(d[2]) > 1.0e-12:
                best = _accept(chart, k, a, b, o, d, (a[1] - o[2]) / d[2], best, r_in)
            continue
        m = seg_r / seg_z                                       # r(z) = a.r + m (z - a.z)
        c0 = a[0] + m * (o[2] - a[1])
        c1 = m * d[2]
        qa = d[0] * d[0] + d[1] * d[1] - c1 * c1
        qb = 2.0 * (o[0] * d[0] + o[1] * d[1] - c0 * c1)
        qc = o[0] * o[0] + o[1] * o[1] - c0 * c0
        if wp.abs(qa) < 1.0e-12:
            if wp.abs(qb) > 1.0e-12:
                t = -qc / qb
                if c0 + c1 * t >= 0.0:
                    best = _accept(chart, k, a, b, o, d, t, best, 0.0)
            continue
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            continue
        sq = wp.sqrt(disc)
        for root in range(2):
            t = (-qb - sq) / (2.0 * qa)
            if root == 1:
                t = (-qb + sq) / (2.0 * qa)
            if c0 + c1 * t >= 0.0:                              # the cone's own nappe, not its mirror
                best = _accept(chart, k, a, b, o, d, t, best, 0.0)
    return best


@wp.func
def ray_hit(chart: Chart, prof: wp.array[wp.vec3], X: wp.transform, o_world: wp.vec3, d_world: wp.vec3) -> Hit:
    """The nearest wetted-side hit of a world ray (``d_world`` unit) with the chart whose frame in world is ``X``."""
    Xi = wp.transform_inverse(X)
    o = wp.transform_point(Xi, o_world)
    d = wp.transform_vector(Xi, d_world)
    h = miss()
    if chart.kind == PLANE:
        h = plane_hit(chart, o, d)
    else:
        h = revolution_hit(chart, prof, o, d)
    if h.cell >= 0:
        h.normal = wp.transform_vector(X, h.normal)
    return h


@wp.func
def cell_window(chart: Chart, prof: wp.array[wp.vec3], i: int, j: int, radius: float) -> wp.vec4i:
    """Cells ``(i0, i1, j0, j1)`` (inclusive) within ``radius`` of cell ``(i, j)``'s centre. ``i`` is clamped to the
    chart; on a revolution chart ``j`` is not (wrap it with ``(j % ny + ny) % ny``), and spans at most one turn."""
    ri = int(wp.ceil(radius / chart.step[0]))
    i0 = wp.max(i - ri, 0)
    i1 = wp.min(i + ri, chart.nx - 1)
    if chart.kind == PLANE:
        rj = int(wp.ceil(radius / chart.step[1]))
        return wp.vec4i(i0, i1, wp.max(j - rj, 0), wp.min(j + rj, chart.ny - 1))
    r = 0.5 * (prof[chart.prof_start + i][0] + prof[chart.prof_start + i + 1][0])
    rj = wp.min(int(wp.ceil(radius / wp.max(r * chart.step[1], 1.0e-9))), (chart.ny - 1) // 2)
    return wp.vec4i(i0, i1, j - rj, j + rj)


@wp.struct
class Near:
    """The chart cell under a point (``point_cell``): ``cell`` (global, -1 outside the chart's extent), ``dist`` the
    signed distance along the outward normal [m] (> 0 on the wetted side), ``normal`` (outward) and ``foot`` (the
    nearest surface point), both in the chart frame."""
    cell: wp.int32
    dist: wp.float32
    normal: wp.vec3
    foot: wp.vec3


@wp.func
def _outside() -> Near:
    n = Near()
    n.cell = -1
    n.dist = MISS_T
    return n


@wp.func
def plane_point(chart: Chart, p: wp.vec3) -> Near:
    """Chart-frame point over a plane chart (surface z = 0, wetted side +z)."""
    n = _outside()
    if wp.abs(p[0]) > chart.half[0] or wp.abs(p[1]) > chart.half[1]:
        return n
    i = wp.clamp(int(wp.floor((p[0] + chart.half[0]) / chart.step[0])), 0, chart.nx - 1)
    j = wp.clamp(int(wp.floor((p[1] + chart.half[1]) / chart.step[1])), 0, chart.ny - 1)
    n.cell = chart.cell_start + i * chart.ny + j
    n.dist = p[2]
    n.normal = wp.vec3(0.0, 0.0, 1.0)
    n.foot = wp.vec3(p[0], p[1], 0.0)
    return n


@wp.func
def revolution_point(chart: Chart, prof: wp.array[wp.vec3], p: wp.vec3) -> Near:
    """Chart-frame point near a revolution chart: the nearest profile segment in the (r, z) half-plane (a foot
    beyond a segment's end clamps to it; the floor's uncharted centre maps to the innermost ring, as ``_accept``).
    A point whose nearest foot lies past the profile's first or last end (above a cup's rim) is outside."""
    r = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    theta = wp.atan2(p[1], p[0])
    if theta < 0.0:
        theta = theta + TWO_PI
    best = _outside()
    best_d2 = float(MISS_T)
    beyond = int(0)
    for k in range(chart.nx):
        a = prof[chart.prof_start + k]
        b = prof[chart.prof_start + k + 1]
        seg = wp.vec2(b[0] - a[0], b[1] - a[1])
        length = wp.length(seg)
        rel = wp.vec2(r - a[0], p[2] - a[1])
        t_raw = wp.dot(rel, seg) / (length * length)
        t = wp.clamp(t_raw, 0.0, 1.0)
        foot = wp.vec2(a[0], a[1]) + t * seg
        flat = wp.abs(seg[1]) <= 1.0e-9 * length
        centre = flat and ((k == 0 and a[0] < b[0]) or (k == chart.nx - 1 and b[0] < a[0]))
        over_centre = centre and r < wp.min(a[0], b[0])
        if over_centre:
            foot = wp.vec2(r, a[1])                             # over the floor's uncharted centre
        past_end = ((k == 0 and t_raw < 0.0) or (k == chart.nx - 1 and t_raw > 1.0)) and not over_centre
        d = wp.vec2(r, p[2]) - foot
        d2 = wp.dot(d, d)
        if d2 < best_d2:
            best_d2 = d2
            beyond = 0
            if past_end:
                beyond = 1
            n_r = -seg[1] / length
            n_z = seg[0] / length
            j = int(wp.floor(theta / chart.step[1])) % chart.ny
            best.cell = chart.cell_start + k * chart.ny + j
            best.dist = n_r * d[0] + n_z * d[1]
            best.normal = wp.vec3(n_r * wp.cos(theta), n_r * wp.sin(theta), n_z)
            best.foot = wp.vec3(foot[0] * wp.cos(theta), foot[0] * wp.sin(theta), foot[1])
    if beyond != 0:
        return _outside()
    return best


@wp.func
def point_cell(chart: Chart, prof: wp.array[wp.vec3], p_local: wp.vec3) -> Near:
    """The cell under a chart-frame point and its distance along the outward normal (the inverse of
    ``cell_center``; drops landing, SPH absorption, design §4.7)."""
    if chart.kind == PLANE:
        return plane_point(chart, p_local)
    return revolution_point(chart, prof, p_local)


__all__ = ["Hit", "MISS_T", "Near", "cell_window", "chart_xform", "miss", "plane_hit", "plane_point", "point_cell",
           "ray_hit", "revolution_hit", "revolution_point"]
