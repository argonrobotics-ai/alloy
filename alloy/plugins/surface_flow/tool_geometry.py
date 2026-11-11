# SPDX-License-Identifier: Apache-2.0
"""Where a tool's underside meets a chart (design note §4.1; phase 2 note, "Applicators"), device side.

On a plane chart the surface is z = 0 and a cell's frame is the chart's. On a revolution chart each cell has its own
frame (``es`` along the profile, ``et`` around, ``n`` out): ``under_at`` puts the tool's pose in it and measures the
underside along the cell's normal, so a flat pad's footprint follows a curved floor or wall (WP8's pan). A window
(``win_of``) is a box of rows and columns; on a revolution chart its columns may run past ``ny`` (``cell_at`` wraps
them) and span at most one turn.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import com_world, point_velocity
from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.geometry import MISS_T, Near, chart_xform
from alloy.plugins.surface_flow.kernels import Cells

DISC = wp.constant(0)
RECT = wp.constant(1)
ROLLER = wp.constant(2)
EDGER = wp.constant(3)              # a rect with GA's edger profile on one long side
FREE_GAP = wp.constant(1.0e3)        # m: no tool over the cell
BEHIND = wp.constant(1.0e-2)         # m: a curved cell's tool acts only down to max(this, its reach) behind it
LIFT = wp.constant(0.01)             # m: an underside this far above a cell's surface (base + h) works nothing there
MIN_FACING = wp.constant(0.2)        # a flat underside more edge-on than this works nothing
V_PUSH = wp.constant(1.0e-3)         # m/s: a slower tool presses film out from under it instead of ploughing


@wp.struct
class ToolSet:
    chart: wp.array[wp.int32]        # [n] the chart each tool works
    shape: wp.array[wp.int32]        # DISC, RECT, ROLLER, EDGER
    center: wp.array[wp.vec3]        # underside centre (disc, rect) or the roller's axis centre, tool body frame
    normal: wp.array[wp.vec3]        # out of the underside (disc, rect), tool body frame
    axis: wp.array[wp.vec3]          # rect: the first half extent's direction; roller: its axis; tool body frame
    size: wp.array[wp.vec2]          # disc (radius, -); rect (half extents); roller (radius, half width) [m]
    stiffness: wp.array[wp.float32]  # pad stiffness on the film's top [Pa/m]
    damping: wp.array[wp.float32]    # Kelvin-Voigt time [s]
    friction: wp.array[wp.float32]   # Coulomb coefficient on a dry surface
    slip_eps: wp.float32             # [m/s] friction regularization
    ceiling: wp.array[wp.int32]      # [n] 1: the underside is a ceiling on the film; 0: a store tool (the nip)
    implicit: wp.array[wp.int32]     # [n] 1: the pad acts implicitly on the tool body (pad_implicit.py)
    porous: wp.array[wp.int32]       # [n] 1: a porous ceiling (a sponge) absorbs its plough while it has room
    edge: wp.array[wp.vec4]          # [n] an edger's (radius, lip depth, lip width, lip side +-1) [m]
    slope: wp.array[wp.float32]      # [n] a disc's pointed tip: its underside rises slope·rho from the centre
    roll_lift: wp.array[wp.float32]  # [n] the roll's edge lift [m] (roll_cap.py); < 0: the roll is uncapped


@wp.struct
class Under:
    gap: float                       # height of the underside above the chart surface [m]
    cover: float                     # share of the cell under the tool, [0, 1]
    point: wp.vec3                   # the underside point over the cell, chart frame
    foot: wp.vec3                    # the cell's surface point, chart frame
    n: wp.vec3                       # the cell's outward normal, chart frame (a plane's +z)
    es: wp.vec3                      # the chart's i direction at the cell (a plane's +x)
    et: wp.vec3                      # the chart's j direction at the cell (a plane's +y)


@wp.struct
class Pose:
    X: wp.transform                  # the chart frame in world
    c: wp.vec3                       # tool centre, chart frame
    n: wp.vec3                       # underside normal, chart frame
    a: wp.vec3                       # tool axis, chart frame
    reach: float                     # the farthest underside point from the centre [m]


@wp.func
def reach_of(shape: int, size: wp.vec2) -> float:
    if shape == DISC:
        return size[0]
    return wp.sqrt(size[0] * size[0] + size[1] * size[1])


@wp.func
def across_of(shape: int, size: wp.vec2, ps: Pose, lat: wp.vec3) -> float:
    """The footprint's half extent along ``lat`` (a unit direction in the surface, chart frame): a disc's radius, a
    rect's or a roller's support along it (a rod sliding along its axis: its radius, not its reach)."""
    if shape == DISC:
        return size[0]
    if shape == ROLLER:
        c = wp.abs(wp.dot(ps.a, lat))
        return size[1] * c + size[0] * wp.sqrt(wp.max(1.0 - c * c, 0.0))
    e1 = wp.normalize(ps.a - wp.dot(ps.a, ps.n) * ps.n)
    e2 = wp.cross(ps.n, e1)
    return size[0] * wp.abs(wp.dot(e1, lat)) + size[1] * wp.abs(wp.dot(e2, lat))


@wp.func
def pose_of(ts: ToolSet, k: int, body_q: wp.array[wp.transform], tool_body: int, chart_body: int,
            chart_local: wp.transform) -> Pose:
    p = Pose()
    p.X = chart_xform(body_q, chart_body, chart_local)
    T = body_q[tool_body]
    inv = wp.transform_inverse(p.X)
    p.c = wp.transform_point(inv, wp.transform_point(T, ts.center[k]))
    p.n = wp.normalize(wp.transform_vector(inv, wp.transform_vector(T, ts.normal[k])))
    p.a = wp.normalize(wp.transform_vector(inv, wp.transform_vector(T, ts.axis[k])))
    p.reach = reach_of(ts.shape[k], ts.size[k])
    return p


@wp.func
def soft_edge(inside: float, cell: float) -> float:
    return wp.clamp(inside / cell + 0.5, 0.0, 1.0)


@wp.func
def underside(shape: int, size: wp.vec2, ps: Pose, p: wp.vec3, cell: float) -> Under:
    """The tool's underside over the chart point ``p`` (chart frame, on the surface z = 0, normal +z)."""
    u = Under()
    u.gap = FREE_GAP
    u.cover = 0.0
    u.point = p
    u.foot = p
    u.n = wp.vec3(0.0, 0.0, 1.0)
    u.es = wp.vec3(1.0, 0.0, 0.0)
    u.et = wp.vec3(0.0, 1.0, 0.0)
    if shape == ROLLER:
        d = p - ps.c
        N = wp.vec3(0.0, 0.0, 1.0)
        D = d - wp.dot(d, ps.a) * ps.a
        M = N - wp.dot(N, ps.a) * ps.a
        mm = wp.dot(M, M)
        if mm < 1.0e-8:
            return u                                    # the axis stands on the surface: no rolling contact
        dm = wp.dot(D, M)
        disc = dm * dm - mm * (wp.dot(D, D) - size[0] * size[0])
        if disc < 0.0:
            return u
        t = (-dm - wp.sqrt(disc)) / mm                  # the lower crossing of the line p + t·N with the cylinder
        along = wp.dot(d + t * N, ps.a)
        u.gap = t
        u.cover = soft_edge(size[1] - wp.abs(along), cell)
        u.point = p + t * N
        return u
    if -ps.n[2] < MIN_FACING:
        return u
    t = ps.c[2] - (ps.n[0] * (p[0] - ps.c[0]) + ps.n[1] * (p[1] - ps.c[1])) / ps.n[2]
    q = wp.vec3(p[0], p[1], t)
    r = q - ps.c
    if shape == DISC:
        u.cover = soft_edge(size[0] - wp.length(r), cell)
    else:
        e1 = wp.normalize(ps.a - wp.dot(ps.a, ps.n) * ps.n)
        e2 = wp.cross(ps.n, e1)
        u.cover = soft_edge(size[0] - wp.abs(wp.dot(r, e1)), cell) * soft_edge(size[1] - wp.abs(wp.dot(r, e2)), cell)
    u.gap = t
    u.point = q
    return u


@wp.func
def lip_side(ps: Pose) -> wp.vec3:
    """An edger's +e2 = normal x axis (the lip's side for edge_side +1), chart frame."""
    e1 = wp.normalize(ps.a - wp.dot(ps.a, ps.n) * ps.n)
    return wp.cross(ps.n, e1)


@wp.func
def lip_in(u: Under, size: wp.vec2, e: wp.vec4, ps: Pose) -> float:
    """How far in from an edger's lip face the underside point lies [m] (the flat is where it is >= the radius)."""
    return size[1] - e[3] * wp.dot(u.point - ps.c, lip_side(ps))


@wp.func
def edged(u: Under, size: wp.vec2, e: wp.vec4, ps: Pose) -> Under:
    """GA's edger sole (paste.py:557-566): flat, then toward the lip side (+e2 = normal x axis for side +1) a quarter
    round of radius r down to the lip, which runs down the form face to depth ``lip``. ``d`` is the distance in from
    the lip's face; the sole point moves down along the underside normal by the profile's depth."""
    d = lip_in(u, size, e, ps)
    r = e[0]
    z = float(0.0)
    if d >= 0.0 and d < r:
        z = -(r - wp.sqrt(wp.max(r * r - (r - d) * (r - d), 0.0)))
    if d < e[2]:
        z = -e[1]
    u.gap = u.gap - z * ps.n[2]
    u.point = u.point - z * ps.n
    return u


@wp.func
def tool_under(ts: ToolSet, k: int, ps: Pose, p: wp.vec3, cell: float) -> Under:
    """Tool ``k``'s underside over the plane chart point ``p``: its shape, and an edger's profile."""
    u = underside(ts.shape[k], ts.size[k], ps, p, cell)
    if ts.shape[k] == EDGER and u.cover > 0.0:
        u = edged(u, ts.size[k], ts.edge[k], ps)
    if ts.shape[k] == DISC and ts.slope[k] > 0.0 and u.cover > 0.0:   # a pointed round brush (GA's round no. 8)
        z = ts.slope[k] * wp.length(u.point - ps.c)
        u.gap = u.gap - z * ps.n[2]
        u.point = u.point - z * ps.n
    return u


@wp.func
def window(ch: Chart, ps: Pose) -> wp.vec4i:
    """Cells (i0, i1, j0, j1) the tool may cover on a plane chart (lift-off is per cell, from its surface: a small
    tool over a deep paste base stands far above the chart's plane; WP6)."""
    r = ps.reach + wp.max(ch.step[0], ch.step[1])
    i0 = wp.max(int(wp.floor((ps.c[0] - r + ch.half[0]) / ch.step[0])), 0)
    i1 = wp.min(int(wp.floor((ps.c[0] + r + ch.half[0]) / ch.step[0])), ch.nx - 1)
    j0 = wp.max(int(wp.floor((ps.c[1] - r + ch.half[1]) / ch.step[1])), 0)
    j1 = wp.min(int(wp.floor((ps.c[1] + r + ch.half[1]) / ch.step[1])), ch.ny - 1)
    return wp.vec4i(i0, i1, j0, j1)


@wp.func
def cell_point(ch: Chart, i: int, j: int) -> wp.vec3:
    return wp.vec3((float(i) + 0.5) * ch.step[0] - ch.half[0], (float(j) + 0.5) * ch.step[1] - ch.half[1], 0.0)


@wp.func
def relative(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
             tool_body: int, chart_body: int, ps: Pose, under: Under) -> wp.vec3:
    """The tool's velocity relative to the surface at the underside point, chart frame (z: off the surface)."""
    pw = wp.transform_point(ps.X, under.point)
    v = point_velocity(body_qd, com_world(body_q, body_com, tool_body), tool_body, pw)
    if chart_body >= 0:
        sw = wp.transform_point(ps.X, under.foot)
        v = v - point_velocity(body_qd, com_world(body_q, body_com, chart_body), chart_body, sw)
    return wp.quat_rotate_inv(wp.transform_get_rotation(ps.X), v)


@wp.func
def tangential(v: wp.vec3) -> wp.vec3:
    return wp.vec3(v[0], v[1], 0.0)


@wp.func
def push_direction(v: wp.vec3, p: wp.vec3, ps: Pose) -> wp.vec2:
    """Where film above the ceiling goes: ahead of a sliding tool, else out from under it (chart frame, unit)."""
    if wp.length(v) > V_PUSH:
        return wp.normalize(wp.vec2(v[0], v[1]))
    out = wp.vec2(p[0] - ps.c[0], p[1] - ps.c[1])
    if wp.length(out) < 1.0e-9:
        return wp.vec2(1.0, 0.0)
    return wp.normalize(out)


@wp.func
def along(v: wp.vec3, u: Under) -> wp.vec3:
    """``v``'s part along the surface at ``u``'s cell (a plane chart's: ``tangential``, exactly)."""
    return v - wp.dot(v, u.n) * u.n


@wp.func
def off(v: wp.vec3, u: Under) -> float:
    """``v``'s part off the surface at ``u``'s cell (a plane chart's ``v[2]``, exactly)."""
    return wp.dot(v, u.n)


@wp.func
def in_cell(v: wp.vec3, u: Under) -> wp.vec2:
    """``v`` on the cell's (i, j) axes (a plane chart's ``(v[0], v[1])``, exactly)."""
    return wp.vec2(wp.dot(v, u.es), wp.dot(v, u.et))


@wp.func
def push_at(v: wp.vec3, u: Under, ps: Pose) -> wp.vec2:
    """``push_direction`` on the cell's own axes: ahead of a sliding tool, else out from under it (unit)."""
    if wp.length(v) > V_PUSH:
        return wp.normalize(in_cell(v, u))
    out = in_cell(u.foot - ps.c, u)
    if wp.length(out) < 1.0e-9:
        return wp.vec2(1.0, 0.0)
    return wp.normalize(out)


@wp.func
def rev_cover(shape: int, size: wp.vec2, loc: Pose, q: wp.vec3, w: wp.vec2) -> float:
    """The share of a cell (widths ``w`` along its es, et) under the footprint, in the cell's frame (``q``: the
    underside point over it): a straight edge ramps over the cell's own width across it, so hoop cells wider than
    the profile step are covered by their area."""
    r = q - loc.c
    if shape == DISC:
        d = wp.length(r)
        rh = r / wp.max(d, 1.0e-12)
        return soft_edge(size[0] - d, wp.abs(rh[0]) * w[0] + wp.abs(rh[1]) * w[1])
    e1 = wp.normalize(loc.a - wp.dot(loc.a, loc.n) * loc.n)
    e2 = wp.cross(loc.n, e1)
    w1 = wp.abs(e1[0]) * w[0] + wp.abs(e1[1]) * w[1]
    w2 = wp.abs(e2[0]) * w[0] + wp.abs(e2[1]) * w[1]
    return soft_edge(size[0] - wp.abs(wp.dot(r, e1)), w1) * soft_edge(size[1] - wp.abs(wp.dot(r, e2)), w2)


@wp.func
def rev_under(shape: int, size: wp.vec2, ps: Pose, foot: wp.vec3, n: wp.vec3, w: wp.vec2) -> Under:
    """The underside over a revolution chart's cell (surface point ``foot``, outward normal ``n``, widths ``w`` along
    the profile and around): the plane rule in the cell's frame (es, et, n), so the gap is along the cell's normal
    and an underside facing away covers nothing; the cover by the cell's own widths (``rev_cover``)."""
    et = wp.vec3(-foot[1], foot[0], 0.0)
    r = wp.length(et)
    if r > 1.0e-9:
        et = et / r
    else:
        et = wp.vec3(0.0, 1.0, 0.0)
    es = wp.cross(et, n)
    rt = wp.mat33(es[0], es[1], es[2], et[0], et[1], et[2], n[0], n[1], n[2])     # rows: chart -> cell frame
    loc = Pose()
    loc.X = ps.X
    loc.c = rt * (ps.c - foot)
    loc.n = rt * ps.n
    loc.a = rt * ps.a
    loc.reach = ps.reach
    u = underside(shape, size, loc, wp.vec3(0.0, 0.0, 0.0), w[0])
    if u.gap < -wp.max(ps.reach, BEHIND):                   # far behind the cell: through the wall, not on it
        u.gap = FREE_GAP                                    # (WP8's pad 0.26 m under a tilted pan's floor read
        u.cover = 0.0                                       # full cover and was thrown at 13.6 m/s)
    if u.gap < FREE_GAP:
        u.cover = rev_cover(shape, size, loc, u.point, w)
    u.point = foot + wp.transpose(rt) * u.point
    u.foot = foot
    u.n = n
    u.es = es
    u.et = et
    return u


@wp.func
def seg_dist(ch: Chart, prof: wp.array[wp.vec3], k: int, r: float, z: float) -> float:
    """Distance in the (r, z) half-plane from (r, z) to profile segment ``k``."""
    a = prof[ch.prof_start + k]
    b = prof[ch.prof_start + k + 1]
    seg = wp.vec2(b[0] - a[0], b[1] - a[1])
    rel = wp.vec2(r - a[0], z - a[1])
    t = wp.clamp(wp.dot(rel, seg) / wp.max(wp.dot(seg, seg), 1.0e-18), 0.0, 1.0)
    return wp.length(rel - t * seg)


@wp.func
def facing_near(ch: Chart, prof: wp.array[wp.vec3], ps: Pose) -> Near:
    """The profile point nearest the tool's centre among the segments its underside faces (a wrapped chart's inner
    floor, not the outer bottom just under it, however far the pad has sunk): cell, signed distance along the
    segment's outward normal, that normal and the foot, chart frame. ``cell`` -1 when no segment faces it."""
    rc = wp.sqrt(ps.c[0] * ps.c[0] + ps.c[1] * ps.c[1])
    theta = wp.atan2(ps.c[1], ps.c[0])
    if theta < 0.0:
        theta = theta + 2.0 * wp.pi
    cs = wp.cos(theta)
    sn = wp.sin(theta)
    best = Near()
    best.cell = -1
    best.dist = MISS_T
    best_d2 = float(MISS_T)
    j = int(wp.floor(theta / ch.step[1])) % ch.ny
    for k in range(ch.nx):
        a = prof[ch.prof_start + k]
        b = prof[ch.prof_start + k + 1]
        seg = wp.vec2(b[0] - a[0], b[1] - a[1])
        length = wp.max(wp.length(seg), 1.0e-12)
        n = wp.vec3(-seg[1] / length * cs, -seg[1] / length * sn, seg[0] / length)
        if wp.dot(n, ps.n) > -MIN_FACING:
            continue                                            # the underside does not face this segment
        rel = wp.vec2(rc - a[0], ps.c[2] - a[1])
        t = wp.clamp(wp.dot(rel, seg) / (length * length), 0.0, 1.0)
        d = rel - t * seg
        d2 = wp.dot(d, d)
        if d2 < best_d2:
            best_d2 = d2
            best.cell = ch.cell_start + k * ch.ny + j
            best.dist = (-seg[1] * d[0] + seg[0] * d[1]) / length
            best.normal = n
            foot = wp.vec2(a[0], a[1]) + t * seg
            best.foot = wp.vec3(foot[0] * cs, foot[0] * sn, foot[1])
    return best


@wp.func
def rev_window(ch: Chart, prof: wp.array[wp.vec3], ps: Pose) -> wp.vec4i:
    """Rows and (unwrapped) columns a tool may cover on a revolution chart: the rows next to the nearest cell its
    underside faces (``facing_near``) whose profile passes within ``reach + LIFT + ds`` of its centre, and the columns
    that distance spans on the innermost of them (the whole turn when it reaches the axis). Empty when nothing faces
    it (lift-off is per cell, from its surface)."""
    near = facing_near(ch, prof, ps)
    if near.cell < 0:
        return wp.vec4i(0, -1, 0, -1)
    loc = near.cell - ch.cell_start
    i = loc // ch.ny
    j = loc - i * ch.ny
    rad = ps.reach + LIFT + ch.step[0]
    rc = wp.sqrt(ps.c[0] * ps.c[0] + ps.c[1] * ps.c[1])
    i0 = i
    while i0 > 0 and seg_dist(ch, prof, i0 - 1, rc, ps.c[2]) <= rad:
        i0 = i0 - 1
    i1 = i
    while i1 < ch.nx - 1 and seg_dist(ch, prof, i1 + 1, rc, ps.c[2]) <= rad:
        i1 = i1 + 1
    r_min = float(1.0e30)
    for k in range(i0, i1 + 2):
        r_min = wp.min(r_min, prof[ch.prof_start + k][0])
    if rad >= r_min:
        return wp.vec4i(i0, i1, j - ch.ny // 2, j - ch.ny // 2 + ch.ny - 1)
    rj = int(wp.ceil(wp.asin(rad / r_min) / ch.step[1])) + 1
    if 2 * rj + 1 >= ch.ny:
        return wp.vec4i(i0, i1, j - ch.ny // 2, j - ch.ny // 2 + ch.ny - 1)
    return wp.vec4i(i0, i1, j - rj, j + rj)


@wp.func
def win_of(ch: Chart, prof: wp.array[wp.vec3], ps: Pose) -> wp.vec4i:
    if ch.kind == PLANE:
        return window(ch, ps)
    return rev_window(ch, prof, ps)


@wp.func
def widen(ch: Chart, win: wp.vec4i, n: int) -> wp.vec4i:
    """``win`` grown by ``n`` cells each way: clamped on a plane chart; at most one turn on a revolution chart."""
    i0 = wp.max(win[0] - n, 0)
    i1 = wp.min(win[1] + n, ch.nx - 1)
    if ch.kind == PLANE:
        return wp.vec4i(i0, i1, wp.max(win[2] - n, 0), wp.min(win[3] + n, ch.ny - 1))
    j0 = win[2] - n
    return wp.vec4i(i0, i1, j0, wp.min(win[3] + n, j0 + ch.ny - 1))


@wp.func
def cell_at(ch: Chart, i: int, j: int) -> int:
    """The cell at row ``i``, column ``j`` (wrapped: a revolution window's columns may run past ``ny``)."""
    return ch.cell_start + i * ch.ny + (j % ch.ny + ch.ny) % ch.ny


@wp.func
def edge_of(ch: Chart) -> float:
    """The soft edge's width: a plane chart's smaller step, a revolution chart's profile step."""
    if ch.kind == PLANE:
        return wp.min(ch.step[0], ch.step[1])
    return ch.step[0]


@wp.func
def under_at(ts: ToolSet, k: int, ch: Chart, cells: Cells, ps: Pose, i: int, j: int, c: int, cell: float) -> Under:
    """Tool ``k``'s underside over cell ``c`` (row ``i``, column ``j``)."""
    if ch.kind == PLANE:
        return tool_under(ts, k, ps, cell_point(ch, i, j), cell)
    p = cells.pos[c]
    hoop = wp.max(wp.sqrt(p[0] * p[0] + p[1] * p[1]) * ch.step[1], 1.0e-9)
    return rev_under(ts.shape[k], ts.size[k], ps, p, cells.nrm[c], wp.vec2(float(cells.area[c]) / hoop, hoop))


@wp.func
def centre_under(ts: ToolSet, k: int, ch: Chart, prof: wp.array[wp.vec3], ps: Pose, cell: float) -> Under:
    """The underside over the surface point nearest the tool's centre (its sliding velocity)."""
    if ch.kind == PLANE:
        return tool_under(ts, k, ps, wp.vec3(ps.c[0], ps.c[1], 0.0), cell)
    near = facing_near(ch, prof, ps)
    if near.cell < 0:
        return tool_under(ts, k, ps, ps.c, cell)
    return rev_under(ts.shape[k], ts.size[k], ps, near.foot, near.normal, wp.vec2(cell, cell))


__all__ = ["DISC", "EDGER", "FREE_GAP", "LIFT", "MIN_FACING", "Pose", "RECT", "ROLLER", "ToolSet", "Under", "V_PUSH",
           "across_of", "along",
           "cell_at", "cell_point", "centre_under", "edge_of", "edged", "facing_near", "in_cell", "off", "pose_of",
           "push_at",
           "push_direction", "reach_of", "relative", "rev_cover", "rev_under", "rev_window", "soft_edge", "tangential",
           "tool_under", "under_at", "underside", "widen", "win_of", "window"]
