# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the sports plug-ins: the pressure-shell ball's contacts and the ball's aerodynamics.

One thread per world. It walks that world's balls and, for each, the pitch and the listed collider shapes, so
reactions on collider bodies are written by that thread alone (deterministic). The formulas are those of
``model.py`` (float32 here, float64 there).
"""

from __future__ import annotations

import math

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, torque_wrench, wrench_at

PI = wp.constant(math.pi)
P_ATM = wp.constant(101325.0)
GAMMA = wp.constant(1.4)
SPHERE, CAPSULE, CYLINDER, BOX = 3, 4, 6, 7
ROLL_EPS = wp.constant(0.1)          # rad/s: rolling resistance ramps in below this rolling rate


@wp.struct
class Ball:
    radius: float
    shell_k: float
    shell_c: float
    k_tan: float
    c_tan: float


@wp.struct
class Pitch:
    on: int
    z: float
    k: float              # N/m^1.5
    c: float              # N·s/m
    mu: float
    roll: float           # C_rr


@wp.struct
class Air:
    rho: float
    mu: float
    radius: float
    cd_sub: float
    cd_sup: float
    re_lo: float
    re_hi: float
    cl_max: float
    cl_s0: float
    cm: float


@wp.func
def shell_force(d: float, p: float, b: Ball) -> float:
    """Elastic force of the ball at overlap d (model.shell_force)."""
    r = b.radius
    x = wp.clamp(d, 0.0, r)
    v0 = 4.0 / 3.0 * PI * r * r * r
    dv = PI * x * x * (3.0 * r - x) / 3.0
    pg = (p + P_ATM) * wp.pow(v0 / (v0 - dv), GAMMA) - P_ATM
    return pg * PI * (2.0 * r * x - x * x) + b.shell_k * x


@wp.func
def shell_stiffness(d: float, p: float, b: Ball) -> float:
    h = 1.0e-5
    lo = wp.max(d - h, 0.0)
    return (shell_force(d + h, p, b) - shell_force(lo, p, b)) / (d + h - lo)


@wp.func
def shape_xform(body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], shape_xf: wp.array[wp.transform],
                s: wp.int32) -> wp.transform:
    b = shape_body[s]
    if b < 0:
        return shape_xf[s]
    return wp.transform_multiply(body_q[b], shape_xf[s])


@wp.func
def box_closest(cl: wp.vec3, h: wp.vec3) -> wp.vec4:
    """(local normal, signed distance from the surface) of point ``cl`` to a box of half extents ``h``."""
    q = wp.vec3(wp.clamp(cl[0], -h[0], h[0]), wp.clamp(cl[1], -h[1], h[1]), wp.clamp(cl[2], -h[2], h[2]))
    dist = wp.length(cl - q)
    if dist > 1.0e-9:
        nl = (cl - q) / dist
        return wp.vec4(nl[0], nl[1], nl[2], dist)
    best = float(1.0e6)                                # centre inside: leave through the nearest face
    nl = wp.vec3(0.0, 0.0, 1.0)
    for i in range(3):
        gap = h[i] - wp.abs(cl[i])
        if gap < best:
            best = gap
            nl = wp.vec3(0.0)
            nl[i] = wp.sign(cl[i]) + float(cl[i] == 0.0)
    return wp.vec4(nl[0], nl[1], nl[2], -best)


@wp.func
def cylinder_closest(cl: wp.vec3, radius: float, half: float) -> wp.vec4:
    """(local normal, signed distance from the surface) of point ``cl`` to a cylinder along local z."""
    rho = wp.sqrt(cl[0] * cl[0] + cl[1] * cl[1])
    if rho <= radius and wp.abs(cl[2]) <= half:
        side, cap = radius - rho, half - wp.abs(cl[2])
        if side < cap and rho > 1.0e-9:
            return wp.vec4(cl[0] / rho, cl[1] / rho, 0.0, -side)
        return wp.vec4(0.0, 0.0, wp.sign(cl[2]) + float(cl[2] == 0.0), -cap)
    k = wp.min(1.0, radius / wp.max(rho, 1.0e-9))
    q = wp.vec3(cl[0] * k, cl[1] * k, wp.clamp(cl[2], -half, half))
    dist = wp.length(cl - q)
    nl = wp.vec3(0.0, 0.0, 1.0)
    if dist > 1.0e-9:
        nl = (cl - q) / dist
    return wp.vec4(nl[0], nl[1], nl[2], dist)


@wp.func
def closest(kind: wp.int32, scale: wp.vec3, xf: wp.transform, c: wp.vec3, r: float) -> wp.vec4:
    """(n, overlap) of a sphere of radius r at c against the shape: n points from the shape to the ball."""
    cl = wp.transform_point(wp.transform_inverse(xf), c)
    g = wp.vec4(0.0, 0.0, 1.0, 1.0e6)
    if kind == SPHERE or kind == CAPSULE:
        a = wp.vec3(0.0)
        if kind == CAPSULE:
            a = wp.vec3(0.0, 0.0, wp.clamp(cl[2], -scale[1], scale[1]))
        dist = wp.length(cl - a)
        nl = wp.vec3(0.0, 0.0, 1.0)
        if dist > 1.0e-9:
            nl = (cl - a) / dist
        g = wp.vec4(nl[0], nl[1], nl[2], dist - scale[0])
    elif kind == BOX:
        g = box_closest(cl, scale)
    elif kind == CYLINDER:
        g = cylinder_closest(cl, scale[0], scale[1])
    n = wp.transform_vector(xf, wp.vec3(g[0], g[1], g[2]))
    return wp.vec4(n[0], n[1], n[2], r - g[3])


@wp.func
def turf_step(d: float, dd: float, u: float, p: float, b: Ball, t: Pitch, dt: float) -> wp.vec2:
    """Implicit step of the turf compression u (model.series_step): returns (u_new, normal force)."""
    cs = b.shell_c + t.c
    g = (b.shell_c * dd + shell_force(d - u, p, b) - t.k * wp.pow(wp.max(u, 0.0), 1.5)) / cs
    jac = -(shell_stiffness(wp.max(d - u, 0.0), p, b) + 1.5 * t.k * wp.sqrt(wp.max(u, 0.0))) / cs
    un = wp.clamp(u + dt * g / (1.0 - dt * jac), 0.0, wp.max(d, 0.0))
    f = t.k * wp.pow(un, 1.5) + t.c * (un - u) / dt
    if f <= 0.0:
        un = u / (1.0 + dt * 1.5 * t.k * wp.sqrt(wp.max(u, 0.0)) / t.c)
        f = 0.0
    return wp.vec2(un, f)


@wp.func
def friction(xi: wp.vec3, n: wp.vec3, vt: wp.vec3, fn: float, mu: float, b: Ball, dt: float) -> wp.spatial_vector:
    """Tangential spring-dashpot capped at μ·F_n: returns (force, new anchor) as a spatial vector."""
    if fn <= 0.0:
        return wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    a = xi - wp.dot(xi, n) * n + vt * dt
    f = -b.k_tan * a - b.c_tan * vt
    cap = mu * fn
    mag = wp.length(f)
    if mag > cap:
        f = f * (cap / mag)
        a = -f / b.k_tan
    return wp.spatial_vector(f, a)


@wp.func
def touch_update(tc: wp.vec2, fn: float, dt: float) -> wp.vec2:
    """(duration [s], peak force [N]) of the current or last contact."""
    if fn <= 0.0:
        return tc
    if tc[1] < 0.0:                                  # a new contact (negative peak marks "ended")
        return wp.vec2(dt, fn)
    return wp.vec2(tc[0] + dt, wp.max(tc[1], fn))


@wp.func
def end_touch(tc: wp.vec2, fn: float) -> wp.vec2:
    if fn <= 0.0 and tc[1] > 0.0:
        return wp.vec2(tc[0], -tc[1])
    return tc


@wp.func
def normal_force(j: int, d: float, dd: float, u: float, p: float, b: Ball, t: Pitch, dt: float) -> wp.vec3:
    """(turf compression, normal force, the ball's own deformation) of contact j: 0 is the pitch (turf in series),
    the others rigid colliders."""
    if j == 0:
        us = turf_step(d, dd, u, p, b, t, dt)
        return wp.vec3(us[0], us[1], d - us[0])
    if d > 0.0:
        return wp.vec3(u, wp.max(0.0, shell_force(d, p, b) + b.shell_c * dd), d)
    return wp.vec3(u, 0.0, d)


@wp.func
def rolling(om: wp.vec3, n: wp.vec3, fn: float, arm: float, t: Pitch) -> wp.vec3:
    """Turf rolling resistance: C_rr·F_n·arm against the ball's rolling (viscous below ROLL_EPS)."""
    wt = om - wp.dot(om, n) * n
    return -t.roll * fn * arm * wt / wp.max(wp.length(wt), ROLL_EPS)


@wp.func
def push(body_f: wp.array[wp.spatial_vector], body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3],
         ball: int, other: int, pc: wp.vec3, f: wp.vec3, tq: wp.vec3):
    """Force ``f`` at ``pc`` (plus torque ``tq``) on the ball, the opposite force on the collider's body."""
    add_wrench(body_f, ball, wrench_at(f, pc, com_world(body_q, body_com, ball)) + torque_wrench(tq))
    if other >= 0:
        add_wrench(body_f, other, wrench_at(-f, pc, com_world(body_q, body_com, other)))


@wp.kernel
def ball_contacts(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                  balls: wp.array2d[wp.int32], pressure: wp.array2d[wp.float32], cols: wp.array2d[wp.int32],
                  col_mu: wp.array[wp.float32], n_cols: int, shape_body: wp.array[wp.int32],
                  shape_xf: wp.array[wp.transform], shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
                  b: Ball, t: Pitch, dt: float, turf_u: wp.array2d[wp.float32], slip: wp.array3d[wp.vec3],
                  touch: wp.array3d[wp.vec2], body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    for i in range(balls.shape[1]):
        bb = balls[w, i]
        com = com_world(body_q, body_com, bb)
        c = wp.transform_get_translation(body_q[bb])
        for j in range(n_cols + 1):
            if j == 0 and t.on == 0:
                continue
            nd = wp.vec4(0.0, 0.0, 1.0, b.radius - (c[2] - t.z))
            other = int(-1)
            mu = t.mu
            if j > 0:
                s = cols[w, j - 1]
                nd = closest(shape_type[s], shape_scale[s], shape_xform(body_q, shape_body, shape_xf, s), c, b.radius)
                other = shape_body[s]
                mu = col_mu[j - 1]
            n = wp.vec3(nd[0], nd[1], nd[2])
            vrel = point_velocity(body_qd, com, bb, c - n * b.radius)
            if other >= 0:
                vrel = vrel - point_velocity(body_qd, com_world(body_q, body_com, other), other, c - n * b.radius)
            nf = normal_force(j, nd[3], -wp.dot(vrel, n), turf_u[w, i], float(pressure[w, i]), b, t, dt)
            if j == 0:
                turf_u[w, i] = nf[0]
            fn = nf[1]
            arm = b.radius - wp.max(nf[2], 0.0)
            fa = friction(slip[w, i, j], n, vrel - wp.dot(vrel, n) * n, fn, mu, b, dt)
            slip[w, i, j] = wp.spatial_bottom(fa)
            touch[w, i, j] = end_touch(touch_update(touch[w, i, j], fn, dt), fn)
            if fn > 0.0:
                tq = wp.vec3(0.0)
                if j == 0 and t.roll > 0.0:
                    tq = rolling(wp.spatial_bottom(body_qd[bb]), n, fn, arm, t)
                push(body_f, body_q, body_com, bb, other, c - n * arm, n * fn + wp.spatial_top(fa), tq)


@wp.func
def drag_coeff(re: float, a: Air) -> float:
    s = wp.clamp((re - a.re_lo) / (a.re_hi - a.re_lo), 0.0, 1.0)
    return a.cd_sub + (a.cd_sup - a.cd_sub) * s * s * (3.0 - 2.0 * s)


@wp.kernel
def ball_aero(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
              balls: wp.array2d[wp.int32], wind: wp.array[wp.vec3], a: Air, body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    q = 0.5 * a.rho * PI * a.radius * a.radius
    for i in range(balls.shape[1]):
        bb = balls[w, i]
        com = com_world(body_q, body_com, bb)
        v = point_velocity(body_qd, com, bb, com) - wind[w]
        om = wp.spatial_bottom(body_qd[bb])
        sp = wp.length(v)
        if sp > 1.0e-6:
            f = -q * drag_coeff(a.rho * sp * 2.0 * a.radius / a.mu, a) * sp * v
            wxv = wp.cross(om, v)
            nw = wp.length(wxv)
            if nw > 1.0e-9:
                f = f + q * a.cl_max * wp.tanh(a.radius * nw / (sp * sp) / a.cl_s0) * sp * sp * wxv / nw
            add_wrench(body_f, bb, wp.spatial_vector(f, -q * a.radius * a.radius * a.cm * sp * om))
