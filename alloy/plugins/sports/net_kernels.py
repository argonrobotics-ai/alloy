# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of ``GoalNet``: a cable net of nodes on a grid, tension-only cords, a ball pressing on the cords.

``net_substep`` advances every node one sub-step: one thread per world and node, Jacobi over ping-pong buffers, so
it is deterministic.
- Cords: EA·strain plus a dashpot plus a knot-friction branch, tension only. The branch is a Coulomb slider with a
  pre-slide spring (Iwan element): it follows the stretch until it reaches ±t_f, then slips, independent of rate.
  Node k owns its right and up cords' slider states; both end nodes read the old state, the owner writes the new one.
- Ball: each cord segment inside the ball (radius + cord radius) pushes back with k·depth·(length inside). The push
  is split between the two end nodes by where that length sits. Both end threads compute the same segment, so the
  shares add to the whole.
- Ground: nodes stay at or above ``ground_z``, with Coulomb friction against the impact.

``net_reduce`` sums the nodes' push on the ball, one thread per world. ``net_apply`` hands it to the ball as a force
in the next exchange.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench


@wp.struct
class Net:
    rows: int                 # nodes up (v)
    cols: int                 # nodes across (u)
    origin: wp.vec3           # node (0, 0)
    du: wp.vec3               # one cell across
    dv: wp.vec3               # one cell up
    node_mass: float
    rest_u: float             # cord rest lengths across and up
    rest_v: float
    ea: float                 # cord axial stiffness EA [N]
    zeta: float               # cord damping ratio (against the node mass)
    t_f: float                # knot friction: the slider's force limit [N]
    k_slip: float             # the slider's pre-slide stiffness, as a multiple of EA/rest
    pin_left: int
    pin_right: int
    pin_bottom: int
    pin_top: int
    ball_r: float             # ball radius + cord radius
    k_c: float                # contact stiffness per metre of cord in the ball [N/m²]
    c_c: float                # contact damping per metre of cord [N·s/m²]
    ground_on: int
    ground_z: float
    ground_mu: float


@wp.func
def pinned(i: int, j: int, n: Net) -> bool:
    return (j == 0 and n.pin_left != 0) or (j == n.cols - 1 and n.pin_right != 0) or \
        (i == 0 and n.pin_bottom != 0) or (i == n.rows - 1 and n.pin_top != 0)


@wp.func
def home(i: int, j: int, n: Net) -> wp.vec3:
    return n.origin + n.du * float(j) + n.dv * float(i)


@wp.func
def cord(xa: wp.vec3, va: wp.vec3, xb: wp.vec3, vb: wp.vec3, rest: float, fb: float, n: Net) -> wp.vec3:
    """Force on node a from the cord to node b: EA·strain, damping and the knot slider ``fb``, tension only."""
    d = xb - xa
    length = wp.length(d)
    if length <= rest or length < 1.0e-9:
        return wp.vec3(0.0)
    e = d / length
    k = n.ea / rest
    c = 2.0 * n.zeta * wp.sqrt(k * 0.5 * n.node_mass)
    t = k * (length - rest) + c * wp.dot(vb - va, e) + fb
    return e * wp.max(t, 0.0)


@wp.func
def slip(xa: wp.vec3, va: wp.vec3, xb: wp.vec3, vb: wp.vec3, rest: float, fb: float, h: float, n: Net) -> float:
    """The knot slider's next force: it follows the stretch (pre-slide spring k_slip·EA/rest) and is capped at
    ±t_f; a slack cord carries none."""
    d = xb - xa
    length = wp.length(d)
    if length <= rest or length < 1.0e-9 or n.t_f <= 0.0:
        return 0.0
    rate = wp.dot(vb - va, d / length)
    return wp.clamp(fb + n.k_slip * n.ea / rest * rate * h, -n.t_f, n.t_f)


@wp.func
def segment_push(xa: wp.vec3, va: wp.vec3, xb: wp.vec3, vb: wp.vec3, c: wp.vec3, vball: wp.vec3, n: Net) -> wp.vec3:
    """Node a's share of the push of the ball on the cord a-b (zero when the cord misses the ball)."""
    d = xb - xa
    length = wp.length(d)
    if length < 1.0e-9:
        return wp.vec3(0.0)
    e = d / length
    sc = wp.dot(c - xa, e)
    r = xa + e * sc - c
    dist = wp.length(r)
    if dist >= n.ball_r or dist < 1.0e-9:
        return wp.vec3(0.0)
    h = wp.sqrt(n.ball_r * n.ball_r - dist * dist)
    lo = wp.max(sc - h, 0.0)
    hi = wp.min(sc + h, length)
    if hi <= lo:
        return wp.vec3(0.0)
    inside = hi - lo
    mid = 0.5 * (lo + hi) / length
    nrm = r / dist
    rate = -wp.dot(va + (vb - va) * mid - vball, nrm)          # closing speed of the cord on the ball
    f = wp.max(n.k_c * (n.ball_r - dist) * inside + n.c_c * rate * inside, 0.0)
    return nrm * (f * (1.0 - mid))


@wp.func
def on_ground(x: wp.vec3, v: wp.vec3, n: Net) -> wp.spatial_vector:
    """(position, velocity) kept at or above the ground; friction takes up to μ·|v_n| of the sliding speed."""
    if n.ground_on == 0 or x[2] >= n.ground_z:
        return wp.spatial_vector(x, v)
    vn = wp.min(v[2], 0.0)
    vt = wp.vec3(v[0], v[1], 0.0)
    sp = wp.length(vt)
    keep = 0.0
    if sp > 1.0e-9:
        keep = wp.max(0.0, 1.0 - n.ground_mu * wp.abs(vn) / sp)
    return wp.spatial_vector(wp.vec3(x[0], x[1], n.ground_z), wp.vec3(vt[0] * keep, vt[1] * keep, v[2] - vn))


@wp.kernel
def net_substep(x: wp.array2d[wp.vec3], v: wp.array2d[wp.vec3], fr: wp.array2d[wp.vec2], n: Net,
                gravity: wp.vec3, ball_c: wp.array[wp.vec3], ball_v: wp.array[wp.vec3], tau: float, dt: float,
                x_out: wp.array2d[wp.vec3], v_out: wp.array2d[wp.vec3], fr_out: wp.array2d[wp.vec2],
                f_c: wp.array2d[wp.vec3]):
    w, k = wp.tid()
    i = k / n.cols
    j = k - i * n.cols
    xa, va = x[w, k], v[w, k]
    c = ball_c[w] + ball_v[w] * tau                  # the ball moves straight through the exchange
    vb = ball_v[w]
    f = gravity * n.node_mass
    fc = wp.vec3(0.0)
    own = wp.vec2(0.0, 0.0)                          # this node's right and up sliders
    if j > 0:
        f = f + cord(xa, va, x[w, k - 1], v[w, k - 1], n.rest_u, fr[w, k - 1][0], n)
        fc = fc + segment_push(xa, va, x[w, k - 1], v[w, k - 1], c, vb, n)
    if j < n.cols - 1:
        f = f + cord(xa, va, x[w, k + 1], v[w, k + 1], n.rest_u, fr[w, k][0], n)
        fc = fc + segment_push(xa, va, x[w, k + 1], v[w, k + 1], c, vb, n)
        own[0] = slip(xa, va, x[w, k + 1], v[w, k + 1], n.rest_u, fr[w, k][0], dt, n)
    if i > 0:
        f = f + cord(xa, va, x[w, k - n.cols], v[w, k - n.cols], n.rest_v, fr[w, k - n.cols][1], n)
        fc = fc + segment_push(xa, va, x[w, k - n.cols], v[w, k - n.cols], c, vb, n)
    if i < n.rows - 1:
        f = f + cord(xa, va, x[w, k + n.cols], v[w, k + n.cols], n.rest_v, fr[w, k][1], n)
        fc = fc + segment_push(xa, va, x[w, k + n.cols], v[w, k + n.cols], c, vb, n)
        own[1] = slip(xa, va, x[w, k + n.cols], v[w, k + n.cols], n.rest_v, fr[w, k][1], dt, n)
    f_c[w, k] = fc                                   # pinned nodes push back on the ball too
    fr_out[w, k] = own
    if pinned(i, j, n):
        x_out[w, k] = home(i, j, n)
        v_out[w, k] = wp.vec3(0.0)
        return
    vn = va + (f + fc) * (dt / n.node_mass)
    g = on_ground(xa + vn * dt, vn, n)
    x_out[w, k] = wp.spatial_top(g)
    v_out[w, k] = wp.spatial_bottom(g)


@wp.kernel
def net_reduce(f_c: wp.array2d[wp.vec3], dt: float, impulse: wp.array[wp.vec3]):
    """impulse[w] += -Σ_k f_c[w, k]·dt: what the nodes pushed back on the ball over this sub-step."""
    w = wp.tid()
    s = wp.vec3(0.0)
    for k in range(f_c.shape[1]):
        s = s + f_c[w, k]
    impulse[w] = impulse[w] - s * dt


@wp.kernel
def net_apply(impulse: wp.array[wp.vec3], balls: wp.array[wp.int32], dt_prev: wp.array[wp.float32],
              body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    h = dt_prev[w]
    if h > 0.0:
        f = impulse[w] / h
        add_wrench(body_f, balls[w], wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0))


@wp.kernel
def ball_motion(body_q_in: wp.array[wp.transform], body_q_out: wp.array[wp.transform], body_com: wp.array[wp.vec3],
                balls: wp.array[wp.int32], dt: float, c0: wp.array[wp.vec3], vel: wp.array[wp.vec3],
                impulse: wp.array[wp.vec3], dt_prev: wp.array[wp.float32]):
    """The ball's centre at the exchange start and its mean velocity over the exchange; clears the impulse."""
    w = wp.tid()
    b = balls[w]
    a = wp.transform_point(body_q_in[b], body_com[b])
    e = wp.transform_point(body_q_out[b], body_com[b])
    c0[w] = a
    vel[w] = (e - a) / dt
    impulse[w] = wp.vec3(0.0)
    dt_prev[w] = dt
