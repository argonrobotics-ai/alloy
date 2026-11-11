# SPDX-License-Identifier: Apache-2.0
"""Warp kernels behind ``BenchMonitors`` (D11 bench-only checks).

Per-world sums are gathers: one thread per world walks that world's items in a fixed order (CSR from ``world_csr``),
so energies and work are bitwise repeatable and use no float atomics (SPEC §3.7). Per-world maxima use
``atomic_max`` on float32 / int32, which is order independent and allowed (§3.1, §3.7).
"""

from __future__ import annotations

import numpy as np
import warp as wp


def world_csr(item_world: np.ndarray, world_count: int) -> tuple[np.ndarray, np.ndarray]:
    """(start [W+1], index [N]) listing the items of each world in ascending item order; world -1 is dropped."""
    item_world = np.asarray(item_world, dtype=np.int64)
    keep = np.nonzero((item_world >= 0) & (item_world < world_count))[0]
    order = keep[np.argsort(item_world[keep], kind="stable")]
    counts = np.bincount(item_world[keep], minlength=world_count)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return start, order.astype(np.int32)


@wp.func
def _pe(m: float, g: wp.vec3, x: wp.vec3, z_ref: wp.float64) -> wp.float64:
    # potential energy above the world's datum (D20): -m g . (x - z_ref e_z), height difference in float64
    dz = wp.float64(x[2]) - z_ref
    horiz = -m * (g[0] * x[0] + g[1] * x[1])
    return wp.float64(horiz) + wp.float64(-m * g[2]) * dz


@wp.kernel
def body_energy_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    z_ref: wp.array[wp.float64],
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    w = wp.tid()
    e = wp.float64(0.0)
    for i in range(start[w], start[w + 1]):
        b = index[i]
        m = body_mass[b]
        if m > 0.0:
            x = wp.transform_point(body_q[b], body_com[b])
            v = wp.spatial_top(body_qd[b])
            om = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), wp.spatial_bottom(body_qd[b]))
            ke = 0.5 * m * wp.dot(v, v) + 0.5 * wp.dot(om, body_inertia[b] * om)
            e += wp.float64(ke) + _pe(m, gravity[w], x, z_ref[w])
    out[w] = e


@wp.kernel
def particle_energy_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_flags: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    z_ref: wp.array[wp.float64],
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    w = wp.tid()
    e = wp.float64(0.0)
    for i in range(start[w], start[w + 1]):
        p = index[i]
        m = particle_mass[p]
        if m > 0.0 and (particle_flags[p] & 1) != 0:  # ParticleFlags.ACTIVE
            v = particle_qd[p]
            e += wp.float64(0.5 * m * wp.dot(v, v)) + _pe(m, gravity[w], particle_q[p], z_ref[w])
    out[w] = out[w] + e


@wp.kernel
def body_force_work_kernel(
    body_f: wp.array[wp.spatial_vector],
    qd_prev: wp.array[wp.spatial_vector],
    qd_cur: wp.array[wp.spatial_vector],
    dt: float,
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    # work of the applied wrench held over the step (linear at the COM, angular in world frame), midpoint velocity
    w = wp.tid()
    work = wp.float64(0.0)
    for i in range(start[w], start[w + 1]):
        b = index[i]
        f = body_f[b]
        v = 0.5 * (qd_prev[b] + qd_cur[b])
        work += wp.float64(dt * (wp.dot(wp.spatial_top(f), wp.spatial_top(v)) + wp.dot(wp.spatial_bottom(f), wp.spatial_bottom(v))))
    out[w] = work


@wp.kernel
def particle_force_work_kernel(
    particle_f: wp.array[wp.vec3],
    qd_prev: wp.array[wp.vec3],
    qd_cur: wp.array[wp.vec3],
    flags: wp.array[wp.int32],
    dt: float,
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    # work of applied particle forces held over the step, at the midpoint velocity; added to the wrench work
    w = wp.tid()
    work = wp.float64(0.0)
    for i in range(start[w], start[w + 1]):
        p = index[i]
        if (flags[p] & 1) != 0:
            work += wp.float64(dt * wp.dot(particle_f[p], 0.5 * (qd_prev[p] + qd_cur[p])))
    out[w] = out[w] + work


@wp.kernel
def shape_work_kernel(
    shape_work: wp.array[wp.float64],
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    # work robot shapes' boundaries did on a solver's particles this step (SCR-WP9-6); added to the wrench work
    w = wp.tid()
    work = wp.float64(0.0)
    for j in range(start[w], start[w + 1]):
        work += shape_work[index[j]]
    out[w] = out[w] + work


@wp.kernel
def ledger_kernel(
    e_now: wp.array[wp.float64],
    e_prev: wp.array[wp.float64],
    w_force: wp.array[wp.float64],
    robot_touch: wp.array[wp.int32],
    e_scale: wp.array[wp.float64],
    cum: wp.array[wp.float64],
    gain_step_max: wp.array[wp.float32],
    gain_cum_max: wp.array[wp.float32],
):
    # passive-subsystem ledger: gain = dE_passive - W(applied wrenches); a step in which the robot touches a passive
    # body is excused (the robot does unmeasured work through contact) and adds nothing to the cumulative gain
    w = wp.tid()
    gain = e_now[w] - e_prev[w] - w_force[w]
    if robot_touch[w] != 0:
        gain = wp.float64(0.0)
    cum[w] = cum[w] + gain
    gain_step_max[w] = wp.max(gain_step_max[w], wp.float32(gain / e_scale[w]))
    gain_cum_max[w] = wp.max(gain_cum_max[w], wp.float32(cum[w] / e_scale[w]))
    e_prev[w] = e_now[w]


@wp.kernel
def contact_touch_kernel(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_robot: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    touch_any: wp.array[wp.int32],
    touch_robot: wp.array[wp.int32],
    world_robot_touch: wp.array[wp.int32],
):
    # per body: 1 if in any contact row, 1 if in a row with a robot body; per world: a robot touched a passive body
    # (int atomics only)
    c = wp.tid()
    if c >= count[0]:
        return
    b0 = -1
    b1 = -1
    if shape0[c] >= 0:
        b0 = shape_body[shape0[c]]
    if shape1[c] >= 0:
        b1 = shape_body[shape1[c]]
    if b0 >= 0:
        wp.atomic_max(touch_any, b0, 1)
        if b1 >= 0 and body_robot[b1] != 0 and body_robot[b0] == 0:
            wp.atomic_max(touch_robot, b0, 1)
            wp.atomic_max(world_robot_touch, body_world[b0], 1)
    if b1 >= 0:
        wp.atomic_max(touch_any, b1, 1)
        if b0 >= 0 and body_robot[b0] != 0 and body_robot[b1] == 0:
            wp.atomic_max(touch_robot, b1, 1)
            wp.atomic_max(world_robot_touch, body_world[b1], 1)


@wp.kernel
def soft_touch_kernel(
    count: wp.array[wp.int32],
    soft_shape: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_robot: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    excuse: int,
    touch_any: wp.array[wp.int32],
    world_robot_touch: wp.array[wp.int32],
):
    # a body touching particles (cloth, liquid, a soft solid) is in contact: its acceleration is explained. With
    # `excuse` (no shape-work source books the robot's work on particles), a robot touching particles excuses its
    # world's step like a rigid robot contact: a blade's work on a soft solid is not measured (WP15, 2026-11-09)
    c = wp.tid()
    if c >= count[0]:
        return
    s = soft_shape[c]
    if s >= 0:
        b = shape_body[s]
        if b >= 0:
            wp.atomic_max(touch_any, b, 1)
            if excuse != 0 and body_robot[b] != 0 and body_world[b] >= 0:
                wp.atomic_max(world_robot_touch, body_world[b], 1)


@wp.struct
class BodyChecks:
    """Per-world worst values written by ``body_checks_kernel`` (float32 maxima, int32 flags)."""

    nonfinite: wp.array[wp.int32]
    max_speed: wp.array[wp.float32]  # m/s, any body
    teleport: wp.array[wp.float32]  # |dx| / (10 v_max dt), S1 above 1
    nontarget_disp: wp.array[wp.float32]  # m, COM shift + radius x rotation angle since reset
    # m/s, D12 as amended by SCR-WP7-7 (``impossible_speed_kernel``): above v_max = speed not explained by robot
    # contact or free fall
    impossible_speed: wp.array[wp.float32]
    impossible_speed_d12: wp.array[wp.float32]  # m/s, D12 before SCR-WP7-7: any free-body speed without robot contact
    contact_free_accel: wp.array[wp.float32]  # m/s^2, |a - g| of free bodies with no contact at all
    joint_sep: wp.array[wp.float32]  # m, anchor separation of joints that lock translation
    max_pen: wp.array[wp.float32]  # m, contact-buffer penetration (fallback when no HealthMonitor)
    pen_s1_ratio: wp.array[wp.float32]  # pen / min(pen_s1, half the thinner part), S1 above 1


@wp.func
def _finite_tf(q: wp.transform, v: wp.spatial_vector) -> bool:
    s = wp.float32(0.0)
    for k in range(7):
        s += q[k]
    for k in range(6):
        s += v[k]
    return wp.isfinite(s)


@wp.func
def _displacement(q0: wp.transform, q1: wp.transform, com: wp.vec3, radius: float) -> float:
    # farthest surface point shift <= |COM shift| + radius * rotation angle
    shift = wp.length(wp.transform_point(q1, com) - wp.transform_point(q0, com))
    dq = wp.transform_get_rotation(q1) * wp.quat_inverse(wp.transform_get_rotation(q0))
    angle = 2.0 * wp.acos(wp.min(wp.abs(dq[3]), 1.0))
    return shift + radius * angle


@wp.kernel
def body_checks_kernel(
    q_prev: wp.array[wp.transform],
    qd_prev: wp.array[wp.spatial_vector],
    q_cur: wp.array[wp.transform],
    qd_cur: wp.array[wp.spatial_vector],
    q_reset: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_world: wp.array[wp.int32],
    body_kind: wp.array[wp.int32],  # bit 0 free, bit 1 robot, bit 2 non-target
    body_radius: wp.array[float],
    touch_any: wp.array[wp.int32],
    touch_robot: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    v_max: float,
    dt: float,
    out: BodyChecks,
):
    b = wp.tid()
    w = body_world[b]
    if w < 0:
        return
    if not _finite_tf(q_cur[b], qd_cur[b]):
        wp.atomic_max(out.nonfinite, w, 1)
        return
    x1 = wp.transform_point(q_cur[b], body_com[b])
    v1 = wp.spatial_top(qd_cur[b])
    speed = wp.length(v1)
    wp.atomic_max(out.max_speed, w, speed)
    x0 = wp.transform_point(q_prev[b], body_com[b])
    wp.atomic_max(out.teleport, w, wp.length(x1 - x0) / (10.0 * v_max * dt))
    kind = body_kind[b]
    if (kind & 4) != 0:
        wp.atomic_max(out.nontarget_disp, w, _displacement(q_reset[b], q_cur[b], body_com[b], body_radius[b]))
    if (kind & 1) != 0 and touch_robot[b] == 0:
        wp.atomic_max(out.impossible_speed_d12, w, speed)
    if (kind & 1) != 0 and touch_any[b] == 0:
        a = (v1 - wp.spatial_top(qd_prev[b])) / dt - gravity[w]
        wp.atomic_max(out.contact_free_accel, w, wp.length(a))



@wp.struct
class FlightState:
    """Per-body state of the SCR-WP7-7 ballistic envelope, carried across substeps (reset per episode)."""

    exit_speed: wp.array[wp.float32]  # m/s, |v| in the last substep with a contact or the one after it
    free_t: wp.array[wp.float32]  # s of contact-free flight since then
    touch_prev: wp.array[wp.int32]  # any contact in the previous substep
    robot_prev: wp.array[wp.int32]  # robot contact in the previous substep


@wp.kernel
def impossible_speed_kernel(
    qd_cur: wp.array[wp.spatial_vector],
    body_world: wp.array[wp.int32],
    body_kind: wp.array[wp.int32],
    touch_any: wp.array[wp.int32],
    touch_robot: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    contacts_known: int,  # 0: no contact buffer, every substep counts as in contact (fail closed)
    v_max: float,
    margin: float,
    dt: float,
    fs: FlightState,
    out: BodyChecks,
):
    # D12 / §4.3 as amended by SCR-WP7-7: in contact or one substep after, the body's own speed (robot contact in
    # either substep excuses it); in contact-free flight, S1 above max(v_max, |v_exit| + g t + margin). The stored
    # value is the speed scaled so that v_max is the threshold in both cases.
    b = wp.tid()
    w = body_world[b]
    if w < 0 or (body_kind[b] & 1) == 0:
        return
    speed = wp.length(wp.spatial_top(qd_cur[b]))
    if not wp.isfinite(speed):
        return  # the nonfinite row has it
    if contacts_known == 0 or touch_any[b] != 0 or fs.touch_prev[b] != 0:
        fs.exit_speed[b] = speed
        fs.free_t[b] = 0.0
        if touch_robot[b] == 0 and fs.robot_prev[b] == 0:
            wp.atomic_max(out.impossible_speed, w, speed)
        return
    t = fs.free_t[b] + dt
    fs.free_t[b] = t
    envelope = wp.max(v_max, fs.exit_speed[b] + wp.length(gravity[w]) * t + margin)
    wp.atomic_max(out.impossible_speed, w, speed * v_max / envelope)


@wp.kernel
def particle_checks_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[wp.int32],
    out: BodyChecks,
):
    # active particles: non-finite state and speed (D12: particles count for the speed checks)
    p = wp.tid()
    w = particle_world[p]
    if w < 0 or (particle_flags[p] & 1) == 0:
        return
    x = particle_q[p]
    v = particle_qd[p]
    if not wp.isfinite(x[0] + x[1] + x[2] + v[0] + v[1] + v[2]):
        wp.atomic_max(out.nonfinite, w, 1)
        return
    wp.atomic_max(out.max_speed, w, wp.length(v))


@wp.kernel
def contact_pen_kernel(
    count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    shape_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_thin: wp.array[float],
    pen_s1: float,
    out: BodyChecks,
):
    # rows detected on state_in, evaluated at state_out (Newton contact_surface_separation, contacts.py:71-91)
    c = wp.tid()
    if c >= count[0]:
        return
    s0 = shape0[c]
    s1 = shape1[c]
    if s0 < 0 or s1 < 0:
        return
    w = wp.max(shape_world[s0], shape_world[s1])
    if w < 0:
        return
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    x0 = point0[c]
    x1 = point1[c]
    limit = pen_s1
    if b0 >= 0:
        x0 = wp.transform_point(body_q[b0], x0)
        limit = wp.min(limit, 0.5 * body_thin[b0])
    if b1 >= 0:
        x1 = wp.transform_point(body_q[b1], x1)
        limit = wp.min(limit, 0.5 * body_thin[b1])
    pen = -(wp.dot(normal[c], x1 - x0) - (margin0[c] + margin1[c]))
    if pen > 0.0:
        wp.atomic_max(out.max_pen, w, pen)
        wp.atomic_max(out.pen_s1_ratio, w, pen / limit)


@wp.kernel
def joint_sep_kernel(
    joint_type: wp.array[wp.int32],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    out: BodyChecks,
):
    # REVOLUTE (1), BALL (2), FIXED (3) lock translation: their two anchors must coincide (maximal-coordinate solvers)
    j = wp.tid()
    t = joint_type[j]
    w = joint_world[j]
    if w < 0 or not joint_enabled[j] or (t != 1 and t != 2 and t != 3):
        return
    xp = joint_X_p[j]
    if joint_parent[j] >= 0:
        xp = body_q[joint_parent[j]] * xp
    xc = body_q[joint_child[j]] * joint_X_c[j]
    wp.atomic_max(out.joint_sep, w, wp.length(wp.transform_get_translation(xp) - wp.transform_get_translation(xc)))


# MuJoCo Warp OverflowType bits that mean dropped data (rows, contacts, CCD/EPA/sensor buffers): everything but
# ITERATIONS (1 << 9) and LS_ITERATIONS (1 << 10), which report an unconverged solve, not a lost constraint
# (mujoco_warp/_src/types.py OverflowType).
CAPACITY_OVERFLOW_MASK = 0xFFFF & ~((1 << 9) | (1 << 10))


@wp.kernel
def overflow_or_kernel(bits: wp.array[int], mask: int, sticky: wp.array[wp.int32]):
    """sticky[w] |= bits[w] & mask (Newton's SolverMuJoCo.step zeroes ``bits`` before every step)."""
    w = wp.tid()
    sticky[w] = sticky[w] | wp.int32(bits[w] & mask)

