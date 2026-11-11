# SPDX-License-Identifier: Apache-2.0
"""Fluid wrenches on bodies, embedded body integration and the proxy harvest (SCR-WP9-2).

Boundary reactions are summed per body and per shape in int64 fixed point (deterministic, no float atomics):
``step`` (force [N], reset every step, drives integration), ``exchange`` (impulse [N·s], reset when the coupler
syncs proxies, drives the direct harvest) and ``meter`` (impulse per shape, reset by the user, for force laws).
``work`` (per shape, reset every step) is the work [J] each shape's boundary did on the liquid at the wall's point
velocity, -J·v_b for the pressure reaction J; the projection pass adds its velocity corrections. A moving wall
(a driven cup) gives the liquid energy that no wrench carries: energy ledgers count it (SCR-WP9-6).

Body modes: 1 = integrated here with the fluid wrench at the view's (effective) mass, so Newton's default momentum
harvest (newton@ac7f7830:newton/_src/solvers/coupled/proxy_utils.py:180-206) returns the fluid wrench and the fluid
sees the body respond inside every substep; 2 = kinematic, advanced with its own velocity; 0 = frozen.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from alloy._vendor.newton import BodyFlags

IMP_SCALE = wp.constant(1125899906842624.0)  # 2^50: impulse fixed point (8.9e-16 N·s resolution)
FORCE_SCALE = wp.constant(1099511627776.0)  # 2^40: force fixed point
WORK_SCALE = wp.constant(1125899906842624.0)  # 2^50: work fixed point (8.9e-16 J resolution, |W| < 8192 J/step)
IMP_SCALE_HOST = float(2**50)
FORCE_SCALE_HOST = float(2**40)
MODE_FROZEN, MODE_INTEGRATE, MODE_KINEMATIC = 0, 1, 2


@wp.func
def _fx(x: float, scale: float) -> wp.int64:
    return wp.int64(wp.round(wp.float64(x) * wp.float64(scale)))


@wp.func
def _ufx(q: wp.int64, scale: float) -> float:
    return float(wp.float64(q) / wp.float64(scale))


@wp.kernel(enable_backward=False)
def accumulate_kernel(
    slot_shape: wp.array2d[wp.int32], slot_xb: wp.array2d[wp.vec3], slot_vb: wp.array2d[wp.vec3],
    slot_impulse: wp.array2d[wp.vec3], shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3], dt: float, one_way: int, step_f: wp.array2d[wp.int64], exch_j: wp.array2d[wp.int64],
    meter_j: wp.array2d[wp.int64], work: wp.array[wp.int64],
):
    i, k = wp.tid()
    s = slot_shape[i, k]
    if s < 0 or s >= shape_body.shape[0]:  # soft faces: soft_walls.py
        return
    J = slot_impulse[i, k]
    if J[0] == 0.0 and J[1] == 0.0 and J[2] == 0.0:
        return
    b = shape_body[s]
    if b >= 0:  # the wall's work on the liquid (module doc); static walls do none
        wp.atomic_add(work, s, _fx(-wp.dot(J, slot_vb[i, k]), WORK_SCALE))
    if one_way != 0:  # the liquid does not push back: no meters, no body wrench
        return
    for c in range(3):
        wp.atomic_add(meter_j, s, c, _fx(J[c], IMP_SCALE))
    if b < 0:
        return
    com = wp.transform_point(body_q[b], body_com[b])
    T = wp.cross(slot_xb[i, k] - com, J)
    inv_dt = 1.0 / dt
    for c in range(3):
        wp.atomic_add(step_f, b, c, _fx(J[c] * inv_dt, FORCE_SCALE))
        wp.atomic_add(step_f, b, 3 + c, _fx(T[c] * inv_dt, FORCE_SCALE))
        wp.atomic_add(exch_j, b, c, _fx(J[c], IMP_SCALE))
        wp.atomic_add(exch_j, b, 3 + c, _fx(T[c], IMP_SCALE))


@wp.func
def _row6(a: wp.array2d[wp.int64], b: int, scale: float) -> wp.spatial_vector:
    return wp.spatial_vector(
        wp.vec3(_ufx(a[b, 0], scale), _ufx(a[b, 1], scale), _ufx(a[b, 2], scale)),
        wp.vec3(_ufx(a[b, 3], scale), _ufx(a[b, 4], scale), _ufx(a[b, 5], scale)),
    )


@wp.kernel(enable_backward=False)
def integrate_bodies_kernel(
    mode: wp.array[wp.int32], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector], has_f: int, body_mass: wp.array[wp.float32],
    body_inv_mass: wp.array[wp.float32], body_inertia: wp.array[wp.mat33], body_inv_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3], body_world: wp.array[wp.int32], gravity: wp.array[wp.vec3],
    step_f: wp.array2d[wp.int64], dt: float, body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    """Semi-implicit Euler with gyroscopic term; body_qd = (v_com, omega) world, body_f at the COM (Newton)."""
    b = wp.tid()
    q = body_q[b]
    qd = body_qd[b]
    m = mode[b]
    if m == 0:
        body_q_out[b] = q
        body_qd_out[b] = qd
        return
    v = wp.spatial_top(qd)
    w = wp.spatial_bottom(qd)
    rot = wp.transform_get_rotation(q)
    if m == 1:
        fl = _row6(step_f, b, FORCE_SCALE)
        F = wp.spatial_top(fl)
        T = wp.spatial_bottom(fl)
        if has_f != 0:
            F += wp.spatial_top(body_f[b])
            T += wp.spatial_bottom(body_f[b])
        wid = body_world[b]
        g = wp.vec3(0.0, 0.0, 0.0)
        if wid >= 0:
            g = gravity[wid]
        v = v + dt * (F * body_inv_mass[b] + g)
        wb = wp.quat_rotate_inv(rot, w)
        tb = wp.quat_rotate_inv(rot, T)
        wb = wb + dt * (body_inv_inertia[b] * (tb - wp.cross(wb, body_inertia[b] * wb)))
        w = wp.quat_rotate(rot, wb)
    com = wp.transform_point(q, body_com[b]) + dt * v
    rot_new = wp.normalize(rot + wp.quat(w, 0.0) * rot * (0.5 * dt))
    pos = com - wp.quat_rotate(rot_new, body_com[b])
    body_q_out[b] = wp.transform(pos, rot_new)
    body_qd_out[b] = wp.spatial_vector(v, w)


@wp.kernel(enable_backward=False)
def harvest_kernel(
    local_to_global: wp.array[int], mode: wp.array[wp.int32], qd_before: wp.array[wp.spatial_vector],
    qd_after: wp.array[wp.spatial_vector], body_q: wp.array[wp.transform], body_mass: wp.array[wp.float32],
    body_inertia: wp.array[wp.mat33], exch_j: wp.array2d[wp.int64], dt: float,
    out: wp.array[wp.spatial_vector],
):
    """Integrated bodies: Newton's momentum harvest m dv/dt added onto the lagged value (one writer per id).
    Others: the exchange-averaged fluid wrench written directly (MPM pattern)."""
    b = wp.tid()
    gid = local_to_global[b]
    if gid < 0:
        return
    if mode[b] == 1:
        dv = wp.spatial_top(qd_after[b]) - wp.spatial_top(qd_before[b])
        dw = wp.spatial_bottom(qd_after[b]) - wp.spatial_bottom(qd_before[b])
        r = wp.transform_get_rotation(body_q[b])
        f = body_mass[b] * dv / dt
        tau = wp.quat_rotate(r, body_inertia[b] * wp.quat_rotate_inv(r, dw)) / dt
        out[gid] = out[gid] + wp.spatial_vector(f, tau)
    else:
        jw = _row6(exch_j, b, IMP_SCALE)
        out[gid] = wp.spatial_vector(wp.spatial_top(jw) / dt, wp.spatial_bottom(jw) / dt)


def body_modes(flags: np.ndarray, inv_mass: np.ndarray, integrate: str) -> np.ndarray:
    """Per-body mode from BodyFlags, inverse mass and Config.integrate_bodies."""
    proxy = (flags & int(BodyFlags.PROXY)) != 0
    kinematic = ((flags & int(BodyFlags.KINEMATIC)) != 0) | (inv_mass <= 0.0)
    dyn = ~kinematic
    integ = dyn & ((proxy & (integrate in ("proxies", "dynamic"))) | (~proxy & (integrate == "dynamic")))
    mode = np.full(len(flags), MODE_KINEMATIC, dtype=np.int32)
    mode[integ] = MODE_INTEGRATE
    return mode


@wp.kernel(enable_backward=False)
def work_to_float_kernel(work: wp.array[wp.int64], out: wp.array[wp.float64]):
    s = wp.tid()
    out[s] = wp.float64(work[s]) / wp.float64(WORK_SCALE)


def fixed_rows_to_float(a: np.ndarray, scale: float) -> np.ndarray:
    """Host conversion of fixed-point int64 rows to float64."""
    return a.astype(np.float64) / scale
