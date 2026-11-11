# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the ``Thermal`` plug-in (engine item 10, heat v1).

Thermal bodies are numbered world-major: t = w·B + b. Each step:
1. **One thread per exported contact.** Conduction G = h_c·a_c between the two sides, if both have a temperature
   (a thermal body or a fixed-temperature plate) and the contact is compressive. Friction power
   P = |f_t · v_slip|, split equally between the two sides; a side that is not a thermal body loses its half to
   the surroundings. Sums are int64 fixed point, so the order of contacts does not matter.
2. **One thread per thermal body**, semi-implicit in its own temperature (unconditionally stable):
   T' = (C·T + dt·(P + Σ G_j·T_j + hA·T_amb)) / (C + dt·(Σ G_j + hA)).
3. **One thread per motor:** a winding and housing two-node model, I²R heating from the torque input, and a derated
   torque limit.

Temperatures are float64: a step's change (~1e-7 K) is below float32's resolution at 300 K.
"""

from __future__ import annotations

import warp as wp

FIXQ = 4294967296.0     # 2^32 counts per W (or W/K): 2.3e-10 resolution, ±2e9 range


@wp.func
def _q(x: float) -> wp.int64:
    return wp.int64(wp.round(wp.float64(x) * wp.float64(FIXQ)))


@wp.func
def _side_temp(b: int, therm_of_body: wp.array(dtype=wp.int32), plate_of_body: wp.array(dtype=wp.int32),
               temp: wp.array(dtype=wp.float64), plate_temp: wp.array(dtype=wp.float64)) -> wp.float64:
    """The side's temperature [K], or -1 if it has none (not a thermal body or plate)."""
    if b < 0:
        return wp.float64(-1.0)
    t = therm_of_body[b]
    if t >= 0:
        return temp[t]
    p = plate_of_body[b]
    if p >= 0:
        return plate_temp[p]
    return wp.float64(-1.0)


@wp.func
def _point_velocity(b: int, p: wp.vec3, body_q: wp.array(dtype=wp.transform),
                    body_qd: wp.array(dtype=wp.spatial_vector), body_com: wp.array(dtype=wp.vec3)) -> wp.vec3:
    if b < 0:
        return wp.vec3(0.0)
    qd = body_qd[b]
    com = wp.transform_point(body_q[b], body_com[b])
    return wp.spatial_top(qd) + wp.cross(wp.spatial_bottom(qd), p - com)


@wp.func
def _deposit(b: int, therm_of_body: wp.array(dtype=wp.int32), g: float, t_other: wp.float64, power: float,
             g_acc: wp.array(dtype=wp.int64), gt_acc: wp.array(dtype=wp.int64), p_acc: wp.array(dtype=wp.int64)):
    if b < 0:
        return
    t = therm_of_body[b]
    if t < 0:
        return
    if g > 0.0 and t_other >= wp.float64(0.0):
        wp.atomic_add(g_acc, t, _q(g))
        wp.atomic_add(gt_acc, t, wp.int64(wp.round(wp.float64(g) * t_other * wp.float64(FIXQ))))
    if power > 0.0:
        wp.atomic_add(p_acc, t, _q(power))


@wp.kernel
def contact_heat(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32), shape1: wp.array(dtype=wp.int32),
                 point0: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3),
                 contact_f: wp.array(dtype=wp.spatial_vector), shape_body: wp.array(dtype=wp.int32),
                 body_q: wp.array(dtype=wp.transform), qd_in: wp.array(dtype=wp.spatial_vector),
                 qd_out: wp.array(dtype=wp.spatial_vector), body_com: wp.array(dtype=wp.vec3),
                 therm_of_body: wp.array(dtype=wp.int32), plate_of_body: wp.array(dtype=wp.int32),
                 temp: wp.array(dtype=wp.float64), plate_temp: wp.array(dtype=wp.float64), g_contact: float,
                 friction: int, g_acc: wp.array(dtype=wp.int64), gt_acc: wp.array(dtype=wp.int64),
                 p_acc: wp.array(dtype=wp.int64)):
    i = wp.tid()
    if i >= count[0]:
        return
    s0 = shape0[i]
    s1 = shape1[i]
    b0 = int(-1)
    b1 = int(-1)
    if s0 >= 0:
        b0 = shape_body[s0]
    if s1 >= 0:
        b1 = shape_body[s1]
    f = wp.spatial_top(contact_f[i])
    n = normal[i]
    if wp.dot(f, n) >= 0.0:
        return                      # not compressive: no contact pressure, no heat path
    t0 = _side_temp(b0, therm_of_body, plate_of_body, temp, plate_temp)
    t1 = _side_temp(b1, therm_of_body, plate_of_body, temp, plate_temp)
    g = float(0.0)
    if t0 >= wp.float64(0.0) and t1 >= wp.float64(0.0):
        g = g_contact
    power = float(0.0)
    if friction != 0:
        p = point0[i]
        if b0 >= 0:
            p = wp.transform_point(body_q[b0], point0[i])
        v = 0.5 * (_point_velocity(b1, p, body_q, qd_in, body_com) + _point_velocity(b1, p, body_q, qd_out, body_com)
                   - _point_velocity(b0, p, body_q, qd_in, body_com) - _point_velocity(b0, p, body_q, qd_out, body_com))
        ft = f - wp.dot(f, n) * n
        vt = v - wp.dot(v, n) * n
        power = 0.5 * wp.abs(wp.dot(ft, vt))
    _deposit(b0, therm_of_body, g, t1, power, g_acc, gt_acc, p_acc)
    _deposit(b1, therm_of_body, g, t0, power, g_acc, gt_acc, p_acc)


@wp.kernel
def update_bodies(capacity: wp.array(dtype=wp.float64), h_area: wp.array(dtype=wp.float64), ambient: wp.float64,
                  dt: float, g_acc: wp.array(dtype=wp.int64), gt_acc: wp.array(dtype=wp.int64),
                  p_acc: wp.array(dtype=wp.int64), ext: wp.array(dtype=wp.int64), temp: wp.array(dtype=wp.float64),
                  conductance: wp.array(dtype=wp.float64), friction_heat: wp.array(dtype=wp.float64),
                  ext_total: wp.array(dtype=wp.float64)):
    """One thread per thermal body: the semi-implicit update; zeroes its accumulators. ``ext`` is heat put in by
    other plug-ins since the last update [µJ, int64] (consumed here, ``ext_total`` keeps the sum in J)."""
    t = wp.tid()
    q = wp.float64(ext[t]) * wp.float64(1.0e-6)
    ext[t] = wp.int64(0)
    ext_total[t] = ext_total[t] + q
    fq = wp.float64(FIXQ)
    g = wp.float64(g_acc[t]) / fq
    gt = wp.float64(gt_acc[t]) / fq
    p = wp.float64(p_acc[t]) / fq
    g_acc[t] = wp.int64(0)
    gt_acc[t] = wp.int64(0)
    p_acc[t] = wp.int64(0)
    c = capacity[t]
    ha = h_area[t]
    h = wp.float64(dt)
    temp[t] = (c * temp[t] + q + h * (p + gt + ha * ambient)) / (c + h * (g + ha))
    conductance[t] = g
    friction_heat[t] = friction_heat[t] + h * p


@wp.kernel
def update_motors(torque: wp.array(dtype=wp.float32), prm: wp.array(dtype=wp.float64), ambient: wp.float64, dt: float,
                  winding: wp.array(dtype=wp.float64), housing: wp.array(dtype=wp.float64),
                  limit: wp.array(dtype=wp.float32)):
    """One thread per (world, motor). prm = (k_t, R, C_w, C_h, R_wh, R_ha, T_derate, T_max, tau_max)."""
    k = wp.tid()
    h = wp.float64(dt)
    cur = wp.float64(torque[k]) / prm[0]
    power = cur * cur * prm[1]
    tw = (prm[2] * winding[k] + h * (power + housing[k] / prm[4])) / (prm[2] + h / prm[4])
    g_h = wp.float64(1.0) / prm[4] + wp.float64(1.0) / prm[5]
    th = (prm[3] * housing[k] + h * (tw / prm[4] + ambient / prm[5])) / (prm[3] + h * g_h)
    winding[k] = tw
    housing[k] = th
    frac = wp.clamp((prm[7] - tw) / (prm[7] - prm[6]), wp.float64(0.0), wp.float64(1.0))
    limit[k] = wp.float32(prm[8] * frac)
