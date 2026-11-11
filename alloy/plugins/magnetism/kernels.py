# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of ``Magnetism``: one thread per world adds every magnetic wrench of that world (no atomics), and a
second kernel turns the drives. The formulas are those of ``field.py`` (μ0/4π = 1e-7):
- body dipole ↔ body dipole: force on each, torque on each; the reaction acts at p_j - r̂·max(d, r_min), so linear
  and angular momentum balance exactly, as in the library ``magnet`` plug-in;
- drive → body dipole: force and torque on the body. The reaction (force and the drive's own torque) goes to the
  drive's support body, or to the world;
- plate → body dipole: the image dipole's force and torque. The plate is static (the reaction goes to the world),
  one-sided (only dipoles in front of it) and bounded by its half extents (0 = unbounded). Heights below
  r_min / 2 are evaluated at r_min / 2;
- uniform field: torque m × B.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, body_point, body_vector, com_world, torque_wrench, wrench_at

MU0_4PI = 1.0e-7
TWO_PI = 6.283185307179586


@wp.func
def dip_field(ms: wp.vec3, rh: wp.vec3, d: float) -> wp.vec3:
    """B at distance d along r̂ from a dipole m_s."""
    return (3.0 * rh * wp.dot(ms, rh) - ms) * (MU0_4PI / (d * d * d))


@wp.func
def dip_force(ms: wp.vec3, m: wp.vec3, rh: wp.vec3, d: float) -> wp.vec3:
    """Force on m at distance d along r̂ from m_s."""
    a = wp.dot(ms, rh)
    b = wp.dot(m, rh)
    return (a * m + b * ms + wp.dot(ms, m) * rh - 5.0 * a * b * rh) * (3.0 * MU0_4PI / (d * d * d * d))


@wp.func
def unit_or_z(r: wp.vec3, d: float) -> wp.vec3:
    if d > 0.0:
        return r / d
    return wp.vec3(0.0, 0.0, 1.0)


@wp.struct
class Drives:
    body: wp.array2d(dtype=wp.int32)      # [W, D] support body, -1 = world
    pos: wp.array(dtype=wp.vec3)          # [D] in the support body's frame (or world)
    e1: wp.array(dtype=wp.vec3)           # [D] moment direction at phase 0
    e2: wp.array(dtype=wp.vec3)           # [D] at phase π/2 (axis × e1)
    moment: wp.array(dtype=float)         # [D] A·m²


@wp.struct
class Plates:
    point: wp.array(dtype=wp.vec3)        # [P] a point of the plate's front face (world)
    normal: wp.array(dtype=wp.vec3)       # [P] outward unit normal
    u: wp.array(dtype=wp.vec3)            # [P] in-plane axes
    v: wp.array(dtype=wp.vec3)
    half: wp.array(dtype=wp.vec2)         # [P] half extents along u, v; <= 0 = unbounded
    kappa: wp.array(dtype=float)          # [P] (μr - 1)/(μr + 1)


@wp.func
def drive_moment(dr: Drives, body_q: wp.array(dtype=wp.transform), w: int, k: int, phase: float) -> wp.vec3:
    m = (wp.cos(phase) * dr.e1[k] + wp.sin(phase) * dr.e2[k]) * dr.moment[k]
    b = dr.body[w, k]
    if b >= 0:
        return body_vector(body_q, b, m)
    return m


@wp.func
def drive_point(dr: Drives, body_q: wp.array(dtype=wp.transform), w: int, k: int) -> wp.vec3:
    b = dr.body[w, k]
    if b >= 0:
        return body_point(body_q, b, dr.pos[k])
    return dr.pos[k]


@wp.func
def plate_wrench(pl: Plates, q: int, p: wp.vec3, m: wp.vec3, com: wp.vec3, r_min: float) -> wp.spatial_vector:
    n = pl.normal[q]
    h = wp.dot(p - pl.point[q], n)
    if h <= 0.0:
        return wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    foot = p - h * n
    half = pl.half[q]
    if half[0] > 0.0 and wp.abs(wp.dot(foot - pl.point[q], pl.u[q])) > half[0]:
        return wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if half[1] > 0.0 and wp.abs(wp.dot(foot - pl.point[q], pl.v[q])) > half[1]:
        return wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    d = 2.0 * wp.max(h, 0.5 * r_min)
    mi = pl.kappa[q] * (2.0 * wp.dot(m, n) * n - m)           # the image's moment; r̂ from the image is +n
    f = dip_force(mi, m, n, d)
    tau = wp.cross(m, dip_field(mi, n, d))
    return wrench_at(f, p, com) + torque_wrench(tau)


@wp.kernel
def magnetism(body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
              bodies: wp.array2d(dtype=wp.int32), offsets: wp.array(dtype=wp.vec3), axes: wp.array(dtype=wp.vec3),
              moments: wp.array(dtype=float), dr: Drives, n_drives: int, phase: wp.array2d(dtype=float), pl: Plates,
              n_plates: int, r_min: float, b_ext: wp.vec3, body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    n = bodies.shape[1]
    for i in range(n):
        bi = bodies[w, i]
        pi = body_point(body_q, bi, offsets[i])
        mi = body_vector(body_q, bi, axes[i]) * moments[i]
        ci = com_world(body_q, body_com, bi)
        add_wrench(body_f, bi, torque_wrench(wp.cross(mi, b_ext)))
        for j in range(i + 1, n):                          # body dipole pairs
            bj = bodies[w, j]
            if bj == bi:
                continue
            pj = body_point(body_q, bj, offsets[j])
            mj = body_vector(body_q, bj, axes[j]) * moments[j]
            r = pj - pi
            d = wp.length(r)
            rh = unit_or_z(r, d)
            dd = wp.max(d, r_min)
            f = dip_force(mi, mj, rh, dd)
            add_wrench(body_f, bj, wrench_at(f, pj, com_world(body_q, body_com, bj))
                       + torque_wrench(wp.cross(mj, dip_field(mi, rh, dd))))
            add_wrench(body_f, bi, wrench_at(-f, pj - rh * dd, ci) + torque_wrench(wp.cross(mi, dip_field(mj, rh, dd))))
        for k in range(n_drives):                          # driven sources
            pk = drive_point(dr, body_q, w, k)
            mk = drive_moment(dr, body_q, w, k, phase[w, k])
            r = pi - pk
            d = wp.length(r)
            rh = unit_or_z(r, d)
            dd = wp.max(d, r_min)
            f = dip_force(mk, mi, rh, dd)
            add_wrench(body_f, bi, wrench_at(f, pi, ci) + torque_wrench(wp.cross(mi, dip_field(mk, rh, dd))))
            bk = dr.body[w, k]
            if bk >= 0:
                add_wrench(body_f, bk, wrench_at(-f, pi - rh * dd, com_world(body_q, body_com, bk))
                           + torque_wrench(wp.cross(mk, dip_field(mi, rh, dd))))
        for q in range(n_plates):                          # ferromagnetic plates (image dipoles)
            add_wrench(body_f, bi, plate_wrench(pl, q, pi, mi, ci, r_min))


@wp.kernel
def advance_phase(rpm: wp.array2d(dtype=float), n_drives: int, dt: float, phase: wp.array2d(dtype=float)):
    """One thread per world: each drive turns by rpm·2π/60·dt (kept in [0, 2π))."""
    w = wp.tid()
    for k in range(n_drives):
        ph = phase[w, k] + rpm[w, k] * (TWO_PI / 60.0) * dt
        phase[w, k] = ph - TWO_PI * wp.floor(ph / TWO_PI)
