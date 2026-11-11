# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the tactile sensor: contact binning, fixed-point readout, the noise layer.

Pads are numbered world-major: pad ``g = w·P + p`` (world w, pad p of P). Taxel slot ``k = g·T + j`` holds 3
channels: the normal force (compression positive) and the two shear forces along the pad's x and y.

Binning runs one thread per contact and adds int64 fixed point (``FIX`` counts per newton). A power-of-two scale
makes each float32 force exact in fixed point, and integer adds commute, so the sums do not depend on contact order.
MuJoCo Warp's contact order varies run to run on the GPU, and float atomics are banned (SPEC §3.7).
"""

from __future__ import annotations

import warp as wp

FIX = 16777216.0  # 2^24 counts per N: 6e-8 N resolution, ±5e11 N range


@wp.func
def _fix(x: float) -> wp.int64:
    return wp.int64(wp.round(x * FIX))


@wp.func
def _nearest(x: wp.vec2, taxel_xy: wp.array(dtype=wp.vec2), n: int) -> wp.vec2:
    """(index, distance) of the taxel nearest to ``x``, as a vec2."""
    best = int(0)
    bd = float(1.0e30)
    for j in range(n):
        d = wp.length(x - taxel_xy[j])
        if d < bd:
            bd = d
            best = j
    return wp.vec2(float(best), bd)


@wp.func
def _deposit(k0: int, ch: wp.vec3, x: wp.vec2, taxel_xy: wp.array(dtype=wp.vec2), n: int, near: int,
             spread: float, acc: wp.array(dtype=wp.int64)):
    """Add ``ch`` to the nearest taxel, or share it by a normalised Gaussian of width ``spread``."""
    if spread <= 0.0:
        for c in range(3):
            wp.atomic_add(acc, (k0 + near) * 3 + c, _fix(ch[c]))
        return
    inv = 0.5 / (spread * spread)
    total = float(0.0)
    for j in range(n):
        d = x - taxel_xy[j]
        total += wp.exp(-wp.dot(d, d) * inv)
    for j in range(n):
        d = x - taxel_xy[j]
        w = wp.exp(-wp.dot(d, d) * inv) / total
        for c in range(3):
            wp.atomic_add(acc, (k0 + j) * 3 + c, _fix(ch[c] * w))


@wp.func
def _side(g: int, b: int, p_body: wp.vec3, f_world: wp.vec3, body_q: wp.array(dtype=wp.transform),
          pads: int, pad_origin: wp.array(dtype=wp.vec3), pad_rot: wp.array(dtype=wp.quat),
          taxel_xy: wp.array(dtype=wp.vec2), n: int, reach: float, capture: float, spread: float,
          acc: wp.array(dtype=wp.int64), pad_acc: wp.array(dtype=wp.int64)):
    """One contact on pad ``g`` (body ``b``): point in the body frame, force on the pad in world frame."""
    p = g % pads
    rp = pad_rot[p]
    x = wp.quat_rotate_inv(rp, p_body - pad_origin[p])
    if wp.abs(x[2]) > capture:
        return
    f = wp.quat_rotate_inv(rp, wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), f_world))
    ch = wp.vec3(-f[2], f[0], f[1])
    for c in range(3):
        wp.atomic_add(pad_acc, g * 3 + c, _fix(ch[c]))
    xy = wp.vec2(x[0], x[1])
    nd = _nearest(xy, taxel_xy, n)
    if nd[1] > reach:
        return
    _deposit(g * n, ch, xy, taxel_xy, n, int(nd[0]), spread, acc)


@wp.kernel
def bin_contacts(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
                 shape1: wp.array(dtype=wp.int32), point0: wp.array(dtype=wp.vec3),
                 point1: wp.array(dtype=wp.vec3), contact_f: wp.array(dtype=wp.spatial_vector),
                 shape_body: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform),
                 pad_of_body: wp.array(dtype=wp.int32), pads: int, pad_origin: wp.array(dtype=wp.vec3),
                 pad_rot: wp.array(dtype=wp.quat), taxel_xy: wp.array(dtype=wp.vec2), n: int, reach: float,
                 capture: float, spread: float, acc: wp.array(dtype=wp.int64), pad_acc: wp.array(dtype=wp.int64)):
    """One thread per exported contact; ``contact_f`` is the force on shape0 (world frame)."""
    i = wp.tid()
    if i >= count[0]:
        return
    f0 = wp.spatial_top(contact_f[i])
    s0 = shape0[i]
    s1 = shape1[i]
    if s0 >= 0:
        b0 = shape_body[s0]
        if b0 >= 0:
            g0 = pad_of_body[b0]
            if g0 >= 0:
                _side(g0, b0, point0[i], f0, body_q, pads, pad_origin, pad_rot, taxel_xy, n, reach, capture,
                      spread, acc, pad_acc)
    if s1 >= 0:
        b1 = shape_body[s1]
        if b1 >= 0:
            g1 = pad_of_body[b1]
            if g1 >= 0:
                _side(g1, b1, point1[i], -f0, body_q, pads, pad_origin, pad_rot, taxel_xy, n, reach, capture,
                      spread, acc, pad_acc)


@wp.kernel
def read_out(acc: wp.array(dtype=wp.int64), pad_acc: wp.array(dtype=wp.int64), n: int,
             raw: wp.array(dtype=wp.float32), pad_force: wp.array(dtype=wp.float32)):
    """One thread per taxel slot: fixed point to newtons, and zero the accumulators for the next step."""
    k = wp.tid()
    for c in range(3):
        raw[k * 3 + c] = float(acc[k * 3 + c]) / FIX
        acc[k * 3 + c] = wp.int64(0)
    if k % n == 0:
        g = k // n
        for c in range(3):
            pad_force[g * 3 + c] = float(pad_acc[g * 3 + c]) / FIX
            pad_acc[g * 3 + c] = wp.int64(0)
