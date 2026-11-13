# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the OpenTac board reading: contacts to bump wrenches (or foam spread), the bump split, counts.

The NumPy reference is ``model.py`` (formulas and the OpenTac sources they port are there, MIT, Copyright (c) 2026
The OpenTac authors). Pads are world-major (``g = w * P + p``), as in ``tactile/kernels.py``. Sums run in int64 fixed
point (``FIX`` per N, and per N mm for the first moments), so they do not depend on contact order.

Bump slot ``s = g * B + b`` holds 5 channels: F_n, F_x, F_y (N, force on the pad in the pad frame) and the first
moments sum F_n dx, sum F_n dy (N mm) about the bump centre. Taxel slot ``k = g * T + t`` holds the normal load of
a bump-less taxel, or of every taxel on a plain skin (foam route).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.tactile.kernels import FIX, _fix, _nearest

MODE_BUMPS = 0
MODE_FOAM = 1


@wp.func
def _share(d: float, hw: float, sigma: float) -> float:
    """One axis of the foam spread: the part of a unit load at offset ``d`` that lands in [-hw, hw]."""
    if sigma <= 0.0:
        if wp.abs(d) <= hw:
            return 1.0
        return 0.0
    r = 1.0 / (sigma * 1.4142135623730951)
    return 0.5 * (wp.erf((hw - d) * r) - wp.erf((-hw - d) * r))


@wp.struct
class BoardArgs:
    """The board's constant arrays and settings for ``bin_board``."""

    taxel_xy: wp.array(dtype=wp.vec2)
    taxel_bump: wp.array(dtype=wp.int32)
    bump_xy: wp.array(dtype=wp.vec2)
    n: int                 # taxels
    nb: int                # bumps (1 on a plain skin, unused)
    reach: float
    capture: float
    mode: int
    foam_hw: float         # m, half width of a taxel's foam square
    foam_sigma: float      # m


@wp.func
def _foam(g: int, xy: wp.vec2, fn: float, a: BoardArgs, tacc: wp.array(dtype=wp.int64)):
    for t in range(a.n):
        d = a.taxel_xy[t] - xy
        w = _share(d[0], a.foam_hw, a.foam_sigma) * _share(d[1], a.foam_hw, a.foam_sigma)
        if w > 0.0:
            wp.atomic_add(tacc, g * a.n + t, _fix(fn * w))


@wp.func
def _board_side(g: int, b: int, p_body: wp.vec3, f_world: wp.vec3, body_q: wp.array(dtype=wp.transform), pads: int,
                pad_origin: wp.array(dtype=wp.vec3), pad_rot: wp.array(dtype=wp.quat), a: BoardArgs,
                bacc: wp.array(dtype=wp.int64), tacc: wp.array(dtype=wp.int64)):
    """One contact on pad ``g``: the same frame and capture as ``tactile.kernels._side``, then board binning."""
    rp = pad_rot[g % pads]
    x = wp.quat_rotate_inv(rp, p_body - pad_origin[g % pads])
    if wp.abs(x[2]) > a.capture:
        return
    f = wp.quat_rotate_inv(rp, wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), f_world))
    fn = -f[2]
    xy = wp.vec2(x[0], x[1])
    if a.mode == MODE_FOAM:
        _foam(g, xy, fn, a, tacc)
        return
    nd = _nearest(xy, a.taxel_xy, a.n)
    if nd[1] > a.reach:
        return
    j = int(nd[0])
    bb = a.taxel_bump[j]
    if bb < 0:
        wp.atomic_add(tacc, g * a.n + j, _fix(fn))
        return
    d = (xy - a.bump_xy[bb]) * 1000.0
    s = (g * a.nb + bb) * 5
    wp.atomic_add(bacc, s + 0, _fix(fn))
    wp.atomic_add(bacc, s + 1, _fix(f[0]))
    wp.atomic_add(bacc, s + 2, _fix(f[1]))
    wp.atomic_add(bacc, s + 3, _fix(fn * d[0]))
    wp.atomic_add(bacc, s + 4, _fix(fn * d[1]))


@wp.kernel
def bin_board(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32), shape1: wp.array(dtype=wp.int32),
              point0: wp.array(dtype=wp.vec3), point1: wp.array(dtype=wp.vec3),
              contact_f: wp.array(dtype=wp.spatial_vector), shape_body: wp.array(dtype=wp.int32),
              body_q: wp.array(dtype=wp.transform), pad_of_body: wp.array(dtype=wp.int32), pads: int,
              pad_origin: wp.array(dtype=wp.vec3), pad_rot: wp.array(dtype=wp.quat), a: BoardArgs,
              bacc: wp.array(dtype=wp.int64), tacc: wp.array(dtype=wp.int64)):
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
                _board_side(g0, b0, point0[i], f0, body_q, pads, pad_origin, pad_rot, a, bacc, tacc)
    if s1 >= 0:
        b1 = shape_body[s1]
        if b1 >= 0:
            g1 = pad_of_body[b1]
            if g1 >= 0:
                _board_side(g1, b1, point1[i], -f0, body_q, pads, pad_origin, pad_rot, a, bacc, tacc)


@wp.func
def _member_load(fz: float, inv_n: float, w: wp.vec2, mx: float, my: float) -> float:
    return wp.max(fz * inv_n + w[0] * my - w[1] * mx, 0.0)


@wp.kernel
def split_bumps(bacc: wp.array(dtype=wp.int64), start: wp.array(dtype=wp.int32), members: wp.array(dtype=wp.int32),
                weights: wp.array(dtype=wp.vec2), nb: int, n: int, height: float, tip: float,
                load: wp.array(dtype=wp.float32), bump_wrench: wp.array(dtype=wp.float32)):
    """One thread per bump slot: the base wrench, OpenTac's split over the member taxels, and the accumulators
    zeroed; ``tip`` (0 or 1) adds the normal load's first moment to the base moment. ``bump_wrench`` [slot, 5]
    keeps the bump's contact load as ground truth: F_n, F_x, F_y (N) and the normal load's first moments about the
    bump centre, sum F_n dx and sum F_n dy (N m)."""
    s = wp.tid()
    g = s // nb
    bb = s - g * nb
    v = wp.vector(dtype=float, length=5)
    for ch in range(5):
        v[ch] = float(bacc[s * 5 + ch]) / FIX
        bacc[s * 5 + ch] = wp.int64(0)
    v[3] = v[3] * 1.0e-3
    v[4] = v[4] * 1.0e-3
    for ch in range(5):
        bump_wrench[s * 5 + ch] = v[ch]
    my = v[1] * height * 0.5 + tip * v[3]
    mx = -v[2] * height * 0.5 - tip * v[4]
    fz = wp.max(v[0], 0.0)
    m0 = start[bb]
    m1 = start[bb + 1]
    inv_n = 1.0 / float(m1 - m0)
    total = float(0.0)
    for m in range(m0, m1):
        total += _member_load(fz, inv_n, weights[members[m]], mx, my)
    scale = float(0.0)
    if total > 0.0:
        scale = fz / total
    for m in range(m0, m1):
        t = members[m]
        load[g * n + t] = _member_load(fz, inv_n, weights[t], mx, my) * scale


@wp.struct
class CountArgs:
    """The readout profile at this pitch (``readout.py``) and the board's settings."""

    a: float               # counts / kPa^b at this taxel area
    b: float
    area_mm2: float
    adc_max: float
    gain: float
    noise: float           # counts std; 0 = off
    quantize: int
    seed: int


@wp.kernel
def finish_board(tacc: wp.array(dtype=wp.int64), taxel_bump: wp.array(dtype=wp.int32), n: int, slots_per_world: int,
                 tick: wp.array(dtype=wp.int32), r: CountArgs, load: wp.array(dtype=wp.float32),
                 counts: wp.array(dtype=wp.float32)):
    """One thread per taxel slot: bump-less (or foam) loads from fixed point, then counts by the readout profile."""
    k = wp.tid()
    t = k % n
    if taxel_bump[t] < 0:
        load[k] = float(tacc[k]) / FIX
        tacc[k] = wp.int64(0)
    kpa = wp.max(load[k], 0.0) / r.area_mm2 * 1.0e3
    c = float(0.0)
    if kpa > 0.0:
        c = r.gain * r.a * wp.pow(kpa, r.b)
    if r.noise > 0.0:
        w = k // slots_per_world
        rs = wp.rand_init(r.seed + 7919 * tick[w], k - w * slots_per_world)   # same draws in identical worlds
        c += r.noise * wp.randn(rs)
    c = wp.clamp(c, 0.0, r.adc_max)
    if r.quantize != 0:
        c = wp.round(c)
    counts[k] = c
