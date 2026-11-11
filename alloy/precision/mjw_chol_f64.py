# SPDX-License-Identifier: Apache-2.0
"""The BUL-0016 guard's fallback (``mjw_chol``): MuJoCo C's ``mju_cholFactor(H, nv, mjMINVAL)`` and the Newton solve,
in float64 (MuJoCo C's precision), on one thread per failing world.

Why float64. A float32 Hessian that MuJoCo Warp's float32 factorization breaks is still positive definite in float64
(E9 100072:1890: smallest eigenvalue 5.0e-6, condition 3.4e7). Clamping pivot 15 to mjMINVAL in float32 left the
following pivots strongly negative: seven were clamped, and dividing by sqrt(mjMINVAL) seven times overflowed the
search direction to inf (WP13, main f97b84be). Factored in float64, the same H needs no clamp and its solve matches
the exact one to 4e-10. The clamp stays MuJoCo C's: a float64 pivot below mjMINVAL (or NaN) is raised to mjMINVAL.

Scratch per world: ``f64`` [W, nv + 1, nv] float64. Rows 0..nv-1 hold the upper factor U (U^T U = H); row nv holds the
solve's vector.
"""

from __future__ import annotations

import math

import warp as wp

MINVAL = 1.0e-15                                            # mujoco.mjMINVAL
MINVAL64 = wp.constant(wp.float64(MINVAL))
SQRT_MINVAL = wp.constant(wp.float32(math.sqrt(MINVAL)))   # float32 diagonal of a clamped pivot; the check's threshold
EPS32 = wp.constant(wp.float64(2.0 ** -24))                  # float32 unit roundoff (factor64_floor)


@wp.func
def _factor64_at(h: wp.array3d[float], w: int, n: int, f64: wp.array3d[wp.float64], floor: wp.float64):
    """Upper U of H's float32 upper triangle, in float64, column order; a pivot below ``floor`` (or NaN) is raised to
    it."""
    for j in range(n):
        s = wp.float64(h[w, j, j])
        for k in range(j):
            r = f64[w, k, j]
            s = s - r * r
        if not (s >= floor):  # MuJoCo C: tmp < mindiag -> mindiag (NaN too)
            s = floor
        d = wp.sqrt(s)
        f64[w, j, j] = d
        inv = wp.float64(1.0) / d
        for i in range(j + 1, n):
            si = wp.float64(h[w, j, i])
            for k in range(j):
                si = si - f64[w, k, i] * f64[w, k, j]
            f64[w, j, i] = si * inv


@wp.func
def factor64(h: wp.array3d[float], w: int, n: int, f64: wp.array3d[wp.float64]):
    """Upper U of H's float32 upper triangle, in float64, column order; a pivot below mjMINVAL is raised to it."""
    _factor64_at(h, w, n, f64, MINVAL64)


@wp.func
def factor64_floor(h: wp.array3d[float], w: int, n: int, f64: wp.array3d[wp.float64], c: wp.float64):
    """``factor64`` with the pivot floor at H's float32 rounding: c * n * eps32 * max_j H_jj (SCR-WP3-CHAINGRIP-GPU).

    A pivot under that floor is rounding noise of the float32 assembly, not curvature. On a CHAIN gripped at impratio
    300, the friction rows' entries (~100) carry float32 errors near 6e-6, above the links' twist curvature (the
    1e-6 armature floor): H comes out indefinite, and mjMINVAL turns the step into inf. At this floor the step stays a
    descent direction and the line search keeps the step size; the clamped directions are those the float32 H cannot
    resolve."""
    hmax = wp.float64(0.0)
    for j in range(n):
        hmax = wp.max(hmax, wp.abs(wp.float64(h[w, j, j])))
    _factor64_at(h, w, n, f64, wp.max(MINVAL64, c * wp.float64(n) * EPS32 * hmax))


@wp.func
def solve64(w: int, n: int, f64: wp.array3d[wp.float64], grad: wp.array2d[float], search: wp.array2d[float],
            search_dot: wp.array[float], newton_decrement: wp.array[float]):
    """x = (U^T U)^-1 grad in float64; search = -x, search_dot = sum x^2, newton_decrement = sum grad x (float32)."""
    for i in range(n):  # U^T y = g
        v = wp.float64(grad[w, i])
        for k in range(i):
            v = v - f64[w, k, i] * f64[w, n, k]
        f64[w, n, i] = v / f64[w, i, i]
    for ii in range(n):  # U x = y, in place
        i = n - 1 - ii
        v = f64[w, n, i]
        for k in range(i + 1, n):
            v = v - f64[w, i, k] * f64[w, n, k]
        f64[w, n, i] = v / f64[w, i, i]
    dot = wp.float64(0.0)
    dec = wp.float64(0.0)
    for i in range(n):
        x = f64[w, n, i]
        dot = dot + x * x
        dec = dec + wp.float64(grad[w, i]) * x
        search[w, i] = wp.float32(-x)
    search_dot[w] = wp.float32(dot)
    newton_decrement[w] = wp.float32(dec)


@wp.func
def store_factor(w: int, n: int, f64: wp.array3d[wp.float64], hfactor: wp.array3d[float]):
    """The float64 factor rounded into MuJoCo Warp's float32 ``hfactor`` [nv_pad, nv_pad] (upper), padding identity
    (MuJoCo Warp pads ``ctx.h`` with identity), so the blocked skip-unchanged path can reuse it. Every diagonal written
    is >= SQRT_MINVAL, so a reused factor passes the guard's check."""
    n_pad = hfactor.shape[1]
    for j in range(n_pad):
        for i in range(j, n_pad):
            v = float(0.0)
            if i < n:
                v = wp.float32(f64[w, j, i])
            elif i == j:
                v = 1.0
            hfactor[w, j, i] = v
