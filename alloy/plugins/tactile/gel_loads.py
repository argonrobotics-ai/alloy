# SPDX-License-Identifier: Apache-2.0
"""GelSensor's loads and marker field (see gel_kernels.py for the pipeline).

Loads sit on a coarse grid, one per block of depth cells, in proportion to the depth in the block.
- **Normal:** p_c = Fn·D_c/ΣD.
- **Shear:** t_c = Ft·D_c/ΣD + τ·ẑ × (c - c̄)·D_c/Σ D|c - c̄|², where c̄ is the depth-weighted centroid.

Their sums are the engine's normal force, its shear force and its twisting moment about c̄, all exact. The
displacement is u(q) = C·Σ_c G(q - c)·(t_x, t_y, p)_c. G is the Green's tensor:
- ``kernel="analytic"``: Boussinesq-Cerruti for an elastic half-space, Poisson ν, regularised at the coarse cell
  size;
- ``kernel="taxim"``: Taxim's FEM tensor, in sensor axes.

C is chosen per sensor so that the normal displacement at the most loaded block equals its mean depth. Row index
order = displacement component, column = load component; z is positive into the gel.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.tactile.gel_kernels import FIX, active

P_PARAMS = 10   # per-sensor scalars: Fn, Ftx, Fty, tau, cx, cy, sumD, sumDr2, C, peak


@wp.func
def green(dx: float, dy: float, a: float, nu: float, mode: int, table: wp.array(dtype=wp.mat33),
          pix: float) -> wp.mat33:
    if mode == 1:
        ix = 450 + int(wp.round(dx / pix))
        iy = 450 - int(wp.round(dy / pix))
        if ix < 0 or ix > 899 or iy < 0 or iy > 899:
            return wp.mat33(0.0)
        return table[iy * 900 + ix]
    r2 = dx * dx + dy * dy + a * a
    r = wp.sqrt(r2)
    r3 = r2 * r
    t = 2.0 * (1.0 - nu) / r
    b = (1.0 - 2.0 * nu) / r2
    return wp.mat33(t + 2.0 * nu * dx * dx / r3, 2.0 * nu * dx * dy / r3, -b * dx,
                    2.0 * nu * dx * dy / r3, t + 2.0 * nu * dy * dy / r3, -b * dy,
                    b * dx, b * dy, t)


@wp.kernel
def gel_coarse(depth: wp.array(dtype=wp.float32), rows: int, cols: int, block: int, sensors: int,
               stamp: wp.array(dtype=wp.int32), every: int, load_d: wp.array(dtype=wp.float32)):
    """One thread per (sensor, coarse cell): the depth summed over its block of cells."""
    k = wp.tid()
    cr = rows // block
    cc = cols // block
    g = k // (cr * cc)
    if not active(stamp, g // sensors, every):
        return
    c = k - g * cr * cc
    ci = c // cc
    cj = c - ci * cc
    total = float(0.0)
    for di in range(block):
        for dj in range(block):
            total += depth[g * rows * cols + (ci * block + di) * cols + cj * block + dj]
    load_d[k] = total


@wp.func
def coarse_xy(c: int, cc: int, cr: int, size: float) -> wp.vec2:
    ci = c // cc
    cj = c - ci * cc
    return wp.vec2((float(cj) + 0.5) * size - float(cc) * size * 0.5, float(cr) * size * 0.5 - (float(ci) + 0.5) * size)


@wp.func
def load_at(c: int, dsum: float, prm: wp.array(dtype=wp.float32), base: int, xy: wp.vec2) -> wp.vec3:
    """The (t_x, t_y, p) load of coarse cell c, whose depth sum is ``dsum``."""
    if prm[base + 6] <= 0.0:
        return wp.vec3(0.0)
    w = dsum / prm[base + 6]
    tx = prm[base + 1] * w
    ty = prm[base + 2] * w
    if prm[base + 7] > 0.0:
        rx = xy[0] - prm[base + 4]
        ry = xy[1] - prm[base + 5]
        tw = prm[base + 3] * dsum / prm[base + 7]
        tx = tx - tw * ry
        ty = ty + tw * rx
    return wp.vec3(tx, ty, prm[base + 0] * w)


@wp.func
def _moments(g: int, n: int, cc: int, cr: int, size: float, load_d: wp.array(dtype=wp.float32)) -> wp.vec4:
    """(ΣD, c̄x, c̄y, index of the largest block) over sensor g's coarse cells."""
    s = float(0.0)
    sx = float(0.0)
    sy = float(0.0)
    peak = int(0)
    best = float(-1.0)
    for c in range(n):
        d = load_d[g * n + c]
        xy = coarse_xy(c, cc, cr, size)
        s += d
        sx += d * xy[0]
        sy += d * xy[1]
        if d > best:
            best = d
            peak = c
    if s > 0.0:
        sx = sx / s
        sy = sy / s
    return wp.vec4(s, sx, sy, float(peak))


@wp.kernel
def gel_sensor(force_acc: wp.array(dtype=wp.int64), torque_acc: wp.array(dtype=wp.int64),
               load_d: wp.array(dtype=wp.float32), cr: int, cc: int, size: float, block_cells: int, a: float,
               nu: float, mode: int, table: wp.array(dtype=wp.mat33), pix: float, sensors: int,
               stamp: wp.array(dtype=wp.int32), every: int, prm: wp.array(dtype=wp.float32),
               force_out: wp.array(dtype=wp.float32), in_contact: wp.array(dtype=wp.int32)):
    """One thread per sensor: engine force and twist (zeroing the accumulators), load moments, and the scale C."""
    g = wp.tid()
    if not active(stamp, g // sensors, every):
        return
    n = cr * cc
    base = g * P_PARAMS
    f = wp.vec3(float(force_acc[g * 3]) / FIX, float(force_acc[g * 3 + 1]) / FIX, float(force_acc[g * 3 + 2]) / FIX)
    tau0 = float(torque_acc[g]) / FIX
    for c in range(3):
        force_acc[g * 3 + c] = wp.int64(0)
    torque_acc[g] = wp.int64(0)
    m = _moments(g, n, cc, cr, size, load_d)
    r2 = float(0.0)
    for c in range(n):
        xy = coarse_xy(c, cc, cr, size)
        r2 += load_d[g * n + c] * ((xy[0] - m[1]) * (xy[0] - m[1]) + (xy[1] - m[2]) * (xy[1] - m[2]))
    vals = wp.vector(f[0], f[1], f[2], tau0 - (m[1] * f[2] - m[2] * f[1]), m[1], m[2], m[0], r2, 0.0, m[3],
                     length=10, dtype=float)
    for k in range(P_PARAMS):
        prm[base + k] = vals[k]
    for c in range(3):
        force_out[g * 3 + c] = f[c]
    in_contact[g] = wp.where(m[0] > 0.0, 1, 0)
    if m[0] <= 0.0 or f[0] <= 0.0:
        return
    peak = int(m[3])
    q = coarse_xy(peak, cc, cr, size)
    u = wp.vec3(0.0)
    for c in range(n):
        d = load_d[g * n + c]
        if d > 0.0:
            xy = coarse_xy(c, cc, cr, size)
            u += green(q[0] - xy[0], q[1] - xy[1], a, nu, mode, table, pix) * load_at(c, d, prm, base, xy)
    if u[2] > 0.0:
        prm[base + 8] = load_d[g * n + peak] / float(block_cells) / u[2]


@wp.kernel
def gel_markers(marker_xy: wp.array(dtype=wp.vec2), markers: int, load_d: wp.array(dtype=wp.float32), cr: int,
                cc: int, size: float, a: float, nu: float, mode: int, table: wp.array(dtype=wp.mat33), pix: float,
                prm: wp.array(dtype=wp.float32), in_contact: wp.array(dtype=wp.int32), sensors: int,
                stamp: wp.array(dtype=wp.int32), every: int, out: wp.array(dtype=wp.vec3)):
    """One thread per (sensor, marker): u = C·Σ G(q - c)·load_c, written as (x, y, -z): z out of the gel."""
    k = wp.tid()
    g = k // markers
    if not active(stamp, g // sensors, every):
        return
    base = g * P_PARAMS
    out[k] = wp.vec3(0.0)
    if in_contact[g] == 0 or prm[base + 8] == 0.0:
        return
    q = marker_xy[k - g * markers]
    n = cr * cc
    u = wp.vec3(0.0)
    for c in range(n):
        d = load_d[g * n + c]
        if d > 0.0:
            xy = coarse_xy(c, cc, cr, size)
            u += green(q[0] - xy[0], q[1] - xy[1], a, nu, mode, table, pix) * load_at(c, d, prm, base, xy)
    s = prm[base + 8]
    out[k] = wp.vec3(s * u[0], s * u[1], -s * u[2])
