# SPDX-License-Identifier: Apache-2.0
"""Herschel-Bulkley viscosity kernels (device only, fixed size: graph-capture safe): from each particle's own strain
rate, or from the strain-rate tensor averaged over its neighbours in the same world (a warp HashGrid with groups)."""

from __future__ import annotations

import warp as wp


@wp.func
def shear_rate(grad: wp.mat33) -> float:
    """gdot = sqrt(2 D':D') of the velocity gradient's symmetric part D (D' deviatoric); simple shear u_x = g z -> g."""
    d = 0.5 * (grad + wp.transpose(grad))
    tr = (d[0, 0] + d[1, 1] + d[2, 2]) / 3.0
    dev = d - tr * wp.identity(n=3, dtype=float)
    return wp.sqrt(2.0 * wp.ddot(dev, dev))


@wp.kernel
def herschel_bulkley_viscosity(
    particles: wp.array(dtype=wp.int32),
    qd_grad: wp.array(dtype=wp.mat33),
    consistency: wp.array(dtype=float),
    flow_index: wp.array(dtype=float),
    gdot_min: float,
    eta_min: float,
    viscosity: wp.array(dtype=float),
):
    i = particles[wp.tid()]
    k = consistency[i]
    n = flow_index[i]
    g = wp.max(shear_rate(qd_grad[i]), gdot_min)          # thinning (n < 1): caps eta at its gdot_min value
    eta_still = k * wp.pow(gdot_min, n - 1.0)               # thickening (n > 1): this is the curve's minimum
    viscosity[i] = wp.max(k * wp.pow(g, n - 1.0), wp.min(eta_min, eta_still))


@wp.func
def deviatoric_rate(grad: wp.mat33) -> wp.mat33:
    """D' = sym(grad) - tr/3 I."""
    d = 0.5 * (grad + wp.transpose(grad))
    return d - (d[0, 0] + d[1, 1] + d[2, 2]) / 3.0 * wp.identity(n=3, dtype=float)


@wp.func
def hb_eta(g_raw: float, k: float, n: float, gdot_min: float, eta_min: float) -> float:
    """K max(gdot, gdot_min)^(n-1), floored as ``herschel_bulkley_viscosity`` does."""
    g = wp.max(g_raw, gdot_min)
    return wp.max(k * wp.pow(g, n - 1.0), wp.min(eta_min, k * wp.pow(gdot_min, n - 1.0)))


@wp.kernel
def gather_strain_rates(
    particles: wp.array(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3),
    qd_grad: wp.array(dtype=wp.mat33),
    world: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    dev: wp.array(dtype=wp.mat33),
    groups: wp.array(dtype=wp.int32),
):
    k = wp.tid()
    i = particles[k]
    points[k] = q[i]
    dev[k] = deviatoric_rate(qd_grad[i])
    groups[k] = world[i]


@wp.kernel
def herschel_bulkley_viscosity_smoothed(
    grid: wp.uint64,
    particles: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    dev: wp.array(dtype=wp.mat33),
    groups: wp.array(dtype=wp.int32),
    radius: float,
    consistency: wp.array(dtype=float),
    flow_index: wp.array(dtype=float),
    gdot_min: float,
    eta_min: float,
    viscosity: wp.array(dtype=float),
):
    """eta from the hat-weighted mean D' of the particle's neighbours within ``radius`` in its own world (itself
    included, weight 1): zero-mean noise cancels, a linear shear field keeps its value."""
    k = wp.tid()
    p = points[k]
    acc = wp.mat33(0.0)
    wsum = float(0.0)
    query = wp.hash_grid_query(grid, p, radius, groups[k])
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        d = wp.length(points[j] - p)
        if d < radius:
            w = 1.0 - d / radius
            acc = acc + w * dev[j]
            wsum = wsum + w
    s = acc / wp.max(wsum, 1.0e-12)
    i = particles[k]
    viscosity[i] = hb_eta(wp.sqrt(2.0 * wp.ddot(s, s)), consistency[i], flow_index[i], gdot_min, eta_min)
