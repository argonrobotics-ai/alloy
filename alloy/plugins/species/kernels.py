# SPDX-License-Identifier: Apache-2.0
"""Kernels of the species plug-in (plugin.py): inflow and retirement bookkeeping, SPH diffusion between neighbours.

Amounts are int64 fixed point, ``AMT_SCALE`` per unit (2^50: 8.9e-16 units resolution, at most 8,192 units per
particle and species). Integer sums are exact and independent of their order, so totals are conserved exactly and a
particle's sum over its neighbours does not depend on the hash grid's visit order.

Diffusion (Cleary and Monaghan 1999, J. Comput. Phys. 148, eq. 8; Brookshaw's Laplacian): the amount n = C V of
species s in particle i changes by
    dn_i = dt sum_k V_i V_k (D_s + D_s) (C_i - C_k) (r_ik . grad W_ik) / (|r_ik|^2 + eta^2)
with V = m / rho0 (rest volume) and W the cubic spline of support h. Each pair term is computed once in canonical
order (lower particle id first, ``r = x_a - x_b``), rounded to fixed point and added to one particle and subtracted
from the other, so a pair exchanges exactly opposite amounts.
"""

from __future__ import annotations

import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.solvers.sph.kernels import KERNEL_CUBIC, kernel_grad

AMT_SCALE = wp.constant(1125899906842624.0)  # 2^50 fixed-point steps per unit
AMT_SCALE_HOST = float(2**50)
_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.func
def to_fx(x: wp.float64) -> wp.int64:
    return wp.int64(wp.round(x * wp.float64(AMT_SCALE)))


@wp.func
def from_fx(q: wp.int64) -> wp.float64:
    return wp.float64(q) / wp.float64(AMT_SCALE)


@wp.kernel(enable_backward=False)
def bookkeeping_kernel(idx: wp.array2d[wp.int32], flags: wp.array[wp.int32], volume: wp.array[wp.float32],
                       inflow: wp.array[wp.float64], prev: wp.array2d[wp.int32], amount: wp.array3d[wp.int64],
                       ledger_in: wp.array2d[wp.int64], ledger_out: wp.array2d[wp.int64]):
    """Particles that became active take the inflow composition (counted in); particles that became inactive (sinks,
    the escape box) give their amounts to the out ledger. Integer atomics on the ledgers: exact in any order."""
    w, j = wp.tid()
    on = int(0)
    if (flags[idx[w, j]] & _ACTIVE) != 0:
        on = 1
    if on == prev[w, j]:
        return
    for s in range(amount.shape[2]):
        if on == 1:
            q = to_fx(inflow[s] * wp.float64(volume[idx[w, j]]))
            amount[w, j, s] = q
            wp.atomic_add(ledger_in, w, s, q)
        else:
            wp.atomic_add(ledger_out, w, s, amount[w, j, s])
            amount[w, j, s] = wp.int64(0)
    prev[w, j] = on


@wp.kernel(enable_backward=False)
def groups_kernel(idx: wp.array2d[wp.int32], prev: wp.array2d[wp.int32], group: wp.array[wp.int32]):
    """Hash grid group of each tracked particle: its world when active, else -1 (untracked particles stay -1)."""
    w, j = wp.tid()
    g = int(-1)
    if prev[w, j] != 0:
        g = w
    group[idx[w, j]] = g


@wp.kernel(enable_backward=False)
def diffusion_kernel(grid: wp.uint64, q: wp.array[wp.vec3], idx: wp.array2d[wp.int32], slot: wp.array[wp.int32],
                     group: wp.array[wp.int32], volume: wp.array[wp.float32], diff: wp.array[wp.float64],
                     h: float, eta2: float, dt: float, amount: wp.array3d[wp.int64], delta: wp.array3d[wp.int64]):
    """delta[w, j, s]: the exact-antisymmetric diffusion exchange of particle j of world w with its neighbours."""
    w, j = wp.tid()
    i = idx[w, j]
    S = amount.shape[2]
    for s in range(S):
        delta[w, j, s] = wp.int64(0)
    if group[i] != w:
        return
    xi = q[i]
    query = wp.hash_grid_query(grid, xi, h, w)
    k = int(0)
    while wp.hash_grid_query_next(query, k):
        if k == i or group[k] != w:
            continue
        a = i
        b = k
        sign = wp.int64(1)
        if k < i:
            a = k
            b = i
            sign = wp.int64(-1)
        r = q[a] - q[b]
        d2 = wp.dot(r, r)
        if d2 >= h * h:
            continue
        va = wp.float64(volume[a])
        vb = wp.float64(volume[b])
        rg = wp.float64(wp.dot(r, kernel_grad(r, h, KERNEL_CUBIC))) / wp.float64(d2 + eta2)
        ja = slot[a]
        jb = slot[b]
        for s in range(S):
            if diff[s] > wp.float64(0.0):
                ca = from_fx(amount[w, ja, s]) / va
                cb = from_fx(amount[w, jb, s]) / vb
                f = to_fx(wp.float64(dt) * va * vb * wp.float64(2.0) * diff[s] * (ca - cb) * rg)
                delta[w, j, s] = delta[w, j, s] + sign * f


@wp.kernel(enable_backward=False)
def apply_kernel(delta: wp.array3d[wp.int64], amount: wp.array3d[wp.int64]):
    w, j, s = wp.tid()
    amount[w, j, s] = amount[w, j, s] + delta[w, j, s]
