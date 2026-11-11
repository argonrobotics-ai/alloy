# SPDX-License-Identifier: Apache-2.0
"""Heat v2 exposure kernel (``particle_area`` doc): each slot's neighbourhood moment M (conduction's free-surface
correction) and its exposed area s² · min(6, F, C). Split from ``particle_kernels`` (file size); the structs live there.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.thermal.particle_kernels import AREA_CAP, F64, Cloud, MatTable, Slots, cubic_dwdr


@wp.func
def closed_view(e: wp.vec3, dirs: wp.array[wp.vec3], cosang: float, lo: wp.uint32, hi: wp.uint32) -> wp.vec2ui:
    """Mark the view directions whose cone holds unit vector e (bits 0-31 in lo, 32-63 in hi)."""
    for k in range(dirs.shape[0]):
        if wp.dot(e, dirs[k]) >= cosang:
            if k < 32:
                lo = lo | (wp.uint32(1) << wp.uint32(k))
            else:
                hi = hi | (wp.uint32(1) << wp.uint32(k - 32))
    return wp.vec2ui(lo, hi)


@wp.func
def bits(x: wp.uint32) -> int:
    n = int(0)
    for k in range(32):
        if (x >> wp.uint32(k)) & wp.uint32(1) != wp.uint32(0):
            n += 1
    return n


@wp.kernel(enable_backward=False)
def surface_moment(sl: Slots, cl: Cloud, mt: MatTable, dirs: wp.array[wp.vec3], cosang: float, area: wp.vec3d,
                   with_exposure: int, moment: wp.array[wp.mat33d], exposure: wp.array[wp.float64]):
    """M of slot s (module doc): the neighbourhood's second-moment tensor, the identity in the bulk; and its exposed
    area s² · min(6, F, C) (``particle_area`` doc; ``area`` = (FLOOR, κ_F, κ_C))."""
    s = wp.tid()
    i = sl.gidx[s]
    xi = cl.q[i]
    eta2 = 0.01 * cl.hs * cl.hs
    acc = wp.mat33d()
    for n in range(cl.count[i]):
        j = cl.nbr[i, n]
        sj = sl.slot_of[j]
        if sj >= 0:
            rij = xi - cl.q[j]
            d2 = wp.dot(rij, rij)
            d = wp.sqrt(d2)
            f = -F64(cl.mass[j]) / mt.density[sl.mat[sj]] * F64(d * cubic_dwdr(d, cl.hs) / (d2 + eta2))
            r = wp.vec3d(F64(rij[0]), F64(rij[1]), F64(rij[2]))
            acc += f * wp.outer(r, r)
    moment[s] = acc
    if with_exposure == 0:  # no convection: nothing reads the exposure (its cone pass was 0.5x of the update)
        exposure[s] = F64(0.0)
        return
    open_faces = F64(AREA_CAP)
    fill = wp.trace(acc) / F64(3.0)
    if fill > F64(0.0):
        open_faces = wp.max(F64(0.0), -wp.log2(fill) - area[0]) / area[1]
    if open_faces > F64(0.0):  # min(…, 0) is 0: the cone pass only where F leaves room (WP13: 58-63 % skip it)
        view = wp.vec2ui(wp.uint32(0), wp.uint32(0))
        for n in range(cl.count[i]):
            j = cl.nbr[i, n]
            if sl.slot_of[j] >= 0:
                rij = xi - cl.q[j]
                d = wp.length(rij)
                if d > 0.0 and d < 2.0 * cl.hs:
                    view = closed_view(-rij / d, dirs, cosang, view[0], view[1])
        closed = bits(view[0]) + bits(view[1])
        if closed > 0:
            open_faces = wp.min(wp.min(open_faces, F64(AREA_CAP)),
                                -wp.log2(F64(closed) / F64(dirs.shape[0])) / area[2])
    exposure[s] = sl.face[s] * wp.min(open_faces, F64(AREA_CAP))


__all__ = ["bits", "closed_view", "surface_moment"]
