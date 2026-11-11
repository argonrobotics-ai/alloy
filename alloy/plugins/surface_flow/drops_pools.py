# SPDX-License-Identifier: Apache-2.0
"""Drops and drips that start or end in a pool's water (the drop stage; WP15's review of the lumped films, WP8's
colander and sink; WP9): a world point is in a pool's water when it lies in that pool's water region (DC5's
``water_of``: a cavity's inside, else a box pool's footprint above its floor) under its free surface (``plane_of``).
Water that goes there joins the pool's tank through DC5's ``pool_add`` (species through ``pool_add_sp``)."""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.plugins.surface_flow.supply_cavity import Cavity
from alloy.plugins.surface_flow.supply_float import plane_of, water_of
from alloy.plugins.surface_flow.supply_kernels import PoolSpec, pool_frame


@wp.struct
class DropPools:
    """The supply's pools as the drop stage sees them (``on`` 0: no pools)."""
    on: int
    pools: wp.array[PoolSpec]
    pbody: wp.array2d[wp.int32]
    cavs: wp.array[Cavity]
    cprof: wp.array[wp.vec2]
    pressure: wp.array2d[wp.float64]
    surf: wp.array2d[wp.vec4]
    add: wp.array2d[wp.float64]       # [W, P] the supply's pool_add (m³; it books lump_pool and zeroes it)
    add_sp: wp.array3d[wp.float64]    # [W, P, K] its pool_add_sp (kg)


@wp.func
def into_pool(dp: DropPools, w: int, x: wp.vec3, body_q: wp.array[wp.transform]) -> int:
    """The pool whose water holds world point x (its region, as DC5's floats find it, under its free surface, above a
    box pool's floor), or -1."""
    if dp.on == 0:
        return -1
    p = water_of(w, x, -1, body_q, dp.pools, dp.pbody, dp.cavs, dp.cprof)
    if p < 0:
        return -1
    if dp.cavs[p].on == 0:
        q = wp.transform_point(wp.transform_inverse(pool_frame(body_q, dp.pbody[w, p])), x) - dp.pools[p].floor
        if q[2] < -1.0e-3:
            return -1
    pl = plane_of(w, p, body_q, dp.pools, dp.pbody, dp.pressure, dp.cavs, dp.surf)
    if wp.dot(wp.vec3(pl[0], pl[1], pl[2]), x) > pl[3]:
        return -1
    return p


def drop_pools(ctx: Any, supply: Any) -> DropPools:
    """``DropPools`` over the supply's pools, or an empty one (no supply, or no pools)."""
    dp = DropPools()
    if supply is not None and getattr(supply, "pools", None):
        dp.on, dp.pools, dp.pbody, dp.cavs, dp.cprof = 1, supply.pool_spec, supply.pbody, supply.cav, supply.cav_prof
        dp.pressure, dp.surf, dp.add, dp.add_sp = supply.fluid.pressure, supply.cav_surf, supply.pool_add, \
            supply.pool_add_sp
        return dp
    dp.on, dp.pools = 0, wp.zeros(1, dtype=PoolSpec, device=ctx.device)
    dp.cavs, dp.cprof = wp.zeros(1, dtype=Cavity, device=ctx.device), ctx.zeros(1, wp.vec2)
    dp.pbody, dp.pressure, dp.surf = ctx.zeros((1, 1), wp.int32), ctx.zeros((1, 1), wp.float64), \
        ctx.zeros((1, 1), wp.vec4)
    dp.add, dp.add_sp = ctx.zeros((1, 1), wp.float64), ctx.zeros((1, 1, 1), wp.float64)
    return dp


__all__ = ["DropPools", "drop_pools", "into_pool"]
