# SPDX-License-Identifier: Apache-2.0
"""Adsorbed loads under a pool's water (SCR-WP8-10; WP9 builds on WP15's ``adsorption.py``, WP15 reviews).

A cell under a flooding pool's water (DC5's ``flooded >= 0``) has given its film to the pool (h = 0), so WP15's
film exchange (``adsorption_kernel``, film above ``H_DRY`` only) never runs there and a submerged wall kept its whole
load. Here, once per exchange (after the film step, before the supply's ``after_film``), each flooded cell's load
exchanges with its pool's water, at the tank's concentration ``C = m/V`` read once per exchange (``m`` the tank's
species with the inflow booked so far, ``V`` its volume, as DC5's ``pool_exchange`` computes them, but before the
supply's drains and pooled film add to the tank this exchange: C lags those additions, and the take cap stays
conservative):
``dΓ/dt = k_d·(K·C·(Γ_max - Γ) - Γ)`` stepped exactly over the exchange toward Langmuir's rest
``Γ_max·K·C/(1 + K·C)`` (linear in Γ at fixed C: no overshoot at any ``k_d·dt``).

- What a wall gives up joins its pool's tank (DC5's ``pool_add_sp``); what it takes from a soapy pool leaves the
  tank through DC5's species take ``pool_take_sp`` (species alone, confirmed in ``pool_taken_sp``). Each pool's
  takes are scaled together to what its tank holds plus what its walls give up this exchange, so the tank never
  goes negative (DC5 also caps a take at the tank after the adds).
- Per world, one thread, over each chart's flood box (DC5's ``flood_cur``) in row order: deterministic, no atomics.
- Book ``adsorbed_to_pools`` [W, K] kg (given minus taken): the film's species ledger counts it with the load.
- A cell that leaves the water goes back to WP15's film exchange, which steps it only while it holds film.
"""

from __future__ import annotations

from typing import Any

import logging

import numpy as np
import warp as wp

from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.kernels import Cells
from alloy.plugins.surface_flow.supply_kernels import PoolSpec
from alloy.plugins.surface_flow.supply_lump import tank_have
from alloy.plugins.surface_flow.supply_species import NetSpecies
from alloy.plugins.surface_flow.supply_species import refresh as ns_refresh

log = logging.getLogger("alloy.plugins.surface_flow")


@wp.func
def pool_conc(ns: NetSpecies, pool: PoolSpec, w: int, k: int, pressure: wp.array2d[wp.float64],
              inflow: wp.array2d[wp.float64], dt: wp.float64) -> wp.vec2d:
    """(C [kg/m³], m [kg]) of film species k in a pool's tank after this exchange's booked transfers (0, 0 when the
    network carries no such species or the tank is empty)."""
    kn = ns.map[k]
    zero = wp.float64(0.0)
    if kn < 0:
        return wp.vec2d(zero, zero)
    m = wp.max(ns.species[w, pool.tank, kn] + ns.inflow[w, pool.tank, kn] * dt, zero)
    v = tank_have(pool, w, pressure, inflow, dt)
    if v <= zero:
        return wp.vec2d(zero, m)
    return wp.vec2d(m / v, m)


@wp.func
def langmuir_step(g: wp.float64, c: wp.float64, K: wp.float64, k_d: wp.float64, g_max: wp.float64,
                  dt: wp.float64) -> wp.float64:
    """Γ after dt at fixed C: exact for dΓ/dt = k_d·(K·C·(Γ_max - Γ) - Γ)."""
    kc = K * c
    rest = g_max * kc / (wp.float64(1.0) + kc)
    return rest + (g - rest) * wp.exp(-k_d * (wp.float64(1.0) + kc) * dt)


@wp.func
def box_cell(ch: Chart, b: wp.vec4i, i: int, jj: int) -> int:
    """Cell (i, b[2] + jj) of a chart's flood box (columns wrap on a revolution chart), or -1 off the chart."""
    j = b[2] + jj
    if ch.wrap != 0:
        j = (j % ch.ny + ch.ny) % ch.ny
    elif j < 0 or j >= ch.ny:
        return -1
    return ch.cell_start + i * ch.ny + j


@wp.kernel
def pool_adsorb_kernel(dt: wp.float64, charts: wp.array[Chart], cells: Cells, flood: wp.array2d[wp.vec4i],
                       flooded: wp.array2d[wp.int32], area: wp.array[wp.float64], K: wp.array[wp.float64],
                       k_d: wp.array[wp.float64], cap: wp.array2d[wp.float64], pools: wp.array[PoolSpec],
                       pressure: wp.array2d[wp.float64], inflow: wp.array2d[wp.float64], ns: NetSpecies,
                       gave: wp.array3d[wp.float64], want: wp.array3d[wp.float64],
                       load: wp.array3d[wp.float64], add_sp: wp.array3d[wp.float64],
                       take_sp: wp.array3d[wp.float64], to_pools: wp.array2d[wp.float64]):
    """Per world: two passes over the flooded cells in row order, the pools' gives and wants, then each load's step
    (wants scaled to what the tank holds plus what is given) into ``add_sp`` / ``take_sp`` (module doc)."""
    w = wp.tid()
    zero = wp.float64(0.0)
    S = K.shape[0]
    for p in range(gave.shape[1]):
        for k in range(S):
            gave[w, p, k] = zero
            want[w, p, k] = zero
    for sweep in range(2):
        for kc in range(charts.shape[0]):
            ch = charts[kc]
            b = flood[w, kc]
            span = wp.min(b[3] - b[2] + 1, ch.ny)
            for i in range(wp.max(b[0], 0), wp.min(b[1], ch.nx - 1) + 1):
                for jj in range(span):
                    c = box_cell(ch, b, i, jj)
                    if c < 0:
                        continue
                    p = flooded[w, c]
                    if p < 0:
                        continue
                    for k in range(S):
                        g_max = cap[cells.chart[c], k]
                        if K[k] <= zero or g_max <= zero:
                            continue
                        cm = pool_conc(ns, pools[p], w, k, pressure, inflow, dt)
                        g = load[w, c, k]
                        d = (langmuir_step(g, cm[0], K[k], k_d[k], g_max, dt) - g) * area[c]    # > 0: taken
                        if sweep == 0:
                            if d > zero:
                                want[w, p, k] = want[w, p, k] + d
                            else:
                                gave[w, p, k] = gave[w, p, k] - d
                            continue
                        if d > zero:                    # the pool's takes scaled to what its tank holds
                            d = d * wp.min(wp.float64(1.0), (cm[1] + gave[w, p, k]) / want[w, p, k])
                            take_sp[w, p, k] = take_sp[w, p, k] + d
                        else:
                            add_sp[w, p, k] = add_sp[w, p, k] - d
                        load[w, c, k] = g + d / area[c]
                        to_pools[w, k] = to_pools[w, k] - d


class PoolAdsorption:
    """The flooded cells' exchange with their pools (module doc); built when the film has adsorption and a flooding
    pool."""

    def __init__(self, flow: Any, ctx: Any) -> None:
        sup, ad = flow.supply, flow.adsorption
        W, P, S = ctx.world_count, max(len(sup.pools), 1), int(ad.K.shape[0])
        self.flow, self.ctx = flow, ctx
        self.gave, self.want = ctx.zeros((W, P, S), wp.float64), ctx.zeros((W, P, S), wp.float64)
        self.to_pools = flow.add_state("adsorbed_to_pools", ctx.zeros((W, S), wp.float64))

    def step(self, dt: float) -> None:
        f, sup = self.flow, self.flow.supply
        ad = f.adsorption
        wp.launch(pool_adsorb_kernel, dim=self.ctx.world_count,
                  inputs=[wp.float64(dt), self.flow.tables.charts, f.cells, sup.flood_cur, f.flooded,
                          f.tables.area, ad.K, ad.k_d, ad.cap, sup.pool_spec, sup.fluid.pressure, sup.fluid.inflow,
                          ns_refresh(sup.ns, sup.fluid), self.gave, self.want],
                  outputs=[ad.load, sup.pool_add_sp, sup.pool_take_sp, self.to_pools],
                  device=self.ctx.device)

    def read(self, out: dict[str, np.ndarray]) -> None:
        """``adsorbed_to_pools`` [W, K] kg, counted in the film's species ledger with the load."""
        W = self.ctx.world_count
        out["adsorbed_to_pools"] = self.to_pools.numpy()[:W].copy()
        out["species_ledger"] = out["species_ledger"] + out["adsorbed_to_pools"]


def setup_pool_adsorption(flow: Any, ctx: Any) -> None:
    """``flow.pool_adsorption``: None unless the film adsorbs and a pool floods. An adsorbing species the network
    does not carry is shed from the walls under water into nothing (DC5's ``away``) and never taken: warned once."""
    sup = getattr(flow, "supply", None)
    flow.pool_adsorption = None
    if flow.adsorption is None or sup is None or not getattr(sup, "flooding", False):
        return
    flow.pool_adsorption = PoolAdsorption(flow, ctx)
    K, names = flow.adsorption.K.numpy(), tuple(flow.config.species)
    lost = [names[k] for k in range(len(names)) if K[k] > 0.0 and (k >= 4 or int(sup.ns.map[k]) < 0)]
    if lost:
        log.warning("%s: adsorbing species %s are not carried by the supply's network: walls under a pool's water "
                    "shed them into nothing (species away) and never take them back; add them to "
                    "network['species']", flow.label, lost)


__all__ = ["PoolAdsorption", "langmuir_step", "pool_adsorb_kernel", "pool_conc", "setup_pool_adsorption"]
