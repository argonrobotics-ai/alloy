# SPDX-License-Identifier: Apache-2.0
"""Species adsorbed on the walls (WP8's gap G6: surfactant on a pan, a can, glassware): Langmuir kinetics.

A species the film carries (mass per area ``m``, concentration ``C = m/h``) adsorbs onto the substrate under it as a
load ``Γ`` [kg/m²] per cell: ``dΓ/dt = k_d·(K·C·(Γ_max - Γ) - Γ)``, at rest ``Γ = Γ_max·K·C/(1 + K·C)`` (``K``
[m³/kg] the Langmuir constant, ``k_d`` [1/s] the desorption rate, ``Γ_max`` the substrate's capacity, per chart).
Each film step exchanges ``m`` and ``Γ`` exactly for the step's free-site share ``1 - Γ/Γ_max`` (taken at the
step's start): ``m + Γ`` is kept bit for bit, neither goes negative, and the stepped rest is Langmuir's exactly.
Only a cell holding film (``h > H_DRY``) exchanges: a dry wall keeps its load until water comes back (a rinse
desorbs it at ``k_d``). ``read()["adsorbed"]`` [W, C, n_species] kg/m²; the species ledger counts the load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.kernels import Cells
from alloy.plugins.surface_flow.process_kernels import due
from alloy.plugins.surface_flow.tiles import Tiles


@dataclass(frozen=True)
class Adsorption:
    """``species`` adsorbs on the walls: Langmuir ``K`` [m³/kg], desorption ``k_d`` [1/s], ``capacity`` Γ_max
    [kg/m²]: one value, or one per chart (0: that chart's surface takes none)."""

    species: str
    K: float
    k_d: float
    capacity: float | tuple[float, ...]


def adsorption_specs(spec: Any) -> tuple[Adsorption, ...]:
    """``Config.adsorption``: None, one Adsorption (or dict), or a sequence of them."""
    if spec is None:
        return ()
    items = spec if isinstance(spec, (list, tuple)) else (spec,)
    out = []
    for s in items:
        if isinstance(s, dict):
            s = Adsorption(**{k: tuple(v) if isinstance(v, list) else v for k, v in s.items()})
        if not isinstance(s, Adsorption):
            raise PluginError(f"adsorption: expected an Adsorption or a dict, got {s!r}")
        out.append(s)
    return tuple(out)


def _capacity(s: Adsorption, n_charts: int, label: str) -> np.ndarray:
    cap = np.atleast_1d(np.asarray(s.capacity, np.float64))
    if cap.size == 1:
        cap = np.full(n_charts, float(cap[0]))
    if cap.size != n_charts or not np.isfinite(cap).all() or (cap < 0.0).any():
        raise PluginError(f"{label}: adsorption of {s.species!r}: capacity must be one value or one per chart "
                          f"({n_charts}), each >= 0 kg/m², got {s.capacity}")
    return cap


@dataclass
class AdsorptionTables:
    K: wp.array            # [S] m³/kg (0: the species does not adsorb)
    k_d: wp.array          # [S] 1/s
    cap: wp.array          # [n_charts, S] kg/m²
    load: wp.array         # [W, C, S] kg/m², the adsorbed load (state)


def setup_adsorption(flow: Any, ctx: Any, W: int, C: int) -> None:
    """``flow.adsorption`` (None unless configured): the Langmuir constants per species and the load per cell."""
    specs = adsorption_specs(flow.config.adsorption)
    flow.adsorption = None
    if not specs:
        return
    names, n_charts = tuple(flow.config.species), len(flow.config.charts)
    S = len(names)
    K, kd, cap = np.zeros(S), np.zeros(S), np.zeros((n_charts, S))
    for s in specs:
        if s.species not in names:
            raise PluginError(f"{flow.label}: adsorption of {s.species!r}: the film carries no such species "
                              f"{names} (Config.species)")
        k = names.index(s.species)
        if K[k] > 0.0:
            raise PluginError(f"{flow.label}: adsorption of {s.species!r} is given twice")
        if not (math.isfinite(s.K) and s.K > 0.0 and math.isfinite(s.k_d) and s.k_d > 0.0):
            raise PluginError(f"{flow.label}: adsorption of {s.species!r}: K [m³/kg] and k_d [1/s] must be finite "
                              f"and > 0, got K={s.K}, k_d={s.k_d}")
        K[k], kd[k], cap[:, k] = s.K, s.k_d, _capacity(s, n_charts, flow.label)
    load = flow.add_state("adsorbed", ctx.zeros((W, C, S), wp.float64))
    flow.adsorption = AdsorptionTables(K=ctx.array(K, wp.float64), k_d=ctx.array(kd, wp.float64),
                                       cap=ctx.array(cap, wp.float64), load=load)


@wp.kernel
def adsorption_kernel(tl: Tiles, tile_on: wp.array2d[wp.int32], tick: wp.array[wp.int32], stride: int,
                      cells: Cells, h: wp.array2d[wp.float64], h_dry: wp.float64, film_dt: wp.float64,
                      K: wp.array[wp.float64], k_d: wp.array[wp.float64], cap: wp.array2d[wp.float64],
                      mass: wp.array3d[wp.float64], load: wp.array3d[wp.float64]):
    """One film step's exchange between each species in the film and its adsorbed load (module doc)."""
    w, t = wp.tid()
    if not due(tick, stride, tile_on, 0, w, t):
        return
    first, pitch = tl.first[t], tl.pitch[t]
    for r in range(tl.rows[t]):
        for j in range(tl.cols[t]):
            c = first + r * pitch + j
            depth = h[w, c]
            if depth <= h_dry:
                continue
            for k in range(K.shape[0]):
                g_max = cap[cells.chart[c], k]
                if K[k] <= wp.float64(0.0) or g_max <= wp.float64(0.0):
                    continue
                m, g = mass[w, c, k], load[w, c, k]
                total = m + g
                free = wp.max(wp.float64(1.0) - g / g_max, wp.float64(0.0))
                a = k_d[k] * K[k] * g_max * free / depth            # [1/s] on m: k_a·C·(Γ_max - Γ)
                rate = a + k_d[k]
                rest = a * total / rate
                g_new = wp.min(rest + (g - rest) * wp.exp(-rate * film_dt), g_max)
                load[w, c, k] = g_new
                mass[w, c, k] = total - g_new


def step_adsorption(flow: Any, dt: float, h_dry: float) -> None:
    """After the film step (``update``): the exchange on the tiles that stepped (film above ``h_dry``)."""
    ad = flow.adsorption
    if ad is None:
        return
    wp.launch(adsorption_kernel, dim=(flow.ctx.world_count, flow.tl.count),
              inputs=[flow.tl.tiles, flow.tile_on, flow.tick, flow.stride, flow.cells, flow.h, h_dry,
                      float(dt) * flow.stride, ad.K, ad.k_d, ad.cap, flow.sp.mass],
              outputs=[ad.load], device=flow.ctx.device)


def set_adsorbed(flow: Any, load: Any, species: str, chart: int | None, k: int) -> None:
    """Host: the adsorbed load [kg/m²] of species index ``k`` (shapes as ``set_depth``)."""
    ad = flow.adsorption
    if ad is None or float(ad.K.numpy()[k]) <= 0.0:
        raise PluginError(f"{flow.label}: species {species!r} does not adsorb (Config.adsorption)")
    full = ad.load.numpy()
    layer = wp.array(np.ascontiguousarray(full[:, :, k]), dtype=wp.float64, device=ad.load.device)
    flow._assign(layer, load, chart, f"adsorbed {species!r}")
    full[:, :, k] = layer.numpy()
    ad.load.assign(full)


def adsorbed_read(flow: Any, out: dict[str, np.ndarray]) -> None:
    """``read()``: the load per cell and species, and the species ledger's share of it."""
    ad = flow.adsorption
    if ad is None:
        return
    out["adsorbed"] = ad.load.numpy()
    out["adsorbed_mass"] = (out["adsorbed"] * flow.tables.area_np[None, :, None]).sum(axis=1)     # [W, S] kg
    out["species_ledger"] = out["species_ledger"] + out["adsorbed_mass"]


__all__ = ["Adsorption", "AdsorptionTables", "adsorbed_read", "adsorption_kernel", "adsorption_specs",
           "set_adsorbed", "setup_adsorption", "step_adsorption"]
