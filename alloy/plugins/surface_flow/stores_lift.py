# SPDX-License-Identifier: Apache-2.0
"""A scraping store's overflow and its lifted gob (GA's carrying knife, concrete_finishing/coupling.py:922-938).

- **Overflow.** With ``flow.overflow_on`` set (WP13's end bands, roll_cap.py, set it), a full blade still cuts all
  that stands above its underside. The cut beyond capacity goes to ``tool_overflow`` [W, n] m³ with its species
  ``tool_overflow_sp`` [W, n, K] kg, at the cut cells' concentrations (it never joins the store, whose
  concentration is its own paint's in v1). The load stays at capacity. Both are pending totals: the stores kernel
  adds, the end bands subtract what they lay. With it off (the default) a full blade cuts only what fits.
- **Lift.** A blade that touches nothing (no film or surface under its underside) and carries less than
  ``LIFT_N`` keeps what its paste's yield stress holds on it,

      V_stick = 0.5·A·tau_y / (rho·g·DROP_STICK) · max(STICK_FLOOR, 0.5·(1 + up)),

  ``A`` its footprint and ``up`` its underside normal along gravity (1 face down, -1 face up). The rest of its load,
  and any pending overflow, falls at once as a gob (``tool_dropped``), through DC5's stage-drips channel when there
  is a supply. GA drops it while the knife's plate force is under 0.05 N; here the blade must also be off the film,
  since the roll in front of a blade riding the film rests on the film.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

from alloy.plugins.surface_flow.cell_kernels import Species
from alloy.plugins.surface_flow.stores_species import StoreSpecies

LIFT_N = wp.constant(0.05)          # N: GA's lift threshold on the plate force (coupling.py:937)
DROP_STICK = 6.0                    # GA's yield-stress mound factor (params.py:137)
STICK_FLOOR = wp.constant(0.02)     # an overturned blade's adhesion share (coupling.py:930)


def stick_volume(tool: Any, material: Any, gravity: Any) -> float:
    """``0.5·A·tau_y / (rho·g·DROP_STICK)`` [m³] for a scraping store (a level blade's stick limit), else 0. ``A`` is
    ``Store(stick_area=)``, GA's whole sole (4·hx·hy, coupling.py:933), else the tool's footprint. No gravity: nothing
    falls."""
    if tool.store is None or not tool.store.scrape:
        return 0.0
    g = math.sqrt(sum(float(v) ** 2 for v in gravity))
    if g <= 0.0:
        return 1.0e30
    tau = float(getattr(material, "yield_stress", 0.0) or 0.0)
    rho = float(getattr(material, "density", 1000.0) or 1000.0)
    area = float(tool.store.stick_area) or float(tool.footprint)
    return 0.5 * area * tau / (rho * g * DROP_STICK)


def gravity_dir(gravity: Any) -> wp.vec3:
    g = math.sqrt(sum(float(v) ** 2 for v in gravity))
    if g <= 0.0:
        return wp.vec3(0.0, 0.0, -1.0)
    return wp.vec3(*(float(v) / g for v in gravity))


@wp.func
def stick_share(normal_world: wp.vec3, down: wp.vec3) -> wp.float64:
    """``max(STICK_FLOOR, 0.5·(1 + up))``, ``up`` the underside's outward normal along gravity."""
    up = wp.dot(wp.normalize(normal_world), down)
    return wp.float64(wp.max(STICK_FLOOR, 0.5 * (1.0 + up)))


@wp.func
def overflow_species(sp: Species, w: int, c: int, cut: wp.float64, over: wp.float64, h: wp.float64,
                     area: wp.float64, osp: wp.array3d[wp.float64], k: int, ssp: StoreSpecies, b: int):
    """Before ``nip_species`` books the cut ``cut`` [m³] (< 0) out of cell c into bin b: the overflow's share
    ``over / -cut`` of the species it takes, into ``osp[w, k]`` [kg], out of the bin (``given``)."""
    if over <= wp.float64(0.0) or cut >= wp.float64(0.0) or h <= wp.float64(0.0):
        return
    share = over / (-cut)
    for s in range(sp.count):
        gone = sp.mass[w, c, s] * wp.min(-cut / (h * area), wp.float64(1.0)) * area
        osp[w, k, s] = osp[w, k, s] + gone * share
        ssp.load[w, b, s] = ssp.load[w, b, s] - gone * share
        ssp.given[w, k, s] = ssp.given[w, k, s] + gone * share


__all__ = ["DROP_STICK", "LIFT_N", "STICK_FLOOR", "gravity_dir", "overflow_species", "stick_share",
           "stick_volume"]
