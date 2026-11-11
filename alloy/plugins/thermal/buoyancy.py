# SPDX-License-Identifier: Apache-2.0
"""Heat v2 with thermal buoyancy: hot liquid rises (gaps-now #8, 2026-11-08).

``BuoyantParticleThermal`` is ``ParticleThermal`` plus, before each solver step, a Boussinesq lift on its liquid
slots from their temperature (``alloy.solvers.sph.buoyancy``: F = m φ (ρ(T) - ρ̄) / ρ̄ g, with ρ̄ the mean density of
that heat material's liquid in the world, so the forces sum to zero and only temperature differences drive flow).
``particle_scene.for_scene`` builds it when the config has ``buoyancy=True``, which is what
``scene.set_heat(buoyancy=True)`` passes. A heat material without a ``density_table`` (only water has one: NIST
WebBook, ``phase_materials.WATER_DENSITY``) gets no lift, and the build says so once. Everything else (conduction,
heaters, phase change, read-outs) is ParticleThermal's.

Example::

    scene.add_liquid("water", size=(0.05, 0.05, 0.04), pos=(0, 0, 0.024), temperature=293.15)
    scene.set_heater("pan_floor", temperature=353.15)
    scene.set_heat(buoyancy=True)      # hot water gets its lift; the flow stalls (docs/known_gaps/heat-no-buoyancy.md)
"""

from __future__ import annotations

import logging
from typing import Any

from alloy._vendor import newton
from alloy.plugins.base import Forces
from alloy.plugins.thermal.particle import ParticleThermal
from alloy.plugins.thermal.phase_materials import material
from alloy.solvers.sph.buoyancy import Buoyancy

log = logging.getLogger("alloy.plugins")


class BuoyantParticleThermal(ParticleThermal):
    """``ParticleThermal`` whose liquid slots also feel thermal buoyancy (module docstring)."""

    name = ParticleThermal.name     # one heat plug-in, one label: sim.plugins["particle_thermal"] with buoyancy too
    summary = "per-particle temperature (heat v2) with thermal buoyancy of liquids"

    def setup(self, ctx: Any) -> None:
        super().setup(ctx)
        tables = {n: material(n).density_table for n in set(self.config.materials)}
        self._buoy = Buoyancy.build(self.config.materials, tables, self._group, self._W, ctx.device)
        if self.config.time_scale != 1.0:
            log.warning("%s: buoyancy with time_scale %g: heat runs %g times faster than the buoyant flow, so "
                        "convection is that much weaker relative to conduction than in reality; use time_scale=1 "
                        "where convection matters", self.label, self.config.time_scale, self.config.time_scale)
        if self._buoy.skipped:
            log.warning("%s: no density table for heat material(s) %s, so they get no thermal buoyancy (only water "
                        "has one)", self.label, list(self._buoy.skipped))

    @property
    def applies_forces(self) -> bool:
        return True

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        """ParticleThermal's flow write-back when it has one, then the buoyancy forces into ``forces.particle_f``."""
        if super().applies_forces:
            super().apply(state, forces, dt)
        self._buoy.apply(self._flat["t"], self._flat["phi"], self._slot_mass, self._gidx, self.ctx.model.gravity,
                         forces.particle_f)
