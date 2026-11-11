# SPDX-License-Identifier: Apache-2.0
"""Bench laws re-run at one card's values for CARD-1 coverage (`alloy.scene.coverage_checks`).

- A15 uniaxial Neo-Hookean (`alloy.bench.analytic.neohookean`, WP6): the same squat pad, frictionless floor and dead
  top load, with the card's density and its Lamé constants from (E, ν) as the SOFT_VBD emitter maps them
  (`alloy.scene.derive.lame`); the applied stress and the stress read back use the card's (μ, λ).
- A13 hydrostatics (`alloy.solvers.sph.laws`, WP9): the same resting column (H/(2r) = 50), with the card's rest
  density and viscosity (as the SPH emitter writes them) at the router's particle radius; the reference
  ρ₀·g·w·H²/2 uses the card's ρ₀.
Full-length runs (not quick): this is the nightly GPU job.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp
from alloy._vendor import newton

from alloy.bench.analytic import neohookean as NH
from alloy.bench.types import Condition
from alloy.cards import types as ct
from alloy.scene.derive import lame
from alloy.solvers.sph import laws as SL
from alloy.solvers.sph.build import add_box_tank, add_fluid_block
from alloy.solvers.sph.twins import HydrostaticTwin

NEOHOOKEAN_LAW = dataclasses.replace(NH.LAW, solvers=("vbd",))
HYDROSTATIC_LAW = SL.A13_HYDROSTATIC
WATER_VISCOSITY = 1.0e-3


class CardNeoHookeanTwin(NH.NeoHookeanTwin):
    """A15 at one material's (μ, λ, ρ)."""

    def __init__(self, mu: float, lam: float, density: float) -> None:
        super().__init__()
        self.mu, self.lam, self.density = mu, lam, density

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"stretch{s:g}", {"stretch": s, "p_applied": self.stress(s)}) for s in NH.STRETCHES)

    def stress(self, a: float) -> float:
        return NH.uniaxial_stress(a, self.mu, self.lam)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        nx, ny, nz = NH.CELLS
        b.add_soft_grid(pos=wp.vec3(-NH.WIDTH / 2, -NH.WIDTH / 2, NH.RADIUS), rot=wp.quat_identity(),
                        vel=wp.vec3(0.0), dim_x=nx, dim_y=ny, dim_z=nz, cell_x=NH.WIDTH / nx, cell_y=NH.WIDTH / ny,
                        cell_z=NH.HEIGHT / nz, density=self.density, k_mu=self.mu, k_lambda=self.lam, k_damp=0.0,
                        tri_ke=0.0, tri_ka=0.0, tri_kd=0.0, add_surface_mesh_edges=False, particle_radius=NH.RADIUS)

    def scalars(self, log) -> dict[str, np.ndarray]:
        t = log.times()
        late = t >= t[-1] - 0.1 * (t[-1] - NH.T_EASE)
        stretch = np.mean(log.series("z_top")[:, late] - log.series("z_bot")[:, late], axis=1) / self._h0
        p_app = self.param("p_applied")
        p_sim = np.array([self.stress(s) if np.isfinite(s) and s > 0 else math.nan for s in stretch])
        self.monitors.update({"stretch_sim": stretch.tolist(), "stretch_target": self.param("stretch").tolist(),
                              "mu": self.mu, "lambda": self.lam, "density": self.density})
        err = np.abs(p_sim - p_app) / np.abs(p_app)
        return {"stress_rel_err": np.where(np.isfinite(err), err, math.inf)}


def card_neohookean(mat: ct.MaterialCard) -> CardNeoHookeanTwin:
    mu, lam = lame(float(mat.elastic.youngs.value), float(mat.elastic.poisson.value))
    return CardNeoHookeanTwin(mu, lam, float(mat.density.value))


class CardHydrostaticTwin(HydrostaticTwin):
    """A13 resting column at one fluid card's (ρ₀, μ) and SPH radius r."""

    def __init__(self, density: float, viscosity: float, radius: float) -> None:
        super().__init__(HYDROSTATIC_LAW.dt)
        self.density, self.viscosity, self.radius = density, viscosity, radius

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        (c,) = super().conditions(quick)
        params = {**c.params, "r": self.radius}
        return (Condition(f"{c.id.rsplit('_r', 1)[0]}_r{self.radius * 1e3:g}mm", params),)

    def build(self, b: newton.ModelBuilder, cond: Condition, cfg: Any) -> None:
        p, s = cond.params, cfg.spacing
        nx, ny, nz = int(p["nx"]), int(p["ny"]), int(p["nz"])
        tank = add_box_tank(b, (0.0, 0.0, 0.0), (nx * s, ny * s, (nz + int(p["headroom"])) * s), wall=cfg.h)
        add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (nx, ny, nz), rest_density=self.density, viscosity=self.viscosity)
        self._walls.append(tank["x_lo"])

    def scalars(self, log_: Any) -> dict[str, np.ndarray]:
        out = super().scalars(log_)                       # reference at 1000 kg/m³; rescale to the card's ρ₀
        force = np.asarray(self.monitors["wall_force_N"], np.float64)
        ref = np.asarray(self.monitors["wall_force_ref_N"], np.float64) * self.density / 1000.0
        self.monitors.update(wall_force_ref_N=ref.tolist(), density=self.density, viscosity=self.viscosity)
        out["wall_force"] = np.abs(force - ref) / ref
        return out


def card_hydrostatic(mat: ct.MaterialCard, radius: float) -> CardHydrostaticTwin:
    mu = float(mat.fluid.viscosity.value) if mat.fluid is not None else WATER_VISCOSITY
    return CardHydrostaticTwin(float(mat.density.value), mu, radius)


__all__ = ["NEOHOOKEAN_LAW", "HYDROSTATIC_LAW", "CardNeoHookeanTwin", "CardHydrostaticTwin", "card_neohookean",
           "card_hydrostatic"]
