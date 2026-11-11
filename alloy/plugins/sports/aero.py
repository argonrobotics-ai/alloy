# SPDX-License-Identifier: Apache-2.0
"""``BallAero``: drag with a drag crisis, Magnus lift and spin decay on balls (``model.py`` gives the sources).

    aero = BallAero(bodies=("ball",))
    aero.set_input("wind", (2.0, 0.0, 0.0))    # m/s, per world

Drag ½ρAC_d(Re)|v|v (C_d from ``cd_sub`` below ``re_lo`` to ``cd_super`` above ``re_hi``), lift ½ρAC_L(S)|v|²
along ω × v with C_L = ``cl_max``·tanh(S/``cl_s0``), S = Rω⊥/|v|, and the spin torque -½ρAR²·C_M·|v|·ω. All act at
the centre of mass; v is relative to the world's wind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.errors import PluginError
from alloy.plugins.sports import kernels as K
from alloy.plugins.sports import model as M


class BallAero(ForcePlugin):
    """Ball aerodynamics: drag crisis, Magnus lift, spin decay, per-world wind."""

    name = "ball_aero"
    summary = "ball aerodynamics: drag (with the drag crisis), Magnus lift, spin decay, wind"

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("ball",)
        radius: float = M.R_BALL                  # m
        air_density: float = M.RHO_AIR            # kg/m³
        air_viscosity: float = M.MU_AIR           # Pa·s
        cd_sub: float = M.CD_SUB                  # below re_lo (ESTIMATE)
        cd_super: float = M.CD_SUPER              # above re_hi (Goff & Carré 2012: 0.17)
        re_lo: float = M.RE_LO
        re_hi: float = M.RE_HI
        cl_max: float = M.CL_MAX                  # C_L = cl_max·tanh(S / cl_s0)
        cl_s0: float = M.CL_S0
        cm: float = M.CM_SPIN                     # spin-decay coefficient (ESTIMATE)

        def __post_init__(self) -> None:
            if not self.bodies or self.radius <= 0.0 or self.air_density < 0.0 or self.air_viscosity <= 0.0:
                raise ValueError("at least one ball body, radius > 0, air_density >= 0, air_viscosity > 0")
            if not self.re_hi > self.re_lo > 0.0 or self.cl_s0 <= 0.0:
                raise ValueError("0 < re_lo < re_hi and cl_s0 > 0 are required")

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self.balls = ctx.array(np.stack([ctx.bodies(b).numpy() for b in c.bodies], axis=1).astype(np.int32), wp.int32)
        self.wind = self.add_input("wind", ctx.zeros(W, wp.vec3))
        a = K.Air()
        a.rho, a.mu, a.radius = c.air_density, c.air_viscosity, c.radius
        a.cd_sub, a.cd_sup, a.re_lo, a.re_hi = c.cd_sub, c.cd_super, c.re_lo, c.re_hi
        a.cl_max, a.cl_s0, a.cm = c.cl_max, c.cl_s0, c.cm
        self.air = a
        b = int(self.balls.numpy()[0, 0])
        self._mass = float(ctx.model.body_mass.numpy()[b])

    def check_dt(self, dt: float) -> None:
        """Explicit drag: its rate ρA·C_d·v/m at 60 m/s times dt stays < 0.2."""
        c = self.config
        rate = c.air_density * math.pi * c.radius ** 2 * max(c.cd_sub, c.cd_super) * 60.0 / self._mass * dt
        if rate > 0.2:
            raise PluginError(f"{self.label}: drag rate·dt {rate:.2f} > 0.2 at 60 m/s: reduce dt or check the mass")

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        m = self.ctx.model
        wp.launch(K.ball_aero, dim=self.ctx.world_count, device=self.ctx.device,
                  inputs=[state.body_q, state.body_qd, m.body_com, self.balls, self.wind, self.air],
                  outputs=[forces.body_f])

    def test_scene(self) -> Any:
        from alloy.plugins.sports.laws_aero import aero_scene  # noqa: PLC0415

        return aero_scene()

    def laws(self) -> list:
        from alloy.plugins.sports.laws_aero import aero_laws  # noqa: PLC0415

        return aero_laws()


__all__ = ["BallAero"]
