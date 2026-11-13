# SPDX-License-Identifier: Apache-2.0
"""Quasi-static mass scaling of a j2 solid on implicit MPM (GAP #6 slice 2b step 2, opt-in; physics ruling 2026-11-10).

A stiff solid (a metal) needs about 0.2 (c dt / h)^2 Gauss-Seidel sweeps a step (``mpm_solid.sweeps_needed``): brass
about 1.6 million, so its stress solve stops far short and a tool's force reads a few % of the truth. Multiplying its
density by s divides that by s. The weight stays true (an upward force (s - 1) m0 g on each particle, every exchange:
``alloy.api._quasi_static``), so only inertia grows: a tool pressing at speed v meets an inertial stress of about
rho s c_s v = v sqrt(E rho s), c_s = sqrt(E / (rho s)). Its share of the yield stress,

    share(s) = v sqrt(E rho s) / yield,

is what the build bounds: ``"auto"`` takes the largest s with share <= ``AUTO_SHARE`` (1 %), a given s above
``WARN_SHARE`` warns and above ``MAX_SHARE`` (5 %) is refused. Measured ([internal note] notes/wp7-dent/slice2, small rig: a
5 mm ball 1 mm into brass, Newton's Delassus regularization at the route's 1e-12): x4,000 (share 1.2 % at 2 mm/s) 0.99
of the converged force in 338 sweeps, x16,000 (2.3 %) 1.00 in 106-151, x64,000 (4.7 %) 1.02; the force before the ball
meets the surface grows with s and v, 3.6 % at x16,000 and 2 mm/s, 1.0 % at 0.5 mm/s.

Quasi-static only: the solid is s times heavier for anything that moves it (a push, a drop, a hit), so the API refuses
it for a solid that does not rest on a support, and the run-time check (``QUASI_STATIC``, DEGRADED; physics ruling
2026-11-11) flags a world while the solid as a whole moves faster than ``MOTION_MARGIN`` (0.5) times the declared
tool speed (its mass-weighted mean velocity: a correct press reads at most 0.14x; a block carried at the declared
speed reads 1x, and its push force read 4.4x the unscaled one at x2,913), or while a body touching it moves faster
than ``CONTACT_MARGIN`` (12) times that speed. A tool up to 12x the declared speed is not flagged; what it costs the
force was measured (brass x16,000, 25 mm ball, 3 and 1.5 mm voxels, 0.2-2 mm deep, each period against the declared
press at the same depth): 3x and 10x -8.7 to +9.2 %, medians +1 to +3 % (single periods of the declared press scatter
up to 9 % on 1.5 mm voxels); 30x (flagged) -9.6 to +7.0 %; 100x 38 % low at 0.2 mm. The formula above gives 7 and
23 % at 3x and 10x: conservative ([internal note] notes/wp7-dent/slice2/probe_qs/api17/README.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from alloy.cards.types import MaterialCard

__all__ = ["AUTO_SHARE", "CONTACT_MARGIN", "MAX_SHARE", "MOTION_MARGIN", "NUMBER", "SHARE_NUMBER", "SPEED_NUMBER",
           "WARN_SHARE", "Scale", "auto_scale", "choose", "inertial_share", "share_factor"]

AUTO_SHARE = 0.01                # "auto": the largest s whose inertial share stays at or under 1 %
WARN_SHARE = 0.01                # a given s above this share warns ...
MAX_SHARE = 0.05                 # ... and above this is refused (SceneError, alloy.api._quasi_static)
NUMBER, SHARE_NUMBER, SPEED_NUMBER = "mass_scale", "inertial_share", "tool_speed"   # RouteDecision.numbers keys
MOTION_MARGIN = 0.5              # run time: the solid's mean velocity over this times the declared tool speed
CONTACT_MARGIN = 12.0            # run time: a body touching the solid faster than this times the declared speed


def share_factor(youngs: float, density: float, yield_stress: float, scale: float) -> float:
    """sqrt(E rho s) / yield (s/m): the inertial share per unit speed. Example: brass x16,000 -> 11.7 s/m."""
    return math.sqrt(youngs * density * scale) / yield_stress


def inertial_share(youngs: float, density: float, yield_stress: float, scale: float, speed: float) -> float:
    """v sqrt(E rho s) / yield. Example: brass (97 GPa, 8,500 kg/m3, 310 MPa) at x16,000 and 2 mm/s -> 0.023."""
    return speed * share_factor(youngs, density, yield_stress, scale)


def auto_scale(youngs: float, density: float, yield_stress: float, speed: float) -> float:
    """The largest whole s >= 1 with ``inertial_share`` <= ``AUTO_SHARE``. Example: brass at 2 mm/s -> 2,913."""
    s = (AUTO_SHARE * yield_stress / speed) ** 2 / (youngs * density)
    return float(max(1, math.floor(s)))


@dataclass(frozen=True)
class Scale:
    """The scale a j2 solid gets and what it costs: ``s``, the predicted inertial ``share`` at ``speed`` (m/s)."""

    s: float
    share: float
    speed: float
    auto: bool

    def numbers(self) -> dict[str, float]:
        return {NUMBER: self.s, SHARE_NUMBER: self.share, SPEED_NUMBER: self.speed}

    def reason(self) -> str:
        how = "auto: the largest within 1 %" if self.auto else "as given"
        return (f"quasi-static mass scaling x{self.s:,.0f} ({how}): density times {self.s:,.0f} with the weight kept "
                f"true; at the tool speed {self.speed * 1e3:g} mm/s its inertial share v sqrt(E rho s) / yield is "
                f"{self.share * 100:.2g} %. Quasi-static only: the solid is {self.s:,.0f} times heavier for anything "
                "that moves it")

    def warning(self, name: str) -> str | None:
        """The build warning for a given s whose share is over ``WARN_SHARE`` (over ``MAX_SHARE`` the API refuses)."""
        if self.auto or self.share <= WARN_SHARE or self.share > MAX_SHARE:
            return None
        return (f"{name}: quasi-static mass scaling x{self.s:,.0f} at {self.speed * 1e3:g} mm/s has an inertial share "
                f"of {self.share * 100:.2g} % (over {WARN_SHARE * 100:g} %): a tool's force reads that much high from "
                "the solid's scaled inertia. Fix: scale='auto', a smaller scale, or a slower tool")


def choose(material: MaterialCard, spec: dict[str, Any]) -> Scale:
    """The scale for ``spec`` (``{"scale": "auto" | s, "tool_speed": v}``, from ``Scene.set_quasi_static_mass_scale``)
    on a j2 card. Example: brass at 2 mm/s, "auto" -> Scale(s=2913, share=0.0100, ...)."""
    e, rho = float(material.elastic.youngs.value), float(material.density.value)
    y, v = float(material.plastic.yield_stress.value), float(spec["tool_speed"])
    auto = spec.get("scale", "auto") == "auto"
    s = auto_scale(e, rho, y, v) if auto else float(spec["scale"])
    return Scale(s, inertial_share(e, rho, y, s, v), v, auto)
