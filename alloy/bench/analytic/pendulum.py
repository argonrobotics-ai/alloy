# SPDX-License-Identifier: Apache-2.0
"""A2 pendulum period and A3 conservative energy drift (bench §4.5, SPEC §5.3).

A2: T = 4 sqrt(I_p / (m g d)) K(k), k = sin(theta0 / 2) (complete elliptic integral of the first kind; equals
2 pi sqrt(I_p / (m g d)) as theta0 -> 0). Pass: |T_sim - T| / T <= 1e-3 at dt 1 ms (§5.3: "A2's 1 ms stands").
The reference is the law's own geometry: a solid sphere of radius RADIUS on an arm d, I_p / m = 2/5 RADIUS^2 + d^2
(mass cancels), g = 9.81. It reads nothing from the model under test (audit AB-laws-2: taken from the finalized model,
I_p followed a builder that got a sphere's inertia or the bob's place wrong, and the law still passed).
A3 (conservative): max_t |E(t) - E0| / E_scale <= 5e-3 (0.5 % over the run; newton@ac7f7830:newton/tests/
test_physics_verification.py:323-329 uses |E0|), with the D20 scale E_scale = max(E0 - PE_min, m g L_char, 1e-3 J).
The angle is read from the bob position (atan2), so maximal- and reduced-coordinate solvers are measured alike.
A2 uses a 5 cm arm so two zero crossings fit the 0.5 s quick run even at 90 deg; A3 uses Newton's 1 m pendulum
(V:156-170, theta0 = 1 rad at V:236) because the symplectic-Euler energy oscillation scales with omega*h.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
from scipy.special import ellipk

from alloy.bench.analytic.common import G, BaseTwin
from alloy.bench.measures import EnergyMeter
from alloy.bench.solvers import RIGID_SOLVERS, SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.core.defaults import E_SCALE_FLOOR_J

ARM = 0.05  # [m] pivot to bob centre
RADIUS = 0.01  # [m] bob radius
DENSITY = 8000.0  # [kg/m^3] steel bob: m g d stays well above the 1e-3 J floor of E_scale
AMPLITUDES_DEG = (5.0, 45.0, 90.0)
ARM_ENERGY = 1.0  # [m] Newton's verification pendulum
AMPLITUDES_ENERGY_DEG = (5.0, math.degrees(1.0), 90.0)
_SETTINGS = {"xpbd": SolverSettings("xpbd", {"iterations": 20})}  # as Newton's pendulum test (V:1657)

LAW_A2 = LawSpec(
    id="A2.pendulum", suite="analytic",
    metrics={"period_rel_err": Tolerance(abs=1e-3, source="bench §4.5 A2: period <= 0.1 % at dt 1 ms")},
    stat={"period_rel_err": "mean"}, dt=1e-3, substeps=1, duration_s=2.0, worlds_per_condition=1,
    solvers=RIGID_SOLVERS, production_dt=False,
)
LAW_A3 = LawSpec(
    id="A3.pendulum_energy", suite="analytic",
    metrics={"energy_drift": Tolerance(abs=5e-3, source="bench §4.5 A3: drift <= 0.5 %; scale per SPEC D20")},
    stat={"energy_drift": "mean"}, dt=1e-3, substeps=1, duration_s=2.0, worlds_per_condition=1,
    solvers=RIGID_SOLVERS, production_dt=False,
)


def elliptic_period(i_pivot: float, mass: float, arm: float, theta0: float, g: float = G) -> float:
    """Exact period of a rigid pendulum released from rest at theta0 [rad]."""
    k = math.sin(theta0 / 2.0)
    return 4.0 * math.sqrt(i_pivot / (mass * g * arm)) * float(ellipk(k * k))


def crossing_period(t: np.ndarray, theta: np.ndarray) -> float:
    """Period from linearly interpolated zero crossings of theta(t): 2 x mean spacing; NaN with < 2 crossings."""
    s = np.signbit(theta)
    idx = np.flatnonzero(s[1:] != s[:-1])
    if idx.size < 2:
        return math.nan
    t0, t1, y0, y1 = t[idx], t[idx + 1], theta[idx], theta[idx + 1]
    tc = t0 - y0 * (t1 - t0) / (y1 - y0)
    return float(2.0 * (tc[-1] - tc[0]) / (tc.size - 1))


class PendulumTwin(BaseTwin):
    law_id = "A2"
    units = {"period_rel_err": "1", "energy_drift": "1"}
    settings = _SETTINGS
    needs_contacts = False
    arm = ARM
    amplitudes_deg = AMPLITUDES_DEG

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"amp{a:.4g}deg", {"theta0": math.radians(a)}) for a in self.amplitudes_deg)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        link = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, -self.arm), wp.quat_identity()), label="bob")
        b.add_shape_sphere(link, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(density=DENSITY))
        j = b.add_joint_revolute(parent=-1, child=link, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform_identity(),
                                 child_xform=wp.transform(wp.vec3(0.0, 0.0, self.arm), wp.quat_identity()),
                                 armature=0.0)
        b.add_articulation([j])
        b.joint_q[-1] = cond.params["theta0"]

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        if step == 0:
            self.energy = EnergyMeter(self.model, state, log)
        self.record_first_body(step, state, log)
        self.energy.update(state, step)

    def _pivot_inertia(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.model
        first = self.ranges.start.numpy()
        mass = m.body_mass.numpy()[first].astype(np.float64)
        i_yy = m.body_inertia.numpy()[first][:, 1, 1].astype(np.float64)
        return mass, i_yy, first

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()
        x, z = log.series("x").astype(np.float64), log.series("z").astype(np.float64)
        arm = np.full(len(self.worlds), self.arm)         # the law's arm and sphere, not the built model's (docstring)
        mass, _, _ = self._pivot_inertia()                  # only scales E_scale below; it cancels in the period
        theta0 = self.param("theta0")
        periods = np.array([crossing_period(t, np.arctan2(-x[w], -z[w])) for w in range(len(self.worlds))])
        law = np.array([elliptic_period(0.4 * RADIUS**2 + self.arm**2, 1.0, self.arm, theta0[w])
                        for w in range(len(self.worlds))])
        e = log.series("energy")
        e_scale = np.maximum.reduce([mass * G * arm * (1 - np.cos(theta0)), mass * G * 2 * RADIUS,
                                     np.full_like(mass, E_SCALE_FLOOR_J)])
        self.monitors.update({"period_sim_s": periods.tolist(), "period_law_s": law.tolist(),
                              "e_scale_j": e_scale.tolist()})
        drift = np.nanmax(np.abs(e - e[:, :1]), axis=1) / e_scale
        drift = np.where(np.all(np.isfinite(e), axis=1), drift, np.inf)
        return {"period_rel_err": np.abs(periods - law) / law, "energy_drift": drift}


class PendulumEnergyTwin(PendulumTwin):
    law_id = "A3"
    arm = ARM_ENERGY
    amplitudes_deg = AMPLITUDES_ENERGY_DEG


__all__ = ["LAW_A2", "LAW_A3", "PendulumTwin", "PendulumEnergyTwin", "elliptic_period", "crossing_period"]
