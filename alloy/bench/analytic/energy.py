# SPDX-License-Identifier: Apache-2.0
"""A3 in contact scenes: a passive box dropped and slid on a plane must not gain energy (SPEC §5.3, D20).

No external work acts (W_ext = 0), so every energy rise is unexplained:
- step_gain = max_n (E_n - E_{n-1}) / E_scale <= 1e-2 (the §5.3 A3 amendment; the S2 ENERGY_JUMP level of §4.3);
- cum_gain = max_n (E_n - E_0) / E_scale <= 0.5 (the S1 cumulative level of §4.3).
E_scale = max(E0 - PE_min, m g L_char, 1e-3 J) with PE measured from the lowest COM height at reset (D20): for a
box resting on the plane PE_min is its height drop h, so E0 - PE_min = m g h + m v^2 / 2; L_char = box edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import G, BaseTwin, set_body_velocities
from alloy.bench.measures import EnergyMeter
from alloy.bench.solvers import RIGID_SOLVERS
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.core.defaults import E_SCALE_FLOOR_J, ENERGY_JUMP_REL, ENERGY_S1_REL

HALF = 0.025  # [m] box half extent
MU = 0.5
DENSITY = 1000.0  # [kg/m^3]
_SRC_STEP = "SPEC §5.3 A3 amendment: per-step unexplained gain <= 1e-2·E_scale (D20)"
_SRC_CUM = "SPEC §4.3 ENERGY_JUMP S1: cumulative unexplained gain <= 0.5·E_scale"

LAW = LawSpec(
    id="A3.contact_energy", suite="analytic",
    metrics={"step_gain": Tolerance(abs=ENERGY_JUMP_REL, source=_SRC_STEP),
             "cum_gain": Tolerance(abs=ENERGY_S1_REL, source=_SRC_CUM)},
    stat={"step_gain": "mean", "cum_gain": "mean"},
    dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,
)

# (id, drop height [m], vx, vy [m/s])
_CASES = (("rest", 0.0, 0.0, 0.0), ("drop2cm", 0.02, 0.0, 0.0), ("slide", 0.0, 0.5, 0.0),
          ("drop5cm_slide", 0.05, 0.5, 0.3))


class ContactEnergyTwin(BaseTwin):
    law_id = LAW.id
    units = {"step_gain": "1", "cum_gain": "1"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(c, {"h": h, "vx": vx, "vy": vy}) for c, h, vx, vy in _CASES)

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        z = HALF + cond.params["h"]
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), label="box")
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(mu=MU, density=DENSITY))

    def init_state(self, model: newton.Model, state: Any) -> None:
        qd = np.zeros((model.body_count, 6))
        qd[:, 0], qd[:, 1] = self.param("vx"), self.param("vy")
        set_body_velocities(model, state, qd)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        if step == 0:
            self.energy = EnergyMeter(self.model, state, log)
        self.energy.update(state, step)

    def e_scale(self) -> np.ndarray:
        mass = self.model.body_mass.numpy()[self.ranges.start.numpy()].astype(np.float64)
        v2 = self.param("vx") ** 2 + self.param("vy") ** 2
        return np.maximum.reduce([mass * G * self.param("h") + 0.5 * mass * v2, mass * G * 2 * HALF,
                                  np.full_like(mass, E_SCALE_FLOOR_J)])

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        e = log.series("energy")
        scale = self.e_scale()
        step_gain = log.series("acc:energy_gain") / scale
        cum = np.max(e - e[:, :1], axis=1) / scale
        finite = np.all(np.isfinite(e), axis=1)
        self.monitors["e_scale_j"] = scale.tolist()
        self.monitors["e_end_minus_e0_j"] = (e[:, -1] - e[:, 0]).tolist()
        return {"step_gain": np.where(finite, np.maximum(step_gain, 0.0), np.inf),
                "cum_gain": np.where(finite, np.maximum(cum, 0.0), np.inf)}


__all__ = ["LAW", "ContactEnergyTwin"]
