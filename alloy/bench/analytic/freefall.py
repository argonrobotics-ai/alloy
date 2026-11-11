# SPDX-License-Identifier: Apache-2.0
"""A1 free fall and projectile: x(t) = x0 + v0 t + g t^2 / 2, v(t) = v0 + g t (bench §4.5).

Bound (Newton's verification test, newton@ac7f7830:newton/tests/test_physics_verification.py:120-122): a first-order
integrator at step h is off by g h t / 2, so |x - x_law| <= 2 * (1/2) g h t and |v - v_law| <= g h. The metrics are
the per-world maxima over time of error / bound, so the tolerance is 1.0 at any dt and duration.

g in the reference and the bound is the literal `G`, never the model's (audit AB-laws-2: read from `model.gravity` the
reference followed a wrong gravity, and the law passed at 9.0 m/s2). The twin sets no gravity of its own, so the
bodies fall under the engine's default, which is what the law holds to 9.81.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import BaseTwin, set_body_velocities
from alloy.bench.solvers import RIGID_SOLVERS
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance

G = 9.81  # [m/s2] standard gravity along -z (AGENTS.md conventions): the reference's own number
Z0 = 1.0  # [m] start height; there is no ground, so nothing limits the fall
RADIUS = 0.05  # [m]
_SRC = "newton@ac7f7830:newton/tests/test_physics_verification.py:120-122 (2·½·g·h·t, g·h)"

LAW = LawSpec(
    id="A1.freefall", suite="analytic",
    metrics={"pos_err_ratio": Tolerance(abs=1.0, source=_SRC), "vel_err_ratio": Tolerance(abs=1.0, source=_SRC)},
    stat={"pos_err_ratio": "mean", "vel_err_ratio": "mean"},
    dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,
)

_VELOCITIES = ((0.0, 0.0, 0.0), (1.0, 0.0, 3.0), (-2.0, 1.5, 5.0), (0.5, -2.0, 0.0))


class FreeFallTwin(BaseTwin):
    law_id = LAW.id
    units = {"pos_err_ratio": "1", "vel_err_ratio": "1"}
    needs_contacts = False

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"v{i}", {"vx": v[0], "vy": v[1], "vz": v[2]}) for i, v in enumerate(_VELOCITIES))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, Z0), wp.quat_identity()), label="ball")
        b.add_shape_sphere(body, radius=RADIUS)

    def init_state(self, model: newton.Model, state: Any) -> None:
        qd = np.zeros((model.body_count, 6))
        qd[:, :3] = np.stack([self.param("vx"), self.param("vy"), self.param("vz")], axis=1)
        set_body_velocities(model, state, qd)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log, velocity=True)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()[1:]
        g = np.tile(np.array([0.0, 0.0, -G]), (len(self.worlds), 1))    # the law's g, not model.gravity (docstring)
        x = np.stack([log.series(a) for a in "xyz"], -1).astype(np.float64)  # [W, T, 3]
        v = np.stack([log.series(f"v{a}") for a in "xyz"], -1).astype(np.float64)
        x0, v0 = x[:, :1, :], v[:, :1, :]
        tt = t[None, :, None]
        x_law = x0 + v0 * tt + 0.5 * g[:, None, :] * tt**2
        v_law = v0 + g[:, None, :] * tt
        h = log.dt / LAW.substeps
        gmag = np.linalg.norm(g, axis=1)[:, None]
        pos_err = np.linalg.norm(x[:, 1:] - x_law, axis=-1)
        vel_err = np.linalg.norm(v[:, 1:] - v_law, axis=-1)
        self.monitors["max_pos_err_m"] = float(np.max(pos_err))
        self.monitors["pos_bound_end_m"] = float(np.max(gmag[:, 0]) * h * t[-1])
        return {"pos_err_ratio": np.max(pos_err / (gmag * h * t[None, :]), axis=1),
                "vel_err_ratio": np.max(vel_err / (gmag * h), axis=1)}


__all__ = ["LAW", "FreeFallTwin"]
