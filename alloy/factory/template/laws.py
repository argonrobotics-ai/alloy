# SPDX-License-Identifier: Apache-2.0
"""The plug-in's analytic law (§3.8 item 7): free fall of an unconstrained particle.

``FAC.<name>.freefall``: a particle released at rest from z0 = 1 m, no contact for T = 0.3 s, ends at
z0 - g T^2 / 2 = 0.5586 m. Symplectic Euler at dt = 1 ms lands low by g·dt·T/2 = 1.47 mm, so the per-world error
bound is 2 mm (derived). The twin follows the bench runner's hooks (``conditions``, ``prepare_builder``,
``build_world``, ``make_solver``, ``record`` into the device log with a kernel, ``scalars`` as per-world errors).
Registered with ``alloy.bench.law.register_law`` when the bench library (WP6) is installed; without it
``evaluate_plugin`` reports the law ``not_run``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bench.types import Condition, LawSpec, Tolerance

from .solver import NAMESPACE, PLUGIN_NAME, SolverTemplate

log = logging.getLogger("alloy.factory")

Z0, T_END, DT, G = 1.0, 0.3, 1.0e-3, 9.81
FREEFALL = LawSpec(
    id=f"FAC.{PLUGIN_NAME}.freefall", suite="analytic",
    metrics={"z_err": Tolerance(abs=2.0e-3, source="derived: symplectic Euler bias g dt T/2 = 1.47 mm at 1 ms")},
    stat={"z_err": "mean"}, dt=DT, substeps=1, duration_s=T_END, worlds_per_condition=4, solvers=(PLUGIN_NAME,),
    producer=f"factory:{PLUGIN_NAME}", quick={"worlds": 4, "duration_s": T_END})


@wp.kernel
def _record_z(q: wp.array[wp.vec3], world: wp.array[wp.int32], col: int, out: wp.array2d[wp.float32]):
    p = wp.tid()
    w = world[p]
    if w >= 0:
        out[w, col] = q[p][2]


class FreeFallTwin:
    """One particle per world, nothing to drive; the particle height is logged on the device every step."""

    law_id = FREEFALL.id
    units = {"z_err": "m"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        del quick
        return (Condition("drop_1m", {"z0": Z0}),)

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        del solver
        SolverTemplate.register_custom_attributes(b)

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        z0 = cond.params.get("z0", Z0) + sample.get("dz0", 0.0)
        b.add_particle((0.0, 0.0, z0), (0.0, 0.0, 0.0), 0.01, radius=0.005,
                       custom_attributes={f"{NAMESPACE}:ground_ke": 0.0, f"{NAMESPACE}:ground_kd": 0.0})

    def make_solver(self, model: newton.Model) -> SolverTemplate:
        self._model = model
        return SolverTemplate(model)

    def make_pipeline(self, model: newton.Model, solver: str) -> None:
        del model, solver  # no shapes: nothing to collide

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        del step, t, state, control  # free fall: no input

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: Any) -> None:
        col = log.column(step)
        if col >= 0:
            wp.launch(_record_z, dim=self._model.particle_count, device=self._model.device,
                      inputs=[state.particle_q, self._model.particle_world, col, log.channel("z")])

    def scalars(self, log: Any) -> dict[str, np.ndarray]:
        z = np.asarray(log.series("z"), dtype=np.float64)
        t = float(log.times()[-1])
        g = float(-self._model.gravity.numpy()[0][2])
        return {"z_err": np.abs(z[:, -1] - (z[:, 0] - 0.5 * g * t * t))}


def register() -> bool:
    """Register the law with the bench library if it is installed; returns whether it was."""
    try:
        law_mod = importlib.import_module("alloy.bench.law")
    except ImportError:
        log.info("%s: alloy.bench.law not installed; law %s not registered", PLUGIN_NAME, FREEFALL.id)
        return False
    law_mod.register_law(FREEFALL, FreeFallTwin)
    return True


REGISTERED = register()
