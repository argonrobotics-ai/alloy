# SPDX-License-Identifier: Apache-2.0
"""R4 drop-and-bounce twin (bench §4.4 R4): restitution e = sqrt(h1 / h0) within +/- 0.03.

The sphere is dropped from each condition's height onto a per-world plane carrying the restitution fitted from the
`calib` trials (mean calib e: a material constant, so one value across conditions). Only XPBD consumes
`ShapeConfig.restitution` (newton@ac7f7830:newton/_src/sim/builder.py:1009-1017); other solvers fail until WP2's
damping path exists, which is the measured baseline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.analytic.restitution import MU, first_rebound
from alloy.bench.solvers import RIGID_SOLVERS, SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigTrial, Tolerance

RADIUS = 0.02  # [m]

LAW = LawSpec(
    id="R4.drop_bounce", suite="rig",
    metrics={"restitution": Tolerance(abs=0.03, source="bench §4.4 R4: e = sqrt(h1/h0) +/- 0.03")},
    stat={"restitution": "mean"}, dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=8,
    solvers=RIGID_SOLVERS,
)


class BounceTwin(BaseTwin):
    law_id = LAW.id
    units = {"restitution": "1"}
    settings = {"xpbd": SolverSettings("xpbd", {"enable_restitution": True})}

    def __init__(self) -> None:
        super().__init__()
        self.e_fit = math.nan

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        self.e_fit = float(np.mean([t.scalars["restitution"] for t in calib]))
        self.monitors["e_fit_calib"] = self.e_fit

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(mu=MU, restitution=self.e_fit)
        b.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0, cfg=cfg)
        z = RADIUS + cond.params["drop_height"]
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), label="ball")
        b.add_shape_sphere(body, radius=RADIUS, cfg=cfg)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t, z = log.times(), log.series("z").astype(np.float64)
        h0 = self.param("drop_height")
        h1 = np.array([first_rebound(t, z[w]) for w in range(len(self.worlds))]) - RADIUS
        e = np.sqrt(np.clip(h1, 0.0, None) / h0)
        self.monitors["h1_m"] = h1.tolist()
        return {"restitution": e}


__all__ = ["LAW", "BounceTwin"]
