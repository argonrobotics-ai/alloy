# SPDX-License-Identifier: Apache-2.0
"""A4 momentum in zero gravity: collisions, torque-free tumbling and a Newton's-cradle row (bench §4.5).

With no external force or torque, linear momentum p = sum m v and angular momentum about the world origin
L = sum (x_com x m v + R I R^T omega) are constant whatever the contacts do (equal and opposite impulses).
Pass: max_t |p(t) - p0| / |p0| <= 5e-4 and max_t |L(t) - L0| / |L0| <= 1e-3 over 10 s, quick mode 0.5 s
(bench §4.5 A4; Newton's free-body version uses 5e-4 for both,
newton@ac7f7830:newton/tests/test_physics_verification.py:577-578).

Collide world conditioning (coordinator 2026-11-01, test conditioning, tolerance unchanged): the two spheres sit
0.2 m off the x axis (lever about the origin 10x the original 2 cm) at the same 1 m/s and 2 cm impact offset, and
the window ends once the centres are `SEP_END` apart; after that the spheres only fly free, which adds float32
position rounding, not solver error. The original world (2 cm lever, 10 s of free flight over ~5 m) stays on the
board as the informational law `A4.collide_small_lever` (`law.INFORMATIONAL`, never gated): 1.18e-3 on MuJoCo full.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import BaseTwin, set_body_velocities
from alloy.bench.measures import MomentumMeter
from alloy.bench.solvers import RIGID_SOLVERS
from alloy.bench.twin import DeviceTwinLog, record_body_position
from alloy.bench.types import Condition, LawSpec, Tolerance

R = 0.05  # [m] sphere radius
SEP_END = 1.0  # [m] the collide window ends when the sphere centres are this far apart (coordinator 2026-11-01)
LEVER = 0.2  # [m] collide world: sphere 0's line of motion off the origin (10x the original 2 cm)
SMALL_LEVER = 0.02  # [m] the original collide world, kept as the informational law
_SRC = "bench §4.5 A4: |dp|/|p| <= 5e-4; |dL|/|L| <= 1e-3"

LAW = LawSpec(
    id="A4.momentum", suite="analytic",
    metrics={"dp_rel": Tolerance(abs=5e-4, source=_SRC), "dL_rel": Tolerance(abs=1e-3, source=_SRC)},
    stat={"dp_rel": "mean", "dL_rel": "mean"},
    dt=1.25e-3, substeps=1, duration_s=10.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,  # bench: "over 10 s"
)
LAW_SMALL_LEVER = dataclasses.replace(LAW, id="A4.collide_small_lever")

# Every world holds the same bodies (SolverMuJoCo needs homogeneous worlds): 4 spheres then 1 box. Each scene sets
# (position, linear velocity, angular velocity) per body; bodies a scene does not use are parked far apart at rest,
# where they add nothing to p or L. Moving rows sit off the x axis so L about the origin is non-zero in every world.
BOX = (0.05, 0.1, 0.15)  # [m] half extents (asymmetric inertia for tumbling)
_ZERO = (0.0, 0.0, 0.0)


def _parked(i: int) -> tuple[tuple, tuple, tuple]:
    return (10.0 + 2.0 * i, 10.0, 10.0), _ZERO, _ZERO


def _collide(lever: float) -> tuple[tuple[tuple, tuple, tuple], ...]:
    """Sphere 0 (1 m/s along +x, spinning 2 rad/s about z) hits resting sphere 1 with a 2 cm impact offset; sphere 0
    moves `lever` off the x axis, so L about the origin is about m * 1 m/s * lever."""
    return (((-0.15, lever, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 2.0)), ((0.0, lever - 0.02, 0.0), _ZERO, _ZERO),
            _parked(2), _parked(3), _parked(4))


_SCENES: dict[str, tuple[tuple[tuple, tuple, tuple], ...]] = {
    "collide": _collide(LEVER),
    "tumble": (_parked(0), _parked(1), _parked(2), _parked(3), (_ZERO, (0.1, 0.0, 0.0), (3.5, -2.5, 4.5))),
    "cradle": (((-0.25, 0.02, 0.0), (1.0, 0.0, 0.0), _ZERO), ((0.0, 0.02, 0.0), _ZERO, _ZERO),
               ((2 * R + 1e-4, 0.02, 0.0), _ZERO, _ZERO), ((4 * R + 2e-4, 0.02, 0.0), _ZERO, _ZERO), _parked(4)),
}


class MomentumTwin(BaseTwin):
    law_id = LAW.id
    units = {"dp_rel": "1", "dL_rel": "1"}
    scenes = _SCENES
    window_sep: Mapping[str, float] = {"collide": SEP_END}  # world -> sphere 0-1 distance that ends its window

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(name, {}) for name in self.scenes)

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        for i, (pos, _, _) in enumerate(self.scenes[cond.id]):
            body = b.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
            if i < 4:
                b.add_shape_sphere(body, radius=R)
            else:
                b.add_shape_box(body, hx=BOX[0], hy=BOX[1], hz=BOX[2])

    def init_state(self, model: newton.Model, state: Any) -> None:
        qd = [list(v) + list(w) for cond in self.worlds for _, v, w in self.scenes[cond.id]]
        set_body_velocities(model, state, np.asarray(qd, dtype=np.float64))

    def on_model(self, model: newton.Model) -> None:
        self._second_body = wp.array(self.ranges.start.numpy() + 1, dtype=wp.int32, device=model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        if step == 0:
            self.meter = MomentumMeter(self.model, log)
        self.meter.update(state, step)
        record_body_position(log, "s0", state.body_q, self._first_body, step)
        record_body_position(log, "s1", state.body_q, self._second_body, step)

    def window_end(self, log: DeviceTwinLog) -> np.ndarray:
        """Per world, the last sample of its window: the first where spheres 0 and 1 are `window_sep` apart, else
        the last sample."""
        sep = np.linalg.norm(_xyz(log, "s0") - _xyz(log, "s1"), axis=-1)  # [W, T]
        ends = np.full(sep.shape[0], sep.shape[1] - 1)
        for w, cond in enumerate(self.worlds):
            limit = self.window_sep.get(cond.id)
            hit = np.flatnonzero(sep[w] >= limit) if limit is not None else ()
            if len(hit):
                ends[w] = hit[0]
        return ends

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        p, ang = log.series("p").astype(np.float64), log.series("L").astype(np.float64)  # [W, T, 3]
        ends = self.window_end(log)
        dp = drift(p, ends)
        dl = drift(ang, ends)
        names = [c.id for c in self.worlds]
        self.monitors.update({"p0": p[:, 0].tolist(), "L0": ang[:, 0].tolist(),
                              "dp_rel_per_world": dict(zip(names, dp.tolist())),
                              "dL_rel_per_world": dict(zip(names, dl.tolist())),
                              "window_end_s_per_world": dict(zip(names, log.times()[ends].tolist())),
                              "window_rule": {k: f"ends when spheres 0-1 are {v} m apart"
                                              for k, v in self.window_sep.items() if k in names}})
        return {"dp_rel": dp, "dL_rel": dl}


class SmallLeverCollideTwin(MomentumTwin):
    """The original collide world (2 cm lever, whole run): informational, it measures float32 position rounding."""

    law_id = LAW_SMALL_LEVER.id
    scenes = {"collide_small_lever": _collide(SMALL_LEVER)}
    window_sep: Mapping[str, float] = {}


def _xyz(log: DeviceTwinLog, prefix: str) -> np.ndarray:
    return np.stack([log.series(prefix + a) for a in "xyz"], axis=-1).astype(np.float64)


def drift(v: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """max over each world's window [0, ends[w]] of |v(t) - v(0)| / |v(0)| for a [W, T, 3] series."""
    dev = np.linalg.norm(v - v[:, :1], axis=-1)
    inside = np.arange(v.shape[1])[None, :] <= ends[:, None]
    return np.where(inside, dev, 0.0).max(axis=1) / np.linalg.norm(v[:, 0], axis=-1)


__all__ = ["LAW", "LAW_SMALL_LEVER", "MomentumTwin", "SmallLeverCollideTwin", "drift"]
