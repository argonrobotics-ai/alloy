# SPDX-License-Identifier: Apache-2.0
"""Analytic law A11.torsion on the CHAIN route (coordinator 2026-11-08 05:06Z: XB4-10 found MuJoCo 4.0 ships a rod twist
test and Alloy had none; producer WP2, with WP10).

A rod clamped at one end, gravity off, takes a constant torque T about its axis at the free end and settles with the
rod self-test's damping (tau = 1.4 / omega1). Pure torsion: the twist at the free end is theta = T L / (G J). The rod
is built by the CHAIN emitter itself (`build_chain`: D6 joints, twist axis along each link with k = G J / l, so the
chain's twist is the series sum T * sum(l_j) / (G J) = T L / (G J) exactly, half-length end links included), so the
law judges what the router emits. The tip twist is read from a point off the axis on the last link: v = p_off - p_axis,
theta = atan2(-v_y, v_z) (rotation about +x), unwrapped in time, averaged over the last 0.5 s. 1 %.

Conditions: theta 0.5 and 1.5 rad on a 30-link 0.5 m rod (gamma 0.5 stiffness, G J = E I / 1.3). T and the expected
twist use the nominal L, so the emitted link lengths are judged too (monitor ``length_ratio``). A dead axial torque on
a cantilever has no static critical torque (Ziegler), so the straight state is checked dynamically: a 1e-4 N lateral
push on the tip for the first 0.1 s seeds a ~1e-4 m bend, and the monitors give its peak and the axis drift at the end
(stable if the seed dies out).

Twist-bend coupling: the CHAIN model claims none (each link's twist and its two bends are separate D6 springs, no
coupling stiffness; rotations compose exactly, so only geometric coupling exists, e.g. the Greenhill buckling of a
twisted rod), so the law has no coupling case.

Twist dynamics (WP10's review; [internal note] notes/wp2-torsion/twist_release.py): a solid rod's first torsional mode at
these lengths is hundreds of Hz (this rod: 3,051 rad/s, period 2.06 ms, under two production steps), so no integrator
resolves it at the production dt. The emitter's implicitfast armature on each twist dof (here 2,478x the link's axial
inertia) keeps it stable: released from 0.5 rad at the card-like damping (bend-mode ratio 0.05) the twist decays with
an e-fold of 3.6 ms (the continuum, damping-limited: 4.5 ms); at a hundredth of that damping it rings at the armature-
limited joint rate (a few steps per period) and dies within 20-50 ms, where the continuum would ring at 2.06 ms.
At a 25 µs step it rings 2.58x slow, held by the route's armature floor (none: 2.061 ms, +0.07 %). Statics are exact;
torsional vibration is not resolved (docs/known_gaps/rod-twist-dynamics.md, not a law).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.scene.analytic_rods import H0, SETTLE_TAU, RodLawTwin, ei_for_gamma, rod_card

TWISTS = (0.5, 1.5)             # rad, the conditions' expected tip twist
GAMMA, LINKS, L = 0.5, 30, 0.5
OFFSET = 0.01                   # m, the off-axis tracked point on the tip link
SEED_F, SEED_S = 1.0e-4, 0.1    # N, s: a lateral push on the tip for the first 0.1 s (~1e-4 m), the straight rod's
#                                 stability check under the dead torque (no static critical torque: Ziegler)
_SRC = "coordinator 2026-11-08 (XB4-10): clamped rod under end torque, theta = T L / (G J)"

LAW_TORSION = LawSpec(
    id="A11.torsion", suite="analytic",
    metrics={"twist_rel_err": Tolerance(abs=0.01, source=f"{_SRC}, within 1 %")},
    stat={"twist_rel_err": "mean"}, dt=1.25e-3, substeps=1, duration_s=2.0, worlds_per_condition=1,
    solvers=("mujoco",), production_dt=True, producer="WP2", quick={"worlds": 2, "duration_s": 2.0},
)


@wp.kernel
def _end_torque(bodies: wp.array(dtype=wp.int32), torque: wp.array(dtype=float), push: float,
                body_f: wp.array(dtype=wp.spatial_vector)):
    """A dead torque T about world +x (the rod's axis) on each world's tip link, plus the seed's lateral push along
    +y while it lasts (Newton order: f, tau)."""
    w = wp.tid()
    body_f[bodies[w]] = wp.spatial_vector(0.0, push, 0.0, torque[w], 0.0, 0.0)


def twist_angle(off: tuple[np.ndarray, ...], axis: tuple[np.ndarray, ...]) -> np.ndarray:
    """[worlds, samples] rotation about +x of the off-axis point around the axis point, unwrapped in time."""
    vy, vz = off[1] - axis[1], off[2] - axis[2]
    return np.unwrap(np.arctan2(-vy, vz), axis=1)


class TorsionTwin(RodLawTwin):
    law_id = LAW_TORSION.id
    units = {"twist_rel_err": "1"}
    dt = LAW_TORSION.dt

    def __init__(self) -> None:
        super().__init__()
        self._torques: list[float] = []
        self._expected: list[float] = []
        self._length_ratio: list[float] = []

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"theta{t:g}", {"theta": t}) for t in TWISTS)

    def world_gravity(self, cond: Condition) -> tuple[float, float, float] | None:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rod = rod_card(L, ei_for_gamma(L, GAMMA), LINKS, damping_ratio=0.5 * SETTLE_TAU)
        h = self.emit(b, rod, [(0.0, 0.0, H0), (L, 0.0, H0)], clamp=True)
        tip = h.bodies[-1]
        self.track([(tip, (0.0, 0.0, OFFSET)), (tip, (0.0, 0.0, 0.0))])
        torque = float(cond.params["theta"]) * rod.GJ / rod.length    # nominal L: the emitted lengths are judged
        self._torques.append(torque)
        self._expected.append(torque * rod.length / rod.GJ)
        self._length_ratio.append(float(sum(h.lengths)) / rod.length)

    def on_model(self, model: newton.Model) -> None:
        super().on_model(model)
        dev = model.device
        self._tip = wp.array([h.bodies[-1] for h in self.chains], dtype=wp.int32, device=dev)
        self._torque = wp.array(self._torques, dtype=float, device=dev)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        push = SEED_F if t < SEED_S else 0.0
        wp.launch(_end_torque, dim=len(self.chains), inputs=[self._tip, self._torque, push], outputs=[state.body_f],
                  device=self.model.device)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        theta = twist_angle(self.point(log_, 0), self.point(log_, 1))
        tail = max(1, int(round(0.5 / (log_.stride * log_.dt))))
        got = theta[:, -tail:].mean(axis=1)
        ref = np.asarray(self._expected)
        _, ay, az = self.point(log_, 1)
        drift = np.hypot(ay, az - H0)                     # the tip axis point's lateral excursion [worlds, samples]
        seed_end = max(1, int(round(SEED_S / (log_.stride * log_.dt))))
        self.monitors.update({"twist_rad": got.tolist(), "expected_rad": ref.tolist(), "torque_n_m": self._torques,
                              "length_ratio": self._length_ratio,
                              "twist_spread_last_0p5s": (theta[:, -tail:].max(1) - theta[:, -tail:].min(1)).tolist(),
                              "seed_drift_peak_m": drift[:, :4 * seed_end].max(1).tolist(),
                              "axis_drift_end_m": drift[:, -1].tolist(),
                              "torque_over_pi_ei_l": [t / (math.pi * r.EI / r.length) for t, r in zip(self._torques,
                                                                                                        self.rods)]})
        return {"twist_rel_err": np.abs(got / ref - 1.0)}


__all__ = ["LAW_TORSION", "TWISTS", "TorsionTwin", "twist_angle"]
