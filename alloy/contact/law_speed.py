# SPDX-License-Identifier: Apache-2.0
"""A5.speed: Coulomb sliding force does not depend on speed (coordinator 2026-11-03; DC6 08:15Z, DC3 C13 evidence).

A peg (20 mm steel cube, 64 g) is pressed on a flat floor (mu = 0.5 on both shapes) by its weight plus an applied
load and dragged along x at a constant speed by a stiff servo, as a tester pulls a part at a set speed: a D6 joint
gives the peg a driven slide x (position + velocity actuator, kp ``KP``, critical kd, target ``v t + x0``) and a free
slide z. Each world starts at the Coulomb steady state (servo stretched to mu N, peg moving at v); a contact with less
sliding friction runs ahead and settles at its own sliding force within the first half. The sliding force is the
servo force ``kp (v (k+1) dt + x0 - x) + kd (v - xdot)`` that the next step applies, averaged over the second half.
Pass: ``|F / (mu N) - 1| <= 2 %`` in every condition (the metric is the worst one), loads ``LOADS`` x speeds ``SPEEDS``.

MuJoCo runs as Alloy routes the scene (``BaseTwin.settings_for``: builtin kwargs, implicitfast because the slide is
actuated). Its soft friction is viscous below saturation: the friction row's force is about ``b v / R_t`` until it
reaches mu N; the saturation speed is about ``mu delta / (2 tau impratio)`` (delta the contact's penetration, tau
its time constant). At impratio 1 (the route without a file impratio) the peg reaches 19 % (1 N) and 3 % (20 N) of
mu N at 0.4 mm/s and 30 % at 20 N, 4 mm/s; it passes from impratio 50 (docs/notes/wp3-results.md, A5.speed rows).
The peg pressed by its weight alone is not a gated condition: a z-only slider that light hops on the elliptic cone's
sliding lift at impratio >= 10 (reported in the results, converging with dt).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.common import G, BaseTwin
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance

MU = 0.5
HALF = 0.01  # [m] cube half extent
DENSITY = 8000.0  # [kg/m^3] steel: 64 g
LOADS = (1.0, 20.0)  # [N] applied normal load on top of the weight
SPEEDS = (0.4e-3, 4.0e-3)  # [m/s]
KP = 1.0e5  # [N/m] servo stiffness (a load cell and fixture)
DT = 1.25e-3
DURATION_S = 1.0
_SRC = ("coordinator 2026-11-03 (DC6 08:15Z, DC3 C13): Coulomb sliding force within 2 % of mu N at 0.4 and 4 mm/s "
        "on the default route")

LAW_SPEED = LawSpec(
    id="A5.speed", suite="analytic",
    metrics={"force_rel_err": Tolerance(abs=0.02, source=f"{_SRC}; worst condition")},
    stat={"force_rel_err": "mean"},
    dt=DT, substeps=1, duration_s=DURATION_S, worlds_per_condition=1, solvers=("mujoco",), producer="WP3",
    quick={"worlds": len(LOADS) * len(SPEEDS), "duration_s": DURATION_S},
)


def peg_mass() -> float:
    return DENSITY * (2.0 * HALF) ** 3


def servo_kd() -> float:
    return 2.0 * (KP * peg_mass()) ** 0.5


@wp.kernel
def _drive(dof: wp.array(dtype=wp.int32), body: wp.array(dtype=wp.int32), speed: wp.array(dtype=float),
           x0: wp.array(dtype=float), load: wp.array(dtype=float), k1: float,
           target_q: wp.array(dtype=float), body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    target_q[dof[w]] = speed[w] * k1 + x0[w]
    body_f[body[w]] = wp.spatial_vector(0.0, 0.0, -load[w], 0.0, 0.0, 0.0)  # (force, torque) at the COM


@wp.kernel
def _record(dof: wp.array(dtype=wp.int32), coord: wp.array(dtype=wp.int32), speed: wp.array(dtype=float),
            x0: wp.array(dtype=float), k1: float, kp: float, kd: float, col: int, q: wp.array(dtype=float),
            qd: wp.array(dtype=float), force: wp.array2d(dtype=float), height: wp.array2d(dtype=float)):
    w = wp.tid()
    i = coord[w]
    force[w, col] = kp * (speed[w] * k1 + x0[w] - q[i]) + kd * (speed[w] - qd[dof[w]])
    height[w, col] = q[i + 1]


class SlidingSpeedTwin(BaseTwin):
    """See module docstring."""

    law_id = LAW_SPEED.id
    units: Mapping[str, str] = {"force_rel_err": "1"}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"load{f:g}N_v{v * 1e3:g}mm_s", {"load": f, "speed": v})
                     for f, v in itertools.product(LOADS, SPEEDS))

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rest = wp.transform(wp.vec3(0.0, 0.0, HALF), wp.quat_identity())
        peg = b.add_link(xform=rest, label="peg")
        b.add_shape_box(peg, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(mu=MU, density=DENSITY))
        dof = newton.ModelBuilder.JointDofConfig
        drive = dof(axis=newton.Axis.X, target_vel=float(cond.params["speed"]), target_ke=KP, target_kd=servo_kd(),
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY)
        joint = b.add_joint_d6(-1, peg, parent_xform=rest, linear_axes=[drive, dof(axis=newton.Axis.Z)])
        b.add_articulation([joint])

    def on_model(self, model: newton.Model) -> None:
        dev = model.device
        starts = model.joint_qd_start.numpy()[:-1]
        if len(starts) != len(self.worlds):
            raise ValueError(f"A5.speed expects one D6 joint per world, got {len(starts)} for {len(self.worlds)}")
        self.normal = np.array([peg_mass() * G + c.params["load"] for c in self.worlds])
        self._dof = wp.array(starts.astype(np.int32), dtype=wp.int32, device=dev)
        self._coord = wp.array(model.joint_q_start.numpy()[:-1].astype(np.int32), dtype=wp.int32, device=dev)
        self._speed = wp.array(self.param("speed").astype(np.float32), dtype=float, device=dev)
        self._x0 = wp.array((MU * self.normal / KP).astype(np.float32), dtype=float, device=dev)
        self._load = wp.array(self.param("load").astype(np.float32), dtype=float, device=dev)

    def init_state(self, model: newton.Model, state: Any) -> None:
        """Start at the Coulomb steady state: the peg slides at v (the servo is pre-stretched in ``drive``)."""
        qd = state.joint_qd.numpy()
        qd[self._dof.numpy()] = self.param("speed")
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_drive, dim=len(self.worlds), device=self.model.device,
                  inputs=[self._dof, self._first_body, self._speed, self._x0, self._load, float((step + 1) * DT)],
                  outputs=[control.joint_target_q, state.body_f])

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        col = log.column(step)
        if col < 0:
            return
        wp.launch(_record, dim=len(self.worlds), device=self.model.device,
                  inputs=[self._dof, self._coord, self._speed, self._x0, float((step + 1) * DT), KP, servo_kd(), col,
                          state.joint_q, state.joint_qd],
                  outputs=[log.channel("force"), log.channel("z")])

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()
        win = t >= 0.5 * t[-1]
        force = log.series("force")[:, win].astype(np.float64)
        z = log.series("z")[:, win].astype(np.float64)
        ratio = force.mean(1) / (MU * self.normal)
        ids = [c.id for c in self.worlds]
        self.monitors.update({
            "force_over_mu_n": dict(zip(ids, ratio.tolist())),
            "force_sd_over_mu_n": dict(zip(ids, (force.std(1) / (MU * self.normal)).tolist())),
            "penetration_um": dict(zip(ids, (-z.mean(1) * 1e6).tolist())),
            "height_peak_to_peak_um": dict(zip(ids, ((z.max(1) - z.min(1)) * 1e6).tolist())),
            "mu_n": dict(zip(ids, (MU * self.normal).tolist()))})
        return {"force_rel_err": np.array([float(np.max(np.abs(ratio - 1.0)))])}


__all__ = ["LAW_SPEED", "SlidingSpeedTwin", "peg_mass", "servo_kd"]
