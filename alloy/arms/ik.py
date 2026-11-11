# SPDX-License-Identifier: Apache-2.0
"""Batched IK for a ``Rig`` with Newton's ``newton.ik`` (Levenberg-Marquardt), for any of the kit's arms.

``ChainIK(rig, n)`` builds the rig alone (bases placed, no visuals) as one Newton world (``newton.ik`` solves
``n`` problems on one copy) and an ``n``-world copy for batched forward kinematics. Per arm: a TCP position objective
and a rotation objective (weight 0.2) on the flange body with the rig's TCP offset; joint limits as an objective,
then checked hard; gripper joints masked out. A miss or a joint at its limit raises ``IKError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
import alloy._vendor.newton.ik as ik

from alloy.arms.binding import ArmBinding
from alloy.arms.poses import angle_between


class IKError(RuntimeError):
    """No IK solution within tolerance, or a solution outside the joint limits."""


@dataclass(frozen=True)
class IKTolerance:
    pos_m: float = 2e-4         # 0.2 mm
    ang_rad: float = 2e-3       # 0.11 deg
    limit_margin: float = 0.02  # rad kept clear of every arm joint limit


class ChainIK:
    """Every arm's TCP pose -> joint angles, batched over ``n`` problems (one per world)."""

    def __init__(self, rig, n: int, *, device: str = "cpu", tol: IKTolerance = IKTolerance(),
                 iterations: int = 30) -> None:  # fmt: skip
        self.rig, self.n, self.tol, self.iterations, self.device = rig, n, tol, iterations, device
        single = rig.builder(visuals=False)
        self.model = single.finalize(device=device)
        self.bind = ArmBinding.bind(self.model, rig)
        multi = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(multi)
        for _ in range(n):
            multi.add_world(single)
        self.fk_model = multi.finalize(device=device)
        self.fk_bind = ArmBinding.bind(self.fk_model, rig)
        self._fk_state = self.fk_model.state()
        self._fk_qd = wp.zeros(self.fk_model.joint_dof_count, dtype=wp.float32, device=device)
        self._build_solver()

    @property
    def arms(self) -> int:
        return int(self.bind.arm_coord.shape[1])

    def _build_solver(self) -> None:
        m, n, dev = self.model, self.n, self.device
        zeros3 = wp.zeros(n, dtype=wp.vec3, device=dev)
        unit = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32), dtype=wp.vec4, device=dev)
        self.pos_obj, self.rot_obj = [], []
        for a in range(self.arms):
            link, off = int(self.bind.flange[0, a]), self.bind.tcp_offset[a]
            self.pos_obj.append(ik.IKObjectivePosition(link, wp.vec3(*off[:3]), wp.clone(zeros3)))
            self.rot_obj.append(ik.IKObjectiveRotation(link, wp.quat(*off[3:]), wp.clone(unit), weight=0.2))
        lim = ik.IKObjectiveJointLimit(m.joint_limit_lower, m.joint_limit_upper, weight=1.0)
        mask = np.zeros(m.joint_dof_count, dtype=bool)
        mask[self.bind.arm_dof[0].ravel()] = True
        self.solver = ik.IKSolver(m, n, [*self.pos_obj, *self.rot_obj, lim], optimizer="lm", jacobian_mode="analytic",
                                  joint_dof_mask=wp.array(mask, dtype=wp.bool, device=dev))  # fmt: skip
        self._q = wp.zeros((n, m.joint_coord_count), dtype=wp.float32, device=dev)
        self._q0 = self.model.joint_q.numpy()   # gripper joints and the rest stay at the model's values

    def fk(self, q: np.ndarray) -> np.ndarray:
        """``[n, A, J]`` arm angles -> ``[n, A, 7]`` TCP poses (the gripper does not move the TCP)."""
        jq = np.tile(self._q0, self.n).astype(np.float32)
        jq[self.fk_bind.arm_coord] = q
        newton.eval_fk(self.fk_model, wp.array(jq, device=self.device), self._fk_qd, self._fk_state)
        return self.fk_bind.tcp(self._fk_state.body_q.numpy())

    def solve(self, target: np.ndarray, seed: np.ndarray, *, check: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``target [n, A, 7]`` TCP poses, ``seed [n, A, J]`` -> ``(q [n, A, J], err [n, A, 2])`` (m, rad)."""
        target = np.broadcast_to(np.asarray(target, dtype=np.float64), (self.n, self.arms, 7))
        for a in range(self.arms):
            self.pos_obj[a].set_target_positions(wp.array(target[:, a, :3].astype(np.float32), dtype=wp.vec3,
                                                          device=self.device))  # fmt: skip
            self.rot_obj[a].set_target_rotations(wp.array(target[:, a, 3:].astype(np.float32), dtype=wp.vec4,
                                                          device=self.device))  # fmt: skip
        q0 = np.tile(self._q0, (self.n, 1)).astype(np.float32)
        q0[:, self.bind.arm_coord[0]] = np.broadcast_to(seed, (self.n, *self.bind.arm_coord.shape[1:]))
        self._q.assign(q0)
        self.solver.step(self._q, self._q, iterations=self.iterations)
        q = self._q.numpy()[:, self.bind.arm_coord[0]].astype(np.float64)
        got = self.fk(q)
        err = np.stack([np.linalg.norm(got[..., :3] - target[..., :3], axis=-1),
                        angle_between(got[..., 3:], target[..., 3:])], axis=-1)  # fmt: skip
        if check:
            self.check(q, err)
        return q, err

    def check(self, q: np.ndarray, err: np.ndarray) -> None:
        """Raise ``IKError`` on a missed target or a joint within ``tol.limit_margin`` of its limit (the guard)."""
        names = self.rig.arms
        bad = (err[..., 0] > self.tol.pos_m) | (err[..., 1] > self.tol.ang_rad)
        if bad.any():
            w, a = map(int, np.argwhere(bad)[0])
            raise IKError(f"IK: {int(bad.sum())} arm targets missed; world {w} {names[a]} off by "
                          f"{err[w, a, 0] * 1e3:.2f} mm / {np.degrees(err[w, a, 1]):.2f} deg (unreachable?)")
        m = self.tol.limit_margin
        low, high = q < self.bind.lower + m, q > self.bind.upper - m
        if (low | high).any():
            w, a, j = map(int, np.argwhere(low | high)[0])
            raise IKError(f"IK: world {w} {names[a]} {self.rig.specs[a].joints[j]} = {q[w, a, j]:.3f} rad within "
                          f"{m} rad of its limit [{self.bind.lower[a, j]:.3f}, {self.bind.upper[a, j]:.3f}]")


__all__ = ["ChainIK", "IKError", "IKTolerance"]
