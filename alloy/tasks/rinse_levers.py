# SPDX-License-Identifier: Apache-2.0
"""Sink plumbing and a hand on its levers, for the rinse_* proof tasks.

- ``SinkPlumbing`` is ``FluidNetwork`` (the lumped-fluid core: open tanks, orifice drains, overflows, a faucet) with
  lever wastes: before each network step it sets each drain's opening from its lever's joint angle,
  ``f = clip(q / q_open, 0, 1)`` (Gym-Anything ``water/system.py:514-519``), read from the body poses after the solve.
- ``LeverHand`` is the privileged scripted hand of the proofs: while it grips a lever it twists it toward a target
  angle with a PD torque about the joint axis, capped at ``max_torque``; released, it applies nothing. Targets and
  grips are inputs set between control periods (``set_input("target", ...)``, ``set_input("grip", ...)``).

Both read joint angles from body poses, as ``FluidNetwork``'s actuators do, so they do not depend on a solver's
joint coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench, torque_wrench
from alloy.plugins.library.fluid_network import FluidNetwork


@wp.func
def parent_frame(body_q: wp.array[wp.transform], j: wp.int32, joint_parent: wp.array[wp.int32],
                 joint_X_p: wp.array[wp.transform]) -> wp.transform:
    """The joint's parent frame in world (a world-anchored joint: its X_p)."""
    xp = joint_X_p[j]
    par = joint_parent[j]
    if par >= 0:
        xp = wp.transform_multiply(body_q[par], xp)
    return xp


@wp.func
def joint_angle(xp: wp.transform, xc: wp.transform, axis: wp.vec3) -> float:
    """Rotation of the child joint frame ``xc`` relative to the parent frame ``xp`` about ``axis`` (parent joint
    frame), in (-π, π]."""
    qr = wp.quat_inverse(wp.transform_get_rotation(xp)) * wp.transform_get_rotation(xc)
    sgn = wp.where(qr[3] < 0.0, -1.0, 1.0)
    return 2.0 * wp.atan2(sgn * wp.dot(wp.vec3(qr[0], qr[1], qr[2]), wp.normalize(axis)), sgn * qr[3])


@wp.struct
class Joints:
    parent: wp.array[wp.int32]
    child: wp.array[wp.int32]
    X_p: wp.array[wp.transform]
    X_c: wp.array[wp.transform]
    axis: wp.array[wp.vec3]
    qd_start: wp.array[wp.int32]


def joints_struct(model: newton.Model) -> Joints:
    s = Joints()
    s.parent, s.child, s.X_p, s.X_c = model.joint_parent, model.joint_child, model.joint_X_p, model.joint_X_c
    s.axis, s.qd_start = model.joint_axis, model.joint_qd_start
    return s


@wp.func
def lever_angle(body_q: wp.array[wp.transform], js: Joints, j: wp.int32) -> float:
    xp = parent_frame(body_q, j, js.parent, js.X_p)
    xc = wp.transform_multiply(body_q[js.child[j]], js.X_c[j])
    return joint_angle(xp, xc, js.axis[js.qd_start[j]])


@wp.kernel
def openings_kernel(body_q: wp.array[wp.transform], js: Joints, levers: wp.array2d[wp.int32],
                    edges: wp.array[wp.int32], q_open: float, opening: wp.array2d[wp.float32]):
    w = wp.tid()
    for k in range(edges.shape[0]):
        q = lever_angle(body_q, js, levers[w, k])
        opening[w, edges[k]] = wp.clamp(q / q_open, 0.0, 1.0)


@wp.kernel
def hand_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], js: Joints,
                levers: wp.array2d[wp.int32], target: wp.array2d[wp.float32], grip: wp.array2d[wp.float32],
                kp: float, kd: float, max_torque: float, torque: wp.array2d[wp.float32],
                body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    for k in range(levers.shape[1]):
        j = levers[w, k]
        c = js.child[j]
        xp = parent_frame(body_q, j, js.parent, js.X_p)
        axis_w = wp.transform_vector(xp, wp.normalize(js.axis[js.qd_start[j]]))
        q = lever_angle(body_q, js, j)
        qd = body_qd[c]
        rate = wp.dot(wp.vec3(qd[3], qd[4], qd[5]), axis_w)
        par = js.parent[j]
        if par >= 0:
            pd = body_qd[par]
            rate = rate - wp.dot(wp.vec3(pd[3], pd[4], pd[5]), axis_w)
        tau = grip[w, k] * wp.clamp(kp * (target[w, k] - q) - kd * rate, -max_torque, max_torque)
        torque[w, k] = tau
        add_wrench(body_f, c, torque_wrench(tau * axis_w))
        if par >= 0:
            add_wrench(body_f, par, torque_wrench(-tau * axis_w))


def _joint_table(ctx: Any, labels: tuple[str, ...]) -> wp.array:
    """``[W, L]`` joint indices of ``labels`` (one per world)."""
    cols = [ctx.joints(name).numpy() for name in labels]
    return ctx.array(np.stack(cols, axis=1).astype(np.int32), wp.int32)


class SinkPlumbing(FluidNetwork):
    """``FluidNetwork`` whose drain edges open with lever joints (``levers``: (edge, joint label) pairs)."""

    name = "sink_plumbing"
    summary = "a lumped sink network (tanks, drains, overflows, a faucet) whose drains open with lever wastes"

    @dataclass
    class Config(FluidNetwork.Config):
        actuators: tuple[Any, ...] = ()
        levers: tuple[tuple[str, str], ...] = ()       # (drain edge, lever joint label)
        q_open: float = 0.5 * math.pi                   # rad: the lever angle of a fully open drain

        def __post_init__(self) -> None:
            super().__post_init__()
            self.levers = tuple((str(e), str(j)) for e, j in self.levers)
            edges = {e.name for e in self.edges}
            for e, _ in self.levers:
                if e not in edges:
                    raise ValueError(f"levers: {e!r} is not an edge of the network")
            if not self.q_open > 0.0:
                raise ValueError(f"q_open must be > 0, got {self.q_open}")

    def setup(self, ctx: Any) -> None:
        super().setup(ctx)
        c = self.config
        self.js = joints_struct(ctx.model)
        if c.levers:
            self.levers = _joint_table(ctx, tuple(j for _, j in c.levers))
            self.lever_edges = ctx.array(np.asarray([self.fluid.net.edge_index(e) for e, _ in c.levers], np.int32),
                                         wp.int32)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        if self.config.levers:
            wp.launch(openings_kernel, dim=self.ctx.world_count,
                      inputs=[state_out.body_q, self.js, self.levers, self.lever_edges, float(self.config.q_open)],
                      outputs=[self.fluid.opening], device=self.ctx.device)
        super().update(state_in, state_out, dt)


class LeverHand(ForcePlugin):
    """A scripted hand twisting lever joints toward target angles (PD torque about each joint axis, capped)."""

    name = "lever_hand"
    summary = "a privileged scripted hand: PD torque on gripped lever joints toward target angles"

    @dataclass
    class Config:
        levers: tuple[str, ...] = field(default_factory=tuple)   # joint labels
        kp: float = 20.0                                          # N·m/rad: friction 0.30 N·m leaves < 1°
        kd: float = 0.4                                           # N·m·s/rad (about critical on a 2e-3 kg·m² lever)
        max_torque: float = 1.5                                   # N·m

        def __post_init__(self) -> None:
            self.levers = tuple(str(j) for j in self.levers)
            if not self.levers:
                raise ValueError("LeverHand needs at least one lever joint")
            if min(self.kp, self.kd, self.max_torque) < 0.0:
                raise ValueError("kp, kd and max_torque must be >= 0")

    def setup(self, ctx: Any) -> None:
        W, L = ctx.world_count, len(self.config.levers)
        self.js = joints_struct(ctx.model)
        self.levers = _joint_table(ctx, self.config.levers)
        self.target = self.add_input("target", ctx.zeros((W, L), wp.float32))
        self.grip = self.add_input("grip", ctx.zeros((W, L), wp.float32))
        self.torque = self.add_state("torque", ctx.zeros((W, L), wp.float32))

    def check_dt(self, dt: float) -> None:
        """Explicit PD on the lever's own inertia: refuse a step where kp·dt²/I or kd·dt/I is unstable."""
        model = self.ctx.model
        armature = model.joint_armature.numpy()
        inertia = model.body_inertia.numpy()
        for k, name in enumerate(self.config.levers):
            j = int(self.levers.numpy()[0, k])
            dof = int(model.joint_qd_start.numpy()[j])
            c = int(model.joint_child.numpy()[j])
            i_min = float(armature[dof]) + float(np.min(np.linalg.eigvalsh(inertia[c])))
            if self.config.kp * dt * dt / i_min > 1.0 or self.config.kd * dt / i_min > 1.0:
                raise PluginError(f"{self.label}: gains on lever {name!r} unstable at dt {dt:.3g} s "
                                  f"(inertia {i_min:.3g} kg·m²): lower kp/kd or the exchange step")

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        c = self.config
        wp.launch(hand_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.js, self.levers, self.target, self.grip, float(c.kp),
                          float(c.kd), float(c.max_torque)],
                  outputs=[self.torque, forces.body_f], device=self.ctx.device)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (x, y, z, w) quaternions."""
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.r_[aw * bv + bw * av + np.cross(av, bv), aw * bw - np.dot(av, bv)]


def lever_angles(model: newton.Model, body_q: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Host: lever angles [rad] of joint indices ``joints`` (any shape, e.g. ``[W, L]``) from numpy ``body_q``
    ([B, 7]: x, y, z, qx, qy, qz, qw), as ``lever_angle`` does on the device."""
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    xp, xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    axis, qd_start = model.joint_axis.numpy(), model.joint_qd_start.numpy()
    out = np.zeros(np.shape(joints))
    for idx, j in np.ndenumerate(np.asarray(joints)):
        qp = xp[j][3:] if parent[j] < 0 else _qmul(body_q[parent[j]][3:], xp[j][3:])
        qr = _qmul(qp * np.r_[-1.0, -1.0, -1.0, 1.0], _qmul(body_q[child[j]][3:], xc[j][3:]))
        sgn = -1.0 if qr[3] < 0 else 1.0
        a = axis[qd_start[j]] / np.linalg.norm(axis[qd_start[j]])
        out[idx] = 2.0 * math.atan2(sgn * float(np.dot(qr[:3], a)), sgn * float(qr[3]))
    return out


__all__ = ["LeverHand", "SinkPlumbing", "lever_angles"]
