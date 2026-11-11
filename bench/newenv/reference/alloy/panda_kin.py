# SPDX-License-Identifier: Apache-2.0
"""Franka Panda kinematics in numpy, batched over worlds: Alloy has no IK or Jacobian call in its public API, so the
reference environments' scripted policies carry their own (chain read from library/robots/emb_e2_panda/scene.xml).

``fk(q)`` gives the hand frame and the grasp point (between the finger pads); ``ik_step`` is one damped
least-squares step toward a pose; ``solve_ik`` iterates it kinematically (used once, for start poses).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_RX_NEG = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])     # quat (c, -s, 0, 0): Rx(-90 deg)
_RX_POS = _RX_NEG.T                                                             # Rx(+90 deg)
#: per joint: (parent-frame offset, parent-frame rotation) of link k's frame; the joint turns about its z
CHAIN = [((0.0, 0.0, 0.333), np.eye(3)), ((0.0, 0.0, 0.0), _RX_NEG), ((0.0, -0.316, 0.0), _RX_POS),
         ((0.0825, 0.0, 0.0), _RX_POS), ((-0.0825, 0.384, 0.0), _RX_NEG), ((0.0, 0.0, 0.0), _RX_POS),
         ((0.088, 0.0, 0.0), _RX_POS)]
HAND_OFFSET = (0.0, 0.0, 0.107)
_C45 = float(np.cos(np.pi / 4))
HAND_ROT = np.array([[_C45, _C45, 0.0], [-_C45, _C45, 0.0], [0.0, 0.0, 1.0]])   # Rz(-45 deg)
GRASP_Z = 0.1034                # hand frame -> centre of the finger pads, along the hand's z
Q_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.7854])
DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])         # hand z down, fingers close along y


@dataclass
class Frames:
    """World frames for joint angles q (N, 7): joint origins and axes, the hand rotation, the grasp point."""

    origins: np.ndarray     # (N, 7, 3)
    axes: np.ndarray        # (N, 7, 3)
    hand_pos: np.ndarray    # (N, 3)
    hand_rot: np.ndarray    # (N, 3, 3)
    grasp: np.ndarray       # (N, 3)


def _rz(angle: np.ndarray) -> np.ndarray:
    c, s, z, o = np.cos(angle), np.sin(angle), np.zeros_like(angle), np.ones_like(angle)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def fk(q: np.ndarray, tool: float = GRASP_Z) -> Frames:
    """Frames for q (N, 7) with the robot base at the world origin. ``tool``: grasp point distance along hand z."""
    q = np.asarray(q, np.float64)
    n = q.shape[0]
    rot, pos = np.broadcast_to(np.eye(3), (n, 3, 3)).copy(), np.zeros((n, 3))
    origins, axes = np.zeros((n, 7, 3)), np.zeros((n, 7, 3))
    for k, (offset, r_parent) in enumerate(CHAIN):
        pos = pos + rot @ np.asarray(offset)
        rot = rot @ r_parent @ _rz(q[:, k])
        origins[:, k], axes[:, k] = pos, rot[:, :, 2]
    hand_pos = pos + rot @ np.asarray(HAND_OFFSET)
    hand_rot = rot @ HAND_ROT
    return Frames(origins, axes, hand_pos, hand_rot, hand_pos + hand_rot[:, :, 2] * tool)


def jacobian(f: Frames, point: np.ndarray) -> np.ndarray:
    """(N, 6, 7) geometric Jacobian of ``point`` (N, 3) fixed in the hand: rows linear xyz, then angular xyz."""
    lin = np.cross(f.axes, point[:, None, :] - f.origins)
    return np.concatenate([lin.transpose(0, 2, 1), f.axes.transpose(0, 2, 1)], axis=1)


def rot_error(r_des: np.ndarray, r_now: np.ndarray) -> np.ndarray:
    """(N, 3) small-angle rotation vector taking ``r_now`` (N, 3, 3) to ``r_des`` (3, 3) or (N, 3, 3), world frame."""
    e = np.zeros(r_now.shape[:-2] + (3,))
    r_des = np.broadcast_to(r_des, r_now.shape)
    for k in range(3):
        e += np.cross(r_now[..., :, k], r_des[..., :, k])
    return 0.5 * e


def ik_step(q: np.ndarray, p_des: np.ndarray, r_des: np.ndarray, *, tool: float = GRASP_Z, max_lin: float = 0.01,
            max_dq: float = 0.05, damping: float = 0.05, posture: float = 0.02,
            rest: np.ndarray | None = None, rot_gain: float = 1.0) -> np.ndarray:
    """One damped least-squares step dq (N, 7) moving the grasp point toward p_des (N, 3) and the hand toward r_des.

    The Cartesian step is capped at ``max_lin`` m, the joint step at ``max_dq`` rad (scaled as a whole, so the
    direction is kept); a small null-space term pulls toward ``rest`` (default ``Q_HOME``)."""
    f = fk(q, tool)
    err = np.asarray(p_des, np.float64) - f.grasp
    norm = np.linalg.norm(err, axis=1, keepdims=True)
    err = err * np.minimum(1.0, max_lin / np.maximum(norm, 1e-9))
    twist = np.concatenate([err, rot_gain * np.clip(rot_error(r_des, f.hand_rot), -0.1, 0.1)], axis=1)
    jac = jacobian(f, f.grasp)
    gram = jac @ jac.transpose(0, 2, 1) + damping**2 * np.eye(6)
    pinv = jac.transpose(0, 2, 1) @ np.linalg.inv(gram)
    dq = (pinv @ twist[:, :, None])[:, :, 0]
    null = np.eye(7) - pinv @ jac
    dq = dq + posture * (null @ ((Q_HOME if rest is None else rest) - q)[:, :, None])[:, :, 0]
    scale = np.minimum(1.0, max_dq / np.maximum(np.abs(dq).max(axis=1, keepdims=True), 1e-9))
    return dq * scale


def solve_ik(p_des: np.ndarray, r_des: np.ndarray = DOWN, q0: np.ndarray = Q_HOME, iters: int = 300,
             tool: float = GRASP_Z) -> np.ndarray:
    """Joint angles (7,) whose grasp point is at p_des (3,) with hand rotation r_des; raises if it does not converge."""
    q = np.asarray(q0, np.float64)[None, :].copy()
    for _ in range(iters):
        q = np.clip(q + ik_step(q, np.asarray(p_des)[None, :], r_des, tool=tool, max_lin=0.05, max_dq=0.2), Q_LOW,
                    Q_HIGH)
    f = fk(q, tool)
    miss = float(np.linalg.norm(f.grasp[0] - p_des)) + float(np.abs(rot_error(r_des, f.hand_rot)).max())
    if miss > 1e-3:
        raise ValueError(f"no IK solution for {p_des}: residual {miss:.4f}")
    return q[0]
