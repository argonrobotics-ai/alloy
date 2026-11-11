# SPDX-License-Identifier: Apache-2.0
"""Point mobilities of the vehicle plug-in's wheel contacts, for the tyre step bounds (``check_dt``) and the low-speed
tread dampers.

A wheel on the chassis sees the chassis's own point mobility ``1/m + (r×d)ᵀ·I⁻¹·(r×d)`` (as in v1). A wheel on a body
hinged to the chassis by a revolute joint (an oscillating steer axle) sees the whole vehicle as one rigid body, plus
the hinge's own freedom: ``((e × (c - p))·d)² / I_e``, with ``e`` the hinge axis through the pin ``p``, ``c`` the
contact and ``I_e`` the body's inertia about the hinge (plus the joint's armature). Its free-body mobility would ignore
the pin (a thin axle pitching about its own y axis) and overstate the response many times over. Any other wheel body
takes its free-body mobility, an upper bound. Everything is evaluated at the build pose, the vehicle level.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from alloy._vendor import newton

from alloy.plugins.removal.stability import body_of

log = logging.getLogger("alloy.plugins.vehicle")


def _pose(model: newton.Model, b: int) -> tuple[np.ndarray, np.ndarray]:
    if b < 0:                                            # the world (a joint's parent -1)
        return np.zeros(3), np.eye(3)
    q = np.asarray(model.body_q.numpy()[b], np.float64)
    return q[:3], Rotation.from_quat(q[3:]).as_matrix()


def _world_body(model: newton.Model, b: int) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, world COM, world inertia about the COM) at the build pose."""
    body = body_of(model, b, "wheel body")
    x, rot = _pose(model, b)
    return body.mass, x + rot @ body.com, rot @ body.inertia @ rot.T


def _composite(model: newton.Model, bodies: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
    parts = [_world_body(model, b) for b in bodies]
    mass = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in parts:
        d = c - com
        inertia += i + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def _hinge(model: newton.Model, child: int, parent: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """(axis e, pin p, I_e) of the revolute joint from ``parent`` to ``child``, or None if there is none."""
    kids, parents = model.joint_child.numpy(), model.joint_parent.numpy()
    types, starts = model.joint_type.numpy(), model.joint_qd_start.numpy()
    for j in np.flatnonzero((kids == child) & (parents == parent)):
        if int(types[j]) != int(newton.JointType.REVOLUTE):
            continue
        xp = np.asarray(model.joint_X_p.numpy()[j], np.float64)
        x, rot = _pose(model, parent)
        joint_rot = rot @ Rotation.from_quat(xp[3:]).as_matrix()
        dof = int(starts[j])
        axis = joint_rot @ np.asarray(model.joint_axis.numpy()[dof], np.float64)
        axis /= np.linalg.norm(axis)
        pin = x + rot @ xp[:3]
        m, com, inertia = _world_body(model, child)
        s = com - pin
        about = inertia + m * (float(s @ s) * np.eye(3) - np.outer(s, s))
        return axis, pin, float(axis @ about @ axis) + float(model.joint_armature.numpy()[dof])
    return None


def _cross_matrix(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _point(mass: float, com: np.ndarray, inertia: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The 3x3 point mobility at ``c``: ``1/m·E + [r]×ᵀ·I⁻¹·[r]×`` (velocity per unit force there, world frame)."""
    cr = _cross_matrix(c - com)
    return np.eye(3) / mass + cr.T @ np.linalg.solve(inertia, cr)


def wheel_mobility(model: newton.Model, index: np.ndarray, chassis: np.ndarray, specs: list[Any]) -> np.ndarray:
    """``[W, N, 4]``: each wheel contact's point mobility at the build pose, in its body's frame: the in-plane 2x2
    block (``w_xx, w_xy, w_yy``) and ``w_zz``. The kernels turn the 2x2 into the steered tread frame (``fwdᵀ·W·fwd``,
    ``latᵀ·W·lat``); ``check_dt`` bounds the treads with its largest eigenvalue, which covers every steer angle."""
    out = np.zeros((*index.shape, 4), np.float64)
    for w in range(index.shape[0]):
        root = int(chassis[w])
        carriers = sorted({int(b) for b in index[w]} | {root})
        whole = _composite(model, carriers)
        for i, s in enumerate(specs):
            b = int(index[w, i])
            x, rot = _pose(model, b)
            contact = x + rot @ np.asarray(s.center, np.float64) - np.array([0.0, 0.0, s.radius])
            hinge = None if b == root else _hinge(model, b, root)
            if b != root and hinge is None:
                log.warning("vehicle: wheel %d's body %d is not hinged to the chassis by a revolute joint: its "
                            "free-body mobility bounds the step", i, b)
            if b == root or hinge is None:
                mob = _point(*_world_body(model, b), contact)
            else:
                axis, pin, i_e = hinge
                g = np.cross(axis, contact - pin)
                mob = _point(*whole, contact) + np.outer(g, g) / i_e
            local = rot.T @ mob @ rot
            out[w, i] = (local[0, 0], local[0, 1], local[1, 1], local[2, 2])
    return out


def in_plane_max(mob: np.ndarray) -> np.ndarray:
    """``[W, N]``: the largest eigenvalue of each wheel's in-plane 2x2 mobility (the worst steer angle)."""
    a, b, c = mob[..., 0], mob[..., 1], mob[..., 2]
    return 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)


__all__ = ["in_plane_max", "wheel_mobility"]
