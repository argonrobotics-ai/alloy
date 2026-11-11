# SPDX-License-Identifier: Apache-2.0
"""Resolve a ``Scene.randomize`` robot row's target to model indices (template ids; a world's are id + world x stride).

``"go2"`` is every link, joint and geom the robot's file made; ``"go2/FL_calf"`` one link (and its geoms);
``"go2/FL_calf_joint"`` one joint (and its actuator); ``"gravity"`` has no indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.api import _robot_random as R
from alloy.api.errors import BuildError, UnknownNameError

FREE_JOINT = 4      # newton.JointType.FREE
_EMPTY = np.zeros(0, np.int64)


@dataclass
class Target:
    """Template indices one row writes. ``servo_rows``: MuJoCo actuator rows that are joint servos reading ctrl
    (``velocity_rows``: the velocity servos among them); ``target_dofs``: dofs driven from joint targets."""

    stride: dict[str, int]
    bodies: np.ndarray = field(default_factory=lambda: _EMPTY)
    shapes: np.ndarray = field(default_factory=lambda: _EMPTY)
    dofs: np.ndarray = field(default_factory=lambda: _EMPTY)
    servo_rows: np.ndarray = field(default_factory=lambda: _EMPTY)
    velocity_rows: frozenset[int] = frozenset()
    target_dofs: np.ndarray = field(default_factory=lambda: _EMPTY)


def resolve(built: Any, idx: Any, row: dict[str, Any]) -> Target:
    """The indices of ``row`` in ``built``; a ``BuildError`` / ``UnknownNameError`` that names the fix otherwise.

    Example: ``resolve(built, NameIndex.from_built(built), {"target": "go2", "field": "mass_scale", ...})``."""
    model, worlds = built.model, int(built.world_count)
    name, fld = row["target"], row["field"]
    if name == R.GRAVITY:
        _gravity_ok(model, worlds)
        return Target({})
    robot = name.split("/", 1)[0]
    handle = built.handles.get(robot)
    if handle is None or robot not in idx.robot_joints:
        raise BuildError(f"randomize: {robot!r} is not a robot in the built scene", "name a robot added with "
                         "scene.add_robot (objects take density, friction, youngs, x, y, z)", field="name")
    acts = _actuators(model, idx)
    stride = {"body": idx.stride["body"], "shape": len(handle.world_stride.shape),
              "dof": int(model.joint_dof_count) // worlds, "actuator": acts.stride}
    if fld in R.START_FIELDS:
        from alloy.api._robot_random_start import free_root  # noqa: PLC0415

        free_root(idx, model, robot)                           # a welded base cannot be placed: says so, with the fix
        return Target(stride)
    if fld in R.BODY_FIELDS:
        return _body_target(model, idx, handle, name, fld, stride)
    return _joint_target(model, idx, handle, acts, name, fld, stride)


def _gravity_ok(model: Any, worlds: int) -> None:
    if int(getattr(model, "particle_count", 0)):
        raise BuildError("gravity cannot be randomized in a scene with cloth, liquid, soft bodies, sand or paste "
                         "(their solvers use one gravity for all worlds)", "randomize gravity in scenes of rigid "
                         "objects and robots only", field="name")
    g = model.gravity
    if g is None or g.shape[0] < worlds:
        raise BuildError("this model has no per-world gravity", "build with the bundled Newton", field="name")


def _actuators(model: Any, idx: Any) -> Any:
    from alloy.api._actuators import Actuators  # noqa: PLC0415

    return Actuators.from_model(model, idx.robot_joints, model.joint_qd_start.numpy())


def _body_target(model: Any, idx: Any, handle: Any, name: str, fld: str, stride: dict[str, int]) -> Target:
    t = handle.template
    whole = "/" not in name
    if not whole and name not in idx.bodies:
        raise _unknown(idx, name, "link", "bodies")
    if whole and fld not in R.ONE_LINK:
        bodies = np.asarray(list(t.body), np.int64)
        shapes = np.asarray(list(t.shape), np.int64)
    else:
        bodies = np.asarray([idx.bodies[name]], np.int64)
        owner = model.shape_body.numpy()
        shapes = np.asarray([s for s in t.shape if int(owner[s]) == int(bodies[0])], np.int64)
    if fld == "friction" and not len(shapes):
        raise BuildError(f"randomize: {name!r} has no collision geoms to give a friction",
                         "name the robot, or a link that has geoms", field="name")
    return Target(stride, bodies=bodies, shapes=shapes if fld == "friction" else _EMPTY)


def _joint_target(model: Any, idx: Any, handle: Any, acts: Any, name: str, fld: str, stride: dict[str, int]) -> Target:
    kind, qd = model.joint_type.numpy(), model.joint_qd_start.numpy()
    if "/" in name:
        if name not in idx.joints:
            raise _unknown(idx, name, "joint", "joints")
        joints = [idx.joints[name]]
    else:
        joints = [j for j in handle.template.joint if int(kind[j]) != FREE_JOINT and qd[j + 1] > qd[j]]
    dofs = np.asarray([d for j in joints for d in range(int(qd[j]), int(qd[j + 1]))], np.int64)
    if fld in ("armature", "armature_scale"):
        if not len(dofs):
            raise BuildError(f"randomize: {name!r} has no moving joint to give an armature",
                             "name a robot with joints, or one of its hinge or slide joints", field="name")
        return Target(stride, dofs=dofs)
    position = {j: acts.position[j] for j in joints if j in acts.position}
    velocity = {j: acts.velocity[j] for j in joints if j in acts.velocity}
    wanted = velocity if fld == "kd_scale" and not position else position
    target = [j for j in joints if j in acts.joint_target]
    if not wanted and not target:
        raise BuildError(f"randomize: {name!r} has no joint servo for {fld} to scale (kp_scale needs a position "
                         "servo; kd_scale a position or velocity servo)", "add an actuator to the joint in the "
                         "robot file, e.g. <position joint='...' kp='200' kv='10'/>", field="name")
    rows = sorted({*position.values(), *velocity.values()})
    return Target(stride, servo_rows=np.asarray(rows, np.int64), velocity_rows=frozenset(velocity.values()),
                  target_dofs=np.asarray([d for j in target for d in range(int(qd[j]), int(qd[j + 1]))], np.int64))


def _unknown(idx: Any, name: str, what: str, kind: str) -> UnknownNameError:
    robot = name.split("/", 1)[0]
    valid = sorted(n for n in getattr(idx, kind) if n.startswith(robot + "/"))
    return UnknownNameError(f"randomize: {name!r} is not a {what} of robot {robot!r}",
                            f"use one of {valid} (sim.names() lists them)", field="name")


__all__ = ["Target", "resolve"]
