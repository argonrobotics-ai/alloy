# SPDX-License-Identifier: Apache-2.0
"""A robot's joint parameters in a built model: read the file's values, write new ones per world.

``RobotJoints.read(built, robot)`` lists the robot's position-servoed hinge and slide joints with the values the
robot file gave them; ``write(model, joints, values)`` sets values for every world (one per world when the array is
``(worlds, joints)``): the fit uses it to give each world a candidate, ``scene.add_robot(sysid=)`` to give every
world the fitted ones. Fields (``FIELDS``), all per joint:

==============  ==================  ==================================================================
field           unit                what it is in the model
==============  ==================  ==================================================================
``kp``          N·m/rad (N/m)       the position servo's stiffness
``kd``          N·m·s/rad (N·s/m)   the servo's velocity gain
``damping``     N·m·s/rad (N·s/m)   the joint's own viscous damping (MJCF ``damping``)
``friction``    N·m (N)             dry friction (MJCF ``frictionloss``)
``armature``    kg·m² (kg)          rotor inertia seen at the joint (MJCF ``armature``)
``torque_limit``  N·m (N)           cap on the actuator's force at the joint
==============  ==================  ==================================================================

The torque limit of a robot file can sit on the joint (URDF ``effort``, MJCF ``actuatorfrcrange``) or on the
actuator (MJCF ``forcerange``); the smaller is the file's value, and ``write`` sets both. A robot built with
``add_robot(sysid=)`` gets a force range on every servo actuator at build (``limit_actuators``), so the limit can
also be raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.sysid.errors import SysidError

FIELDS = ("kp", "kd", "damping", "friction", "armature", "torque_limit")
DOF_ARRAYS = {"damping": "joint_damping", "friction": "joint_friction", "armature": "joint_armature",
              "torque_limit": "joint_effort_limit"}
NO_LIMIT = 1.0e5            # N·m: an effort limit at or above this is "none" (the importer's default is 1e6)
NO_RANGE = 1.0e6            # N·m written for "no limit"
JOINT_TARGET = 0            # the solver's ctrl source of an actuator driven from joint targets
PRISMATIC, REVOLUTE = 0, 1  # newton.JointType


@dataclass
class RobotJoints:
    """The servoed single-axis joints of one robot (template = world 0 indices) and their nominal values."""

    robot: str
    names: list[str]                       # joint names as in the robot file (no "<robot>/" prefix)
    dofs: np.ndarray                       # (J,) template dof of each joint
    q_index: np.ndarray                    # (J,) template coordinate of each joint in joint_q
    servo_rows: np.ndarray                 # (J,) MuJoCo actuator row that reads ctrl, or -1 (driven from joint targets)
    hinge: np.ndarray                      # (J,) bool: an angle (rad); False: a slide (m)
    force_rows: np.ndarray                 # (J,) the joint's one actuator row (its force range applies), or -1
    gear: np.ndarray                       # (J,) that actuator's gear (joint torque = gear x actuator force)
    lower: np.ndarray                      # (J,) joint range, -inf / inf without one
    upper: np.ndarray
    stride: dict[str, int] = field(default_factory=dict)    # dof / coord / actuator rows per world
    nominal: dict[str, np.ndarray] = field(default_factory=dict)   # field -> (J,) the robot file's values

    def full(self, joint: str) -> str:
        """The joint's name in the sim. Example: ``joints.full("elbow")`` -> ``"arm/elbow"``."""
        return f"{self.robot}/{joint}"

    def pick(self, names: list[str]) -> np.ndarray:
        """Column of each of ``names``; a ``SysidError`` listing the valid names for an unknown one."""
        missing = [n for n in names if n not in self.names]
        if missing:
            raise SysidError(f"robot {self.robot!r} has no position-servoed joint {missing[0]!r}",
                             f"its servoed joints are {self.names}", field="joints")
        return np.asarray([self.names.index(n) for n in names], np.int64)

    @classmethod
    def read(cls, built: Any, robot: str) -> RobotJoints:
        """Index ``robot`` in a built scene. Example: ``RobotJoints.read(sim.built, "panda").names``."""
        from alloy.api._actuators import Actuators  # noqa: PLC0415
        from alloy.api.names import NameIndex  # noqa: PLC0415

        model, worlds = built.model, int(built.world_count)
        idx = NameIndex.from_built(built)
        if robot not in idx.robot_joints:
            raise SysidError(f"{robot!r} is not a robot in this scene", f"robots: {sorted(idx.robot_joints)}",
                             field="robot")
        qd, q0, kind = model.joint_qd_start.numpy(), model.joint_q_start.numpy(), model.joint_type.numpy()
        acts = Actuators.from_model(model, idx.robot_joints, qd)
        ke = model.joint_target_ke.numpy()
        by_id = {j: n for n, j in idx.joints.items() if n.startswith(robot + "/")}
        names, dofs, coords, rows, force = [], [], [], [], []
        for j in idx.robot_joints[robot]:
            single = int(kind[j]) in (PRISMATIC, REVOLUTE)
            row = acts.position.get(j, -1)
            driven = row >= 0 or (j in acts.joint_target and float(ke[int(qd[j])]) > 0.0) \
                or (not acts.stride and float(ke[int(qd[j])]) > 0.0)
            if single and driven and j in by_id:
                names.append(by_id[j].split("/", 1)[1])
                dofs.append(int(qd[j]))
                coords.append(int(q0[j]))
                rows.append(int(row))
                force.append(_only_row(acts, int(qd[j])))
        dofs_a = np.asarray(dofs, np.int64)
        joint_of = np.searchsorted(qd, dofs_a, side="right") - 1
        out = cls(robot, names, dofs_a, np.asarray(coords, np.int64), np.asarray(rows, np.int64),
                  kind[joint_of] == REVOLUTE if len(dofs) else np.zeros(0, bool),
                  np.asarray(force, np.int64), np.asarray([acts.gear_of(r) if r >= 0 else 1.0 for r in force]),
                  model.joint_limit_lower.numpy()[dofs_a].astype(np.float64),
                  model.joint_limit_upper.numpy()[dofs_a].astype(np.float64),
                  {"dof": int(model.joint_dof_count) // worlds, "coord": int(model.joint_coord_count) // worlds,
                   "actuator": acts.stride})
        unranged = ~(out.lower < out.upper) | (np.abs(out.lower) > 1e6) | (np.abs(out.upper) > 1e6)
        out.lower[unranged], out.upper[unranged] = -np.inf, np.inf
        out.nominal = _nominal(model, out)
        return out


def _only_row(acts: Any, dof: int) -> int:
    """The actuator row on ``dof`` when exactly one actuator drives it, else -1."""
    rows = [r for r, d in acts.dof_of_row.items() if d == dof]
    return rows[0] if len(rows) == 1 else -1


def _mj(model: Any, name: str) -> np.ndarray | None:
    arr = getattr(getattr(model, "mujoco", None), name, None)
    return None if arr is None else arr.numpy()


def _actuator_caps(model: Any, joints: RobotJoints) -> np.ndarray:
    """(J,) joint torque at which each joint's actuator saturates (MJCF ``forcerange`` x gear), inf without one."""
    cap = np.full(len(joints.names), np.inf)
    span, limited, has = (_mj(model, n) for n in ("actuator_forcerange", "actuator_forcelimited",
                                                  "actuator_has_forcerange"))
    rows = joints.force_rows
    at = rows >= 0
    if span is None or limited is None or not at.any():
        return cap
    r = rows[at]
    on = (limited[r] != 0) & (span[r, 0] < span[r, 1]) & (True if has is None else has[r].astype(bool))
    cap[at] = np.where(on, np.abs(span[r]).max(axis=1) * np.abs(joints.gear[at]), np.inf)
    return cap


def _nominal(model: Any, joints: RobotJoints) -> dict[str, np.ndarray]:
    d, rows = joints.dofs, joints.servo_rows
    direct = rows >= 0
    out = {f: getattr(model, a).numpy()[d].astype(np.float64) for f, a in DOF_ARRAYS.items()}
    kp, kd = model.joint_target_ke.numpy()[d].astype(np.float64), model.joint_target_kd.numpy()[d].astype(np.float64)
    gain, bias = _mj(model, "actuator_gainprm"), _mj(model, "actuator_biasprm")
    if direct.any():
        kp[direct], kd[direct] = gain[rows[direct], 0], -bias[rows[direct], 2]
    limit = np.minimum(out["torque_limit"], _actuator_caps(model, joints))
    out["torque_limit"] = np.where(limit >= NO_LIMIT, np.inf, limit)
    return {"kp": kp, "kd": kd, **out}


def limit_actuators(model: Any, joints: RobotJoints) -> None:
    """Give the one actuator of each servoed joint a force range, in every world: the file's own, or one too
    wide to act when the file has none. Nothing moves differently; ``write`` can then lower or raise the torque
    limit, also in a running simulation. Call before the solvers are made.

    Example: ``limit_actuators(built.model, RobotJoints.read(built, "panda"))``."""
    cols = np.flatnonzero(joints.force_rows >= 0)
    mj = getattr(model, "mujoco", None)
    if not len(cols) or mj is None or not hasattr(mj, "actuator_has_forcerange"):
        return
    cap = _actuator_caps(model, joints)[cols]
    _write_force_range(model, joints, cols, np.where(np.isfinite(cap), cap, NO_RANGE)[None, :])
    rows = _world_rows(model, joints, joints.force_rows[cols], "actuator").ravel()
    for name in ("actuator_has_forcerange", "actuator_forcelimited"):
        host = getattr(mj, name).numpy()
        host[rows] = 1
        getattr(mj, name).assign(host)


def _world_rows(model: Any, joints: RobotJoints, rows: np.ndarray, what: str) -> np.ndarray:
    """(worlds, n) rows of every world for world-0 ``rows`` of ``what`` ("dof", "coord" or "actuator")."""
    worlds = int(model.joint_dof_count) // joints.stride["dof"]
    return rows[None, :] + np.arange(worlds)[:, None] * joints.stride[what]


def _write_force_range(model: Any, joints: RobotJoints, cols: np.ndarray, torque: np.ndarray) -> None:
    """Set the actuator force range of joints ``cols`` (those with one actuator) to +-``torque`` / gear."""
    has = joints.force_rows[cols] >= 0
    span = getattr(getattr(model, "mujoco", None), "actuator_forcerange", None)
    if span is None or not has.any():
        return
    rows = _world_rows(model, joints, joints.force_rows[cols][has], "actuator")
    force = np.broadcast_to(torque, (rows.shape[0], len(cols)))[:, has] / np.abs(joints.gear[cols][has])[None, :]
    host = span.numpy()
    host[rows, 0], host[rows, 1] = -force, force
    span.assign(host)


def write(model: Any, joints: RobotJoints, values: dict[str, np.ndarray], cols: np.ndarray | None = None) -> None:
    """Set ``values[field]`` (``(worlds, n)`` or ``(n,)`` for every world) on joints ``cols`` (all when None) of
    every world of ``model``. Unknown fields raise. Tell running solvers with ``notify`` afterwards.

    Example: ``write(model, joints, {"kp": np.full((worlds, len(joints.names)), 300.0)})``."""
    cols = np.arange(len(joints.names)) if cols is None else np.asarray(cols, np.int64)
    worlds = int(model.joint_dof_count) // joints.stride["dof"]
    w = np.arange(worlds)[:, None]
    vals = {}
    for f, v in values.items():
        if f not in FIELDS:
            raise SysidError(f"{f!r} is not a joint parameter", f"use one of {FIELDS}", field="field")
        vals[f] = np.broadcast_to(np.asarray(v, np.float64), (worlds, len(cols)))
    dof = joints.dofs[cols][None, :] + w * joints.stride["dof"]
    if "torque_limit" in vals:
        vals["torque_limit"] = np.where(np.isfinite(vals["torque_limit"]), vals["torque_limit"], NO_RANGE)
        _write_force_range(model, joints, cols, vals["torque_limit"])
    for f, a in DOF_ARRAYS.items():
        if f in vals:
            host = getattr(model, a).numpy()
            host[dof] = vals[f]
            getattr(model, a).assign(host)
    if "kp" in vals or "kd" in vals:
        _write_gains(model, joints, cols, vals, dof, w)


def _write_gains(model: Any, joints: RobotJoints, cols: np.ndarray, vals: dict[str, np.ndarray], dof: np.ndarray,
                 w: np.ndarray) -> None:
    rows = joints.servo_rows[cols]
    direct, target = rows >= 0, rows < 0
    if direct.any():
        mj = model.mujoco
        gain, bias = mj.actuator_gainprm.numpy(), mj.actuator_biasprm.numpy()
        g = rows[direct][None, :] + w * joints.stride["actuator"]
        if "kp" in vals:
            gain[g, 0], bias[g, 1] = vals["kp"][:, direct], -vals["kp"][:, direct]
        if "kd" in vals:
            bias[g, 2] = -vals["kd"][:, direct]
        mj.actuator_gainprm.assign(gain)
        mj.actuator_biasprm.assign(bias)
    if target.any():
        for f, a in (("kp", "joint_target_ke"), ("kd", "joint_target_kd")):
            if f in vals:
                host = getattr(model, a).numpy()
                host[dof[:, target]] = vals[f][:, target]
                getattr(model, a).assign(host)


def notify(engine: Any) -> None:
    """Tell the running solvers that joint and actuator values changed. Example: ``notify(sim.engine)``."""
    from alloy._vendor import newton  # noqa: PLC0415

    flags = int(newton.ModelFlags.ACTUATOR_PROPERTIES) | int(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    engine._notify(flags)
    for solver in getattr(engine.stepper, "solvers", lambda: [])():
        _sync_target_force_ranges(engine.model, solver)


def _sync_target_force_ranges(model: Any, solver: Any) -> None:
    """The solver's notification leaves the force range of ``<position>`` / ``<velocity>`` actuators (driven from
    joint targets) as built; copy each world's range from the model for those."""
    source = getattr(solver, "mjc_actuator_ctrl_source", None)
    index = getattr(solver, "mjc_actuator_to_newton_actuator_idx", None)
    span = getattr(getattr(model, "mujoco", None), "actuator_forcerange", None)
    held = getattr(getattr(solver, "mjw_model", None), "actuator_forcerange", None)
    if source is None or index is None or span is None or held is None:
        return
    source, index = source.numpy(), index.numpy()
    acts = np.flatnonzero((source == JOINT_TARGET) & (index >= 0))
    if not len(acts):
        return
    host, mine = held.numpy(), span.numpy()
    worlds = host.shape[0]
    host[:, acts] = mine.reshape(worlds, -1, 2)[:, index[acts]]
    held.assign(host)
    native = getattr(solver, "mj_model", None)
    if native is not None and getattr(solver, "use_mujoco_cpu", False):     # MuJoCo C keeps world 0's own copy
        native.actuator_forcerange[acts] = host[0, acts]


__all__ = ["FIELDS", "RobotJoints", "limit_actuators", "notify", "write"]
