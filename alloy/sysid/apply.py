# SPDX-License-Identifier: Apache-2.0
"""``scene.add_robot(..., sysid="my_robot.sysid.json")``: give a robot its fitted joint parameters at build.

``item_value`` checks the file when the robot is added and keeps its ``joints`` block on the scene item (so the
scene's JSON carries the values and needs no file). At build, ``install`` writes the values into the model of every
world before the solvers are made and before ``scene.randomize`` rows are drawn: those rows then scale the fitted
values. ``attach`` puts the fitted action delay in front of the robot's servos: each control period the commanded
targets are read, the delayed ones are stepped, and the commanded ones are put back (host work, outside any
captured graph: two array copies per control period for a robot with a delay). The first command after the build
or a reset is delayed as well: until it arrives the servos hold each joint where it is at that moment.

A scene without ``sysid=`` never imports this module and builds and steps exactly as before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.api.errors import BuildError
from alloy.sysid import model as M
from alloy.sysid.delay import SimDelay
from alloy.sysid.result import load

log = logging.getLogger("alloy.sysid")


def item_value(name: str, sysid: Any) -> dict[str, Any]:
    """What ``Scene.add_robot(name, ..., sysid=)`` stores: ``{"robot", "joints": {joint: {field: value}}}``.

    Example: ``item_value("panda", "my_panda.sysid.json")["joints"]["joint1"]["kp"]``."""
    data = load(sysid)
    return {"robot": data["robot"], "joints": data["joints"]}


def _robots(scene: Any) -> dict[str, dict[str, dict[str, float]]]:
    return {n: i["sysid"]["joints"] for n, i in scene.items.items() if i.get("kind") == "robot" and i.get("sysid")}


def refuse_rerun_worlds(scene: Any, rerun_worlds: Any) -> None:
    """``alloy.build(..., rerun_worlds=n)`` re-runs worlds in a side simulation that has no action delay."""
    if rerun_worlds and _robots(scene):
        raise BuildError("rerun_worlds cannot be used with add_robot(sysid=) (the side simulation that re-runs "
                         "worlds would not carry the fitted action delay)", "build with rerun_worlds=0 (the default)",
                         field="rerun_worlds")


def _columns(name: str, joints: M.RobotJoints, fitted: dict[str, dict[str, float]]) -> None:
    unknown = [j for j in fitted if j not in joints.names]
    if unknown:
        raise BuildError(f"robot {name!r}: its sysid file names joint {unknown[0]!r}, which is not a "
                         f"position-servoed joint of this robot", f"the robot's servoed joints are {joints.names}; "
                         "fit the file with this robot (python -m alloy.sysid fit --robot ...)", field="sysid")


def install(scene: Any, built: Any) -> None:
    """Write every robot's fitted values into ``built.model``, the same in every world.

    Example: ``install(scene, built)`` after assembly, before ``scene.randomize`` rows are drawn."""
    for name, fitted in _robots(scene).items():
        joints = M.RobotJoints.read(built, name)
        _columns(name, joints, fitted)
        M.limit_actuators(built.model, joints)
        for f in M.FIELDS:
            names = [j for j, vals in fitted.items() if f in vals]
            if names:
                M.write(built.model, joints, {f: np.asarray([fitted[j][f] for j in names])}, joints.pick(names))
        log.info("robot %s: fitted joint parameters of %d joints applied (sysid)", name, len(fitted))


@dataclass
class _Rows:
    """The delayed control rows of one kind (``ctrl`` or ``position``), a column per joint, a row per world."""

    rows: list[np.ndarray] = field(default_factory=list)       # control row of the joint in every world
    seconds: list[float] = field(default_factory=list)
    coords: list[np.ndarray] = field(default_factory=list)     # the joint's coordinate in state.joint_q
    gear: list[float] = field(default_factory=list)            # control value = gear x (joint position + ref)
    ref: list[np.ndarray] = field(default_factory=list)

    def hold(self, engine: Any) -> np.ndarray:
        """(worlds, joints) control values that hold every joint where it is now."""
        q = engine.state.joint_q.numpy().astype(np.float64)
        return np.stack([g * (q[c] + r) for c, g, r in zip(self.coords, self.gear, self.ref)], axis=1)


def _delayed_rows(scene: Any, built: Any) -> dict[str, _Rows]:
    from alloy.api._actuators import Actuators  # noqa: PLC0415
    from alloy.api.names import NameIndex  # noqa: PLC0415

    model, worlds = built.model, int(built.world_count)
    world = np.arange(worlds)
    qd_start = model.joint_qd_start.numpy()
    acts = Actuators.from_model(model, NameIndex.from_built(built).robot_joints, qd_start)
    start = getattr(model, "joint_target_q_start", None)
    start = start.numpy() if start is not None else model.joint_q_start.numpy()
    dof_ref = getattr(getattr(model, "mujoco", None), "dof_ref", None)
    dof_ref = None if dof_ref is None else dof_ref.numpy()
    out = {"ctrl": _Rows(), "position": _Rows()}
    for name, fitted in _robots(scene).items():
        joints = M.RobotJoints.read(built, name)
        for j, vals in fitted.items():
            if float(vals.get("delay", 0.0)) <= 0.0:
                continue
            col = int(joints.pick([j])[0])
            row, dof = int(joints.servo_rows[col]), int(joints.dofs[col])
            kind = out["ctrl" if row >= 0 else "position"]
            if row >= 0:
                kind.rows.append(row + world * joints.stride["actuator"])
                kind.gear.append(acts.gear_of(row))
                kind.ref.append(np.zeros(worlds) if dof_ref is None else dof_ref[dof + world * joints.stride["dof"]])
            else:
                joint = int(np.searchsorted(qd_start, dof, side="right")) - 1
                kind.rows.append(start[joint + world * (int(model.joint_count) // worlds)])
                kind.gear.append(1.0)
                kind.ref.append(np.zeros(worlds))
            kind.coords.append(int(joints.q_index[col]) + world * joints.stride["coord"])
            kind.seconds.append(float(vals["delay"]))
    return {k: v for k, v in out.items() if v.rows}


def attach(scene: Any, engine: Any) -> None:
    """Put the fitted action delays in front of the servos of ``engine`` (nothing when no joint has one).

    Example: ``attach(scene, engine)`` once the simulation is wired."""
    built = engine.built
    worlds, dt = int(built.world_count), float(built.control_dt)
    kinds = _delayed_rows(scene, built)
    if not kinds:
        return
    delay = SimDelay(engine, {k: np.stack(v.rows, axis=1) for k, v in kinds.items()},
                     {k: np.broadcast_to(np.asarray(v.seconds), (worlds, len(v.seconds))) for k, v in kinds.items()},
                     dt, {k: (lambda v=v: v.hold(engine)) for k, v in kinds.items()})
    inner = engine.step_period

    def step_period() -> None:
        delay.before()
        try:
            inner()
        finally:
            delay.after()

    engine.step_period = step_period
    engine.reset_hooks.append(delay.reset)
    engine.sysid_delay = delay
    log.info("sysid: action delay on %d joints (largest %.1f ms)", sum(len(v.seconds) for v in kinds.values()),
             1e3 * max(max(v.seconds) for v in kinds.values()))


__all__ = ["attach", "install", "item_value", "refuse_rerun_worlds"]
