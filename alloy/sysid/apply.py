# SPDX-License-Identifier: Apache-2.0
"""``scene.add_robot(..., sysid="my_robot.sysid.json")``: give a robot its fitted joint parameters at build.

``item_value`` checks the file when the robot is added and keeps its ``joints`` block on the scene item (so the
scene's JSON carries the values and needs no file). At build, ``install`` writes the values into the model of every
world before the solvers are made and before ``scene.randomize`` rows are drawn: those rows then scale the fitted
values. ``attach`` puts the fitted action delay in front of the robot's servos: each control period the commanded
targets are read, the delayed ones are stepped, and the commanded ones are put back (host work, outside any
captured graph: two array copies per control period for a robot with a delay).

A scene without ``sysid=`` never imports this module and builds and steps exactly as before.
"""

from __future__ import annotations

import logging
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
        for f in M.FIELDS:
            names = [j for j, vals in fitted.items() if f in vals]
            if names:
                M.write(built.model, joints, {f: np.asarray([fitted[j][f] for j in names])}, joints.pick(names))
        log.info("robot %s: fitted joint parameters of %d joints applied (sysid)", name, len(fitted))


def attach(scene: Any, engine: Any) -> None:
    """Put the fitted action delays in front of the servos of ``engine`` (nothing when no joint has one).

    Example: ``attach(scene, engine)`` once the simulation is wired."""
    built = engine.built
    worlds, dt = int(built.world_count), float(built.control_dt)
    rows: dict[str, list[np.ndarray]] = {"ctrl": [], "position": []}
    seconds: dict[str, list[float]] = {"ctrl": [], "position": []}
    world = np.arange(worlds)
    for name, fitted in _robots(scene).items():
        joints = M.RobotJoints.read(built, name)
        start = getattr(built.model, "joint_target_q_start", None)
        start = start.numpy() if start is not None else built.model.joint_q_start.numpy()
        for j, vals in fitted.items():
            if float(vals.get("delay", 0.0)) <= 0.0:
                continue
            col = int(joints.pick([j])[0])
            row = int(joints.servo_rows[col])
            if row >= 0:
                rows["ctrl"].append(row + world * joints.stride["actuator"])
            else:
                joint = int(np.searchsorted(built.model.joint_qd_start.numpy(), joints.dofs[col], side="right")) - 1
                rows["position"].append(start[joint + world * (int(built.model.joint_count) // worlds)])
            seconds["ctrl" if row >= 0 else "position"].append(float(vals["delay"]))
    kinds = [k for k in rows if rows[k]]
    if not kinds:
        return
    delay = SimDelay(engine, {k: np.stack(rows[k], axis=1) for k in kinds},
                     {k: np.broadcast_to(np.asarray(seconds[k]), (worlds, len(seconds[k]))) for k in kinds}, dt)
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
    log.info("sysid: action delay on %d joints (largest %.1f ms)", sum(len(seconds[k]) for k in kinds),
             1e3 * max(max(seconds[k]) for k in kinds))


__all__ = ["attach", "install", "item_value", "refuse_rerun_worlds"]
