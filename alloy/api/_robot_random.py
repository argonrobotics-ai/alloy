# SPDX-License-Identifier: Apache-2.0
"""``Scene.randomize`` for robots and gravity: the fields, their checks, and the rows a scene keeps.

A row is ``{"target", "field", "lo", "hi", "when"}`` in ``scene.robot_random``. ``target`` is a robot's name (every
link, joint or geom of its file), ``"<robot>/<link or joint>"`` (one of them), or ``"gravity"``. One value is drawn
per row and world, uniformly in [lo, hi]; ``when="reset"`` draws a new one each time ``sim.reset()`` runs.
``alloy.api._robot_random_apply`` writes the values into the built model.

Fields:

==================  =========  ==========================================================================
field               target     value
==================  =========  ==========================================================================
``mass_scale``      link(s)    factor on the link's mass and inertia
``mass_add``        one link   kg added to the link (its inertia scales with its mass)
``com_x/_y/_z``     one link   m added to the link's centre of mass, in the link's frame
``friction``        geoms      sliding friction coefficient of the geoms (absolute)
``kp_scale``        joint(s)   factor on the position servo's stiffness
``kd_scale``        joint(s)   factor on the servo's damping (a velocity servo's gain)
``armature_scale``  joint(s)   factor on the joint's armature
``armature``        joint(s)   the joint's armature, kg m^2 (absolute; m for a slide: kg)
``x``, ``y``, ``z``  gravity    that component of gravity, m/s^2 (absolute)
``start_x/_y/_z``   robot      m added to a free-base robot's start position at each reset (``when="reset"``)
``start_yaw``       robot      rad added to its start heading (about z) at each reset (``when="reset"``)
==================  =========  ==========================================================================
"""

from __future__ import annotations

from typing import Any

from alloy.api.errors import SceneError

GRAVITY = "gravity"
BODY_FIELDS = ("mass_scale", "mass_add", "com_x", "com_y", "com_z", "friction")
JOINT_FIELDS = ("kp_scale", "kd_scale", "armature_scale", "armature")
ONE_LINK = ("mass_add", "com_x", "com_y", "com_z")      # on a bare robot name: its root link
START_FIELDS = ("start_x", "start_y", "start_z", "start_yaw")      # a free-base robot, when="reset" only
FIELDS = BODY_FIELDS + JOINT_FIELDS + START_FIELDS
GRAVITY_AXES = {"x": 0, "y": 1, "z": 2}
POSITIVE = ("mass_scale", "kp_scale", "kd_scale", "armature_scale")      # lo must be > 0
NON_NEGATIVE = ("friction", "armature")
WHEN = ("build", "reset")
#: names people reach for -> the field to use
INSTEAD = {"mass": "mass_scale (a factor) or mass_add (kg)", "density": "mass_scale", "com": "com_x, com_y, com_z",
           "kp": "kp_scale", "stiffness": "kp_scale", "kd": "kd_scale", "damping": "kd_scale", "kv": "kd_scale",
           "gains": "kp_scale and kd_scale", "mu": "friction", "gravity": "scene.randomize('gravity', 'z', lo, hi)", "yaw": "start_yaw (when='reset')",
           "push": "scene.add_push(name, force=(fx, fy, fz), every=(lo, hi))"}


def is_robot_row(scene: Any, name: str, field: str) -> bool:
    """True when ``scene.randomize(name, field, ...)`` is a row of this module (gravity, a robot part, or a robot
    field other than its base position). Example: ``is_robot_row(scene, "go2", "mass_scale")`` -> True."""
    if name == GRAVITY and name not in scene.items:
        return True
    robot = name.split("/", 1)[0]
    item = scene.items.get(robot)
    if item is None or item["kind"] != "robot":
        return False
    return "/" in name or field not in ("x", "y", "z", "pose.0", "pose.1", "pose.2")


def add(scene: Any, name: str, field: str, lo: float, hi: float, when: str = "build") -> None:
    """Check and record one row. Example: ``add(scene, "go2", "mass_scale", 0.8, 1.2)``."""
    lo, hi = float(lo), float(hi)
    if when not in WHEN:
        raise SceneError(f"randomize: when={when!r} is not known", f"pass when='build' (drawn once, the default) or "
                         "when='reset' (drawn again at every sim.reset())", field="when")
    if name == GRAVITY:
        _check_gravity(field)
    else:
        _check_field(name, field, lo)
        _check_start(name, field, when)
    scene.robot_random.append({"target": name, "field": field, "lo": lo, "hi": hi, "when": when})


def _check_gravity(field: str) -> None:
    if field not in GRAVITY_AXES:
        raise SceneError(f"gravity: cannot randomize {field!r}", "randomize a component of gravity: 'x', 'y' or 'z' "
                         "(m/s^2, absolute: scene.randomize('gravity', 'z', -10.3, -9.3))", field="field")


def _check_field(name: str, field: str, lo: float) -> None:
    if field not in FIELDS:
        instead = INSTEAD.get(str(field).lower())
        hint = f"; use {instead}" if instead else ""
        raise SceneError(f"robot {name!r}: cannot randomize {field!r}{hint}",
                         f"use one of {sorted(FIELDS)}, or 'x', 'y', 'z' (a free-base robot's base position)",
                         field="field")
    if field in POSITIVE and not lo > 0.0:
        raise SceneError(f"{name!r}: {field} is a factor and must stay above 0, got lo={lo}",
                         "pass a range around 1, such as (0.8, 1.2)", field="lo")
    if field in NON_NEGATIVE and lo < 0.0:
        raise SceneError(f"{name!r}: {field} cannot be negative, got lo={lo}", "pass lo >= 0", field="lo")


def _check_start(name: str, field: str, when: str) -> None:
    if field not in START_FIELDS:
        return
    if "/" in name:
        raise SceneError(f"{name!r}: {field} moves a whole robot, not one link or joint",
                         f"name the robot: scene.randomize({name.split('/', 1)[0]!r}, {field!r}, lo, hi, "
                         "when='reset')", field="name")
    if when != "reset":
        raise SceneError(f"{name!r}: {field} is drawn at every reset, not at build", "pass when='reset' (for a "
                         "start drawn once at build, use the fields 'x', 'y', 'z')", field="when")


def refuse_rerun_worlds(scene: Any, rerun_worlds: Any) -> None:
    """``alloy.build(..., rerun_worlds=n)`` re-runs worlds in a side simulation that does not carry these draws.

    Example: ``refuse_rerun_worlds(scene, 4)`` raises when the scene has robot or gravity rows."""
    if rerun_worlds and getattr(scene, "robot_random", None):
        from alloy.api.errors import BuildError  # noqa: PLC0415

        raise BuildError("rerun_worlds cannot be used with randomized robot fields or gravity (the side simulation "
                         "that re-runs worlds would not hold each world's values)", "build with rerun_worlds=0 (the "
                         "default), or drop the robot / gravity rows of scene.randomize", field="rerun_worlds")


def key(row: dict[str, Any]) -> str:
    """The row's name in ``sim.randomized()`` and in its draws. Example: ``"go2/base.mass_add"``."""
    return f"{row['target']}.{row['field']}"


__all__ = ["BODY_FIELDS", "FIELDS", "GRAVITY", "GRAVITY_AXES", "JOINT_FIELDS", "ONE_LINK", "START_FIELDS", "add",
           "is_robot_row", "key", "refuse_rerun_worlds"]
