# SPDX-License-Identifier: Apache-2.0
"""HELD: the route's friction default where a robot grips things (coordinator ruling, [internal note]
2026-11-06T18:14Z; the cause: docs/notes/held-grip-v1.md; the route numbers: docs/notes/held-grip-route.md).

A pinched object under MuJoCo's soft friction creeps under a steady load and walks under an oscillating one. With
the route's settings as they were (elliptic cone, the FR3 file's impratio 1, no noslip), the FR3 turned a 1.1 kg
bottle 9.9 deg in 2 s under a steady 0.5 N m about the jaw axis, and 7.0 deg in 3 s under +-0.3 N m at 3.5 Hz. With
Alloy's scoped noslip at 100 iterations the turn is 0.0002 deg at impratio 1, 3 or 300; under +-0.5 N m it is 2.0 deg
at impratio 1, 0.0002 deg at 3 and 0.03 deg at 300.

The rule (``held_reason``): a scene whose robot has a gripper or a hand, and which has at least one free rigid
object, runs its MuJoCo entry with ``HELD``'s noslip (100 iterations, tolerance 1e-6; scoped to the rows that touch
a robot, ``noslip_route.install_entry_noslip``). Limits of the rule:
- A file's own noslip request stands (T11, T12, T14, H1 and YAM ask for 50); HELD fills in only where none asks.
- Elliptic cones: the route's already (a file's pyramidal cone is overridden, ``entries.ELLIPTIC_ONLY``).
- Impratio at least 3: a file's lower value (the library's FR3, Panda and UR10e files set 1) is raised to 3 (scene
  effect ``HELD_EFFECT``, ``entries.mujoco_overrides``); the route's 300 where no file sets one stays (SCR-WP3-10).
- Free objects are the scene's object cards (``motion: dynamic``, MuJoCo entry). Whole-file scenes, one MJCF holding
  the robot and its objects (``objects: []``: the XB4 embodiments and the task packs), keep their file's options,
  so benchmark rows do not move. A file can ask for noslip in its own ``<option>``.
- A gripper is an actuated MJCF or URDF robot with a joint or body named as a finger, gripper, jaw, knuckle, thumb,
  claw, palm, hand or pad (``robot_grip``). USD robots are not read.

Code that builds ``SolverMuJoCo`` itself on a model where a robot holds an object (``liquid_robot.CoupledSim``)
builds it with ``held_options`` (elliptic cones, impratio at least 3) and installs HELD's noslip.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("alloy.scene")

CONE = {"pyramidal": 0, "elliptic": 1}
GRIP_NAME = re.compile(r"finger|gripper|jaw|knuckle|thumb|claw|palm|(?:^|[^a-z])(?:hand|pad)(?:$|[^a-z])", re.I)
_ACTUATORS = ("motor", "position", "velocity", "intvelocity", "general", "damper", "cylinder", "muscle", "adhesion")
_MAX_INCLUDE_DEPTH = 8
HELD_EFFECT = "mujoco:held"         # scene effect of a HELD scene: entries.mujoco_overrides raises a low impratio
HELD_REASON = ("HELD, a robot grips a free object: at impratio 1 the scoped noslip still let a pinched bottle walk "
               "2 deg under +-0.5 N m (docs/notes/held-grip-route.md)")


@dataclass(frozen=True)
class HeldGrip:
    """MuJoCo friction settings of a scene where a robot holds an object: their wine sim's, without multiccd, at
    MuJoCo's default noslip tolerance (1e-6). At their 1e-10 the float32 noslip never converges and runs all 100
    sweeps: GripperLift on an L4 at 64 worlds stepped 29 % slower than without noslip, against 14 % at 1e-6, and
    the bottle still holds (HG-1 0.0013 deg, HG-2 0.0022 deg, +-0.5 N m 0.0019 deg)."""

    cone: str = "elliptic"
    impratio: float = 3.0
    noslip_iterations: int = 100
    noslip_tolerance: float = 1.0e-6


HELD = HeldGrip()


def _mjcf_roots(path: Path, depth: int = 0) -> list[ET.Element]:
    """The file's root and the roots of the files it ``<include>``s (relative to the including file)."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        log.warning("held route: cannot read %s (%s); no gripper read from it", path, exc)
        return []
    out = [root]
    if depth < _MAX_INCLUDE_DEPTH:
        for inc in root.iter("include"):
            if inc.get("file"):
                out += _mjcf_roots(path.parent / inc.get("file", ""), depth + 1)
    return out


def _grip_name(elements: list[ET.Element]) -> str | None:
    for e in elements:
        name = e.get("name", "")
        if name and GRIP_NAME.search(name):
            return name
    return None


def mjcf_grip(path: str) -> str | None:
    """The first joint or body name that names a gripper part in an MJCF with actuators, None if there is none."""
    roots = _mjcf_roots(Path(path))
    if not any(a.tag in _ACTUATORS for r in roots for act in r.iter("actuator") for a in act):
        return None
    return (_grip_name([j for r in roots for j in r.iter("joint")])
            or _grip_name([b for r in roots for b in r.iter("body")]))


def urdf_grip(path: str) -> str | None:
    """The first moving joint or link name that names a gripper part in a URDF, None if there is none."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        log.warning("held route: cannot read %s (%s); no gripper read from it", path, exc)
        return None
    moving = [j for j in root.iter("joint") if j.get("type", "fixed") in ("revolute", "prismatic", "continuous")]
    if not moving:
        return None
    return _grip_name(moving) or _grip_name(list(root.iter("link")))


def robot_grip(path: str, kind: str) -> str | None:
    """The gripper part a robot file names (``mjcf_grip`` / ``urdf_grip``); None for USD or no gripper."""
    if kind == "mjcf":
        return mjcf_grip(path)
    if kind == "urdf":
        return urdf_grip(path)
    return None


def free_objects(objects: Sequence[Any]) -> list[str]:
    """Names of the scene's free rigid objects: object cards with ``motion: dynamic`` routed to the MuJoCo entry."""
    return [t.name for t in objects if t.obj is not None and getattr(t.obj.card, "motion", None) == "dynamic"
            and getattr(getattr(t, "decision", None), "entry", None) == "mujoco"]


def held_reason(robots: Sequence[Any], asset: Callable[[str], str], objects: Sequence[Any]) -> str | None:
    """Why the scene takes HELD's noslip ("robot fr3 grips (finger_joint1), free object cube"), None when it does
    not: no robot with a gripper, or no free rigid object."""
    free = free_objects(objects)
    if not free:
        return None
    for r in robots:
        part = robot_grip(asset(r.path), r.kind)
        if part is not None:
            more = f" (+{len(free) - 1})" if len(free) > 1 else ""
            return f"robot {r.name} grips ({part}), free object {free[0]}{more}"
    return None


def held_options(model: Any, grip: HeldGrip = HELD) -> dict[str, object]:
    """``SolverMuJoCo`` constructor options for ``grip`` on ``model`` (they override the model's per-world values):
    ``grip``'s cone, and impratio at least ``grip.impratio`` (a higher one in ``model.mujoco`` stays)."""
    imp = getattr(getattr(model, "mujoco", None), "impratio", None)
    base = float(np.max(imp.numpy())) if imp is not None and imp.shape[0] > 0 else 1.0
    return {"cone": grip.cone, "impratio": max(base, float(grip.impratio))}


__all__ = ["CONE", "HELD", "HELD_EFFECT", "HELD_REASON", "HeldGrip", "free_objects", "held_options", "held_reason",
           "mjcf_grip", "robot_grip", "urdf_grip"]
