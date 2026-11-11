# SPDX-License-Identifier: Apache-2.0
"""Names -> engine indices for a built scene: bodies, joints, particle objects, per world.

Objects keep the names given to ``alloy.Scene``. A robot's bodies and joints are "<robot>/<name in its file>"; a joint
name that is unique in the scene also works without the robot prefix, and "<robot>" alone is its root body.
Static objects have no state (they never move). A kinematic fixture (``Scene.add_open_box(..., kinematic=True)``) is
one body: "<name>" and "<name>_floor" name it, and its other parts ("<name>_wall0", ...) have no state of their own.
Example::

    idx = NameIndex.from_built(built)
    idx.body("cube")            # template body index of the cube (add ``world * idx.stride["body"]``)
    idx.names()                 # {"bodies": [...], "joints": [...], "particles": [...], "static": [...]}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.api.errors import UnknownNameError


@dataclass
class NameIndex:
    """Template indices (world 0) of every named thing, and the per-world strides."""

    worlds: int
    stride: dict[str, int]                                       # body, joint, particle, tri: items per world
    bodies: dict[str, int] = field(default_factory=dict)
    joints: dict[str, int] = field(default_factory=dict)        # name -> joint index (template)
    particles: dict[str, range] = field(default_factory=dict)   # cloth / liquid / soft: particle range (template)
    tris: dict[str, range] = field(default_factory=dict)        # cloth: triangle range (template)
    static: list[str] = field(default_factory=list)
    free_joint: dict[str, int] = field(default_factory=dict)    # object name -> its free joint (template)
    robot_joints: dict[str, range] = field(default_factory=dict)  # robot name -> its joints (template)
    parts: dict[str, str] = field(default_factory=dict)          # kinematic fixture part on its root's body -> fixture

    @classmethod
    def from_built(cls, built: Any) -> NameIndex:
        """Index a ``BuiltScene`` (``alloy.scene.types``). Example: ``NameIndex.from_built(sim.built)``."""
        m = built.model
        any_handle = next(iter(built.handles.values()), None)
        st = any_handle.world_stride if any_handle is not None else None
        stride = {k: (len(getattr(st, k)) if st is not None else 0) for k in ("body", "joint", "particle", "tri")}
        idx = cls(int(built.world_count), stride)
        body_labels = [str(x) for x in m.body_label]
        joint_labels = [str(x) for x in m.joint_label]
        joint_type = m.joint_type.numpy() if m.joint_count else np.zeros(0, np.int32)
        plan = getattr(built, "_plan", None)
        robots = {r.name for r in plan.scene.robots} if plan is not None else set()
        for name, h in built.handles.items():
            idx._add(name, h, name in robots, (body_labels, joint_labels, joint_type))
        idx._add_short_joint_names()
        idx._add_fixtures(getattr(built, "kinematic_fixtures", None) or {})
        return idx

    def _add(self, name: str, h: Any, robot: bool, labels: tuple[list[str], list[str], Any]) -> None:
        body_labels, joint_labels, joint_type = labels
        t = h.template
        robot = robot or len(t.body) > 1
        if len(t.particle):
            self.particles[name] = t.particle
            if len(t.tri):
                self.tris[name] = t.tri
        elif not len(t.body):
            self.static.append(name)
        if len(t.body) == 1 and not robot:
            self.bodies[name] = t.body.start
            for j in t.joint:
                if int(joint_type[j]) == FREE_JOINT:
                    self.free_joint[name] = j
            return
        if robot:
            self.robot_joints[name] = t.joint
        for k, b in enumerate(t.body):
            self.bodies[f"{name}/{_short(body_labels[b])}"] = b
            if k == 0:
                self.bodies[name] = b
        for j in t.joint:
            self.joints[f"{name}/{_short(joint_labels[j])}"] = j

    def _add_fixtures(self, fixtures: Mapping[str, Sequence[str]]) -> None:
        """Each kinematic fixture's name is its root part's body (``alloy.api.fixtures``); its other parts are shapes
        on that body, so they are neither bodies nor static. An object of the fixture's own name keeps the name."""
        for name, parts in fixtures.items():
            root = parts[0]
            if root not in self.bodies:
                continue
            self.bodies.setdefault(name, self.bodies[root])
            if root in self.free_joint:
                self.free_joint.setdefault(name, self.free_joint[root])
            for p in parts[1:]:
                if p in self.static:
                    self.static.remove(p)
                self.parts[p] = name

    def _add_short_joint_names(self) -> None:
        shorts: dict[str, list[str]] = {}
        for full in self.joints:
            shorts.setdefault(full.split("/", 1)[1], []).append(full)
        for short, fulls in shorts.items():
            if len(fulls) == 1 and short not in self.joints:
                self.joints[short] = self.joints[fulls[0]]

    # --- lookups -------------------------------------------------------------------------------------------------

    def names(self) -> dict[str, list[str]]:
        """Every name by kind. Example: ``{"bodies": ["cube"], "joints": [], "particles": [], "static": ["ground"]}``."""
        return {"bodies": list(self.bodies), "joints": list(self.joints), "particles": list(self.particles),
                "static": list(self.static)}

    def body(self, name: str) -> int:
        """Template body index of ``name``. Example: ``idx.body("cube")``."""
        if name in self.bodies:
            return self.bodies[name]
        raise self._unknown(name, "body")

    def joint(self, name: str) -> int:
        """Template joint index of ``name`` ("robot/joint" or a unique "joint"). Example: ``idx.joint("elbow")``."""
        if name in self.joints:
            return self.joints[name]
        raise self._unknown(name, "joint")

    def particle_range(self, name: str) -> range:
        """Template particle range of a cloth / liquid / soft object. Example: ``idx.particle_range("towel")``."""
        if name in self.particles:
            return self.particles[name]
        raise self._unknown(name, "particle object")

    def per_world(self, template: int, kind: str) -> np.ndarray:
        """Indices of one template item in every world. Example: ``idx.per_world(idx.body("cube"), "body")``."""
        return template + np.arange(self.worlds, dtype=np.int64) * self.stride[kind]

    def _unknown(self, name: str, kind: str) -> UnknownNameError:
        if name in self.parts:
            fixture = self.parts[name]
            return UnknownNameError(f"{name!r} is a part of the kinematic fixture {fixture!r}, which moves as one "
                                    "body: the part has no state of its own", f"read or drive {fixture!r} instead "
                                    "(its state is the floor's frame; every part is fixed in it)", field=name)
        if name in self.static:
            return UnknownNameError(f"{name!r} is static: it has no {kind} state", "static objects never move; read "
                                    "their pose from the scene you built", field=name)
        every = self.names()
        listing = "; ".join(f"{k}: {', '.join(v[:12]) or '-'}" for k, v in every.items())
        return UnknownNameError(f"no {kind} named {name!r}", f"use a name from sim.names() ({listing})", field=name)


FREE_JOINT = 4      # newton.JointType.FREE


def _short(label: str) -> str:
    return label.rsplit("/", 1)[-1]
