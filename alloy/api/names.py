# SPDX-License-Identifier: Apache-2.0
"""Names -> engine indices for a built scene: bodies, joints, particle objects, per world.

Objects keep the names given to ``alloy.Scene``. A robot's bodies and joints are "<robot>/<name in its file>" (the
name may hold "/", as in MuJoCo Menagerie's two-arm files); a joint name that is unique in the scene also works
without the robot prefix, and "<robot>" alone is its root body.
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
    joint_short: list[str] = field(default_factory=list)        # template joint -> its name in its file

    @classmethod
    def from_built(cls, built: Any) -> NameIndex:
        """Index a ``BuiltScene`` (``alloy.scene.types``). Example: ``NameIndex.from_built(sim.built)``."""
        m = built.model
        any_handle = next(iter(built.handles.values()), None)
        st = any_handle.world_stride if any_handle is not None else None
        stride = {k: (len(getattr(st, k)) if st is not None else 0) for k in ("body", "joint", "particle", "tri")}
        idx = cls(int(built.world_count), stride)
        body_short, idx.joint_short = _file_names(m, [str(x) for x in m.body_label], [str(x) for x in m.joint_label])
        joint_type = m.joint_type.numpy() if m.joint_count else np.zeros(0, np.int32)
        robots = _robot_names(built)
        for name, h in built.handles.items():
            idx._add(name, h, name in robots, (body_short, idx.joint_short, joint_type))
        idx._add_short_joint_names()
        idx._add_fixtures(getattr(built, "kinematic_fixtures", None) or {})
        return idx

    def _add(self, name: str, h: Any, robot: bool, labels: tuple[list[str], list[str], Any]) -> None:
        body_names, joint_names, joint_type = labels
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
            self.bodies[f"{name}/{body_names[b]}"] = b
            if k == 0:
                self.bodies[name] = b
        for j in t.joint:
            self.joints[f"{name}/{joint_names[j]}"] = j

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


def _file_names(model: Any, body_labels: list[str], joint_labels: list[str]) -> tuple[list[str], list[str]]:
    """Each body's and joint's name in its file. The importer labels a body "<parent body's label>/<name>" (a root
    body "<model>/worldbody/<name>") and a joint "<its body's label>/<name>", and a name may hold "/" itself, so the
    parent's label is taken off the front rather than the label split at its last "/" (which named MuJoCo Menagerie's
    "left/waist" and "right/waist" both "waist", one overwriting the other). Labels of another form keep their last
    part. Example: ``bodies, joints = _file_names(sim.built.model, body_labels, joint_labels)``."""
    n = int(getattr(model, "joint_count", 0) or 0)
    child = model.joint_child.numpy() if n else np.zeros(0, np.int64)
    parent = model.joint_parent.numpy() if n else np.zeros(0, np.int64)
    up = {int(c): int(p) for c, p in zip(child, parent)}
    bodies = [_strip(lab, body_labels[up[b]] if 0 <= up.get(b, -1) < len(body_labels) else None)
              for b, lab in enumerate(body_labels)]
    joints = [_strip(lab, body_labels[int(child[j])] if 0 <= int(child[j]) < len(body_labels) else None)
              for j, lab in enumerate(joint_labels)]
    return bodies, joints


def _strip(label: str, parent: str | None) -> str:
    """``label`` without its parent's label (or, with no parent body, without "<model>/worldbody/")."""
    if parent is not None and label.startswith(parent + "/"):
        return label[len(parent) + 1:]
    if parent is None:
        root = "worldbody/"
        at = 0 if label.startswith(root) else label.find("/" + root) + 1
        if at > 0 or label.startswith(root):
            return label[at + len(root):]
    return label.rsplit("/", 1)[-1]


def _robot_names(built: Any) -> set[str]:
    """The names ``Scene.add_robot`` gave (the assembler's plan), empty without a plan."""
    plan = getattr(built, "_plan", None)
    return {r.name for r in plan.scene.robots} if plan is not None else set()


def scene_shape_names(built: Any) -> tuple[str, ...]:
    """Every model shape's name in the scene's words, for messages (health's start overlaps): an object's shapes by
    the object's name, a robot's by "<robot>/<body>" as ``sim.names()`` names its bodies, or "<robot>/<shape>" for a
    shape fixed to the world (a whole-scene file's floor); a shape no object owns keeps its model label. The model's
    own labels start with the file's model name ("emb_e1_fr3/worldbody/floor"): not the name the robot was added
    with, and the same for two robots from one file. Computed once per built scene.

    Example: ``scene_shape_names(sim.built)[1]`` -> ``"counter_arm/fr3_link0"``."""
    cached = getattr(built, "_scene_shape_names", None)
    if cached is not None:
        return cached
    m = built.model
    out = [str(x) for x in (m.shape_label or [])]
    labels = [str(x) for x in (m.body_label or [])]
    bodies = _file_names(m, labels, [str(x) for x in (m.joint_label or [])])[0]
    shape_body = m.shape_body.numpy() if out else np.zeros(0, np.int32)
    robots, worlds = _robot_names(built), int(built.world_count)
    for name, h in built.handles.items():
        stride = len(h.world_stride.shape)
        ids = [s + w * stride for w in range(worlds) for s in h.template.shape]
        for i in (i for i in ids if i < len(out)):
            b = int(shape_body[i])
            out[i] = f"{name}/{bodies[b] if b >= 0 else _strip(out[i], None)}" if name in robots else name
    names = tuple(out)
    try:
        built._scene_shape_names = names
    except AttributeError:
        pass
    return names
