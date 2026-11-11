# SPDX-License-Identifier: Apache-2.0
"""Split an ``IOModel`` into groups (one per object or robot): the unit URDF files and card imports work on.

A group is one kinematic tree: its root body (joined to the world by a free joint, a fixed joint, or nothing), the
bodies below it, and the static world shapes of the same object. Bodies with no group name are grouped by
connectivity and named after their root body.

Example::

    from alloy.io.groups import split_groups
    for g in split_groups(io):
        print(g.name, g.root.name if g.root else "static", len(g.bodies), g.fixed_base)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alloy.io.model import IDENTITY, IOBody, IOJoint, IOModel, IOShape, Pose


@dataclass
class Group:
    """One object or robot. ``joints`` excludes the root's own world joint (kept in ``root_joint``)."""

    name: str
    kind: str = "object"
    bodies: list[IOBody] = field(default_factory=list)
    joints: list[IOJoint] = field(default_factory=list)
    root: IOBody | None = None
    root_joint: IOJoint | None = None
    world_shapes: list[IOShape] = field(default_factory=list)
    fixed_base: bool = False

    @property
    def base_joint(self) -> IOJoint | None:
        """The root's joint to the world when it is a hinge, slider or ball (a base that moves on a joint)."""
        rj = self.root_joint
        return rj if rj is not None and rj.kind in ("revolute", "prismatic", "ball") else None

    @property
    def articulated(self) -> bool:
        """True when the group has joints besides a free or fixed joint from its root to the world."""
        return bool(self.joints) or self.base_joint is not None

    def base_pose(self) -> Pose:
        """World pose of the group's link frame: the root body's pose (identity for static shapes)."""
        return IDENTITY if self.root is None else self.root.pose


def _root_of(io: IOModel) -> dict[str, str]:
    parent = {j.child: j.parent for j in io.joints if j.kind != "free" and j.parent is not None}
    out = {}
    for b in io.bodies:
        r = b.name
        seen = set()
        while r in parent and r not in seen:
            seen.add(r)
            r = parent[r]
        out[b.name] = r
    return out


def _group_names(io: IOModel) -> dict[str, str]:
    """Group of each body: its own name, else its root's, else the root body name. A named group holding several
    trees (a robot file with free objects in it) keeps its name for the first tree; the others get
    ``<group>_<root>``."""
    roots = _root_of(io)
    named = {b.name: b.group for b in io.bodies if b.group}
    base = {b.name: named.get(b.name) or named.get(roots[b.name]) or roots[b.name] for b in io.bodies}
    first_root: dict[str, str] = {}
    out = {}
    for b in io.bodies:
        g, r = base[b.name], roots[b.name]
        first_root.setdefault(g, r)
        out[b.name] = g if first_root[g] == r else f"{g}_{r}"
    return out


def _welded(io: IOModel) -> set[str]:
    """Bodies welded to the world by an equality constraint."""
    return {e.body1 for e in io.equalities if e.kind == "weld" and e.body2 is None}


def _finish(g: Group, io: IOModel, welded: set[str]) -> None:
    names = {b.name for b in g.bodies}
    tops = [b for b in g.bodies if (j := io.joint_of(b.name)) is None or j.parent not in names]
    g.root = tops[0] if tops else None
    if g.root is None:
        g.fixed_base = True
        return
    g.root_joint = io.joint_of(g.root.name)
    rj = g.root_joint
    g.fixed_base = (rj is not None and rj.kind != "free") or g.root.name in welded or g.root.motion == "kinematic"


def split_groups(io: IOModel) -> list[Group]:
    """Groups in model order: every body-bearing group, then static-only groups."""
    by_name: dict[str, Group] = {}
    gname = _group_names(io)
    for b in io.bodies:
        g = by_name.setdefault(gname[b.name], Group(gname[b.name], io.groups.get(gname[b.name], "object")))
        g.bodies.append(b)
    for s in io.world_shapes:
        name = io.shape_groups.get(s.name, s.name)
        g = by_name.setdefault(name, Group(name, io.groups.get(name, "object")))
        g.world_shapes.append(s)
    welded = _welded(io)
    for g in by_name.values():
        names = {b.name for b in g.bodies}
        g.joints = [j for j in io.joints if j.child in names and j.parent in names]
        _finish(g, io, welded)
    return list(by_name.values())
