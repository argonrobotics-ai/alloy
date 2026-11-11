# SPDX-License-Identifier: Apache-2.0
"""``IOModel`` to URDF: one file per object or robot, plus a JSON index with each file's base pose.

URDF holds one kinematic tree, so a scene becomes several files. Each link carries its inertial, collision and
visual geometry (boxes, spheres and cylinders natively; capsules, cones and ellipsoids as OBJ meshes) and PyBullet's
``<contact>`` element for friction and restitution. Joint frames follow URDF: a child link's frame is its joint
frame, so shapes and inertia are re-expressed when the source joint frame differs from the body frame.

Example::

    from alloy.io.urdf_write import write_urdf
    index = write_urdf(io, "out/urdf")          # io: an IOModel (alloy.io.build.scene_model)
    print(index["groups"][0]["file"], index["groups"][0]["fixed_base"])
"""

from __future__ import annotations

import json
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from alloy.io.geometry import compose, inverse, mesh_digest, quat_matrix, tessellate, write_obj
from alloy.io.groups import Group, split_groups
from alloy.io.model import IDENTITY, IOBody, IOJoint, IOModel, IOShape, Pose
from alloy.io.usd_text import num

log = logging.getLogger("alloy.io")
UNLIMITED_TEXT = "1e+06"        # effort/velocity written when the source has no limit (read back as "no limit")
NATIVE = ("box", "sphere", "cylinder", "mesh")


def rpy(q) -> tuple[float, float, float]:
    """URDF roll-pitch-yaw (fixed X, Y, Z) of an (x, y, z, w) quaternion."""
    r = quat_matrix(q)
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(r[2, 0]) < 1.0 - 1e-12:
        return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])
    return math.atan2(-r[1, 2], r[1, 1]), pitch, 0.0


def _fmt(values) -> str:
    return " ".join(num(v) for v in values)


def _origin(parent: ET.Element, pose: Pose) -> None:
    ET.SubElement(parent, "origin", xyz=_fmt(pose[:3]), rpy=_fmt(rpy(pose[3:])))


class _Writer:
    """State for one group's URDF: output folder and the model (for unmapped notes)."""

    def __init__(self, io: IOModel, out_dir: Path) -> None:
        self.io, self.out = io, out_dir

    def geometry(self, parent: ET.Element, shape: IOShape, owner: str) -> None:
        g = ET.SubElement(parent, "geometry")
        s = shape.size
        if shape.kind == "box":
            ET.SubElement(g, "box", size=_fmt([2 * x for x in s]))
        elif shape.kind == "sphere":
            ET.SubElement(g, "sphere", radius=_fmt(s[:1]))
        elif shape.kind == "cylinder":
            ET.SubElement(g, "cylinder", radius=_fmt(s[:1]), length=_fmt([2 * s[1]]))
        elif shape.kind == "plane":
            hx, hy = (x if x > 0 else 50.0 for x in s[:2])
            ET.SubElement(g, "box", size=_fmt([2 * hx, 2 * hy, 0.002]))
            self.io.note("urdf", owner, "URDF has no plane: written as a 2 mm thick box (top 1 mm below z=0)")
        else:
            mesh = tessellate(shape)
            rel = f"meshes/{shape.kind}_{mesh_digest(mesh)}.obj"
            if not (self.out / rel).exists():
                write_obj(mesh, self.out / rel)
            ET.SubElement(g, "mesh", filename=rel)
            if shape.kind not in NATIVE:
                self.io.note("urdf", owner, f"URDF has no {shape.kind}: written as a triangle mesh")
            elif not shape.convex and shape.collide:
                self.io.note("urdf", owner, "non-convex mesh collision: most URDF loaders use the convex hull")

    def shape(self, link: ET.Element, shape: IOShape, shift: Pose, owner: str) -> None:
        pose = compose(shift, shape.pose)
        if shape.kind == "plane":
            pose = compose(pose, (0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 1.0))
        tags = [t for t, on in (("collision", shape.collide), ("visual", shape.visible)) if on]
        for tag in tags:
            el = ET.SubElement(link, tag, name=shape.name)
            _origin(el, pose)
            self.geometry(el, shape, owner)

    def contact(self, link: ET.Element, shapes: list[IOShape], name: str) -> None:
        mats = [s.material for s in shapes if s.collide and s.material in self.io.materials]
        if not mats:
            return
        if len(set(mats)) > 1:
            self.io.note("urdf", name, f"one friction per URDF link: {mats[0]} used for all of {sorted(set(mats))}")
        m = self.io.materials[mats[0]]
        c = ET.SubElement(link, "contact")
        for tag, v in (("lateral_friction", m.mu_dynamic), ("rolling_friction", m.rolling_lever),
                       ("spinning_friction", m.torsional_lever), ("restitution", m.restitution)):
            ET.SubElement(c, tag, value=_fmt([v]))
        if m.mu_static != m.mu_dynamic:
            self.io.note("urdf", m.name, f"URDF has one friction: dynamic {m.mu_dynamic} written, static "
                                         f"{m.mu_static} dropped")


def _inertial(link: ET.Element, body: IOBody, shift: Pose) -> None:
    if body.mass <= 0.0:
        return
    el = ET.SubElement(link, "inertial")
    com = compose(shift, (*body.com, *IDENTITY[3:]))
    _origin(el, (*com[:3], *IDENTITY[3:]))
    ET.SubElement(el, "mass", value=_fmt([body.mass]))
    r = quat_matrix(shift[3:])
    i = r @ np.asarray(body.inertia, np.float64) @ r.T
    ET.SubElement(el, "inertia", ixx=_fmt([i[0, 0]]), ixy=_fmt([i[0, 1]]), ixz=_fmt([i[0, 2]]),
                  iyy=_fmt([i[1, 1]]), iyz=_fmt([i[1, 2]]), izz=_fmt([i[2, 2]]))


def _joint(robot: ET.Element, j: IOJoint, shift: dict[str, Pose], io: IOModel, group: str) -> None:
    kind = {"revolute": "revolute" if j.lower is not None and j.upper is not None else "continuous",
            "prismatic": "prismatic", "fixed": "fixed"}.get(j.kind)
    if kind is None:
        io.note("urdf", j.name, f"URDF has no {j.kind} joint: written as fixed")
        kind = "fixed"
    el = ET.SubElement(robot, "joint", name=j.name, type=kind)
    _origin(el, compose(inverse(shift.get(j.parent, IDENTITY)), j.parent_pose))
    ET.SubElement(el, "parent", link=j.parent)
    ET.SubElement(el, "child", link=j.child)
    if kind == "fixed":
        return
    ET.SubElement(el, "axis", xyz=_fmt(j.axis))
    lim = {"effort": _fmt([j.effort]) if j.effort is not None else UNLIMITED_TEXT,
           "velocity": _fmt([j.velocity]) if j.velocity is not None else UNLIMITED_TEXT}
    if kind != "continuous":
        lim.update(lower=_fmt([j.lower if j.lower is not None else -1e6]),
                   upper=_fmt([j.upper if j.upper is not None else 1e6]))
    ET.SubElement(el, "limit", **lim)
    if j.damping or j.friction:
        ET.SubElement(el, "dynamics", damping=_fmt([j.damping]), friction=_fmt([j.friction]))
    for what, v in (("armature", j.armature), ("drive stiffness", j.stiffness), ("drive damping", j.drive_damping),
                    ("initial position", j.q0)):
        if v:
            io.note("urdf", group, f"URDF has no joint {what}: set it in the loading simulator (source values in "
                                   "the MJCF or USD export)")


def group_urdf(io: IOModel, g: Group, out_dir: Path) -> ET.Element:
    """The ``<robot>`` element of one group (meshes are written under ``out_dir/meshes``)."""
    w = _Writer(io, out_dir)
    robot = ET.Element("robot", name=g.name)
    joints = list(g.joints)
    if g.base_joint is not None:        # a base on a joint to the world: an empty base link stands for the world
        base = f"{g.name}_base"
        ET.SubElement(robot, "link", name=base)
        joints.insert(0, replace(g.base_joint, parent=base))    # the base link is the world: joint frame as is
    shift = {j.child: j.child_pose for j in joints}        # child link frame = joint frame
    if g.root is None:
        link = ET.SubElement(robot, "link", name=f"{g.name}_base")
        for s in g.world_shapes:
            w.shape(link, s, IDENTITY, g.name)
        w.contact(link, g.world_shapes, g.name)
        return robot
    for b in g.bodies:
        link = ET.SubElement(robot, "link", name=b.name)
        inv = inverse(shift.get(b.name, IDENTITY))
        _inertial(link, b, inv)
        for s in b.shapes:
            w.shape(link, s, inv, g.name)
        w.contact(link, list(b.shapes), b.name)
    for j in joints:
        _joint(robot, j, shift, io, g.name)
    if g.world_shapes:
        io.note("urdf", g.name, f"{len(g.world_shapes)} static shapes of this object are not in its URDF")
    return robot


def write_urdf(io: IOModel, out_dir: str | Path) -> dict:
    """Write ``<group>.urdf`` per object/robot and ``index.json``; returns the index (files, base poses, notes)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = []
    for g in split_groups(io):
        robot = group_urdf(io, g, out)
        tree = ET.ElementTree(robot)
        ET.indent(tree, space="  ")
        path = out / f"{g.name}.urdf"
        tree.write(path, encoding="unicode", xml_declaration=True)
        pose = IDENTITY if g.base_joint is not None else g.base_pose()   # a jointed base carries its own place
        groups.append({"name": g.name, "kind": g.kind, "file": path.name, "base_pose": list(pose),
                       "fixed_base": g.fixed_base, "articulated": g.articulated})
    for e in io.equalities:
        io.note("urdf", e.name, f"{e.kind} between {e.body1} and {e.body2 or 'world'}: URDF has no loop constraints"
                                + ("; a weld to the world is written as fixed_base" if e.body2 is None else ""))
    index = {"format": "urdf", "scene": io.name, "groups": groups,
             "unmapped": [r.as_dict() for r in io.unmapped if r.format in ("all", "urdf", "model")]}
    (out / "index.json").write_text(json.dumps(index, indent=2))
    log.info("wrote %d URDF files to %s", len(groups), out)
    return index
