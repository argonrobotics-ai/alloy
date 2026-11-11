# SPDX-License-Identifier: Apache-2.0
"""One group of an ``IOModel`` (an object or a robot) as a small, self-contained MJCF file.

Used by the importers for what a single card shape cannot hold: compound rigid bodies, articulated objects, and
robots taken out of a larger file. The root body sits at the origin (the card's placement or the robot's base pose
puts it in the world); meshes are written inline. Objects get geom densities (Alloy cards set mass from density),
robots keep explicit inertials and get position actuators for driven joints.

Example::

    from alloy.io.groups import split_groups
    from alloy.io.mjcf_write import write_group_mjcf
    g = split_groups(io)[0]
    write_group_mjcf(io, g, "out/arm.xml", robot=True)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from alloy.io.geometry import compose, inverse, mesh_digest, quat_rotate, shape_volume, tessellate
from alloy.io.groups import Group
from alloy.io.model import IDENTITY, IOBody, IOJoint, IOModel, IOShape, Pose
from alloy.io.usd_text import num

log = logging.getLogger("alloy.io")
_TYPES = {"box": "box", "sphere": "sphere", "capsule": "capsule", "cylinder": "cylinder", "ellipsoid": "ellipsoid",
          "plane": "plane"}
_JOINTS = {"revolute": "hinge", "prismatic": "slide", "ball": "ball"}


def _txt(values) -> str:
    return " ".join(num(v) for v in values)


def _pose_attrs(el: ET.Element, pose: Pose) -> None:
    if any(abs(x) > 0 for x in pose[:3]):
        el.set("pos", _txt(pose[:3]))
    x, y, z, w = pose[3:]
    if abs(w) < 1.0 - 1e-12:
        el.set("quat", _txt((w, x, y, z)))


class _MjcfWriter:
    def __init__(self, io: IOModel, g: Group, robot: bool, density: float | None) -> None:
        self.io, self.g, self.robot, self.density = io, g, robot, density
        self.root = ET.Element("mujoco", model=g.name)
        ET.SubElement(self.root, "compiler", angle="radian", autolimits="true")
        self.asset = ET.SubElement(self.root, "asset")
        self.world = ET.SubElement(self.root, "worldbody")
        self.meshes: set[str] = set()

    def geom(self, parent: ET.Element, s: IOShape, pose: Pose) -> None:
        el = ET.SubElement(parent, "geom", name=s.name)
        if s.kind in _TYPES:
            el.set("type", _TYPES[s.kind])
            size = s.size if s.kind != "plane" else tuple(x if x > 0 else 0.0 for x in s.size) + (1.0,)
            el.set("size", _txt(size))
        else:
            mesh = tessellate(s)
            name = f"mesh_{mesh_digest(mesh)}"
            if name not in self.meshes:
                self.meshes.add(name)
                ET.SubElement(self.asset, "mesh", name=name, vertex=_txt(mesh.vertices.reshape(-1)),
                              face=" ".join(str(int(i)) for i in mesh.faces.reshape(-1)))
            el.set("type", "mesh")
            el.set("mesh", name)
        _pose_attrs(el, pose)
        mat = self.io.materials.get(s.material)
        if mat is not None:
            el.set("friction", _txt((mat.mu_dynamic, mat.torsional_lever or 0.005, mat.rolling_lever or 0.0001)))
        if not s.collide:
            el.set("contype", "0")
            el.set("conaffinity", "0")
            el.set("group", "2")
            el.set("density", "0")
        elif self.density is not None:
            el.set("density", num(self.density))

    def inertial(self, el: ET.Element, b: IOBody) -> None:
        """Robots keep their authored inertials; objects get mass from geom densities (card semantics), except a
        body with no colliding volume, which keeps its own mass so MuJoCo accepts it."""
        if not self.robot and any(s.collide and shape_volume(s) > 0 for s in b.shapes):
            return
        m, i = max(b.mass, 1e-6), np.asarray(b.inertia, np.float64)
        if b.mass <= 0:
            i = np.eye(3) * 1e-9
            self.io.note("mjcf", b.name, "massless body given 1e-6 kg so MuJoCo accepts it")
        ET.SubElement(el, "inertial", pos=_txt(b.com), mass=num(m),
                      fullinertia=_txt((i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])))

    def joint(self, el: ET.Element, j: IOJoint) -> None:
        """MJCF joints sit in the child body: position and axis from the joint frame in that body."""
        typ = _JOINTS.get(j.kind)
        if typ is None:
            return
        frame = j.child_pose
        jel = ET.SubElement(el, "joint", name=j.name, type=typ)
        if any(abs(x) > 0 for x in frame[:3]):
            jel.set("pos", _txt(frame[:3]))
        if typ != "ball":
            jel.set("axis", _txt(quat_rotate(frame[3:], j.axis)))
        if j.lower is not None and j.upper is not None:
            jel.set("range", _txt((j.lower, j.upper)))
        for attr, v in (("damping", j.damping), ("frictionloss", j.friction), ("armature", j.armature)):
            if v:
                jel.set(attr, num(v))
        if j.q0:
            self.io.note("mjcf", j.name, f"initial position {j.q0:.6g} not kept: the file starts at 0")

    def body(self, parent: ET.Element, b: IOBody, rel: Pose, j: IOJoint | None) -> ET.Element:
        """One body at ``rel`` (its frame in the parent body's frame) with its inertial, joint and geoms."""
        el = ET.SubElement(parent, "body", name=b.name)
        _pose_attrs(el, rel)
        self.inertial(el, b)
        if j is None and parent is self.world and not self.g.fixed_base:
            ET.SubElement(el, "freejoint", name=f"{b.name}_free")
        elif j is not None:
            self.joint(el, j)
        for s in b.shapes:
            self.geom(el, s, s.pose)
        return el

    def tree(self) -> None:
        g = self.g
        for s in g.world_shapes:
            self.geom(self.world, s, s.pose)
        if g.root is None:
            return
        by_name = {b.name: b for b in g.bodies}
        els = {g.root.name: self.body(self.world, g.root, IDENTITY, g.base_joint)}
        first: dict[str, IOJoint] = {}
        for j in g.joints:                  # a body's first joint is its tree joint; later ones close loops
            first.setdefault(j.child, j)
        pending = [j for j in g.joints if first[j.child] is j]
        while pending:
            ready = [j for j in pending if j.parent in els]
            if not ready:
                break
            for j in ready:
                rel = compose(j.parent_pose, inverse(j.child_pose))    # child body in parent body at q = 0
                els[j.child] = self.body(els[j.parent], by_name[j.child], rel, j)
                pending.remove(j)
        for j in g.joints:
            if first[j.child] is not j:
                self.loop(j)

    def loop(self, j: IOJoint) -> None:
        """A loop-closing joint (a body's second joint): ball and fixed become connect and weld equalities."""
        tag = {"ball": "connect", "fixed": "weld"}.get(j.kind)
        if tag is None:
            self.io.note("mjcf", j.name, f"loop-closing {j.kind} joint between {j.parent} and {j.child} not kept "
                                         "(MJCF bodies have one parent); add an <equality> for it")
            return
        eq = self.root.find("equality")
        eq = eq if eq is not None else ET.SubElement(self.root, "equality")
        el = ET.SubElement(eq, tag, name=j.name, body1=j.parent, body2=j.child)
        if tag == "connect":
            el.set("anchor", _txt(j.parent_pose[:3]))

    def actuators(self) -> None:
        joints = ([self.g.base_joint] if self.g.base_joint is not None else []) + list(self.g.joints)
        driven = [j for j in joints if j.kind in ("revolute", "prismatic") and j.stiffness > 0]
        if not driven:
            return
        act = ET.SubElement(self.root, "actuator")
        for j in driven:
            a = ET.SubElement(act, "position", name=f"{j.name}_act", joint=j.name, kp=num(j.stiffness))
            if j.drive_damping:
                a.set("kv", num(j.drive_damping))
            if j.effort is not None:
                a.set("forcerange", _txt((-j.effort, j.effort)))


def group_mjcf(io: IOModel, g: Group, *, robot: bool = False, density: float | None = None) -> ET.Element:
    """The ``<mujoco>`` element of one group (``density`` sets every colliding geom's density for objects)."""
    w = _MjcfWriter(io, g, robot, density)
    w.tree()
    if robot:
        w.actuators()
    if not len(w.asset):
        w.root.remove(w.asset)
    return w.root


def write_group_mjcf(io: IOModel, g: Group, path: str | Path, *, robot: bool = False,
                     density: float | None = None) -> Path:
    """Write :func:`group_mjcf` to ``path``; MuJoCo must load it (raises ValueError with MuJoCo's message)."""
    import mujoco  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(group_mjcf(io, g, robot=robot, density=density))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode")
    try:
        mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"generated MJCF {path} does not load in MuJoCo: {exc}") from exc
    return path
