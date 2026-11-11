# SPDX-License-Identifier: Apache-2.0
"""URDF to ``IOModel`` with no extra dependencies (meshes: OBJ and STL).

Links become bodies at their pose for zero joint positions; each link's frame is its URDF link frame. Collision
elements become colliding shapes, visual elements visible-only shapes (a visual identical to a collision of the same
name merges into it). PyBullet's ``<contact>`` element gives the link's friction and restitution.

Example::

    from alloy.io.urdf_read import read_urdf
    io = read_urdf("robot.urdf", floating=False)
    print([b.name for b in io.bodies], [j.kind for j in io.joints])
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from alloy.io.errors import ImportFileError
from alloy.io.geometry import compose, quat_matrix, quat_mul, read_obj
from alloy.io.model import IDENTITY, IOBody, IOJoint, IOMaterial, IOMesh, IOModel, IOShape, Pose

log = logging.getLogger("alloy.io")
_JOINTS = {"revolute": "revolute", "continuous": "revolute", "prismatic": "prismatic", "fixed": "fixed",
           "floating": "free"}




def _attr(el: ET.Element | None, key: str) -> float | None:
    """A float attribute of ``el`` (None if the element or attribute is missing)."""
    return float(el.get(key)) if el is not None and el.get(key) is not None else None


def _floats(text: str | None, n: int, default: float = 0.0) -> list[float]:
    vals = [float(x) for x in (text or "").split()]
    return (vals + [default] * n)[:n]


def rpy_quat(r: float, p: float, y: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion of URDF roll-pitch-yaw (fixed X, then Y, then Z)."""
    qx = (math.sin(r / 2), 0.0, 0.0, math.cos(r / 2))
    qy = (0.0, math.sin(p / 2), 0.0, math.cos(p / 2))
    qz = (0.0, 0.0, math.sin(y / 2), math.cos(y / 2))
    q = quat_mul(qz, quat_mul(qy, qx))
    return tuple(float(c) for c in q / np.linalg.norm(q))


def origin_pose(el: ET.Element | None) -> Pose:
    """Pose of an ``<origin xyz rpy>`` element (identity when absent)."""
    if el is None:
        return IDENTITY
    o = el.find("origin")
    if o is None:
        return IDENTITY
    return (*_floats(o.get("xyz"), 3), *rpy_quat(*_floats(o.get("rpy"), 3)))


def read_stl(path: Path) -> IOMesh:
    """Triangles of a binary or ASCII STL file (duplicate vertices merged)."""
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if n >= 0 and len(data) == 84 + 50 * n:
        rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        tris = rec["v"].astype(np.float64)
    else:
        nums = [line.split()[1:4] for line in data.decode(errors="replace").splitlines()
                if line.strip().startswith("vertex")]
        tris = np.asarray(nums, np.float64).reshape(-1, 3, 3)
    verts, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    return IOMesh(verts, inv.reshape(-1, 3).astype(np.int32))


def resolve_mesh(filename: str, urdf_dir: Path, package_dirs: tuple[Path, ...] = ()) -> Path:
    """Find a URDF mesh file: ``package://pkg/rel`` under the given package folders or the URDF's parents,
    ``file://`` and relative paths from the URDF folder."""
    name = filename.replace("file://", "")
    if name.startswith("package://"):
        pkg, _, rel = name[len("package://"):].partition("/")
        for root in (*package_dirs, *[urdf_dir, *urdf_dir.parents][:4]):
            for cand in (root / pkg / rel, root / rel):
                if cand.exists():
                    return cand
        raise ImportFileError(f"mesh {filename} not found; pass package_dirs=[<folder holding '{pkg}'>]")
    p = Path(name)
    p = p if p.is_absolute() else urdf_dir / p
    if not p.exists():
        raise ImportFileError(f"mesh {filename} not found at {p}")
    return p


def load_mesh_file(path: Path, scale) -> IOMesh:
    """An OBJ or STL mesh, scaled; other formats raise with the fix."""
    suffix = path.suffix.lower()
    if suffix == ".obj":
        mesh = read_obj(path)
    elif suffix == ".stl":
        mesh = read_stl(path)
    else:
        raise ImportFileError(f"{path.name}: {suffix} meshes are not read; convert it to .obj or .stl")
    return IOMesh(mesh.vertices * np.asarray(scale, np.float64), mesh.faces)


class _Reader:
    """One URDF document."""

    def __init__(self, path: Path, package_dirs: tuple[Path, ...]) -> None:
        self.path, self.dirs = path, package_dirs
        self.io = IOModel(name=path.stem)
        self.root = ET.parse(path).getroot()
        if self.root.tag != "robot":
            raise ImportFileError(f"{path}: root element is <{self.root.tag}>, a URDF needs <robot>")

    def geometry(self, el: ET.Element, name: str) -> IOShape | None:
        g = el.find("geometry")
        child = g[0] if g is not None and len(g) else None
        if child is None:
            return None
        pose, tag = origin_pose(el), child.tag
        if tag == "box":
            return IOShape(name, "box", tuple(0.5 * x for x in _floats(child.get("size"), 3)), pose)
        if tag == "sphere":
            return IOShape(name, "sphere", (float(child.get("radius")),), pose)
        if tag in ("cylinder", "capsule"):
            return IOShape(name, tag, (float(child.get("radius")), 0.5 * float(child.get("length"))), pose)
        if tag == "mesh":
            src = resolve_mesh(child.get("filename", ""), self.path.parent, self.dirs)
            mesh = load_mesh_file(src, _floats(child.get("scale"), 3, 1.0) if child.get("scale") else (1, 1, 1))
            return IOShape(name, "mesh", (), pose, mesh=mesh, convex=True)
        self.io.note("urdf", name, f"<{tag}> geometry is not read")
        return None

    def shapes(self, link: ET.Element, lname: str) -> list[IOShape]:
        out: list[IOShape] = []
        for i, el in enumerate(link.findall("collision")):
            s = self.geometry(el, el.get("name") or f"{lname}_collision_{i}")
            if s is not None:
                out.append(IOShape(**{**s.__dict__, "visible": False}))
        for i, el in enumerate(link.findall("visual")):
            s = self.geometry(el, el.get("name") or f"{lname}_visual_{i}")
            if s is None:
                continue
            twin = next((k for k, c in enumerate(out) if c.name == s.name and c.kind == s.kind and c.size == s.size
                         and c.pose == s.pose), None)
            if twin is not None:
                out[twin] = IOShape(**{**out[twin].__dict__, "visible": True})
            else:
                out.append(IOShape(**{**s.__dict__, "collide": False}))
        return out

    def material(self, link: ET.Element, lname: str) -> str:
        c = link.find("contact")
        val = lambda tag: _attr(c.find(tag), "value") if c is not None else None  # noqa: E731
        mu = val("lateral_friction")
        if mu is None:
            return ""
        m = IOMaterial(f"{lname}_surface", 0.0, mu, mu, val("restitution") or 0.0, val("spinning_friction") or 0.0,
                       val("rolling_friction") or 0.0, origin="file")
        self.io.materials[m.name] = m
        return m.name

    def body(self, link: ET.Element, pose: Pose) -> IOBody:
        name = link.get("name")
        mat = self.material(link, name)
        shapes = tuple(IOShape(**{**s.__dict__, "material": mat}) for s in self.shapes(link, name))
        inert = link.find("inertial")
        if inert is None:
            return IOBody(name, pose, 0.0, shapes=shapes)
        o = origin_pose(inert)
        i = inert.find("inertia")
        get = lambda k: float(i.get(k, 0.0)) if i is not None else 0.0  # noqa: E731
        tensor = np.array([[get("ixx"), get("ixy"), get("ixz")], [get("ixy"), get("iyy"), get("iyz")],
                           [get("ixz"), get("iyz"), get("izz")]])
        r = quat_matrix(o[3:])
        tensor = r @ tensor @ r.T
        mass = float(inert.find("mass").get("value")) if inert.find("mass") is not None else 0.0
        return IOBody(name, pose, mass, tuple(o[:3]), tuple(tuple(float(x) for x in row) for row in tensor), shapes)

    def joint(self, el: ET.Element) -> IOJoint:
        typ = el.get("type", "")
        kind = _JOINTS.get(typ)
        name = el.get("name")
        if kind is None:
            self.io.note("urdf", name, f"{typ} joints are not read: written as fixed")
            kind = "fixed"
        lim, dyn, axis = el.find("limit"), el.find("dynamics"), el.find("axis")
        g = _attr
        if el.find("mimic") is not None:
            self.io.note("urdf", name, "mimic joints are read as independent joints")
        limited = typ in ("revolute", "prismatic")
        effort, vel = g(lim, "effort"), g(lim, "velocity")
        return IOJoint(name, kind, el.find("parent").get("link"), el.find("child").get("link"), origin_pose(el),
                       IDENTITY, tuple(_floats(axis.get("xyz") if axis is not None else "1 0 0", 3)),
                       g(lim, "lower") if limited else None, g(lim, "upper") if limited else None,
                       effort if effort is None or effort < 1e5 else None, vel if vel is None or vel < 1e5 else None,
                       damping=g(dyn, "damping") or 0.0, friction=g(dyn, "friction") or 0.0)

    def read(self, base_pose: Pose, floating: bool) -> IOModel:
        links = {lk.get("name"): lk for lk in self.root.findall("link")}
        joints = [self.joint(el) for el in self.root.findall("joint")]
        children = {j.child for j in joints}
        roots = [n for n in links if n not in children]
        if len(roots) != 1:
            raise ImportFileError(f"{self.path}: URDF needs exactly one root link, found {roots}")
        poses = {roots[0]: base_pose}
        pending = list(joints)
        while pending:
            ready = [j for j in pending if j.parent in poses]
            if not ready:
                raise ImportFileError(f"{self.path}: joints {[j.name for j in pending]} hang off unknown links")
            for j in ready:
                poses[j.child] = compose(poses[j.parent], j.parent_pose)
                pending.remove(j)
        self.io.bodies = [self.body(links[n], poses[n]) for n in links if n in poses]
        root_joint = IOJoint(f"{roots[0]}_root", "free" if floating else "fixed", None, roots[0],
                             base_pose if not floating else IDENTITY)
        self.io.joints = [root_joint, *joints]
        self.io.groups = {self.root.get("name", self.path.stem): "robot" if joints else "object"}
        for k, b in enumerate(self.io.bodies):
            self.io.bodies[k] = IOBody(**{**b.__dict__, "group": self.root.get("name", self.path.stem)})
        return self.io


def read_urdf(path: str | Path, *, floating: bool | None = None, base_pose: Pose = IDENTITY,
              package_dirs: tuple[str | Path, ...] = ()) -> IOModel:
    """``IOModel`` of a URDF file. ``floating`` None: free root when the file has no joints (a loose object),
    fixed root otherwise (a robot); True/False forces it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"URDF {path} not found")
    r = _Reader(path, tuple(Path(p) for p in package_dirs))
    has_joints = r.root.find("joint") is not None
    io = r.read(tuple(base_pose), (not has_joints) if floating is None else floating)
    log.info("read URDF %s: %d links, %d joints", path.name, len(io.bodies), len(io.joints) - 1)
    return io
