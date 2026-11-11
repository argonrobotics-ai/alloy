# SPDX-License-Identifier: Apache-2.0
"""Write the Alloy MJCF for an imported asset: visual meshes (no mass, no contact), collision hulls and mass.

An object is one body named after its card, with an explicit ``<inertial>`` (from ``massprops``), its CoACD hulls as
``collision`` mesh geoms and its visual meshes as ``visual`` geoms. A fixture keeps RoboCasa's body tree and its
hinge/slide joints; its collision primitives are kept, mesh colliders are re-hulled to at most 64 vertices, and each
collision geom carries an effective density (see ``pipeline``). Task regions (``reg_bbox``, ``reg_int``...) become
sites in group 4. Newton's MJCF importer treats class ``visual`` as visual-only and class ``collision`` as colliders.

Example::

    >>> from alloy.importers.robocasa.mjcf_out import ModelWriter
    >>> w = ModelWriter("robocasa_glass_cup_003", header="from RoboCasa GlassCup003 (CC BY 4.0)")
    >>> # w.add_visual(...); w.add_hull(...); w.write(Path("out/model.xml"))
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy.importers.robocasa.massprops import Inertial
from alloy.importers.robocasa.meshio import write_msh
from alloy.importers.robocasa.mjcf_in import IDENTITY_QUAT, CollisionGeom, JointInfo, Region, VisualPart

REGION_GROUP = "4"


def fmt(values) -> str:
    """Space-separated floats with 9 significant digits (stable text across runs)."""
    return " ".join(f"{float(v):.9g}" for v in values)


def _pose(el: ET.Element, pos, quat) -> None:
    if any(abs(float(x)) > 0 for x in pos):
        el.set("pos", fmt(pos))
    if tuple(float(x) for x in quat) != IDENTITY_QUAT:
        el.set("quat", fmt(quat))


@dataclass
class ModelWriter:
    """Builds one MJCF file; mesh/texture paths are relative to the file's folder."""

    model: str
    header: str = ""
    friction: tuple[float, float, float] = (0.5, 0.005, 0.0001)
    _assets: list[ET.Element] = field(default_factory=list)
    _textures: dict[str, str] = field(default_factory=dict)
    _materials: dict[tuple, str] = field(default_factory=dict)
    _n_mesh: int = 0
    flat: dict[str, tuple] = field(default_factory=dict)    # mesh file without UVs -> rgba that replaces its texture

    def _mesh(self, file: str, scale=(1.0, 1.0, 1.0), prefix: str = "m") -> str:
        name = f"{prefix}{self._n_mesh}"
        self._n_mesh += 1
        el = ET.Element("mesh", name=name, file=file)
        if tuple(float(s) for s in scale) != (1.0, 1.0, 1.0):
            el.set("scale", fmt(scale))
        self._assets.append(el)
        return name

    def _material(self, rgba, texture: str | None, specular: float, shininess: float) -> str:
        key = (tuple(rgba), texture, specular, shininess)
        if key in self._materials:
            return self._materials[key]
        name = f"mat{len(self._materials)}"
        el = ET.Element("material", name=name, rgba=fmt(rgba), specular=f"{specular:.6g}",
                        shininess=f"{shininess:.6g}")
        if texture:
            if texture not in self._textures:
                self._textures[texture] = f"tex{len(self._textures)}"
                self._assets.append(ET.Element("texture", name=self._textures[texture], type="2d", file=texture))
            el.set("texture", self._textures[texture])
        self._assets.append(el)
        self._materials[key] = name
        return name

    def visual(self, body: ET.Element, part: VisualPart, mesh_file: str) -> None:
        """A drawn mesh on ``body`` (``mesh_file`` relative to the model folder)."""
        g = ET.SubElement(body, "geom", {"class": "visual", "mesh": self._mesh(mesh_file, part.scale, "vis")})
        m = part.material
        if m is not None:
            tex, rgba = (None, self.flat[mesh_file]) if mesh_file in self.flat else (m.texture, m.rgba)
            g.set("material", self._material(rgba, tex, m.specular, m.shininess))
        _pose(g, part.pos, part.quat)

    def hull(self, body: ET.Element, mesh_file: str, density: float | None = None) -> None:
        """A convex collision mesh (already in the body frame)."""
        g = ET.SubElement(body, "geom", {"class": "collision", "mesh": self._mesh(mesh_file, prefix="hull")})
        if density is not None:
            g.set("density", f"{density:.9g}")

    def visual_box(self, body: ET.Element, half, pos, rgba) -> None:
        """A drawn (non-colliding, massless) box, e.g. a cabinet board."""
        g = ET.SubElement(body, "geom", {"class": "visual", "type": "box", "size": fmt(half),
                                         "material": self._material(tuple(rgba), None, 0.2, 0.3)})
        _pose(g, pos, IDENTITY_QUAT)

    def primitive(self, body: ET.Element, c: CollisionGeom, density: float | None = None) -> None:
        """A collision primitive copied from the source model."""
        g = ET.SubElement(body, "geom", {"class": "collision", "type": c.kind, "size": fmt(c.size)})
        if c.name:
            g.set("name", c.name)
        if density is not None:
            g.set("density", f"{density:.9g}")
        _pose(g, c.pos, c.quat)

    def _defaults(self, root: ET.Element) -> None:
        d = ET.SubElement(root, "default")
        vis = ET.SubElement(d, "default", {"class": "visual"})
        ET.SubElement(vis, "geom", type="mesh", contype="0", conaffinity="0", group="2", density="0")
        col = ET.SubElement(d, "default", {"class": "collision"})
        ET.SubElement(col, "geom", type="mesh", group="3", density="0", friction=fmt(self.friction))

    def document(self, body: ET.Element) -> ET.ElementTree:
        """The whole MJCF with ``body`` as the only child of ``<worldbody>``."""
        root = ET.Element("mujoco", model=self.model)
        ET.SubElement(root, "compiler", angle="radian")
        asset = ET.SubElement(root, "asset")
        asset.extend(self._assets)
        self._defaults(root)
        ET.SubElement(root, "worldbody").append(body)
        ET.indent(root, space="  ")
        return ET.ElementTree(root)

    def write(self, body: ET.Element, path: Path) -> Path:
        """Write the file (with the header as a leading comment) and return its path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = ET.tostring(self.document(body).getroot(), encoding="unicode")
        comment = f"<!-- {self.header.replace('--', '-')} -->\n" if self.header else ""
        path.write_text(comment + text + "\n")
        return path


def body_element(name: str, pos=(0.0, 0.0, 0.0), quat=IDENTITY_QUAT) -> ET.Element:
    """A ``<body>`` element with an optional pose."""
    el = ET.Element("body", name=name)
    _pose(el, pos, quat)
    return el


def add_inertial(body: ET.Element, inertial: Inertial) -> None:
    """Explicit mass properties (Newton and MuJoCo use them instead of geom densities)."""
    el = ET.Element("inertial", pos=fmt(inertial.pos), mass=f"{inertial.mass:.9g}",
                    diaginertia=fmt(inertial.diaginertia))
    if tuple(inertial.quat) != IDENTITY_QUAT:
        el.set("quat", fmt(inertial.quat))
    body.insert(0, el)


def add_joint(body: ET.Element, j: JointInfo) -> None:
    """A hinge/slide joint with RoboCasa's limits and passive terms."""
    el = ET.SubElement(body, "joint", name=j.name, type=j.kind, axis=fmt(j.axis))
    if any(abs(x) > 0 for x in j.pos):
        el.set("pos", fmt(j.pos))
    if j.range:
        el.set("range", fmt(j.range))
        el.set("limited", "true")
    for key in ("damping", "frictionloss", "armature", "stiffness"):
        value = getattr(j, key)
        if value:
            el.set(key, f"{value:.6g}")


def add_region(body: ET.Element, r: Region) -> None:
    """A task region as a site (group 4, no physics)."""
    kind = r.kind if r.kind in ("sphere", "capsule", "ellipsoid", "cylinder", "box") else "box"
    el = ET.SubElement(body, "site", name=r.name or "region", type=kind, size=fmt(r.size), group=REGION_GROUP)
    _pose(el, r.pos, r.quat)


def write_hull(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """One convex hull as a MuJoCo ``.msh`` file."""
    write_msh(path, np.asarray(verts, np.float64), np.asarray(faces, np.int64))


__all__ = ["ModelWriter", "add_inertial", "add_joint", "add_region", "body_element", "fmt", "write_hull"]
