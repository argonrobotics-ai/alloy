# SPDX-License-Identifier: Apache-2.0
"""Parse a RoboCasa ``model.xml`` into plain data: bodies, joints, visual parts, collision geoms and regions.

RoboCasa models follow robosuite conventions: geoms of class ``visual`` draw, class ``collision`` collide, class
``region`` (``reg_bbox``, ``reg_int``, ``liquid``) and ``spawn`` mark volumes for task logic. Mesh, texture and
material assets are resolved to file paths relative to the model folder. Only ``pos`` / ``quat`` poses are read
(RoboCasa's exporter writes no other form); anything else raises with the attribute named.

Example::

    >>> from alloy.importers.robocasa.mjcf_in import read_model
    >>> model = read_model(Path("glass_cup/GlassCup003"))          # doctest: +SKIP
    >>> [v.mesh for v in model.root.visuals]                       # doctest: +SKIP
    ['visual/Clear.obj']
    >>> len(model.root.collisions), model.joints()                 # doctest: +SKIP
    (16, [])
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Vec = tuple[float, ...]
IDENTITY_QUAT: Vec = (1.0, 0.0, 0.0, 0.0)   # MuJoCo order (w, x, y, z)
_POSE_ATTRS = ("euler", "axisangle", "xyaxes", "zaxis")


@dataclass(frozen=True)
class MaterialInfo:
    """A MuJoCo material: colour factor, optional texture file, specular and shininess (0..1)."""

    name: str
    rgba: Vec = (1.0, 1.0, 1.0, 1.0)
    texture: str | None = None
    specular: float = 0.5
    shininess: float = 0.5


@dataclass(frozen=True)
class VisualPart:
    """A drawn mesh: file relative to the model folder, its scale, pose in its body and material."""

    mesh: str
    scale: Vec
    pos: Vec
    quat: Vec
    material: MaterialInfo | None


@dataclass(frozen=True)
class CollisionGeom:
    """A colliding geom: a primitive (``size`` in MuJoCo's convention) or a mesh file."""

    kind: str
    size: Vec
    pos: Vec
    quat: Vec
    mesh: str | None = None
    scale: Vec = (1.0, 1.0, 1.0)
    name: str = ""


@dataclass(frozen=True)
class Region:
    """A task region (non-colliding, invisible) or site: name, primitive kind, size and pose."""

    name: str
    kind: str
    size: Vec
    pos: Vec
    quat: Vec


@dataclass(frozen=True)
class JointInfo:
    """A hinge or slide joint with RoboCasa's limits and passive terms."""

    name: str
    kind: str
    axis: Vec
    pos: Vec
    range: Vec | None
    damping: float = 0.0
    frictionloss: float = 0.0
    armature: float = 0.0
    stiffness: float = 0.0


@dataclass
class BodyInfo:
    """A body and everything attached to it; ``children`` keeps file order."""

    name: str
    pos: Vec = (0.0, 0.0, 0.0)
    quat: Vec = IDENTITY_QUAT
    joints: list[JointInfo] = field(default_factory=list)
    visuals: list[VisualPart] = field(default_factory=list)
    collisions: list[CollisionGeom] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    children: list[BodyInfo] = field(default_factory=list)

    def walk(self) -> Iterator[BodyInfo]:
        """This body, then every descendant depth-first."""
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class ModelInfo:
    """A parsed model: its folder and the root body (RoboCasa's ``object`` body)."""

    folder: Path
    root: BodyInfo

    def bodies(self) -> list[BodyInfo]:
        return list(self.root.walk())

    def joints(self) -> list[JointInfo]:
        return [j for b in self.root.walk() for j in b.joints]

    def flattened(self) -> BodyInfo:
        """One body holding every geom and region in the root frame (for jointless models); ``ValueError`` if any
        body below the root has a joint (use the body tree instead)."""
        moving = [b.name or "?" for b in self.root.children for _ in b.walk() if any(x.joints for x in b.walk())]
        if moving or self.root.joints:
            raise ValueError(f"{self.folder}: jointed bodies {sorted(set(moving))}; this model is articulated")
        flat = BodyInfo(self.root.name)
        _flatten_into(flat, self.root, np.zeros(3), np.array(IDENTITY_QUAT), top=True)
        return flat


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_matrix(q: Vec | np.ndarray) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion (normalised first)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _moved(item, pos: np.ndarray, quat: np.ndarray):
    """A geom/region record re-expressed in the parent frame given its body's pose (pos, quat) in that frame."""
    p = pos + quat_matrix(quat) @ np.asarray(item.pos)
    q = quat_mul(quat, np.asarray(item.quat))
    return dataclasses.replace(item, pos=tuple(float(x) for x in p), quat=tuple(float(x) for x in q))


def _flatten_into(flat: BodyInfo, body: BodyInfo, pos: np.ndarray, quat: np.ndarray, top: bool = False) -> None:
    if not top:
        pos = pos + quat_matrix(quat) @ np.asarray(body.pos)
        quat = quat_mul(quat, np.asarray(body.quat))
    flat.visuals += [_moved(v, pos, quat) for v in body.visuals]
    flat.collisions += [_moved(c, pos, quat) for c in body.collisions]
    flat.regions += [_moved(r, pos, quat) for r in body.regions]
    for child in body.children:
        _flatten_into(flat, child, pos, quat)


def _vec(el: ET.Element, key: str, default: Vec) -> Vec:
    raw = el.get(key)
    return tuple(float(x) for x in raw.split()) if raw else default


def _check_pose(el: ET.Element, where: str) -> None:
    bad = [a for a in _POSE_ATTRS if el.get(a) is not None]
    if bad:
        raise ValueError(f"{where}: <{el.tag}> uses {bad}; only pos/quat poses are supported (convert the file)")


@dataclass
class _Assets:
    meshes: dict[str, tuple[str, Vec]]
    materials: dict[str, MaterialInfo]


def _read_assets(root: ET.Element) -> _Assets:
    textures = {t.get("name", ""): t.get("file", "") for t in root.iter("texture") if t.get("file")}
    meshes = {m.get("name") or Path(m.get("file", "")).stem: (m.get("file", ""), _vec(m, "scale", (1.0, 1.0, 1.0)))
              for m in root.iter("mesh") if m.get("file")}
    materials = {}
    for m in root.iter("material"):
        name = m.get("name", "")
        materials[name] = MaterialInfo(name, _vec(m, "rgba", (1.0, 1.0, 1.0, 1.0)), textures.get(m.get("texture", "")),
                                       float(m.get("specular", 0.5)), float(m.get("shininess", 0.5)))
    return _Assets(meshes, materials)


def _class_defaults(root: ET.Element) -> dict[str, dict[str, str]]:
    """class name -> its <geom> default attributes (nested classes inherit)."""
    out: dict[str, dict[str, str]] = {}

    def visit(node: ET.Element, inherited: dict[str, str]) -> None:
        geom = node.find("geom")
        own = {**inherited, **(geom.attrib if geom is not None else {})}
        out[node.get("class", "main")] = own
        for child in node.findall("default"):
            visit(child, own)

    for d in root.findall("default"):
        visit(d, {})
    return out


def _geom_role(attrs: dict[str, str], cls: str) -> str:
    if cls in ("region", "spawn"):
        return "region"
    if cls == "visual":
        return "visual"
    if cls == "collision":
        return "collision"
    no_contact = attrs.get("contype") == "0" and attrs.get("conaffinity") == "0"
    return ("visual" if attrs.get("type") == "mesh" else "region") if no_contact else "collision"


def _add_geom(body: BodyInfo, g: ET.Element, assets: _Assets, defaults: dict[str, dict[str, str]]) -> None:
    cls = g.get("class", "main")
    attrs = {**defaults.get("main", {}), **defaults.get(cls, {}), **g.attrib}
    _check_pose(g, f"geom {attrs.get('name', '?')}")
    kind = attrs.get("type", "sphere")
    pos, quat = _vec(g, "pos", (0.0, 0.0, 0.0)), _vec(g, "quat", IDENTITY_QUAT)
    role = _geom_role(attrs, cls)
    mesh_file, scale = assets.meshes.get(attrs.get("mesh", ""), (None, (1.0, 1.0, 1.0)))
    if role == "visual" and mesh_file:
        body.visuals.append(VisualPart(mesh_file, scale, pos, quat, assets.materials.get(attrs.get("material", ""))))
    elif role == "collision":
        size = tuple(float(x) for x in attrs.get("size", "0").split())
        body.collisions.append(CollisionGeom(kind, size, pos, quat, mesh_file, scale, attrs.get("name", "")))
    else:
        size = tuple(float(x) for x in attrs.get("size", "0").split())
        body.regions.append(Region(attrs.get("name", ""), kind, size, pos, quat))


def _joint(j: ET.Element) -> JointInfo:
    limited = j.get("limited", "auto") != "false" and j.get("range") is not None
    return JointInfo(j.get("name", ""), j.get("type", "hinge"), _vec(j, "axis", (0.0, 0.0, 1.0)),
                     _vec(j, "pos", (0.0, 0.0, 0.0)), _vec(j, "range", ()) if limited else None,
                     float(j.get("damping", 0)), float(j.get("frictionloss", 0)), float(j.get("armature", 0)),
                     float(j.get("stiffness", 0)))


def _body(el: ET.Element, assets: _Assets, defaults: dict[str, dict[str, str]]) -> BodyInfo:
    _check_pose(el, f"body {el.get('name', '?')}")
    body = BodyInfo(el.get("name", ""), _vec(el, "pos", (0.0, 0.0, 0.0)), _vec(el, "quat", IDENTITY_QUAT))
    for child in el:
        if child.tag == "joint":
            body.joints.append(_joint(child))
        elif child.tag == "geom":
            _add_geom(body, child, assets, defaults)
        elif child.tag == "site":
            body.regions.append(Region(child.get("name", ""), child.get("type", "sphere"),
                                       _vec(child, "size", (0.005,)), _vec(child, "pos", (0.0, 0.0, 0.0)),
                                       _vec(child, "quat", IDENTITY_QUAT)))
        elif child.tag == "body":
            body.children.append(_body(child, assets, defaults))
    return body


def _object_body(world: ET.Element, path: Path) -> ET.Element:
    named = world.find(".//body[@name='object']")
    if named is not None:
        return named
    bodies = world.findall("body")
    if len(bodies) != 1:
        raise ValueError(f"{path}: expected one root body or a body named 'object', found {len(bodies)} roots")
    return bodies[0]


def read_model(folder: Path) -> ModelInfo:
    """Parse ``folder/model.xml``; raises ``ValueError`` naming the file and element on anything unsupported."""
    path = folder / "model.xml"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"{path}: cannot read the MJCF ({exc})") from exc
    if root.find("compiler[@angle='degree']") is not None or root.find("compiler") is None:
        if any(j.get("range") for j in root.iter("joint")):
            raise ValueError(f"{path}: joint ranges must be in radians (<compiler angle=\"radian\"/>)")
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{path}: no <worldbody>")
    body = _body(_object_body(world, path), _read_assets(root), _class_defaults(root))
    return ModelInfo(folder, body)


__all__ = ["BodyInfo", "quat_matrix", "quat_mul", "CollisionGeom", "JointInfo", "MaterialInfo", "ModelInfo", "Region", "VisualPart",
           "read_model"]
