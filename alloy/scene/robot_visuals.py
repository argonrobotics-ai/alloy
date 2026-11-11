# SPDX-License-Identifier: Apache-2.0
"""Robot visual layers: display meshes stored next to a robot MJCF (``<dir>/visuals/``) and attached when it loads.

A robot pack's ``scene.xml`` keeps only physics (the library arm packs drop their source's visual geoms); its layer
holds the source's visual meshes (built by :mod:`alloy.tasks.emb_visuals`):

- ``visuals/visuals.json``: the source record and one entry per visual geom (pack body, mesh file, pos / quat in the
  body frame, rgba), or ``same_as`` another pack's layer, once per ``prefixes`` entry (body-name prefixes);
- ``visuals/*.msh``: the meshes (MuJoCo ``.msh``).

Attaching adds display geometry only: no collision, density 0, so no mass, inertia or contact changes.

- :func:`add_mjcf`: Newton's ``add_mjcf``, then the layer's display shapes (scene loads call this; set
  ``ALLOY_ROBOT_VISUALS=0`` to skip the layer, e.g. for training runs that never render);
- :func:`attach_newton`: the layer onto bodies a Newton builder already holds;
- :func:`attach_mjcf`: into a parsed MJCF tree (``<mesh>`` + ``<geom>``: contype = conaffinity = 0, group 2,
  density 0); names already in the tree are skipped;
- :func:`scene_with_visuals`: the scene with its layer as one MJCF file (cached), for MuJoCo's own renderer.

Example::

    from alloy._vendor import newton
    from alloy.scene.robot_visuals import add_mjcf
    b = newton.ModelBuilder()
    add_mjcf(b, "library/robots/emb_e1_fr3/scene.xml")      # the FR3 pack with its menagerie meshes
"""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from alloy.tasks.msh import MeshData, read_msh

VISUALS = "visuals"
MANIFEST = "visuals.json"
GROUP = 2
ENV_SWITCH = "ALLOY_ROBOT_VISUALS"      # "0": scene loads skip the layer


@dataclass(frozen=True)
class VisualGeom:
    name: str                       # geom name (``<body>_vis<k>_<material>``)
    body: str                       # pack body it rides on
    mesh: str                       # mesh asset name
    file: Path                      # the .msh file
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]   # w x y z, in the body frame
    rgba: tuple[float, float, float, float]
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


def layer_dir(xml: Path) -> Path:
    return Path(xml).parent / VISUALS


def has_visuals(xml: Path) -> bool:
    """Whether the MJCF file ``xml`` has a visual layer next to it."""
    return (layer_dir(xml) / MANIFEST).is_file()


def enabled() -> bool:
    """Whether scene loads attach layers (``ALLOY_ROBOT_VISUALS`` unset or not ``0``)."""
    return os.environ.get(ENV_SWITCH, "1").strip() != "0"


def visual_geoms(xml: Path) -> tuple[VisualGeom, ...]:
    """The resolved visual geoms of the scene ``xml`` (``same_as`` followed, prefixes applied); none without a layer."""
    return _resolve(Path(xml).resolve().parent, "", 0) if has_visuals(xml) else ()


@lru_cache(maxsize=64)
def _resolve(pack: Path, prefix: str, depth: int) -> tuple[VisualGeom, ...]:
    if depth > 4:
        raise ValueError(f"{pack}: visual layer same_as chain too deep")
    rec = json.loads((pack / VISUALS / MANIFEST).read_text())
    if "same_as" in rec:
        other = pack.parent / rec["same_as"]
        return tuple(g for p in rec.get("prefixes", [""]) for g in _resolve(other, prefix + p, depth + 1))
    return tuple(VisualGeom(prefix + g["name"], prefix + g["body"], prefix + g["mesh"], pack / VISUALS / g["file"],
                            tuple(g["pos"]), tuple(g["quat"]), tuple(g["rgba"]), tuple(g.get("scale", (1.0, 1.0, 1.0))))
                 for g in rec["geoms"])  # fmt: skip


@lru_cache(maxsize=256)
def _mesh(path: Path) -> MeshData:
    return read_msh(path)


def clear_cache() -> None:
    """Forget resolved layers and loaded meshes (after a layer is rewritten in this process)."""
    _resolve.cache_clear()
    _mesh.cache_clear()


def _f(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def attach_mjcf(root: ET.Element, xml: Path, prefix: str = "") -> int:
    """Add the layer of the scene ``xml`` to the MJCF tree ``root`` (bodies named as in the pack, plus ``prefix``):
    a ``<mesh>`` (absolute path) and a visual-only ``<geom>`` per entry; a geom name already in the tree is skipped.
    Returns how many geoms were added."""
    geoms = visual_geoms(xml)
    names = {el.get("name") for el in root.iter() if el.get("name")}
    bodies = {b.get("name"): b for b in root.find("worldbody").iter("body")}
    asset = root.find("asset") if root.find("asset") is not None else ET.SubElement(root, "asset")
    added = 0
    for g in geoms:
        name, mesh = prefix + g.name, prefix + g.mesh
        if name in names:
            continue
        body = bodies.get(prefix + g.body)
        if body is None:
            raise KeyError(f"{xml}: visual geom {name!r} needs body {prefix + g.body!r}, not in the tree")
        if mesh not in names:
            attrs = {"name": mesh, "file": str(g.file)} | ({"scale": _f(g.scale)} if g.scale != (1.0, 1.0, 1.0) else {})
            ET.SubElement(asset, "mesh", attrs)
            names.add(mesh)
        ET.SubElement(body, "geom", {
            "name": name, "type": "mesh", "mesh": mesh, "pos": _f(g.pos), "quat": _f(g.quat), "contype": "0",
            "conaffinity": "0", "group": str(GROUP), "density": "0", "rgba": _f(g.rgba)})  # fmt: skip
        names.add(name)
        added += 1
    return added


def scene_with_visuals(xml: Path, cache: Path | None = None) -> Path:
    """The scene ``xml`` with its layer attached, as one MJCF file with absolute mesh paths (content-addressed in
    ``cache``, default ``~/.cache/alloy/visuals``); for MuJoCo's own renderer. The physics is the pack's."""
    xml = Path(xml).resolve()
    tree = ET.parse(xml)
    compiler = tree.getroot().find("compiler")
    meshdir = compiler.attrib.pop("meshdir", "") if compiler is not None else ""
    for m in tree.getroot().iter("mesh"):
        if m.get("file"):
            m.set("file", str((xml.parent / meshdir / m.get("file")).resolve()))
    attach_mjcf(tree.getroot(), xml)
    text = ET.tostring(tree.getroot(), encoding="unicode")
    cache = Path(cache or os.environ.get("ALLOY_VISUALS_CACHE", Path.home() / ".cache" / "alloy" / "visuals"))
    out = cache / f"{xml.parent.name}-{hashlib.sha256(text.encode()).hexdigest()[:12]}.xml"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        tmp.replace(out)
    return out


def attach_newton(builder, xml: Path, bodies: range) -> list[int]:
    """Add the layer of the scene ``xml`` to a Newton ``builder`` as display shapes (no collision, density 0) on the
    bodies in ``bodies`` that carry the pack's names (the bodies one ``add_mjcf`` of that scene created), and hide
    those bodies' collision shapes from display (as Newton's importer does for an MJCF with visuals; the flag is
    display only). Returns the new shape indices."""
    from alloy._vendor import newton
    import warp as wp

    geoms = visual_geoms(xml)
    if not geoms:
        return []
    index: dict[str, int] = {}
    for i in bodies:
        index.setdefault(str(builder.body_label[i]).rsplit("/", 1)[-1], i)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False,
                                          is_visible=True)  # fmt: skip
    meshes: dict[Path, newton.Mesh] = {}
    out = []
    for g in geoms:
        body = index.get(g.body)
        if body is None:
            raise KeyError(f"{xml}: visual geom {g.name!r} needs body {g.body!r}, not among the imported bodies")
        if g.file not in meshes:
            data = _mesh(g.file)
            meshes[g.file] = newton.Mesh(data.vertices, data.faces.reshape(-1), compute_inertia=False)
        w, x, y, z = g.quat
        out.append(builder.add_shape_mesh(body, xform=wp.transform(wp.vec3(*g.pos), wp.quat(x, y, z, w)),
                                          mesh=meshes[g.file], scale=wp.vec3(*g.scale), cfg=cfg, color=g.rgba[:3],
                                          opacity=g.rgba[3], label=g.name))  # fmt: skip
    _hide_hulls(builder, {index[g.body] for g in geoms}, set(out))
    return out


def _hide_hulls(builder, dressed: set[int], keep: set[int]) -> None:
    from alloy._vendor import newton

    collide, visible = int(newton.ShapeFlags.COLLIDE_SHAPES), int(newton.ShapeFlags.VISIBLE)
    for s in range(builder.shape_count):
        if s not in keep and builder.shape_body[s] in dressed and builder.shape_flags[s] & collide:
            builder.shape_flags[s] &= ~visible


def add_mjcf(builder, path, **kw) -> list[int]:
    """``builder.add_mjcf(path, **kw)``, then the file's visual layer when it has one, ``parse_visuals`` is on
    (Newton's default) and :func:`enabled`. Returns the display shapes added."""
    first = builder.body_count
    builder.add_mjcf(str(path), **kw)
    if not kw.get("parse_visuals", True) or not enabled() or not has_visuals(Path(path)):
        return []
    return attach_newton(builder, Path(path), range(first, builder.body_count))
