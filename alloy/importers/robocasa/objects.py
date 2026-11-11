# SPDX-License-Identifier: Apache-2.0
"""Import one RoboCasa object: scale check, CoACD hulls, mass from part densities, Alloy MJCF.

The object's bodies are flattened into one rigid body (RoboCasa nests jointless bodies for lids and contents). Each
visual part gets a density from the category table by its role (body, Clear, Prop, Lid); closed parts are solids,
open parts thin shells. The plausible mass and size ranges of the category flag objects whose numbers look wrong;
flagged objects are still written, and the flags go to the report and the sourcing queue.

Example::

    >>> from alloy.importers.robocasa.objects import import_object
    >>> res = import_object(Path("work/glass_cup/GlassCup003"), "glass_cup/GlassCup003", Path("out"))  # doctest: +SKIP
    >>> res.name, round(res.mass_kg, 3), res.flags                                                      # doctest: +SKIP
    ('robocasa_glass_cup_003', 0.276, [])
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo
from alloy.importers.robocasa import mjcf_out as mo
from alloy.importers.robocasa import naming
from alloy.importers.robocasa.hulls import HullParams, decompose, hull_mesh, reduce_hull
from alloy.importers.robocasa.massprops import Inertial, PartMass, body_inertial
from alloy.importers.robocasa.materials import MaterialTable, ObjectSpec, default_table, part_role
from alloy.importers.robocasa.meshio import read_obj_mesh, read_obj_mujoco, write_msh
from alloy.importers.robocasa.mjcf_in import BodyInfo, VisualPart, quat_matrix, read_model

log = logging.getLogger("alloy.importers.robocasa")

TOLERANCE = 1.25        # a value within [lo / 1.25, hi * 1.25] of the plausible range is not flagged


@dataclass
class Part:
    """A visual part in the body frame with its role, material and closedness."""

    name: str
    role: str
    material: str
    mesh: geo.MeshData
    closed: bool


@dataclass
class ObjectResult:
    """What one import produced (JSON-able through ``dataclasses.asdict``)."""

    name: str
    asset: str
    category: str
    model: str
    surface: str
    mass_kg: float
    mass_parts: dict[str, float]
    mass_note: str
    extents_m: tuple[float, float, float]
    hulls: int
    flags: list[str] = field(default_factory=list)
    inertial: Inertial | None = None


def part_mesh(folder: Path, part: VisualPart) -> geo.MeshData:
    """A visual part's mesh scaled and posed into the body frame."""
    m = geo.read_obj(folder / part.mesh)
    v = (m.vertices * np.asarray(part.scale)) @ quat_matrix(part.quat).T + np.asarray(part.pos)
    return geo.MeshData(v, m.faces)


def load_parts(folder: Path, body: BodyInfo, spec: ObjectSpec) -> list[Part]:
    """Every visual part with its role and density material; ``ValueError`` if the model draws nothing."""
    if not body.visuals:
        raise ValueError(f"{folder}: no visual meshes; nothing to import")
    out = []
    for v in body.visuals:
        mesh = part_mesh(folder, v)
        role = part_role(Path(v.mesh).stem)
        out.append(Part(Path(v.mesh).stem, role, spec.material_for(role), mesh, geo.is_watertight(mesh)))
    return out


def merged(parts: list[Part]) -> geo.MeshData:
    """All parts as one triangle soup (input for CoACD and extents)."""
    offsets = np.cumsum([0] + [len(p.mesh.vertices) for p in parts])
    verts = np.concatenate([p.mesh.vertices for p in parts])
    faces = np.concatenate([p.mesh.faces + o for p, o in zip(parts, offsets)])
    return geo.MeshData(verts, faces)


def range_flag(what: str, value: float, bounds: tuple[float, float], unit: str) -> list[str]:
    """``[]`` if ``value`` sits in the tolerant range, else one flag naming the value and the range."""
    lo, hi = bounds
    if lo / TOLERANCE <= value <= hi * TOLERANCE:
        return []
    return [f"{what} {value:.4g} {unit} outside plausible {lo:g}-{hi:g} {unit}"]


def object_mass(parts: list[Part], spec: ObjectSpec, table: MaterialTable) -> tuple[Inertial, str]:
    """Mass properties and the provenance note (densities, solid or shell per part)."""
    pm, notes = [], []
    for p in parts:
        rho = table.density(p.material).value
        pm.append(PartMass(p.name, p.mesh.vertices, p.mesh.faces, rho, p.closed, spec.shell_mm / 1000.0))
        how = "solid" if p.closed else f"shell {spec.shell_mm:g} mm"
        notes.append(f"{p.name}: {p.material} {rho:g} kg/m3, {how}")
    try:
        inertial = body_inertial(pm)
    except ValueError:
        pm, notes = _shell_fallback(pm, notes)
        inertial = body_inertial(pm)
    return inertial, "ESTIMATE: density x visual-mesh volume; " + "; ".join(notes)


def _shell_fallback(pm: list[PartMass], notes: list[str]) -> tuple[list[PartMass], list[str]]:
    """Closed parts whose solid inertia MuJoCo rejects (self-intersecting shells) are recomputed as shells."""
    out, new_notes = [], []
    for p, note in zip(pm, notes):
        if p.closed:
            try:
                body_inertial([p])
            except ValueError:
                log.info("part %s: solid inertia rejected, using a %.3g mm shell", p.name, p.shell_t * 1000)
                p, note = dataclasses.replace(p, closed=False), note.replace("solid", "shell (solid rejected)")
        out.append(p)
        new_notes.append(note)
    return out, new_notes


def dominant_colour(png: Path) -> tuple[float, float, float]:
    """The most common colour of a texture (4 bits per channel bins, averaged within the bin), 0..1."""
    import mujoco  # noqa: PLC0415 - MuJoCo decodes the PNG; heavy import only when needed

    m = mujoco.MjModel.from_xml_string(f'<mujoco><asset><texture name="t" type="2d" file="{png}"/></asset></mujoco>')
    n, h, w = int(m.tex_nchannel[0]), int(m.tex_height[0]), int(m.tex_width[0])
    px = np.asarray(m.tex_data[m.tex_adr[0]: m.tex_adr[0] + n * h * w]).reshape(-1, n)[:, :3].astype(np.int64)
    bins = (px[:, 0] >> 4) * 256 + (px[:, 1] >> 4) * 16 + (px[:, 2] >> 4)
    top = np.bincount(bins).argmax()
    return tuple(float(x) for x in px[bins == top].mean(axis=0) / 255.0)


def visual_mesh(path: Path):
    """A visual mesh as MuJoCo reads it (RoboCasa's reference look: MuJoCo leaves out the UV-less collider
    sub-objects some Lightwheel OBJs embed); this module's own OBJ reader if MuJoCo cannot load the file."""
    try:
        verts, uvs, normals, faces = read_obj_mujoco(path)
    except ValueError as exc:
        log.warning("%s: MuJoCo cannot read it (%s); using the plain OBJ reader", path, exc)
        return read_obj_mesh(path)
    if uvs is not None and path.suffix.lower() == ".obj" and not _has_uvs(path):
        uvs = None                    # MuJoCo makes up coordinates for an unmapped mesh; the atlas cannot use them
    return verts, uvs, normals, faces


def _has_uvs(path: Path) -> bool:
    """True if an OBJ file has any ``vt`` line."""
    with open(path, errors="replace") as fh:
        return any(line.startswith("vt ") for line in fh)


def convert_visual_files(src: Path, dst: Path, visuals: list[VisualPart]) -> tuple[dict[str, str], dict[str, tuple]]:
    """Write each visual mesh as ``.msh`` (UVs kept) and copy the textures it uses. Returns (source -> written mesh
    path, written mesh path -> rgba) where the second map lists textured parts whose mesh has no UVs at all: they
    are drawn in the texture's dominant colour instead (``ModelWriter.flat``)."""
    written: dict[str, str] = {}
    flat: dict[str, tuple] = {}
    textures: set[str] = set()
    for v in visuals:
        if v.mesh in written:
            continue
        rel = str(Path(v.mesh).with_suffix(".msh"))
        verts, uvs, normals, faces = visual_mesh(src / v.mesh)
        write_msh(dst / rel, verts, faces, uvs, normals)
        written[v.mesh] = rel
        tex = v.material.texture if v.material is not None else None
        if tex and uvs is None:
            flat[rel] = (*dominant_colour(src / tex), float(v.material.rgba[3]))
            log.info("%s has no UVs: drawn in %s's dominant colour", v.mesh, tex)
        elif tex:
            textures.add(tex)
    for tex in sorted(textures):
        (dst / tex).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / tex, dst / tex)
    return written, flat


def given_hulls(folder: Path, body: BodyInfo, max_verts: int = 64) -> list[tuple[np.ndarray, np.ndarray]]:
    """RoboCasa's own collision meshes, posed into the body frame and capped at ``max_verts`` vertices (fallback
    when CoACD cannot decompose a mesh)."""
    out = []
    for c in body.collisions:
        if c.kind != "mesh" or not c.mesh:
            continue
        m = geo.read_obj(folder / c.mesh)
        v = (m.vertices * np.asarray(c.scale)) @ quat_matrix(c.quat).T + np.asarray(c.pos)
        try:
            out.append(hull_mesh(reduce_hull(v, max_verts)))
        except ValueError as exc:
            log.info("given hull %s dropped: %s", c.mesh, exc)
    if not out:
        raise ValueError(f"{folder}: no usable collision meshes in the source model to fall back on")
    return out


def write_object_model(name: str, out_dir: Path, body: BodyInfo, hulls: list, inertial: Inertial,
                       header: str, meshes: tuple[dict[str, str], dict[str, tuple]]) -> Path:
    """The object's MJCF: one body, explicit inertial, visual parts, hull colliders, regions as sites."""
    meshes, flat = meshes
    writer = mo.ModelWriter(name, header, flat=flat)
    el = mo.body_element(name)
    mo.add_inertial(el, inertial)
    for v in body.visuals:
        writer.visual(el, v, meshes[v.mesh])
    for i, (verts, faces) in enumerate(hulls):
        rel = f"hulls/hull_{i:02d}.msh"
        mo.write_hull(out_dir / rel, verts, faces)
        writer.hull(el, rel)
    for r in body.regions:
        mo.add_region(el, r)
    return writer.write(el, out_dir / "model.xml")


def import_object(folder: Path, asset: str, out_root: Path, table: MaterialTable | None = None,
                  params: HullParams | None = None, header: str = "", hulls_from: str = "coacd") -> ObjectResult:
    """Import one object asset from its extracted ``folder`` into ``out_root`` (see module doc). ``hulls_from``:
    ``coacd`` (default) or ``given`` (RoboCasa's shipped collision meshes, capped at 64 vertices)."""
    table = table or default_table()
    cat = naming.category(asset)
    spec = table.object_spec(cat)
    body = read_model(folder).flattened()
    parts = load_parts(folder, body, spec)
    soup = merged(parts)
    extents = geo.obb_extents(soup.vertices)
    inertial, note = object_mass(parts, spec, table)
    hulls = decompose(soup.vertices, soup.faces, params) if hulls_from == "coacd" else given_hulls(folder, body)
    name, out_dir = naming.card_name(asset), out_root / naming.model_dir(asset)
    meshes = convert_visual_files(folder, out_dir, body.visuals)
    write_object_model(name, out_dir, body, hulls, inertial, header, meshes)
    flags = range_flag("size", extents[0], spec.size_m, "m") + range_flag("mass", inertial.mass, spec.mass_kg, "kg")
    if hulls_from != "coacd":
        note += f"; hulls: {hulls_from} (RoboCasa's collision meshes, CoACD failed)"
    log.info("object %s: %.4g kg, %d hulls, flags %s", name, inertial.mass, len(hulls), flags)
    return ObjectResult(name, asset, cat, naming.model_rel(asset), spec.surface, inertial.mass, inertial.part_mass,
                        note, extents, len(hulls), flags, inertial)


__all__ = ["ObjectResult", "Part", "import_object", "load_parts", "merged", "object_mass", "range_flag"]
