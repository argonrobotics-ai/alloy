# SPDX-License-Identifier: Apache-2.0
"""Scene to MJCF through Newton's own MuJoCo converter (``SolverMuJoCo(save_to_mjcf=...)``).

Alloy's MuJoCo entry runs exactly the model this converter builds, with Alloy's per-pair friction and contact
stiffness rows (``alloy.scene.mujoco_pairs``) already in it. Writing that model keeps MuJoCo's physics close to
Alloy's. This module adds what the converter leaves out (the timestep, a model name and a header listing what did
not map) and checks that MuJoCo loads the file.

Example::

    from alloy.cards.library import CardLibrary
    from alloy.io.build import scene_model
    from alloy.io.mjcf_export import write_mjcf
    lib = CardLibrary()
    io, built = scene_model(lib.scene("demo_peg_d12"), lib, "demo_peg_d12")
    write_mjcf(built, io, "out/demo_peg_d12.xml")
"""

from __future__ import annotations

import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from alloy.io.errors import ExportError
from alloy.io.from_model import clean_name
from alloy.io.model import IOModel
from alloy.scene.types import BuiltScene

log = logging.getLogger("alloy.io")


def _solver_kwargs(built: BuiltScene) -> dict:
    from alloy.core.registry import solver as capability  # noqa: PLC0415
    from alloy.scene.steppers import solver_kwargs  # noqa: PLC0415

    entry = next((e for e in built.entries if e.capability == "mujoco"), None)
    if entry is None:
        return dict(capability("mujoco").default_kwargs)
    return solver_kwargs(entry, "mujoco")


def convert(built: BuiltScene, path: Path) -> None:
    """Run Newton's MuJoCo converter on the one-world Model and save its MJCF at ``path``."""
    from alloy.core.registry import solver as capability  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

    register_builtin_solvers()
    kw = _solver_kwargs(built)
    kw.update(save_to_mjcf=str(path), use_mujoco_cpu=True)
    try:
        capability("mujoco").factory(built.model, kw)
    except Exception as exc:  # noqa: BLE001 - re-raised with the fix to try
        if path.exists():   # Alloy's post-construction hooks run after the converter has written the file
            log.warning("MuJoCo solver set-up failed after the MJCF was written (%s); the file is kept", exc)
            return
        raise ExportError(f"Newton's MuJoCo converter failed on this scene ({exc}); export the rigid objects only "
                          "(drop deformable and liquid objects from the scene card) and retry") from exc
    if not path.exists():
        raise ExportError(f"the MuJoCo converter wrote no file at {path}")


def _header(io: IOModel) -> str:
    rows = [r for r in io.unmapped if r.format in ("all", "mjcf", "model")]
    lines = [f" Exported by Alloy (alloy.io) from scene '{io.name}'. Units: SI, radians. "]
    if rows:
        lines.append(" Not carried by this file: ")
        lines += [f"   {r.item}: {r.reason} " for r in rows]
    return "\n".join(lines).replace("--", "- -")


def _plain_geom_names(root: ET.Element) -> None:
    """The converter names geoms ``<label>_<shape index>``; drop the index where the label alone is unique."""
    geoms = root.iter("geom")
    names = [g.get("name") for g in root.iter("geom") if g.get("name")]
    stem = {n: n.rsplit("_", 1)[0] for n in names if n.rsplit("_", 1)[-1].isdigit()}
    counts: dict[str, int] = {}
    for s in stem.values():
        counts[s] = counts.get(s, 0) + 1
    taken = set(names)
    rename = {n: s for n, s in stem.items() if counts[s] == 1 and s not in taken}
    for g in geoms:
        if g.get("name") in rename:
            g.set("name", rename[g.get("name")])
    for el in root.iter("pair"):
        for key in ("geom1", "geom2"):
            if el.get(key) in rename:
                el.set(key, rename[el.get(key)])


def _open_masks(root: ET.Element, object_shapes: set[int]) -> None:
    """The converter gives a colliding geom with no partner in this model contype = conaffinity = 0. For object
    geoms that would make the exported object pass through anything added later, so they get MuJoCo's defaults.
    Robot geoms keep the converter's masks (they encode self-collision filters)."""
    for g in root.iter("geom"):
        tail = (g.get("name") or "").rsplit("_", 1)[-1]
        if tail.isdigit() and int(tail) in object_shapes and g.get("contype") == "0" and g.get("conaffinity") == "0":
            del g.attrib["contype"], g.attrib["conaffinity"]


# MuJoCo's defaults for the solver arrays; MuJoCo's writer drops trailing default values ("0.02" for "0.02 1"),
# and Newton's MJCF reader takes a missing value as 0 (dampratio 0), which Alloy's MJCF route rejects.
FULL_ARRAYS = {"solref": (0.02, 1.0), "solreffriction": (0.02, 1.0), "solreflimit": (0.02, 1.0),
               "solimp": (0.9, 0.95, 0.001, 0.5, 2.0), "solimpfriction": (0.9, 0.95, 0.001, 0.5, 2.0),
               "solimplimit": (0.9, 0.95, 0.001, 0.5, 2.0)}


def _full_arrays(root: ET.Element) -> None:
    """Write every solref/solimp array in full, so no reader has to fill the shorthand."""
    for el in root.iter():
        for key, default in FULL_ARRAYS.items():
            text = el.get(key)
            if text is None:
                continue
            vals = text.split()
            if 0 < len(vals) < len(default):
                el.set(key, " ".join(vals + [f"{v:g}" for v in default[len(vals):]]))


def _mocap(root: ET.Element, io: IOModel) -> None:
    """Write kinematic bodies as MuJoCo mocap bodies (moved by the user, not by forces): the converter's free joint
    and its weld to the world are removed."""
    names = {b.name for b in io.bodies if b.motion == "kinematic"}
    world = root.find("worldbody")
    if not names or world is None:
        return
    done = set()
    for body in world.findall("body"):
        if clean_name(body.get("name"), "") not in names:
            continue
        for j in [el for el in body if el.tag == "freejoint" or (el.tag == "joint" and el.get("type") == "free")]:
            body.remove(j)
        if body.find("joint") is None:
            body.set("mocap", "true")
            done.add(body.get("name"))
    for sec in root.findall("equality"):
        for el in list(sec):
            pair = {el.get("body1"), el.get("body2")} - {None, "world"}
            if el.tag == "weld" and len(pair) == 1 and pair <= done:
                sec.remove(el)


def finish(path: Path, io: IOModel, object_shapes: set[int] = frozenset()) -> None:
    """Add the model name, the timestep and the header comment to a converter MJCF; plain geom names."""
    tree = ET.parse(path)
    root = tree.getroot()
    _open_masks(root, object_shapes)
    _plain_geom_names(root)
    _mocap(root, io)
    _full_arrays(root)
    root.set("model", io.name)
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(1, opt)
    if io.timestep:
        opt.set("timestep", f"{io.timestep:.9g}")
    root.insert(0, ET.Comment(_header(io)))
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode", xml_declaration=False)


def colliding_object_shapes(built: BuiltScene, io: IOModel) -> set[int]:
    """Model shape indices of the scene's objects (not robots) that collide."""
    from alloy._vendor import newton  # noqa: PLC0415

    flags = built.model.shape_flags.numpy()
    out = set()
    for name, h in built.handles.items():
        if io.groups.get(name) != "robot":
            out.update(s for s in h.template.shape if int(flags[s]) & int(newton.ShapeFlags.COLLIDE_SHAPES))
    return out


def write_static(io: IOModel, path: Path) -> None:
    """A scene with no bodies (Newton's converter needs a joint): its static shapes as world geoms."""
    from alloy.io.groups import Group  # noqa: PLC0415
    from alloy.io.mjcf_write import group_mjcf  # noqa: PLC0415

    root = group_mjcf(io, Group(io.name, world_shapes=list(io.world_shapes), fixed_base=True))
    if io.timestep:
        ET.SubElement(root, "option", timestep=f"{io.timestep:.9g}")
    if not io.world_shapes:
        io.note("mjcf", io.name, "nothing rigid to export: the file has an empty world")
    root.insert(0, ET.Comment(_header(io)))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="unicode")


def check_loads(path: Path) -> dict[str, int]:
    """Load ``path`` with MuJoCo; returns its counts or raises ExportError with MuJoCo's message."""
    import mujoco  # noqa: PLC0415

    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"MuJoCo cannot load {path}: {exc}") from exc
    return {"bodies": m.nbody - 1, "geoms": m.ngeom, "joints": m.njnt, "actuators": m.nu, "pairs": m.npair,
            "equalities": m.neq}


def note_limits(io: IOModel) -> None:
    """MJCF rows of ``io.unmapped``: MuJoCo collides every mesh through its convex hull."""
    for b in io.bodies:
        if any(s.kind == "mesh" and not s.convex and s.collide for s in b.shapes):
            io.note("mjcf", b.group or b.name, "non-convex mesh: MuJoCo collides its convex hull (holes and "
                                               "pockets fill); use collision: decompose on the card to keep them")
    for s in io.world_shapes:
        if s.kind == "mesh" and not s.convex and s.collide:
            io.note("mjcf", io.shape_groups.get(s.name, s.name), "non-convex mesh: MuJoCo collides its convex hull "
                    "(holes and pockets fill); use collision: decompose on the card to keep them")


def write_mjcf(built: BuiltScene, io: IOModel, path: str | Path) -> dict[str, int]:
    """Write the MJCF of a one-world assembled scene at ``path`` and return MuJoCo's counts of what it loaded."""
    note_limits(io)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not built.model.joint_count:
        write_static(io, path)
        return check_loads(path)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.xml"
        convert(built, raw)
        finish(raw, io, colliding_object_shapes(built, io))
        path.write_text(raw.read_text())
    counts = check_loads(path)
    log.info("wrote MJCF %s: %s", path, counts)
    return counts
