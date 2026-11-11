# SPDX-License-Identifier: Apache-2.0
"""Articulated kitchen cabinets built from imported Lightwheel door panels and handles: hinged doors, sliding drawers.

RoboCasa builds its cabinets procedurally from box boards plus a door-panel mesh and a handle mesh. This module does
the same from the imported panel and handle models (``robocasa_cabinet_door_panel_NNN``, ``robocasa_cabinet_handle_NNN``
in the import root): an 18 mm particleboard carcass (top, bottom, sides, back, a shelf), and per layout

- ``single_left`` / ``single_right``: one door hinged at the left / right front edge (0-90 degrees);
- ``double``: two doors meeting in the middle;
- ``drawers``: three drawers on slides (each a tray of 12 mm boards behind a front panel);
- ``wall``: a shallow wall cabinet with one door.

Boards are solid boxes, so the mass is exact for the chosen density (particleboard, ``robocasa_particleboard``);
door panels and handles are drawn from their meshes, scaled to the door, and collide as boxes and the handle's own
primitives. Joint damping follows RoboCasa's templates (doors 2, drawers 10 N s/m). The base is fixed
(``motion: static``).

Example::

    >>> from alloy.importers.robocasa.cabinets import CabinetSpec, LAYOUTS
    >>> spec = CabinetSpec("double", panel="robocasa_cabinet_door_panel_008", handle="robocasa_cabinet_handle_001")
    >>> spec.size_m, sorted(LAYOUTS)
    ((0.9, 0.58, 0.72), ['double', 'drawers', 'single_left', 'single_right', 'wall'])
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.importers.robocasa import mjcf_out as mo
from alloy.importers.robocasa.mjcf_in import CollisionGeom, JointInfo, VisualPart, quat_matrix, quat_mul, read_model

BOARD = 0.018                    # m, carcass board
TRAY = 0.012                     # m, drawer tray board
GAP = 0.003                      # m, clearance around doors and drawers
PANEL_HALF = (0.25, 0.014, 0.35)  # m, half sizes of a Lightwheel door panel model
DOOR_DAMPING, DRAWER_DAMPING = 2.0, 10.0
LAYOUTS: dict[str, tuple[float, float, float]] = {     # width, depth, height (m): common base / wall cabinet sizes
    "single_left": (0.45, 0.58, 0.72), "single_right": (0.45, 0.58, 0.72), "double": (0.9, 0.58, 0.72),
    "drawers": (0.6, 0.58, 0.72), "wall": (0.6, 0.33, 0.72)}
Q_DRAWER_HANDLE = (0.7071068, 0.0, 0.7071068, 0.0)    # handle bar turned horizontal (about y)
COMPOSED_KIND = "cabinets_composed"
COMPOSED_DIR = f"mjcf/{COMPOSED_KIND}"


@dataclass(frozen=True)
class CabinetSpec:
    """One cabinet: layout, door-panel model and handle model (card names in the import root)."""

    layout: str
    panel: str
    handle: str

    @property
    def size_m(self) -> tuple[float, float, float]:
        if self.layout not in LAYOUTS:
            raise KeyError(f"unknown cabinet layout {self.layout!r}; use one of {sorted(LAYOUTS)}")
        return LAYOUTS[self.layout]

    @property
    def name(self) -> str:
        return f"robocasa_cabinet_{self.layout}_{self.panel.rsplit('_', 1)[-1]}_{self.handle.rsplit('_', 1)[-1]}"


@dataclass(frozen=True)
class Part:
    """A drawn mesh from another imported model: mesh/texture paths relative to the new model's folder."""

    visual: VisualPart
    mesh_rel: str
    colliders: tuple[CollisionGeom, ...]


def _rel(target: Path, start: Path) -> str:
    return str(Path(*[".."] * len(start.parts)) / target) if start.parts else str(target)


def load_part(root: Path, model_rel: str, cabinet_dir_rel: str) -> Part:
    """The first visual mesh and the colliders of an imported model, re-pathed for a cabinet folder."""
    model_dir = Path(model_rel).parent
    info = read_model(root / model_dir).flattened()
    if not info.visuals:
        raise ValueError(f"{root / model_rel}: no visual mesh to reuse")
    v = info.visuals[0]
    up = _rel(model_dir, Path(cabinet_dir_rel))
    mat = v.material
    if mat is not None and mat.texture:
        mat = type(mat)(mat.name, mat.rgba, f"{up}/{mat.texture}", mat.specular, mat.shininess)
    vis = VisualPart(f"{up}/{v.mesh}", v.scale, v.pos, v.quat, mat)
    return Part(vis, vis.mesh, tuple(info.collisions))


BOARD_RGBA = (0.93, 0.92, 0.89, 1.0)      # white melamine carcass


def _box(writer: mo.ModelWriter, body: ET.Element, half, pos, density: float, name: str = "",
         drawn: bool = True) -> None:
    writer.primitive(body, CollisionGeom("box", tuple(half), tuple(pos), (1.0, 0.0, 0.0, 0.0), name=name), density)
    if drawn:
        writer.visual_box(body, half, pos, BOARD_RGBA)


def carcass(writer: mo.ModelWriter, body: ET.Element, size, density: float, shelf: bool = True) -> None:
    """Top, bottom, sides, back (and a middle shelf) of an open-front box centred at the origin."""
    w, d, h = (s / 2 for s in size)
    t = BOARD / 2
    _box(writer, body, (w, d, t), (0, 0, h - t), density, "top")
    _box(writer, body, (w, d, t), (0, 0, -h + t), density, "bottom")
    _box(writer, body, (t, d, h - BOARD), (-w + t, 0, 0), density, "left")
    _box(writer, body, (t, d, h - BOARD), (w - t, 0, 0), density, "right")
    _box(writer, body, (w - BOARD, t, h - BOARD), (0, d - t, 0), density, "back")
    if shelf:
        _box(writer, body, (w - BOARD, d - BOARD, t), (0, 0, 0), density, "shelf")


def posed(c: CollisionGeom, pos, quat) -> CollisionGeom:
    """A collider re-expressed in the frame where its model sits at (pos, quat)."""
    p = np.asarray(pos) + quat_matrix(quat) @ np.asarray(c.pos)
    q = quat_mul(np.asarray(quat), np.asarray(c.quat))
    return dataclasses.replace(c, pos=tuple(float(x) for x in p), quat=tuple(float(x) for x in q))


def _panel_visual(panel: Part, half_w: float, half_h: float, pos) -> VisualPart:
    v = panel.visual
    scale = (v.scale[0] * half_w / PANEL_HALF[0], v.scale[1] * (BOARD / 2) / PANEL_HALF[1],
             v.scale[2] * half_h / PANEL_HALF[2])
    return VisualPart(v.mesh, scale, tuple(float(x) for x in pos), (1.0, 0.0, 0.0, 0.0), v.material)


@dataclass(frozen=True)
class Densities:
    """kg/m^3 for boards (particleboard) and handle primitives (metal)."""

    board: float
    handle: float


def _local_half(c: CollisionGeom) -> np.ndarray:
    sz = list(c.size) + [0.0] * 3
    if c.kind == "box" or c.kind == "ellipsoid":
        return np.array(sz[:3])
    if c.kind == "sphere":
        return np.full(3, sz[0])
    if c.kind == "cylinder":
        return np.array([sz[0], sz[0], sz[1]])
    if c.kind == "capsule":
        return np.array([sz[0], sz[0], sz[1] + sz[0]])
    raise ValueError(f"handle collider type {c.kind!r} is not supported for mounting")


def collider_bounds(colliders: tuple[CollisionGeom, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned (lo, hi) bounds of primitive colliders in their model frame."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for c in colliders:
        half = np.abs(quat_matrix(c.quat)) @ _local_half(c)
        lo, hi = np.minimum(lo, np.asarray(c.pos) - half), np.maximum(hi, np.asarray(c.pos) + half)
    if not np.isfinite(lo).all():
        raise ValueError("handle has no primitive colliders to mount by")
    return lo, hi


def mount_offset(handle: Part) -> float:
    """y shift that puts the back of a handle (turned to face -y) on the panel's front face."""
    lo, _ = collider_bounds(handle.colliders)
    return -BOARD / 2 + float(lo[1])


def _fit(preferred: float, half_len: float, half_panel: float, margin: float = 0.02) -> float:
    """``preferred`` clipped so a handle of half length ``half_len`` stays inside the panel (centre if it can't)."""
    room = half_panel - half_len - margin
    return 0.0 if room <= 0 else float(np.clip(preferred, -room, room))


def add_handle(writer: mo.ModelWriter, body: ET.Element, handle: Part, pos, quat, rho: float) -> None:
    """The handle mesh plus its colliders at (pos, quat) in ``body``; it sticks out towards -y (the front)."""
    v = handle.visual
    writer.visual(body, VisualPart(v.mesh, v.scale, tuple(pos), tuple(quat), v.material), v.mesh)
    for c in handle.colliders:
        writer.primitive(body, posed(c, pos, quat), rho)


def door(writer: mo.ModelWriter, parts: tuple[Part, Part], size, side: str, rho: Densities, name: str,
         x_hinge: float, width: float) -> ET.Element:
    """A door body hinged at ``x_hinge`` on the front face; ``side`` left opens to -90 degrees, right to +90."""
    panel, handle = parts
    _, d, h = size
    sign = 1.0 if side == "left" else -1.0          # the door extends from the hinge towards +x (left) or -x
    body = mo.body_element(name, (x_hinge, -d / 2 - BOARD / 2 - GAP, 0.0))
    rng = (-1.5708, 0.0) if side == "left" else (0.0, 1.5708)
    mo.add_joint(body, JointInfo(f"{name}_hinge", "hinge", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), rng, DOOR_DAMPING))
    hw, hh = width / 2 - GAP, h / 2 - GAP
    centre = (sign * width / 2, 0.0, 0.0)
    writer.visual(body, _panel_visual(panel, hw, hh, centre), panel.mesh_rel)
    _box(writer, body, (hw, BOARD / 2, hh), centre, rho.board, f"{name}_panel", drawn=False)
    lo, hi = collider_bounds(handle.colliders)
    half_len = float(max(-lo[2], hi[2]))
    z = _fit(-hh + 0.12 if size[1] < 0.4 else hh - 0.12, half_len, hh)   # wall cabinets low, base cabinets high
    add_handle(writer, body, handle, (sign * (width - 0.05), mount_offset(handle), z), (0.0, 0.0, 0.0, 1.0),
               rho.handle)
    return body


def drawer(writer: mo.ModelWriter, parts: tuple[Part, Part], size, z: float, height: float, rho: Densities,
           name: str) -> ET.Element:
    """A drawer at height ``z``: tray behind a front panel, on a slide along -y."""
    panel, handle = parts
    w, d, _ = size
    body = mo.body_element(name, (0.0, -d / 2 - BOARD / 2 - GAP, z))
    travel = round(0.75 * d, 3)
    mo.add_joint(body, JointInfo(f"{name}_slide", "slide", (0.0, -1.0, 0.0), (0.0, 0.0, 0.0), (0.0, travel),
                                 DRAWER_DAMPING))
    hw, hh = w / 2 - GAP, height / 2 - GAP
    writer.visual(body, _panel_visual(panel, hw, hh, (0.0, 0.0, 0.0)), panel.mesh_rel)
    _box(writer, body, (hw, BOARD / 2, hh), (0.0, 0.0, 0.0), rho.board, f"{name}_front", drawn=False)
    tw, td, th = w / 2 - BOARD - GAP, (d - 2 * BOARD) / 2 - GAP, hh - 0.02
    y0 = BOARD / 2 + td
    t = TRAY / 2
    _box(writer, body, (tw, td, t), (0.0, y0, -th + t), rho.board, f"{name}_bottom")
    _box(writer, body, (t, td, th), (-tw + t, y0, 0.0), rho.board, f"{name}_left")
    _box(writer, body, (t, td, th), (tw - t, y0, 0.0), rho.board, f"{name}_right")
    _box(writer, body, (tw, t, th), (0.0, y0 + td - t, 0.0), rho.board, f"{name}_back")
    add_handle(writer, body, handle, (0.0, mount_offset(handle), 0.0),
               quat_mul(np.array((0.0, 0.0, 0.0, 1.0)), np.array(Q_DRAWER_HANDLE)), rho.handle)
    return body


def build_cabinet(root: Path, spec: CabinetSpec, parts: tuple[Part, Part], rho: Densities, header: str) -> Path:
    """Write ``mjcf/cabinets_composed/<name>/model.xml`` under ``root`` and return its path."""
    size = spec.size_m
    w, _, h = size
    writer = mo.ModelWriter(spec.name, header)
    base = mo.body_element(spec.name)
    carcass(writer, base, size, rho.board, shelf=spec.layout != "drawers")
    if spec.layout in ("single_left", "wall"):
        base.append(door(writer, parts, size, "left", rho, "door", -w / 2, w))
    elif spec.layout == "single_right":
        base.append(door(writer, parts, size, "right", rho, "door", w / 2, w))
    elif spec.layout == "double":
        base.append(door(writer, parts, size, "left", rho, "door_left", -w / 2, w / 2))
        base.append(door(writer, parts, size, "right", rho, "door_right", w / 2, w / 2))
    else:
        inner = h - 2 * BOARD
        for i in range(3):
            z = -h / 2 + BOARD + inner * (i + 0.5) / 3
            base.append(drawer(writer, parts, size, z, inner / 3, rho, f"drawer{i}"))
    return writer.write(base, root / COMPOSED_DIR / spec.name / "model.xml")


def _model_mass(path: Path) -> float:
    import mujoco  # noqa: PLC0415 - heavy import, only when composing

    return float(mujoco.MjModel.from_xml_path(str(path)).body_mass.sum())


def compose_one(root: Path, spec: CabinetSpec, rho: Densities, validate: bool = True, thumbs: bool = True) -> dict:
    """Build one cabinet, its card, checks, thumbnail and record (``pipeline`` record format)."""
    from alloy.importers.robocasa.fixtures import FixtureResult  # noqa: PLC0415 - avoid an import cycle
    from alloy.importers.robocasa.pipeline import _record  # noqa: PLC0415

    asset = f"{COMPOSED_KIND}/{spec.name.removeprefix('robocasa_')}"
    cab_dir = f"{COMPOSED_DIR}/{spec.name}"
    parts = (load_part(root, f"mjcf/cabinets/{spec.panel}/model.xml", cab_dir),
             load_part(root, f"mjcf/handles/{spec.handle}/model.xml", cab_dir))
    header = f"{spec.name}: cabinet composed by alloy.importers.robocasa.cabinets from {spec.panel} + {spec.handle}"
    path = build_cabinet(root, spec, parts, rho, header)
    info = read_model(path.parent)
    note = (f"boards {rho.board:g} kg/m3 (robocasa_particleboard), handle {rho.handle:g} kg/m3 "
            f"(robocasa_handle_metal), solid boxes: exact for these densities (ESTIMATE densities); composed from "
            f"{spec.panel} + {spec.handle}")
    result = FixtureResult(spec.name, asset, COMPOSED_KIND, f"{cab_dir}/model.xml", "robocasa_particleboard",
                           _model_mass(path), rho.board, note, [j.name for j in info.joints()], len(info.bodies()))
    return _record("fixtures_lightwheel", asset, "fixture", result, root, validate, thumbs)


def compose_all(root: Path, layouts: tuple[str, ...] = tuple(LAYOUTS), validate: bool = True,
                thumbs: bool = True) -> dict[str, int]:
    """Every imported door panel x its paired handle (round robin) x ``layouts``; returns status counts."""
    import collections  # noqa: PLC0415
    import json  # noqa: PLC0415

    from alloy.importers.robocasa.materials import default_table  # noqa: PLC0415
    from alloy.importers.robocasa.pipeline import RECORDS, _status  # noqa: PLC0415

    panels = sorted(p.name for p in (root / "mjcf/cabinets").glob("robocasa_cabinet_door_panel_*"))
    handles = sorted(p.name for p in (root / "mjcf/handles").glob("robocasa_cabinet_handle_*"))
    if not panels or not handles:
        raise FileNotFoundError(f"{root}: import the fixtures_lightwheel 'cabinets' and 'handles' first")
    table = default_table()
    rho = Densities(table.density("robocasa_particleboard").value, table.density("robocasa_handle_metal").value)
    counts: collections.Counter = collections.Counter()
    (root / RECORDS).mkdir(parents=True, exist_ok=True)
    for i, panel in enumerate(panels):
        for layout in layouts:
            rec = compose_one(root, CabinetSpec(layout, panel, handles[i % len(handles)]), rho, validate, thumbs)
            rec["status"] = _status(rec)
            (root / RECORDS / f"{rec['name']}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
            counts[rec["status"]] += 1
    return dict(counts)


__all__ = ["BOARD", "COMPOSED_DIR", "COMPOSED_KIND", "collider_bounds", "compose_all", "compose_one", "mount_offset", "LAYOUTS", "CabinetSpec", "Densities", "Part", "build_cabinet", "carcass",
           "door", "drawer", "load_part", "posed"]
