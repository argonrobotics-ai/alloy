# SPDX-License-Identifier: Apache-2.0
"""Import one RoboCasa fixture (appliance) as a fixed-base articulated model: doors, drawers, knobs keep their joints.

The body tree and RoboCasa's hinge/slide joints (limits, damping, friction loss, armature, spring) are kept. Collision
primitives are copied; mesh colliders are re-hulled to at most 64 vertices. Mass: the kind's typical whole-unit mass
(``categories.yaml``) is spread over the collision volume as one effective density, so a door weighs its share of
the unit (ESTIMATE). A jointed body with no collider gets a small box inertial from its visual bounds. The card is
``motion: static``; Alloy's router gives such models the MuJoCo entry, which steps their joints.

Example::

    >>> from alloy.importers.robocasa.fixtures import import_fixture
    >>> res = import_fixture(Path("work/microwaves/Microwave011"), "microwaves/Microwave011", Path("out"))  # doctest: +SKIP
    >>> res.joints                                                                                       # doctest: +SKIP
    ['microjoint', 'Disc001_joint']
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy.cards import geometry as geo
from alloy.importers.robocasa import mjcf_out as mo
from alloy.importers.robocasa import naming
from alloy.importers.robocasa.hulls import hull_mesh, hull_volume, reduce_hull
from alloy.importers.robocasa.massprops import Inertial
from alloy.importers.robocasa.materials import MaterialTable, default_table
from alloy.importers.robocasa.mjcf_in import BodyInfo, CollisionGeom, quat_matrix, read_model
from alloy.importers.robocasa.objects import convert_visual_files, part_mesh

log = logging.getLogger("alloy.importers.robocasa")

FALLBACK_FILL = 0.2      # fraction of a collider-less moving body's visual box treated as material


@dataclass
class FixtureResult:
    """What one fixture import produced (JSON-able through ``dataclasses.asdict``)."""

    name: str
    asset: str
    kind: str
    model: str
    surface: str
    mass_kg: float
    density_eff: float
    mass_note: str
    joints: list[str]
    bodies: int
    flags: list[str] = field(default_factory=list)


def primitive_volume(c: CollisionGeom) -> float:
    """Volume of a MuJoCo primitive from its ``size`` (half sizes / radius, half length)."""
    s = c.size
    if c.kind == "box":
        return 8.0 * s[0] * s[1] * s[2]
    if c.kind == "sphere":
        return 4.0 / 3.0 * math.pi * s[0] ** 3
    if c.kind == "cylinder":
        return 2.0 * math.pi * s[0] ** 2 * s[1]
    if c.kind == "capsule":
        return 2.0 * math.pi * s[0] ** 2 * s[1] + 4.0 / 3.0 * math.pi * s[0] ** 3
    if c.kind == "ellipsoid":
        return 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2]
    raise ValueError(f"collision geom type {c.kind!r} is not supported; use box/sphere/cylinder/capsule/ellipsoid/mesh")


def mesh_collider(folder: Path, c: CollisionGeom) -> tuple[np.ndarray, np.ndarray]:
    """A mesh collider as a <= 64-vertex hull in the body frame."""
    m = geo.read_obj(folder / c.mesh)
    v = (m.vertices * np.asarray(c.scale)) @ quat_matrix(c.quat).T + np.asarray(c.pos)
    return hull_mesh(reduce_hull(v))


@dataclass
class _Colliders:
    """Per body: primitives and hulls (body frame), and the total collision volume."""

    primitives: dict[str, list[CollisionGeom]] = field(default_factory=dict)
    hulls: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    volume: float = 0.0


def collect_colliders(folder: Path, model_bodies: list[BodyInfo]) -> _Colliders:
    out = _Colliders()
    for b in model_bodies:
        prims = [c for c in b.collisions if c.kind != "mesh"]
        hulls = [mesh_collider(folder, c) for c in b.collisions if c.kind == "mesh" and c.mesh]
        out.primitives[b.name], out.hulls[b.name] = prims, hulls
        out.volume += sum(primitive_volume(c) for c in prims) + sum(hull_volume(*h) for h in hulls)
    return out


def fallback_inertial(folder: Path, body: BodyInfo, density: float) -> Inertial | None:
    """A box inertial from the visual bounds of a jointed body that has no collider (None if it draws nothing)."""
    if not body.visuals:
        return None
    pts = np.concatenate([part_mesh(folder, v).vertices for v in body.visuals])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ext = np.maximum(hi - lo, 1e-3)
    mass = max(density * float(np.prod(ext)) * FALLBACK_FILL, 1e-3)
    diag = tuple(float(mass / 12.0 * (ext[(i + 1) % 3] ** 2 + ext[(i + 2) % 3] ** 2)) for i in range(3))
    return Inertial(mass, tuple(float(x) for x in 0.5 * (lo + hi)), (1.0, 0.0, 0.0, 0.0), diag, {})


def _body_xml(folder: Path, out_dir: Path, b: BodyInfo, ctx: dict, name: str | None = None):
    """The ``<body>`` element for ``b`` and its subtree (recursive)."""
    writer, cols, rho, meshes = ctx["writer"], ctx["cols"], ctx["rho"], ctx["meshes"]
    el = mo.body_element(name or b.name or f"body{ctx['n']}", b.pos, b.quat)
    ctx["n"] += 1
    for j in b.joints:
        mo.add_joint(el, j)
    if b.joints and not cols.primitives[b.name] and not cols.hulls[b.name]:
        inertial = fallback_inertial(folder, b, rho)
        if inertial is not None:
            mo.add_inertial(el, inertial)
    for v in b.visuals:
        writer.visual(el, v, meshes[v.mesh])
    for c in cols.primitives[b.name]:
        writer.primitive(el, c, rho)
    for verts, faces in cols.hulls[b.name]:
        rel = f"hulls/hull_{ctx['h']:03d}.msh"
        ctx["h"] += 1
        mo.write_hull(out_dir / rel, verts, faces)
        writer.hull(el, rel, rho)
    for r in b.regions:
        mo.add_region(el, r)
    for child in b.children:
        el.append(_body_xml(folder, out_dir, child, ctx))
    return el


OVERLAP_EXCLUDE = 5e-4   # m: parts of one fixture overlapping deeper than this when closed get a contact exclude


def overlapping_pairs(model_path: Path, depth: float = OVERLAP_EXCLUDE) -> list[tuple[str, str]]:
    """Body pairs of a fixture whose colliders overlap deeper than ``depth`` in the closed pose (mounted parts
    such as knobs set into a panel). MuJoCo filters parent-child contacts only, and not when the parent is welded to
    the world (a fixed base), so such parts would push apart."""
    import mujoco  # noqa: PLC0415 - heavy import, only when importing fixtures

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = sorted((int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])))
        if b1 > 0 and c.dist < -depth:
            pairs.add((m.body(b1).name, m.body(b2).name))
    return sorted(pairs)


def add_contact_excludes(model_path: Path, pairs: list[tuple[str, str]]) -> None:
    """Append ``<contact><exclude body1 body2/></contact>`` for ``pairs`` to the model file."""
    if not pairs:
        return
    rows = "".join(f'\n    <exclude body1="{a}" body2="{b}" />' for a, b in pairs)
    text = model_path.read_text()
    if "<contact>" in text:
        raise ValueError(f"{model_path}: already has a <contact> section; re-import instead of patching")
    model_path.write_text(text.replace("</mujoco>", f"  <contact>{rows}\n  </contact>\n</mujoco>", 1))


def import_fixture(folder: Path, asset: str, out_root: Path, table: MaterialTable | None = None,
                   header: str = "") -> FixtureResult:
    """Import one fixture asset from its extracted ``folder`` into ``out_root`` (see module doc)."""
    table = table or default_table()
    kind = naming.category(asset)
    spec = table.fixture_spec(kind)
    model = read_model(folder)
    bodies = model.bodies()
    cols = collect_colliders(folder, bodies)
    if cols.volume <= 0:
        raise ValueError(f"{folder}: no collision geometry; cannot give the fixture a mass")
    rho = spec.mass_kg / cols.volume
    name, out_dir = naming.card_name(asset), out_root / naming.model_dir(asset)
    visuals = [v for b in bodies for v in b.visuals]
    meshes, flat = convert_visual_files(folder, out_dir, visuals)
    ctx = {"writer": mo.ModelWriter(name, header, flat=flat), "cols": cols, "rho": rho, "n": 0, "h": 0,
           "meshes": meshes}
    root = _body_xml(folder, out_dir, model.root, ctx, name=name)
    model_path = ctx["writer"].write(root, out_dir / "model.xml")
    excluded = overlapping_pairs(model_path)
    add_contact_excludes(model_path, excluded)
    note = (f"ESTIMATE: typical {kind} mass {spec.mass_kg:g} kg ({spec.note}) spread over {cols.volume * 1e3:.3g} l "
            f"of collision volume: effective density {rho:.4g} kg/m3")
    if excluded:
        note += f"; contacts excluded between parts overlapping when closed: {len(excluded)} pairs"
    joints = [j.name for j in model.joints()]
    log.info("fixture %s: %d bodies, joints %s", name, len(bodies), joints)
    return FixtureResult(name, asset, kind, naming.model_rel(asset), spec.surface, spec.mass_kg, rho, note, joints,
                         len(bodies))


__all__ = ["FixtureResult", "add_contact_excludes", "collect_colliders", "import_fixture", "overlapping_pairs", "mesh_collider", "primitive_volume"]
