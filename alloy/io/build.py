# SPDX-License-Identifier: Apache-2.0
"""Cards to an ``IOModel``: assemble the scene for one world on the CPU with Alloy's own router and emitters, then
read the compiled Model. Exports therefore hold what Alloy simulates (masses from card densities, routed shapes,
attachments), not a second reading of the cards.

Example::

    from alloy.cards.library import CardLibrary
    from alloy.io.build import scene_model
    lib = CardLibrary()
    io, built = scene_model(lib.scene("demo_peg_d12"), lib, "demo_peg_d12")
    print([b.name for b in io.bodies], io.unmapped)
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from alloy.cards import types as ct
from alloy.cards.library import CardLibrary
from alloy.io.from_model import ModelNames, read_model
from alloy.io.model import IOMaterial, IOModel
from alloy.scene.types import BuiltScene

log = logging.getLogger("alloy.io")
RIGID_ROUTES = ("STATIC", "RIGID", "RIGID_COMPLIANT", "RIGID_PILE", "CHAIN", "GRAIN_RIGID", "PANELS")


def material_of(card: ct.MaterialCard, name: str | None = None) -> IOMaterial:
    """``IOMaterial`` of a material card (levers default to 0, stiffness to the engine default)."""
    s = card.surface
    lever = lambda q: float(q.value) if q is not None else 0.0  # noqa: E731
    k = float(s.contact_stiffness.value) if s.contact_stiffness is not None else None
    return IOMaterial(name or card.name, float(card.density.value), float(s.mu_static.value),
                      float(s.mu_dynamic.value), float(s.restitution.value), lever(s.torsional_lever),
                      lever(s.rolling_lever), k, origin="card")


def object_scene(card: ct.ObjectCard, pose: tuple[float, ...] = (0, 0, 0, 0, 0, 0, 1)) -> ct.SceneCard:
    """A one-world scene holding only ``card`` at ``pose`` (used to export a single object card)."""
    return ct.SceneCard(robots=(), objects=(ct.Placement(card.name, tuple(pose)),), worlds=1)


def assemble_one_world(scene: ct.SceneCard, lib: CardLibrary) -> BuiltScene:
    """Alloy's assembler on the CPU with one world (randomization draws are not applied: nominal values)."""
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415 - heavy import

    plain = replace(scene, objects=tuple(replace(p, randomize=()) for p in scene.objects), worlds=1)
    return SceneAssembler(lib).assemble(plain, "cpu", world_count=1)


def _key_material(key: str, lib: CardLibrary) -> IOMaterial | None:
    """Pair-table key -> material ("wood", "wood@box" own row); robot keys come from the robot file."""
    if key.startswith("robot:"):
        return None
    base = key.split("@", 1)[0]
    return material_of(lib.material(base), name=key.replace("@", "_at_"))


def scene_names(built: BuiltScene, lib: CardLibrary, robots: set[str]) -> ModelNames:
    """Group and material names of every body and shape of a one-world assembled scene."""
    names = ModelNames()
    keys = list(built.pair_table.materials)
    mats = {k: _key_material(k, lib) for k in keys}
    names.materials = {m.name: m for m in mats.values() if m is not None}
    attr = getattr(getattr(built.model, "alloy", None), "shape_material", None)
    shape_mat = attr.numpy() if attr is not None else np.full(built.model.shape_count, -1)
    for group, handle in built.handles.items():
        names.groups[group] = "robot" if group in robots else "object"
        for b in handle.template.body:
            names.body_group[b] = group
        for s in handle.template.shape:
            names.shape_group[s] = group
            m = mats.get(keys[int(shape_mat[s])]) if 0 <= int(shape_mat[s]) < len(keys) else None
            if m is not None:
                names.shape_material[s] = m.name
    return names


def _timestep(built: BuiltScene) -> float | None:
    mj = [e for e in built.entries if e.capability == "mujoco"]
    entries = mj or list(built.entries)
    return min(e.target_dt for e in entries) if entries else None


def scene_limits(scene: ct.SceneCard, built: BuiltScene, io: IOModel) -> None:
    """Record what no export format carries for this scene (each writer adds its own rows)."""
    for p in scene.objects:
        if p.randomize:
            io.note("all", f"{p.object}.randomize", "exported at nominal values; per-world draws are Alloy-only")
    for name, d in built.report.decisions.items():
        handle = built.handles.get(name)
        route = getattr(d.route, "name", str(d.route))
        if route not in RIGID_ROUTES and handle is not None and len(handle.template.particle):
            io.note("all", name, f"route {route} ({len(handle.template.particle)} particles): deformables, "
                                 "liquids and grains are not exported; export the rigid parts only")
    for a in scene.attachments:
        if a.break_force is not None or a.break_strain is not None:
            io.note("all", f"attachment {a.a}-{a.b}", "breakable attachment exported as unbreakable")
    if scene.pairs:
        for fmt in ("usd", "urdf"):
            io.note(fmt, "pair friction", f"{len(scene.pairs)} pair rows (Scene.set_friction, pair cards) are not "
                                          "carried: the file has each material's own friction; the MJCF export "
                                          "keeps the pairs")
    for b in io.bodies:
        if b.motion == "kinematic":
            io.note("all", b.group or b.name, "kinematic object: a mocap body in MJCF, kinematicEnabled in USD, "
                    "fixed in URDF. A file loaded with alloy.load is one robot, whose bodies sim.drive does not "
                    "move: alloy.io.import_file(file, folder) makes a kinematic card that can be driven")
    if scene.emitters:
        io.note("all", "emitters", f"{len(scene.emitters)} liquid emitters/sinks are Alloy-only")
    labels = list(built.model.shape_label)
    for pp in built.precision_pairs:
        io.note("all", f"precision pair {labels[pp.shape_a]}-{labels[pp.shape_b]}",
                "SDF/hydroelastic precision contact is Alloy-only; files keep the plain collision shapes")


def without_rods(scene: ct.SceneCard, built: BuiltScene) -> tuple[ct.SceneCard, list[str]]:
    """The scene without its elastic rods (route ROD_VBD: cable joints no format here has) and their names."""
    rods = [n for n, d in built.report.decisions.items() if "ROD" in getattr(d.route, "name", str(d.route))]
    if not rods:
        return scene, []
    gone = lambda ref: str(ref).split(":")[0] in rods  # noqa: E731
    keep = tuple(p for p, n in zip(scene.objects, placement_names(scene)) if n not in rods)
    return replace(scene, objects=keep, attachments=tuple(a for a in scene.attachments
                                                          if not gone(a.a) and not gone(a.b))), rods


def placement_names(scene: ct.SceneCard) -> list[str]:
    """Scene names of the placements: the card name, then ``name#1``, ``name#2`` for further copies."""
    seen: dict[str, int] = {}
    out = []
    for p in scene.objects:
        k = seen.get(p.object, 0)
        seen[p.object] = k + 1
        out.append(p.object if k == 0 else f"{p.object}#{k}")
    return out


def scene_model(scene: ct.SceneCard, lib: CardLibrary, name: str) -> tuple[IOModel, BuiltScene]:
    """``IOModel`` and the one-world ``BuiltScene`` of a scene card. Elastic rods are left out and listed."""
    built = assemble_one_world(scene, lib)
    scene, rods = without_rods(scene, built)
    if rods:
        built = assemble_one_world(scene, lib)
    names = scene_names(built, lib, {r.name for r in scene.robots})
    io = read_model(built.model, name, names)
    io.timestep = _timestep(built)
    for b in io.bodies:
        if not b.group:
            log.debug("body %s has no object group", b.name)
    scene_limits(scene, built, io)
    for rod in rods:
        io.note("all", rod, "elastic rod (route ROD_VBD): not exported, as cloth and liquid are not; the rest of the "
                            "scene is. A rod without a clamp exports as a chain of bodies")
    return io, built
