# SPDX-License-Identifier: Apache-2.0
"""The export plan (SCR-WP13-EXPORT): for a built simulation and the picked worlds, every global row the gather reads,
made once on the host, plus the topology part's static facts (names, shapes, faces, particle sets).

- Bodies: every body of a picked world, in template order (``world * stride + template``).
- Meshes: every particle of a cloth, film or VBD solid (objects with triangles), faces local to the object.
- Particle sets (SPH, MPM: particles without triangles): a fixed subset of each world's block (``_subset``: a
  sorted random choice with a fixed seed); the sets share ``max_points`` in proportion to their sizes (at least 1
  each). Their live flags come from the solver's
  own view when a coupled entry steps them (``alloy.api._liquid``), else from the model.
- Rods: every segment body's capsule start in its body frame, then the last segment's end (``alloy.api._rods``).
- Topology version: the picked worlds' bond ranges in the topology controller, and their bodies' active flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.export import shapes as S


@dataclass
class ParticleSet:
    name: str
    route: str
    material: str
    material_id: int
    count: int                       # particles per world
    local: np.ndarray                # [k] the subset, object-local
    flags: Any                       # device int32 array holding the set's live flags
    flag_rows: np.ndarray            # [P, k] rows into ``flags``


@dataclass
class ExportPlan:
    worlds: list[int]
    max_points: int
    body_names: list[str]
    body_rows: np.ndarray            # [P, B]
    mesh: dict[str, slice] = field(default_factory=dict)          # name -> its vertices in mesh_rows' columns
    mesh_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    faces: dict[str, np.ndarray] = field(default_factory=dict)
    tet: dict[str, slice] = field(default_factory=dict)           # name -> its tets in tet_rows' columns
    tet_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    tets: dict[str, np.ndarray] = field(default_factory=dict)     # name -> [T, 4] int32, into the mesh's vertices
    cut_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))  # [P, K] cuttable whole tets
    sets: list[ParticleSet] = field(default_factory=list)
    part: dict[str, slice] = field(default_factory=dict)
    part_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    rod: dict[str, slice] = field(default_factory=dict)
    rod_rows: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.int32))   # [P, R] segment body rows
    rod_local: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))  # [R] body-frame points
    rod_info: dict[str, dict] = field(default_factory=dict)       # name -> route, bodies, drawn_by_bodies
    bond_range: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))   # [P, 2] into world_order
    static: dict[int, list] = field(default_factory=dict)          # world -> static shapes
    shapes: list = field(default_factory=list)                    # per body (template order) its shapes


def _columns(rows: list[np.ndarray], P: int) -> np.ndarray:
    return (np.concatenate(rows, axis=1) if rows else np.zeros((P, 0), np.int64)).astype(np.int32)


def _bodies(sim: Any, plan: ExportPlan) -> None:
    m, B = sim.built.model, int(sim.index.stride["body"])
    labels = [str(x) for x in m.body_label]
    by_row = {row: name for name, row in sim.index.bodies.items()}
    plan.body_names = [by_row.get(b, labels[b].rsplit("/", 1)[-1]) for b in range(B)]
    plan.body_rows = np.array([[w * B + b for b in range(B)] for w in plan.worlds], np.int32).reshape(len(plan.worlds),
                                                                                                        B)
    arrays = S.model_shapes(m)
    plan.shapes = S.body_shapes(m, arrays, np.arange(B))
    plan.static = {w: S.statics(m, arrays, w) for w in plan.worlds}


def _meshes(sim: Any, plan: ExportPlan) -> None:
    idx, stride, cols, at = sim.index, int(sim.index.stride["particle"]), [], 0
    for name, r in idx.particles.items():
        if name not in idx.tris:
            continue
        cols.append(np.array([[w * stride + p for p in r] for w in plan.worlds], np.int64))
        plan.mesh[name] = slice(at, at + len(r))
        plan.faces[name] = sim.faces(name).astype(np.int32)
        at += len(r)
    plan.mesh_rows = _columns(cols, len(plan.worlds))
    _tets(sim, plan)


def _tets(sim: Any, plan: ExportPlan) -> None:
    """Every tet of a meshed soft body per picked world, with its corners local to the mesh's vertices."""
    m, cols, at = sim.built.model, [], 0
    any_h = next(iter(sim.built.handles.values()), None)
    stride = len(any_h.world_stride.tet) if any_h is not None else 0
    indices = m.tet_indices.numpy().reshape(-1, 4) if getattr(m, "tet_count", 0) else np.zeros((0, 4), np.int32)
    for name in plan.mesh:
        r = sim.built.handles[name].template.tet
        if not len(r):
            continue
        cols.append(np.array([[w * stride + t for t in r] for w in plan.worlds], np.int64))
        plan.tet[name] = slice(at, at + len(r))
        plan.tets[name] = (indices[r.start:r.stop] - sim.index.particles[name].start).astype(np.int32)
        at += len(r)
    plan.tet_rows = _columns(cols, len(plan.worlds))


SUBSET_SEED = 0


def _subset(n: int, k: int) -> np.ndarray:
    """``k`` of ``n`` items, the same every frame and for every exporter (all of them when k >= n): a sorted random
    choice with a fixed seed. An even stride would alias with an MPM lattice's row and plane lengths and draw
    stripes (WP12)."""
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(SUBSET_SEED).choice(n, k, replace=False)).astype(np.int64)


def _flag_source(sim: Any, rows: np.ndarray) -> tuple[Any, np.ndarray]:
    """The device flags array holding these global particle rows' live flags, and the rows in it."""
    from alloy.api import _liquid  # noqa: PLC0415

    for solver in _liquid.liquid_solvers(sim.engine):
        ids = _liquid._local_to_global(sim.engine, solver)
        if ids is None or not len(ids):
            continue
        local = {int(g): i for i, g in enumerate(ids)}
        if all(int(g) in local for g in rows.ravel()):
            return solver.model.particle_flags, np.vectorize(local.__getitem__)(rows).astype(np.int32)
    return sim.engine.model.particle_flags, rows.astype(np.int32)


def _particle_sets(sim: Any, plan: ExportPlan, routes: dict[str, Any]) -> None:
    idx, stride = sim.index, int(sim.index.stride["particle"])
    sets = [(n, r) for n, r in idx.particles.items() if n not in idx.tris and n in routes]
    total = sum(len(r) for _, r in sets)
    cols, at = [], 0
    for mid, (name, r) in enumerate(sets):
        k = max(1, int(plan.max_points * len(r) / max(total, 1))) if total > plan.max_points else len(r)
        local = _subset(len(r), k)
        rows = np.array([[w * stride + r.start + i for i in local] for w in plan.worlds], np.int64)
        flags, flag_rows = _flag_source(sim, rows)
        material = str(getattr(sim.scene, "items", {}).get(name, {}).get("material", ""))
        plan.sets.append(ParticleSet(name, str(routes[name].get("route", "")), material, mid, len(r), local, flags,
                                     flag_rows))
        cols.append(rows)
        plan.part[name] = slice(at, at + len(local))
        at += len(local)
    plan.part_rows = _columns(cols, len(plan.worlds))


def _rods(sim: Any, plan: ExportPlan) -> None:
    from alloy.api._rods import rod_lines  # noqa: PLC0415

    B, cols, local, at = int(sim.index.stride["body"]), [], [], 0
    for name, line in rod_lines(sim.built).items():
        bodies = list(line.bodies) + [line.bodies[-1]]
        pts = list(line.start) + [line.end[-1]]
        cols.append(np.array([[w * B + int(b) for b in bodies] for w in plan.worlds], np.int64))
        local += [np.asarray(p, np.float32) for p in pts]
        plan.rod[name] = slice(at, at + len(pts))
        segs = [int(b) for b in line.bodies]
        plan.rod_info[name] = {"segments": len(segs), "route": str(sim.built.handles[name].decision.route),
                               "bodies": segs, "drawn_by_bodies": any(plan.shapes[b] for b in segs)}
        at += len(pts)
    plan.rod_rows = _columns(cols, len(plan.worlds))
    plan.rod_local = np.array(local, np.float32).reshape(-1, 3)


def _cut_tets(sim: Any, plan: ExportPlan) -> None:
    """Each picked world's cuttable whole tets (a virtual-node cutter's ``copies.tet``; -1 pads a shorter world).
    A split kills exactly one of them, fold or not (WP10), so their dead count is a faithful cut counter."""
    copies = getattr(getattr(getattr(sim.engine, "topology", None), "cutter", None), "copies", None)
    tets = np.asarray(getattr(copies, "tet", np.zeros(0, np.int64)), np.int64)
    if not len(tets):
        plan.cut_rows = np.zeros((len(plan.worlds), 0), np.int32)
        return
    m = sim.engine.model
    world = m.particle_world.numpy()[m.tet_indices.numpy().reshape(-1, 4)[tets, 0]]
    per = [tets[world == w] for w in plan.worlds]
    k = max(len(x) for x in per)
    plan.cut_rows = np.array([np.pad(x, (0, k - len(x)), constant_values=-1) for x in per], np.int32)


def _bounds(plan: ExportPlan, bonds_per_world: int) -> None:
    """topology_version packs three counts (broken bonds, retired bodies, dead cuttable tets) at 2**0, 2**20 and
    2**40: each must stay below its field, or two different topologies could read alike."""
    from alloy.api.errors import SceneError  # noqa: PLC0415

    fields = {"bonds": bonds_per_world, "bodies": plan.body_rows.shape[1], "cuttable tets": plan.cut_rows.shape[1]}
    limits = {"bonds": 1 << 20, "bodies": 1 << 20, "cuttable tets": 1 << 23}
    over = {k: v for k, v in fields.items() if v >= limits[k]}
    if over:
        raise SceneError(f"frame export: too many {', '.join(over)} per world for topology_version {over}",
                         "export a scene with fewer per world (bonds and bodies < 2**20, cuttable tets < 2**23)",
                         field="worlds")


def _bonds(sim: Any, plan: ExportPlan) -> None:
    topo = getattr(sim.engine, "topology", None)
    bonds = getattr(topo, "bonds", None)                # a virtual-node cutter (VNodeTopology) has no bonds
    if bonds is None or not bonds.count:
        plan.bond_range = np.zeros((len(plan.worlds), 2), np.int32)
        return
    ws = topo.tables.world_start.numpy()
    plan.bond_range = np.array([[ws[w], ws[w + 1]] for w in plan.worlds], np.int32)


def build_plan(sim: Any, worlds: list[int], max_points: int) -> ExportPlan:
    """The plan for ``sim`` (an ``alloy.api`` Sim) and the picked ``worlds``."""
    routes = {name: r for name, r in sim.routes().items()}
    plan = ExportPlan(worlds=[int(w) for w in worlds], max_points=int(max_points), body_names=[],
                      body_rows=np.zeros((len(worlds), 0), np.int32))
    _bodies(sim, plan)
    _meshes(sim, plan)
    _particle_sets(sim, plan, routes)
    _rods(sim, plan)
    _bonds(sim, plan)
    _cut_tets(sim, plan)
    _bounds(plan, int((plan.bond_range[:, 1] - plan.bond_range[:, 0]).max()) if len(plan.bond_range) else 0)
    return plan


__all__ = ["ExportPlan", "ParticleSet", "build_plan"]
