# SPDX-License-Identifier: Apache-2.0
"""Turn a Newton Model plus the recorder's declarations into a fixed export layout."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .bundle import BundleError
from .chains import SegmentSpec, apply_pose, build_chain, invert_apply, order_segments
from .extract import (MeshStore, WorldPlan, extract_shapes, host_shape_arrays, mark_replicas, model_worlds,
                      plan_worlds, shape_geometry, to_np, visible_shapes)
from .surfaces import SurfacePlan, detect_surfaces, make_surface, read_twin_pairs

log = logging.getLogger("alloy.render")

NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
CHAIN_GEOS = (4, 6)   # capsule, cylinder
DEFAULT_RADIUS = 0.005


@dataclass
class Decls:
    """What the user declared explicitly through ``add_*``."""

    points: list = field(default_factory=list)       # (name, ids, radius)
    surfaces: list = field(default_factory=list)     # (name, tri, kind)
    curve_particles: list = field(default_factory=list)  # (name, ids, radius)
    curve_bodies: list = field(default_factory=list)     # (name, selector, radius)
    instancers: list = field(default_factory=list)       # (name, selector)
    skip: list = field(default_factory=list)             # particle id arrays never auto-exported
    leftover_points: bool = True                         # export particles no set covers as particles_<k>

    def names(self) -> dict[str, list[str]]:
        return {"surfaces": [d[0] for d in self.surfaces], "points": [d[0] for d in self.points],
                "curves": [d[0] for d in self.curve_particles + self.curve_bodies],
                "instancers": [d[0] for d in self.instancers]}


@dataclass
class PointPlan:
    name: str
    world: int
    ids: np.ndarray       # model particle ids
    radius: float


@dataclass
class CurvePlan:
    name: str
    world: int
    source: str           # "bodies" or "particles"
    radius: float
    ids: np.ndarray | None = None      # particle ids
    rows: np.ndarray | None = None     # captured-body rows in chain order
    spec: SegmentSpec | None = None

    @property
    def count(self) -> int:
        return len(self.ids) if self.source == "particles" else len(self.rows) + 2


@dataclass
class InstancerPlan:
    name: str
    world: int
    rows: np.ndarray      # captured-body rows
    geom: dict


@dataclass
class Layout:
    """Everything the recorder needs to capture and write a bundle."""

    worlds: WorldPlan
    store: MeshStore
    shapes: list[dict]
    bodies: list[dict]
    body_rows: np.ndarray        # captured-body rows of the exported bodies, in column order
    cap_bodies: np.ndarray       # model body ids to copy each frame
    cap_body_off: np.ndarray     # [len(cap_bodies), 3] world offsets
    cap_particles: np.ndarray    # model particle ids to copy each frame
    p_lookup: np.ndarray         # model particle id -> row in cap_particles (-1 if absent)
    p_off: np.ndarray            # [len(cap_particles), 3] offsets
    surfaces: list[SurfacePlan]
    points: list[PointPlan]
    curves: list[CurvePlan]
    instancers: list[InstancerPlan]

    @property
    def need_velocity(self) -> bool:
        return bool(self.points)


def check_names(decls: Decls) -> None:
    for kind, names in decls.names().items():
        for n in names:
            if not NAME_RE.match(n):
                raise BundleError(f"{kind} name {n!r} must match [A-Za-z0-9_.-]+")
        if len(set(names)) != len(names):
            raise BundleError(f"duplicate {kind} names in {names}")


def split_by_world(name: str, ids: np.ndarray, world_of: np.ndarray,
                   plan: WorldPlan) -> list[tuple[str, int, np.ndarray]]:
    """Group ``ids`` by world, dropping worlds that are not exported; suffix ``_w<k>`` when split."""
    ids = np.asarray(ids, np.int64)
    ids = ids[plan.selects(world_of[ids])]
    worlds = sorted(set(int(w) for w in world_of[ids]))
    if not worlds:
        log.warning("%s: no members in the exported worlds; skipped", name)
    out = [(name if len(worlds) == 1 else f"{name}_w{w}", w, ids[world_of[ids] == w]) for w in worlds]
    return out


def _match_bodies(model, name: str, selector) -> np.ndarray:
    if isinstance(selector, str):
        rx = re.compile(selector)
        ids = [i for i, label in enumerate(model.body_label) if rx.search(label)]
    else:
        ids = [int(i) for i in selector]
    n = len(model.body_label)
    if not ids or any(not 0 <= i < n for i in ids):
        raise BundleError(f"{name}: selector {selector!r} matched no valid bodies (model has {n})")
    return np.asarray(sorted(set(ids)), np.int64)


def _body_shape(host: dict, body: int, kinds: tuple | None, visible: np.ndarray) -> list[int]:
    idx = np.flatnonzero(host["body"] == body)
    if kinds is None:
        return [int(i) for i in idx if visible[i]]
    return [int(i) for i in idx if int(host["type"][i]) in kinds]


def plan_instancers(model, decls: Decls, host: dict, world_b: np.ndarray, wplan: WorldPlan,
                    store: MeshStore) -> list[tuple[str, int, np.ndarray, dict]]:
    """Resolve instancers: (name, world, body ids, shared geometry record)."""
    visible = visible_shapes(host)
    out = []
    for name, selector in decls.instancers:
        bodies = _match_bodies(model, name, selector)
        geoms = {}
        for b in bodies:
            shapes = _body_shape(host, int(b), None, visible)
            if len(shapes) != 1:
                raise BundleError(f"instancer {name!r}: body {int(b)} ({model.body_label[b]!r}) has {len(shapes)} "
                                  "visible shapes, need exactly 1")
            g = shape_geometry(model, shapes[0], store, host)
            geoms[int(b)] = g
        first = next(iter(geoms.values()))
        for b, g in geoms.items():
            if g != first:
                raise BundleError(f"instancer {name!r}: body {b} has different geometry than the first body")
        for sub, world, ids in split_by_world(name, bodies, world_b, wplan):
            out.append((sub, world, ids, first))
    return out


def _chain_for_group(model, name: str, bodies: np.ndarray, host: dict, bq0: np.ndarray, radius: float | None):
    ends, to_local, radii = [], [], []
    for b in bodies:
        shapes = _body_shape(host, int(b), CHAIN_GEOS, visible_shapes(host))
        if not shapes:
            raise BundleError(f"curve {name!r}: body {int(b)} ({model.body_label[b]!r}) has no capsule/cylinder shape")
        s = shapes[0]
        hh = float(host["scale"][s][1])
        local = apply_pose(host["xform"][s].astype(np.float64), np.array([[0, 0, -hh], [0, 0, hh]], np.float64))
        pose = bq0[b].astype(np.float64)
        ends.append(apply_pose(pose, local))
        to_local.append(lambda p, pose=pose: invert_apply(pose, p))
        radii.append(float(host["scale"][s][0]))
    ends_a = np.asarray(ends)
    r = float(radius) if radius is not None else radii[0]
    order = order_segments(ends_a, r)
    return order, build_chain(ends_a, to_local, order), r


def _rows_lookup(cap_bodies: np.ndarray, n_bodies: int) -> np.ndarray:
    lut = np.full(n_bodies, -1, np.int64)
    lut[cap_bodies] = np.arange(len(cap_bodies))
    return lut


def plan_curves_from_bodies(model, decls, host, world_b, wplan, lut, bq0) -> list[CurvePlan]:
    out = []
    for name, selector, radius in decls.curve_bodies:
        bodies = _match_bodies(model, name, selector)
        for sub, world, ids in split_by_world(name, bodies, world_b, wplan):
            order, spec, r = _chain_for_group(model, sub, ids, host, bq0, radius)
            out.append(CurvePlan(sub, world, "bodies", r, rows=lut[ids[order]], spec=spec))
    return out


def _median_radius(model, ids: np.ndarray, given: float | None) -> float:
    if given is not None:
        return float(given)
    rad = to_np(getattr(model, "particle_radius", None))
    return float(np.median(rad[ids])) if rad is not None and len(rad) and len(ids) else DEFAULT_RADIUS


def _check_ids(name: str, ids, n: int) -> np.ndarray:
    a = np.asarray(ids, np.int64).reshape(-1)
    if len(a) == 0 or a.min() < 0 or a.max() >= n:
        raise BundleError(f"{name}: particle ids must be non-empty and in [0, {n}), got range "
                          f"{(int(a.min()), int(a.max())) if len(a) else 'empty'}")
    return a


def plan_particle_sets(model, decls: Decls, world_p: np.ndarray, wplan: WorldPlan, twin_pairs: np.ndarray):
    """Explicit surfaces, points, particle curves, and the particles they cover."""
    n = len(world_p)
    covered = np.zeros(n, bool)
    surfaces, points, curves = [], [], []
    for name, tri, kind in decls.surfaces:
        tri = np.asarray(tri, np.int64).reshape(-1, 3)
        _check_ids(name, tri, n)
        for sub, world, _ in split_by_world(name, tri[:, 0], world_p, wplan):
            sel = tri[world_p[tri[:, 0]] == world]
            surfaces.append(make_surface(sub, kind, world, sel, twin_pairs))
        covered[tri.reshape(-1)] = True
    for name, ids, radius in decls.points:
        ids = _check_ids(name, ids, n)
        for sub, world, g in split_by_world(name, ids, world_p, wplan):
            points.append(PointPlan(sub, world, g, _median_radius(model, g, radius)))
        covered[ids] = True
    for name, ids, radius in decls.curve_particles:
        ids = _check_ids(name, ids, n)
        for sub, world, g in split_by_world(name, ids, world_p, wplan):
            curves.append(CurvePlan(sub, world, "particles", _median_radius(model, g, radius), ids=g))
        covered[ids] = True
    return surfaces, points, curves, covered


def plan_auto(model, world_p: np.ndarray, wplan: WorldPlan, covered: np.ndarray,
              decls: Decls) -> tuple[list[SurfacePlan], list[PointPlan]]:
    """Auto-detected surfaces (skipping ones fully covered by explicit or skipped sets) and leftover particles."""
    skipped = np.zeros(len(world_p), bool)
    for _, ids in decls.skip:
        skipped[_check_ids("skip_particles", ids, len(world_p))] = True
    auto, auto_cover = detect_surfaces(model, world_p, set(wplan.worlds))
    surfaces = [s for s in auto if not (covered | skipped)[s.src].all()]
    cover = covered | auto_cover | skipped
    leftovers = np.flatnonzero(~cover & wplan.selects(world_p))
    points = []
    if len(leftovers) and decls.leftover_points:
        for k, (_, world, ids) in enumerate(split_by_world("particles", leftovers, world_p, wplan)):
            points.append(PointPlan(f"particles_{k}", world, ids, _median_radius(model, ids, None)))
    return surfaces, points


def _captured_particles(surfaces, points, curves) -> np.ndarray:
    """Model particle ids any exported set reads: only these are copied to host each frame."""
    parts = [s.src for s in surfaces] + [p.ids for p in points] + [c.ids for c in curves]
    if not parts:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate([np.asarray(x, np.int64) for x in parts]))


def build_layout(model, decls: Decls, worlds, grid_m: float | None) -> Layout:
    """Plan the whole export: worlds, bodies, shapes, particle sets, curves, instancers."""
    check_names(decls)
    wa = model_worlds(model)
    wplan = plan_worlds(wa, worlds, grid_m)
    host = host_shape_arrays(model)
    store = MeshStore()
    world_b, world_s, world_p = wa["body"], wa["shape"], wa["particle"]

    inst = plan_instancers(model, decls, host, world_b, wplan, store)
    skip = {int(b) for _, _, ids, _ in inst for b in ids}
    exported = np.flatnonzero(wplan.selects(world_b))
    body_ids = np.asarray([b for b in exported if int(b) not in skip], np.int64)
    cap_bodies = np.unique(np.concatenate([body_ids, np.asarray(sorted(skip), np.int64)]))
    lut = _rows_lookup(cap_bodies, len(world_b))
    bodies = [{"id": c, "label": str(model.body_label[b]), "world": int(world_b[b])} for c, b in enumerate(body_ids)]
    col = {int(b): c for c, b in enumerate(body_ids)}
    shapes = extract_shapes(model, host, world_s, col, wplan, store, skip)
    mark_replicas(shapes, world_s, wplan)

    bq0 = to_np(model.body_q)
    curves = plan_curves_from_bodies(model, decls, host, world_b, wplan, lut, bq0)
    twin_pairs = read_twin_pairs(model)
    e_surf, e_pts, e_curves, covered = plan_particle_sets(model, decls, world_p, wplan, twin_pairs)
    a_surf, a_pts = plan_auto(model, world_p, wplan, covered, decls)
    surfaces, points, curves = e_surf + a_surf, e_pts + a_pts, curves + e_curves
    cap_p = _captured_particles(surfaces, points, e_curves)
    p_lookup = np.full(len(world_p), -1, np.int64)
    p_lookup[cap_p] = np.arange(len(cap_p))
    instancers = [InstancerPlan(n, w, lut[ids], g) for n, w, ids, g in inst]
    return Layout(wplan, store, shapes, bodies, lut[body_ids], cap_bodies, wplan.offset_of(world_b[cap_bodies]),
                  cap_p, p_lookup, wplan.offset_of(world_p[cap_p]), surfaces, points, curves, instancers)
