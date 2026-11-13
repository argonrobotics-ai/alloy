# SPDX-License-Identifier: Apache-2.0
"""Object and scene validation (SPEC §3.2): shape, loads, thin members, watertightness, precision partners,
topology prerequisites, and scene cross-references (placements, randomization, attachments, emitters)."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from alloy.cards import geometry as geo
from alloy.cards import types as ct
from alloy.cards.defaults import default_leaf_paths, is_routing_leaf
from alloy.cards.paths import CardView
from alloy.cards.validate import (DEFAULT_LEAF_WARNING, NAME_RE, ROUND_SHAPES, Issues, check_leaves, check_nonneg,
                                  check_plain_floats, check_positive, fmt)
from alloy.core.errors import CardError

if TYPE_CHECKING:
    from alloy.cards.library import CardLibrary

log = logging.getLogger("alloy.cards")

MIN_CHAIN_LINKS = 3                     # features.chain_links floor (SCR-DC3-7)
SHELL_RATIO = 0.2                       # thickness <= 0.2 x smallest in-plane extent
ANALYTIC_RE = re.compile(r"^[a-z][a-z0-9_]*:[^\s]+$")
ISO_THREAD_RE = re.compile(r"^iso_thread:M(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?):(\d[a-hA-H])$")


def _route_values() -> frozenset[str] | None:
    try:
        from alloy.route.types import Route  # noqa: PLC0415 - WP0 type, optional at import time
    except ImportError:
        return None
    return frozenset(r.value for r in Route)


# ------------------------------------------------------------------------------------------------ shapes


def _asset(lib: CardLibrary, iss: Issues, path_field: str, rel: str) -> Path | None:
    try:
        p = lib.asset_path(rel)
    except CardError as exc:                     # the path exists under both library/ and alloy-assets
        iss.error(path_field, exc.message)
        return None
    if p is None or not p.is_file():
        roots = getattr(lib, "search_roots", lib.roots)
        iss.error(path_field, f"file not found: {rel!r} (library roots: {', '.join(map(str, roots))})")
        return None
    return p


def _mesh(lib: CardLibrary, iss: Issues, path_field: str, rel: str, scale: ct.Vec3 = (1.0, 1.0, 1.0)) -> geo.MeshData | None:
    p = _asset(lib, iss, path_field, rel)
    if p is None:
        return None
    try:
        mesh = lib.mesh(rel).scaled(scale)
    except (ValueError, OSError) as exc:
        iss.error(path_field, f"cannot read mesh {rel!r}: {exc}")
        return None
    bad = int((~np.isfinite(mesh.vertices)).sum())
    if bad:                                      # NaN breaks every mesh fact below (eigh, volume, welding)
        iss.error(path_field, f"mesh {rel!r} has {bad} non-finite vertex coordinate(s) (NaN or inf)")
        return None
    return mesh


def _check_primitive(iss: Issues, p: ct.Primitive) -> None:
    want = geo.PRIMITIVE_ARITY[p.kind]
    if len(p.size) != want:
        iss.error("shape.size", f"{p.kind} needs {want} size values, got {len(p.size)}")
        return
    for i, s in enumerate(p.size):
        if p.kind == "plane":
            check_nonneg(iss, f"shape.size.{i}", s, " m")
        else:
            check_positive(iss, f"shape.size.{i}", s, " m")


def _check_mesh_source(iss: Issues, card: ct.ObjectCard, s: ct.MeshSource, lib: CardLibrary) -> geo.MeshData | None:
    for i, k in enumerate(s.scale):
        check_positive(iss, f"shape.scale.{i}", k)
    if s.max_parts < 1 or s.max_parts > card.budget.collision_parts:
        iss.error("shape.max_parts", f"{s.max_parts} outside [1, budget.collision_parts={card.budget.collision_parts}]")
    path = _asset(lib, iss, "shape.path", s.path)
    if path is None:
        return None
    digest = geo.sha256_file(path)
    if digest != s.sha256:
        iss.error("shape.sha256", f"file {s.path!r} has sha256 {digest[:12]}..., card says {s.sha256[:12]}...")
    mesh = _mesh(lib, iss, "shape.path", s.path, s.scale)
    if mesh is not None and geo.is_closed(mesh) and geo.signed_volume(mesh) < 0:
        iss.error("shape.path", f"mesh {s.path!r} is inside out (negative signed volume): flip its winding")
    elif mesh is not None and s.collision in ("sdf", "mesh") and not geo.is_watertight(mesh):
        iss.error("shape.path", f"collision={s.collision!r} needs a watertight, consistently wound mesh")
    return mesh


def _watertight_mesh(lib: CardLibrary, iss: Issues, fld: str, rel: str, what: str) -> geo.MeshData | None:
    mesh = _mesh(lib, iss, fld, rel)
    if mesh is not None and not geo.is_watertight(mesh):
        iss.error(fld, f"{what} must be watertight")
    return mesh


def _check_sdf(iss: Issues, s: ct.SDFSource, lib: CardLibrary) -> geo.MeshData | None:
    if (s.path is None) == (s.analytic_id is None):
        iss.error("shape", "SDFSource needs exactly one of path / analytic_id")
    check_positive(iss, "shape.voxel", s.voxel, " m")
    if s.analytic_id is not None and not ANALYTIC_RE.match(s.analytic_id):
        iss.error("shape.analytic_id", f"expected '<family>:<args>', got {s.analytic_id!r}")
    if s.path is None:
        return None
    if Path(s.path).suffix.lower() in (".obj", ".stl"):
        return _watertight_mesh(lib, iss, "shape.path", s.path, "an SDF source mesh")
    _asset(lib, iss, "shape.path", s.path)
    return None


def _check_tet(iss: Issues, s: ct.TetSource, lib: CardLibrary) -> geo.MeshData | None:
    if (s.path is None) == (s.from_mesh is None):
        iss.error("shape", "TetSource needs exactly one of path / from_mesh")
    check_positive(iss, "shape.edge_length", s.edge_length, " m")
    if s.from_mesh is not None:
        return _watertight_mesh(lib, iss, "shape.from_mesh", s.from_mesh, "a tet source mesh")
    if s.path is not None:
        _asset(lib, iss, "shape.path", s.path)
    return None


def _check_curve(iss: Issues, s: ct.CurveSource, lib: CardLibrary) -> None:
    if len(s.points) < 2:
        iss.error("shape.points", f"a curve needs >= 2 points, got {len(s.points)}")
    elif geo.curve_length(s) <= 0:
        iss.error("shape.points", "zero-length curve")
    check_positive(iss, "shape.radius", s.radius, " m")


def _check_particles(iss: Issues, s: ct.ParticleSource, lib: CardLibrary) -> None:
    lo, hi = s.region
    if any(hi[i] <= lo[i] for i in range(3)):
        iss.error("shape.region", f"region lo {lo} must be below hi {hi} on every axis")
    check_positive(iss, "shape.spacing", s.spacing, " m")
    check_positive(iss, "shape.grain_diameter", s.grain_diameter, " m")


def _check_file_source(iss: Issues, s: ct.USDSource | ct.MJCFSource, lib: CardLibrary) -> None:
    if _asset(lib, iss, "shape.path", s.path) is None or not isinstance(s, ct.MJCFSource):
        return
    try:
        lib.mjcf_cloud(s)
    except KeyError as exc:
        iss.error("shape.body", str(exc).strip("'\""))
    except Exception as exc:  # noqa: BLE001 - Newton's MJCF importer raises many types for a bad file
        iss.error("shape.path", f"Newton cannot import {s.path!r}: {type(exc).__name__}: {exc}")


_SOURCE_CHECKS = {ct.SDFSource: _check_sdf, ct.TetSource: _check_tet, ct.CurveSource: _check_curve,
                  ct.ParticleSource: _check_particles, ct.USDSource: _check_file_source, ct.MJCFSource: _check_file_source}


def _check_other_source(iss: Issues, s: object, lib: CardLibrary) -> geo.MeshData | None:
    """Rules of the non-primitive, non-mesh sources; returns the backing mesh where there is one."""
    return _SOURCE_CHECKS[type(s)](iss, s, lib)


def shape_extents(card: ct.ObjectCard, mesh: geo.MeshData | None) -> tuple[float, float, float] | None:
    s = card.shape
    if isinstance(s, ct.Primitive) and len(s.size) == geo.PRIMITIVE_ARITY[s.kind]:
        return geo.primitive_extents(s)
    if isinstance(s, ct.CurveSource) and len(s.points) >= 2:
        return geo.curve_extents(s)
    if isinstance(s, ct.ParticleSource):
        return geo.box_extents(*s.region)
    if mesh is not None:
        return geo.obb_extents(mesh.vertices)
    return None


def _min_member(card: ct.ObjectCard, mesh: geo.MeshData | None, within: float = math.inf) -> float | None:
    """Smallest solid dimension of shapes that become rigid geometry; None if unknown or not applicable. A mesh wall
    of ``within`` or more reads inf (only thinner walls matter to the caller; the probe skips far triangles)."""
    s = card.shape
    cands: list[float] = []
    if isinstance(s, ct.Primitive) and len(s.size) == geo.PRIMITIVE_ARITY[s.kind]:
        cands.append(geo.primitive_min_member(s))
    elif isinstance(s, ct.CurveSource):
        cands.append(2 * s.radius)
    elif mesh is not None and isinstance(s, (ct.MeshSource, ct.SDFSource)) and geo.is_closed(mesh):
        cands.append(geo.wall_thickness(mesh, within=within))
    if card.features.thickness is not None:
        cands.append(card.features.thickness.value)
    return min(cands) if cands else None


# ------------------------------------------------------------------------------------------------ objects


def _check_chain_links(iss: Issues, card: ct.ObjectCard) -> None:
    """``features.chain_links`` (SCR-DC3-7): a whole number >= 3 of CHAIN links; only curve sources route to CHAIN."""
    n = card.features.chain_links
    if n is None:
        return
    if isinstance(n, bool) or not isinstance(n, int):
        iss.error("features.chain_links", f"a link count is a whole number (not randomizable), got {n!r}")
    elif n < MIN_CHAIN_LINKS:
        iss.error("features.chain_links", f"need >= {MIN_CHAIN_LINKS} links, got {n}")
    if not isinstance(card.shape, ct.CurveSource):
        iss.warn("features.chain_links", "a CHAIN link count only applies to curve (rod/cable) sources")


def _check_loads_features(iss: Issues, card: ct.ObjectCard, ext: tuple[float, float, float] | None) -> None:
    ld, ft = card.loads, card.features
    for name in ("grip_force", "bend_force"):
        check_nonneg(iss, f"loads.{name}", getattr(ld, name), " N")
    for name, unit in (("design_force", " N"), ("contact_speed", " m/s"), ("tolerance", " m")):
        check_positive(iss, f"loads.{name}", getattr(ld, name), unit)
    for name in ("thickness", "passage_width", "min_wall"):
        check_positive(iss, f"features.{name}", getattr(ft, name), " m")
    for name in ("bend_stiffness", "twist_stiffness"):
        check_positive(iss, f"features.{name}", getattr(ft, name), " N·m^2")
        if getattr(ft, name) is not None and not isinstance(card.shape, ct.CurveSource):
            iss.warn(f"features.{name}", "a measured rod stiffness only applies to curve (rod/cable) sources")
    _check_chain_links(iss, card)
    for i, c in enumerate(ft.creases):
        check_positive(iss, f"features.creases.{i}.stiffness", c.stiffness, " N·m/rad")
        if c.a == c.b:
            iss.error(f"features.creases.{i}", "crease endpoints coincide")
    if ft.thickness is not None and ext is not None and ext[1] > 0 and math.isfinite(ext[1]):
        if ft.thickness.value > SHELL_RATIO * ext[1]:
            iss.error("features.thickness", f"shell thickness {fmt(ft.thickness.value)} m > {SHELL_RATIO} x smallest "
                      f"in-plane extent {fmt(ext[1])} m (not a shell)")
    for i, b in enumerate((card.budget.collision_parts, card.budget.tets, card.budget.particles)):
        if b < 1:
            iss.error(("budget.collision_parts", "budget.tets", "budget.particles")[i], f"budget must be >= 1, got {b}")


def _check_thin_member(iss: Issues, card: ct.ObjectCard, mesh: geo.MeshData | None, rigid_dt: float) -> None:
    if isinstance(card.shape, (ct.ParticleSource, ct.TetSource)):
        return
    limit = 2.0 * card.loads.contact_speed.value * rigid_dt
    declared = card.features.min_wall
    t_min = _min_member(card, mesh, max(limit, declared.value / 1.5) if declared is not None else limit)
    if t_min is None or t_min >= limit:
        return
    if card.features.min_wall is None:
        iss.error("features.min_wall", f"thinnest member {fmt(t_min)} m < 2·contact_speed·rigid_dt = {fmt(limit)} m "
                  "(tunnels in one step, GOTCHAS:413); declare features.min_wall so CCD/margins are sized")
    elif card.features.min_wall.value > 1.5 * t_min:
        iss.warn("features.min_wall", f"declared {fmt(card.features.min_wall.value)} m but the geometry has "
                 f"{fmt(t_min)} m")


def _material(iss: Issues, lib: CardLibrary, field: str, name: str) -> ct.MaterialCard | None:
    """The validated material ``name``; an error at ``field`` if it is unknown, invalid or defined twice."""
    if not lib.has_material(name):
        iss.error(field, f"unknown material {name!r}")
        return None
    try:
        return lib.material(name)
    except CardError as exc:
        iss.error(field, f"material {name!r}: {exc}")
        return None


def _check_materials(iss: Issues, card: ct.ObjectCard, lib: CardLibrary,
                     mesh: geo.MeshData | None) -> list[tuple[str, ct.MaterialCard]]:
    """(path prefix, material) for the main material and each part material that resolves."""
    mats: list[tuple[str, ct.MaterialCard]] = []
    for field, mname in [("material", card.material),
                         *((f"part_materials.{part}", m) for part, m in sorted(card.part_materials.items()))]:
        mat = _material(iss, lib, field, mname)
        if mat is not None:
            mats.append((field, mat))
    for part in sorted(card.part_materials):
        if isinstance(card.shape, ct.MeshSource) and mesh is not None and part not in mesh.groups:
            iss.error(f"part_materials.{part}", f"mesh {card.shape.path!r} has no group {part!r} "
                      f"(groups: {sorted(mesh.groups)})")
    return mats


def _check_rolling(iss: Issues, card: ct.ObjectCard, mats: list[tuple[str, ct.MaterialCard]]) -> None:
    rolls = isinstance(card.shape, ct.Primitive) and card.shape.kind in ROUND_SHAPES
    rolls = rolls or isinstance(card.shape, ct.CurveSource)
    for prefix, m in mats:
        if m.surface.rolling_lever is not None and m.surface.rolling_lever.value > 0 and not rolls:
            iss.warn(f"{prefix}.surface.rolling_lever", f"material {m.name!r} sets a rolling lever but "
                     f"{card.name!r} is not a round shape (GOTCHAS:471)")


def _feature_with_clearance(card: ct.ObjectCard, feature: str | None) -> ct.PrecisionFeature | None:
    for f in card.precision_features:
        if (feature is None or f.name == feature) and f.clearance is not None:
            return f
    return None


def _check_precision(iss: Issues, card: ct.ObjectCard, lib: CardLibrary, mesh: geo.MeshData | None) -> None:
    names: set[str] = set()
    for i, f in enumerate(card.precision_features):
        p = f"precision_features.{i}"
        if f.name in names:
            iss.error(f"{p}.name", f"duplicate feature name {f.name!r}")
        names.add(f.name)
        if any(f.crop[1][k] <= f.crop[0][k] for k in range(3)):
            iss.error(f"{p}.crop", f"crop lo {f.crop[0]} must be below hi {f.crop[1]} on every axis")
        check_positive(iss, f"{p}.voxel", f.voxel, " m")
        check_positive(iss, f"{p}.clearance", f.clearance, " m")
        if f.analytic_id is not None and not (ANALYTIC_RE.match(f.analytic_id)):
            iss.error(f"{p}.analytic_id", f"expected '<family>:<args>', got {f.analytic_id!r}")
        elif f.analytic_id is not None and f.analytic_id.startswith("iso_thread:") and not ISO_THREAD_RE.match(f.analytic_id):
            iss.error(f"{p}.analytic_id", f"expected 'iso_thread:M<d>x<pitch>:<class>' (e.g. M10x1.5:6g), got {f.analytic_id!r}")
        if f.mode == "hydroelastic" and mesh is not None and not geo.is_watertight(mesh):
            iss.error(f"{p}.mode", "hydroelastic features need a watertight mesh")
    for i, entry in enumerate(card.precision_partners):
        _check_partner(iss, card, lib, i, entry)


def _check_partner(iss: Issues, card: ct.ObjectCard, lib: CardLibrary, i: int, entry: str) -> None:
    p = f"precision_partners.{i}"
    other, _, feature = entry.partition(":")
    mine = _feature_with_clearance(card, None)
    if mine is None:
        iss.error("precision_features", f"partner {entry!r} needs a precision feature with clearance on {card.name!r}")
    if not lib.has_object(other):
        iss.error(p, f"unknown partner object {other!r}")
        return
    try:
        partner = lib.peek_object(other)
    except CardError as exc:
        iss.error(p, f"partner {other!r} does not load: {exc}")
        return
    theirs = _feature_with_clearance(partner, feature or None)
    if theirs is None:
        what = f"feature {feature!r}" if feature else "a precision feature"
        iss.error(p, f"partner {other!r} has no {what} with clearance")
    elif mine is not None and abs(mine.clearance.value - theirs.clearance.value) > 1e-3 * mine.clearance.value:
        iss.warn(p, f"clearance {fmt(mine.clearance.value)} m here vs {fmt(theirs.clearance.value)} m on {other!r}")


def _check_topology_route(iss: Issues, card: ct.ObjectCard, mats: list[tuple[str, ct.MaterialCard]]) -> None:
    needs = card.blade is not None or any(
        m.failure is not None and (m.failure.tear_energy is not None or m.failure.cut_min_force is not None)
        for _, m in mats)
    film = card.route is not None and card.route.route == "cloth_film" and card.blade is None
    if needs and card.topology is None and not film:    # a film tears only along the tear lines its budget declares
        iss.error("topology", "a blade or a tear/cut failure needs a topology budget (spares are preallocated, D19)")
    if card.topology is not None:
        for name in ("spare_particles", "spare_vertices", "spare_tets", "max_events_per_step"):
            if getattr(card.topology, name) < 0:
                iss.error(f"topology.{name}", "must be >= 0")
    if card.blade is not None:
        if card.blade.edge_a == card.blade.edge_b:
            iss.error("blade", "blade edge endpoints coincide")
        check_positive(iss, "blade.edge_radius", card.blade.edge_radius, " m")
    if card.route is not None:
        allowed = _route_values()
        if allowed is not None and card.route.route not in allowed:
            iss.error("route.route", f"unknown route {card.route.route!r} (one of {sorted(allowed)})")
        if not card.route.reason.strip():
            iss.error("route.reason", "a route override needs a reason")


def _warn_defaults(iss: Issues, card: ct.ObjectCard, mats: list[tuple[str, ct.MaterialCard]]) -> None:
    if not mats or mats[0][0] != "material":
        return
    parts = {prefix.split(".", 1)[1]: m for prefix, m in mats[1:]}
    for path in default_leaf_paths(CardView(card, mats[0][1], parts)):
        if is_routing_leaf(path):
            iss.warn(path, DEFAULT_LEAF_WARNING)


def object_issues(card: ct.ObjectCard, lib: CardLibrary, iss: Issues, rigid_dt: float) -> None:
    if not NAME_RE.match(card.name):
        iss.error("name", f"object names use [A-Za-z0-9_-] only, got {card.name!r}")
    check_leaves(iss, card)
    check_plain_floats(iss, card)
    mesh: geo.MeshData | None = None
    if isinstance(card.shape, ct.Primitive):
        _check_primitive(iss, card.shape)
    elif isinstance(card.shape, ct.MeshSource):
        mesh = _check_mesh_source(iss, card, card.shape, lib)
    else:
        mesh = _check_other_source(iss, card.shape, lib)
    mats = _check_materials(iss, card, lib, mesh)
    _check_loads_features(iss, card, shape_extents(card, mesh))
    _check_thin_member(iss, card, mesh, rigid_dt)
    _check_rolling(iss, card, mats)
    _check_precision(iss, card, lib, mesh)
    _check_topology_route(iss, card, mats)
    _warn_defaults(iss, card, mats)


def validate_object(card: ct.ObjectCard, lib: CardLibrary, rigid_dt: float = 1.25e-3) -> list[ct.CardIssue]:
    """All object rules (SPEC §3.2); ``rigid_dt`` is the rigid step on the D4 grid used by the thin-member rule."""
    iss = Issues(card.name)
    object_issues(card, lib, iss, rigid_dt)
    return iss.items

