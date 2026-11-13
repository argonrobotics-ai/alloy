# SPDX-License-Identifier: Apache-2.0
"""Rigid emitters: STATIC, RIGID, RIGID_COMPLIANT, RIGID_PILE (one body per object) and GRAIN_RIGID (one sphere body
per grain on a lattice filling the particle region).

A kinematic fixture (``alloy.Scene.add_open_box(..., kinematic=True)`` and the other fixtures) is one object per part,
but one body: ``emit_part`` emits its root part (decision number ``FIXTURE_ROOT``) as a kinematic body and every other
part of the same ``FIXTURE_GROUP`` as a shape on that body, at its pose relative to the root. A kinematic cup was 25
kinematic bodies (150 dofs beside a 2-dof robot) and is now 1 (6 dofs; FIX-LIST item 6, 2026-11-09). Each part keeps
its object, shape label and material, so sim.routes(), pair tables and shape-label lookups see the same parts. Each
shape adds its own part's mass, so the body has the whole fixture's mass, centre of mass and inertia (readback sums the
parts, ``readback_checks._riders_mass``). With the Newton bump's soft weld on kinematic roots (cced8ced1) a kinematic
body steps with its physical inertia, so that mass is the fixture's, not the floor's alone."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import (MeshSource, MJCFSource, ParticleSource, Primitive, ResolvedObject, SDFSource,
                               USDSource)
from alloy.core.errors import AssemblyError
from alloy.route.types import Route, RouteDecision
from alloy.scene.emitters.common import TriMesh, add_primitive, load_hulls, load_mesh, newton_mesh, shape_config

log = logging.getLogger("alloy.scene")

CONVEX_SOURCE_MAX = 50_000      # vertices: a larger collision="convex" mesh goes to Newton as its capped hull
FIXTURE_GROUP = "kinematic_fixture"          # decision number: the kinematic fixture a part belongs to (1, 2, ...)
FIXTURE_ROOT = "kinematic_fixture_root"      # decision number 1.0: the part whose body carries the others


def emit_rigid(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    """One body (or world shapes for STATIC) carrying the object's shapes, mass from the card density."""
    card = obj.card
    if isinstance(card.shape, (MJCFSource, USDSource)):
        _emit_imported(b, obj, d)
        return
    static = d.route == Route.STATIC and card.motion == "static"
    lever = d.numbers.get("torsional_lever")
    cfg = shape_config(obj.material, torsional_lever=lever)
    if isinstance(card.shape, MeshSource) and card.shape.collision == "decompose":
        _emit_decomposed(b, obj, cfg, static)
        return
    body = -1 if static else b.add_body(label=card.name, is_kinematic=card.motion == "kinematic")
    _add_shapes(b, body, obj, cfg)
    if body >= 0 and card.motion == "dynamic" and isinstance(card.shape, MeshSource):
        _require_mass(b, body, card.shape, card.name)
    if body >= 0 and card.motion == "kinematic":
        log.debug("%s: kinematic body %d", card.name, body)


def _add_shapes(b: newton.ModelBuilder, body: int, obj: ResolvedObject,
                cfg: newton.ModelBuilder.ShapeConfig) -> None:
    shape = obj.card.shape
    name = obj.card.name
    if isinstance(shape, Primitive):
        add_primitive(b, body, shape, cfg, label=name)
    elif isinstance(shape, MeshSource):
        _add_mesh(b, body, shape, cfg, name)
    elif isinstance(shape, SDFSource):
        raise AssemblyError(f"{name}: SDFSource shapes are precision geometry (WP3 shadow shapes), not a rigid "
                            "collision shape; give the object a mesh or primitive shape")
    else:
        raise AssemblyError(f"{name}: shape source {type(shape).__name__} cannot be emitted as a rigid body")


def _add_mesh(b: newton.ModelBuilder, body: int, src: MeshSource, cfg: newton.ModelBuilder.ShapeConfig,
              name: str) -> None:
    raw = load_mesh(src)
    mesh = newton_mesh(raw)
    if src.collision == "convex" and len(raw.vertices) > CONVEX_SOURCE_MAX:
        _add_large_convex(b, body, raw, mesh, cfg, name)
    elif src.collision == "convex":
        b.add_shape_convex_hull(body, mesh=mesh, cfg=cfg, label=name)
    elif src.collision in ("mesh", "sdf"):
        b.add_shape_mesh(body, mesh=mesh, cfg=cfg, label=name)
    else:
        raise AssemblyError(f"{name}: unknown collision mode {src.collision!r} (convex, decompose, mesh, sdf)")


def _add_large_convex(b: newton.ModelBuilder, body: int, raw: TriMesh, mesh: newton.Mesh,
                      cfg: newton.ModelBuilder.ShapeConfig, name: str) -> None:
    """A convex mesh of more than CONVEX_SOURCE_MAX vertices (a raw scan): its collision shape is the hull MuJoCo
    collides with anyway (``Mesh.maxhullvert`` vertices), carrying the whole mesh's mass, centre and inertia
    (``Mesh.compute_convex_hull`` keeps them), and the mesh itself is a display-only shape. Newton
    builds the support adjacency of a convex source vertex by vertex, which ran past 19 min on a 655,362-vertex
    sphere; meshes this size had failed earlier at a 6.25 TiB width probe, so no scene that built changes."""
    first = b.shape_count
    b.add_shape_convex_hull(body, mesh=mesh.compute_convex_hull(), cfg=cfg, label=name)
    _add_display_mesh(b, body, raw, cfg, name, range(first, b.shape_count))
    log.info("%s: %d mesh vertices collide as their %d-vertex hull (display: the full mesh)", name,
             len(raw.vertices), mesh.maxhullvert)


def _require_mass(b: newton.ModelBuilder, body: int, src: MeshSource, name: str) -> None:
    """An AssemblyError for a dynamic body whose mesh gave it no mass: Newton gives the body density x the mesh's
    signed volume, which is 0 or less for an open, flat or inconsistently wound mesh (half a cube's faces flipped:
    0), and the rigid step's contact gains divide by that mass. Static and kinematic bodies are not judged."""
    mass = float(b.body_mass[body])
    if not mass > 0.0:
        raise AssemblyError(f"{name}: mesh {src.path} encloses no volume, so the moving body has mass {mass:.3g} kg: "
                            "the mesh is open, flat or its faces are wound inconsistently")


def _inertia(mesh: TriMesh, density: float, kind: int) -> tuple[float, wp.vec3, wp.mat33]:
    """Newton's own mass, COM and inertia of a mesh (as alloy.cards.mass.mesh_mass and add_shape_* compute them)."""
    from alloy._vendor.newton.geometry import compute_inertia_shape

    nm = newton.Mesh(mesh.vertices.astype(np.float32), mesh.indices.reshape(-1).astype(np.int32), compute_inertia=True)
    return compute_inertia_shape(kind, wp.vec3(1.0, 1.0, 1.0), nm, float(density))


def _emit_decomposed(b: newton.ModelBuilder, obj: ResolvedObject, cfg: newton.ModelBuilder.ShapeConfig,
                     static: bool) -> None:
    """SCR-OSS-10: one convex shape per precomputed hull; mass and inertia from the card mesh when it is watertight
    (hull density 0), else hull density scaled so the body mass equals the card mass (inertia from the hulls)."""
    from alloy.cards.geometry import MeshData, is_watertight

    card, src = obj.card, obj.card.shape
    if obj.part_materials:
        raise AssemblyError(f"{card.name}: collision='decompose' with part_materials is not supported yet (the body "
                            "mass would not equal card_mass); use one material or collision='convex'/'mesh'")
    mesh, hulls = load_mesh(src), load_hulls(src)
    body = -1
    if not static:
        rho = float(obj.material.density.value)
        mass, com, inertia = _inertia(mesh, rho, int(newton.GeoType.MESH))
        if mass <= 0.0:
            raise AssemblyError(f"{card.name}: card mesh {src.path} has mass {mass:.3g} kg (open or inside-out); "
                                "repair it (watertight, outward winding) with the importer that made it")
        if is_watertight(MeshData(mesh.vertices, mesh.indices)):
            body = b.add_body(label=card.name, is_kinematic=card.motion == "kinematic", mass=mass, com=com,
                              inertia=inertia, lock_inertia=True)
            cfg.density = 0.0
        else:
            vol = sum(_inertia(h, 1.0, int(newton.GeoType.CONVEX_MESH))[0] for h in hulls)
            cfg.density = mass / vol
            body = b.add_body(label=card.name, is_kinematic=card.motion == "kinematic")
            log.info("%s: card mesh not watertight; inertia from hulls (density %.6g kg/m^3)", card.name, cfg.density)
    first = b.shape_count
    for k, h in enumerate(hulls):
        b.add_shape_convex_hull(body, mesh=newton_mesh(h), cfg=cfg, label=f"{card.name}/hull_{k}")
    _add_display_mesh(b, body, mesh, cfg, card.name, range(first, b.shape_count))


def _add_display_mesh(b: newton.ModelBuilder, body: int, mesh: TriMesh, cfg: newton.ModelBuilder.ShapeConfig,
                      name: str, hulls: range) -> int:
    """SCR OSS-9-5 item 4: the card mesh as a display-only shape on the body, and VISIBLE cleared on its hulls, so
    cameras draw the object rather than its convex decomposition (SOFT-3's robot visual layer does the same).
    Physics is unchanged: the shape has density 0 (no mass update) and no shape or particle collision (SolverMuJoCo
    skips visual-only geoms), and no solver reads VISIBLE. It keeps the card's surface values for the readback."""
    vis = dataclasses.replace(cfg, density=0.0, has_shape_collision=False, has_particle_collision=False,
                              is_visible=True, is_hydroelastic=False)                  # hydroelastic needs mesh.sdf
    display = newton.Mesh(mesh.vertices.astype(np.float32), mesh.indices.reshape(-1).astype(np.int32),
                          compute_inertia=False)
    shape = b.add_shape_mesh(body, mesh=display, cfg=vis, label=f"{name}/visual")
    visible = int(newton.ShapeFlags.VISIBLE)
    for s in hulls:
        b.shape_flags[s] &= ~visible
    return shape


def _emit_imported(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    """An object described by an MJCF or USD file: imported as is; the card's material is applied by the
    assembler to every imported shape."""
    shape = obj.card.shape
    floating = obj.card.motion == "dynamic"
    first = b.shape_count
    if isinstance(shape, MJCFSource):
        b.add_mjcf(shape.path, floating=floating)
    else:
        b.add_usd(shape.path, floating=floating)
    cfg = shape_config(obj.material, torsional_lever=d.numbers.get("torsional_lever"))
    _restyle(b, range(first, b.shape_count), cfg)


def _restyle(b: newton.ModelBuilder, shapes: range, cfg: newton.ModelBuilder.ShapeConfig) -> None:
    """The card's surface values on imported shapes (the file's own friction, e.g. MJCF's default μ = 1, would
    otherwise reach the Model: CARD-1's readback caught hinged_box_plywood at μ 1.0 for a 0.3 card)."""
    for s in shapes:
        b.shape_material_mu[s] = cfg.mu
        b.shape_material_restitution[s] = cfg.restitution
        b.shape_material_mu_torsional[s] = cfg.mu_torsional
        b.shape_material_mu_rolling[s] = cfg.mu_rolling
        b.shape_material_ke[s], b.shape_material_kd[s] = cfg.ke, cfg.kd


def emit_grains(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    """GRAIN_RIGID: sphere bodies of the card's grain diameter on a cubic lattice over the region."""
    src = obj.card.shape
    if not isinstance(src, ParticleSource) or src.grain_diameter is None:
        raise AssemblyError(f"{obj.card.name}: GRAIN_RIGID needs a ParticleSource with grain_diameter")
    r = 0.5 * src.grain_diameter
    cfg = shape_config(obj.material)
    centers = grain_centers(src.region, max(src.spacing, src.grain_diameter))
    cap = int(d.numbers.get("n_grains", len(centers)))
    for i, c in enumerate(centers[:cap]):
        body = b.add_body(xform=wp.transform(wp.vec3(*c), wp.quat_identity()), label=f"{obj.card.name}/{i}")
        b.add_shape_sphere(body, radius=r, cfg=cfg)


def grain_centers(region: tuple, spacing: float) -> np.ndarray:
    """Lattice points with `spacing` inside the axis-aligned region, centred, deterministic order (x, y, z)."""
    lo, hi = np.asarray(region[0], np.float64), np.asarray(region[1], np.float64)
    n = np.maximum(1, np.floor((hi - lo) / spacing + 1e-9).astype(int))
    start = lo + 0.5 * ((hi - lo) - (n - 1) * spacing)
    axes = [start[k] + spacing * np.arange(n[k]) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def grain_mass(d_grain: float, rho: float) -> float:
    return rho * math.pi * d_grain**3 / 6.0


def emit_part(b: newton.ModelBuilder, t: Any, cap: Any, make_sub: Any, roots: dict[float, tuple[int, Any]]) -> Any:
    """``template.emit_object(b, t, cap, make_sub)``, except a kinematic fixture's part other than its root: its shape
    on the root's body at its pose relative to the root (module docstring). ``roots`` maps each fixture group to its
    root's (body, placement pose), filled as the roots are emitted. Returns the object's index ranges."""
    from alloy.scene import template as tp

    numbers = t.decision.numbers if t.obj is not None else {}
    group = numbers.get(FIXTURE_GROUP)
    if group is None or numbers.get(FIXTURE_ROOT):
        r = tp.emit_object(b, t, cap, make_sub)
        if group is not None:
            roots[group] = (r.body.start, t.placement.pose)
        return r
    if group not in roots:
        raise AssemblyError(f"{t.name}: a kinematic fixture's part was emitted before its root part; keep the "
                            "fixture's parts in the order alloy.Scene added them (root first)")
    if not isinstance(t.obj.card.shape, Primitive):
        raise AssemblyError(f"{t.name}: a kinematic fixture's part must be a primitive to ride on its root's body, "
                            f"not {type(t.obj.card.shape).__name__}; build the fixture from boxes (add_open_box ...)")
    body, pose = roots[group]
    rel = wp.transform_multiply(wp.transform_inverse(tp.pose_transform(pose)), tp.pose_transform(t.placement.pose))
    cfg = shape_config(t.obj.material, torsional_lever=numbers.get("torsional_lever"))      # the part's own mass
    before = tp.counts(b)
    add_primitive(b, body, t.obj.card.shape, cfg, xform=rel, label=t.obj.card.name)
    return tp.ranges_between(before, tp.counts(b))
