# SPDX-License-Identifier: Apache-2.0
"""Film emitter: a triangle mesh + FilmSpec -> cloth particles, triangles and hinges with the film attributes.

- membrane: VBD tri_ke = membrane_scale·E·h/(1−ν²), tri_ka = 0 (as the stock emitter, derive.cloth_stiffness),
  tri_kd = β·tri_ke; the strain cap carries the in-plane stiffness (design note §3);
- bending: per-hinge ke from D (topology.bending_ke), kd = β·ke; fold lines (creases: a card mesh's OBJ ``l``
  lines, or ``creases=`` pairs) take D/(crease_ratio·h) and crease_yield where those are set; a J2 card's interior
  hinges carry its plane-strain section, their length l_h = (A1 + A2)/|e| and ke = D/l_h (``film:j2_*``,
  ``film:hinge_length``, sheet_j2.py);
- ``film:strain_cap`` = yield_stress/E, ``film:compress_cap``, ``film:air`` per triangle; ``film:yield_angle`` per
  hinge; particle radius = the film contact radius;
- mass: ρ·h per area (Newton lumps a third of each triangle's mass to its vertices).

``emit_film_object`` is the plug-in's ``Emitter`` for route ``cloth_film`` (a card's mesh via the stock
``cloth_mesh``); ``add_film_mesh`` is the same for code-built meshes (laws, clips).
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import warp as wp

from alloy._vendor import newton

from .config import (AIR, COMPRESS_CAP, HINGE_LENGTH, J2_HARDENING, J2_MODULUS, J2_THICKNESS, J2_YIELD, LIMIT_KE,
                     SEAM_OVERLOAD_STRAIN, STRAIN_CAP, TEAR_FORCE, YIELD_ANGLE, register_custom_attributes)
from .materials import FILM_DEFAULTS, FilmExtras, FilmSpec
from .topology import bending_ke

log = logging.getLogger("alloy.film")


def crease_mask(edges: np.ndarray, creases: np.ndarray | None) -> np.ndarray:
    """[E] bool: hinge rows (o0, o1, v0, v1, local vertex ids) whose edge v0-v1 is one of ``creases`` [K, 2]."""
    mask = np.zeros(len(edges), dtype=bool)
    if creases is None or len(creases) == 0:
        return mask
    pairs = {(int(min(a, c)), int(max(a, c))) for a, c in np.asarray(creases, np.int64).reshape(-1, 2)}
    for n, (_, _, a, c) in enumerate(edges):
        mask[n] = (int(min(a, c)), int(max(a, c))) in pairs
    return mask


def _set_edge_values(b: newton.ModelBuilder, name: str, ids: np.ndarray, value: float) -> None:
    """Per-edge custom attribute values (Newton keeps them as {edge id: value})."""
    attr = b.custom_attributes[name]
    attr.values = attr.values or {}
    for e in ids:
        attr.values[int(e)] = float(value)


def _cloth_kwargs(spec: FilmSpec, *, pos, rot, vel, label, tear_force: float = math.inf) -> dict:
    """add_cloth_mesh keywords of a film: scaled VBD membrane, no stock bending (per hinge below), film attributes."""
    ex = spec.extras
    nu, h, E = spec.poisson, spec.thickness, spec.youngs
    tri_ke = ex.membrane_scale * E * h / (1.0 - nu * nu)
    return dict(pos=wp.vec3(*pos), rot=wp.quat_identity() if rot is None else rot, scale=1.0, vel=wp.vec3(*vel),
                density=spec.areal_density, tri_ke=tri_ke, tri_ka=0.0, tri_kd=spec.rayleigh_beta * tri_ke,
                edge_ke=0.0, edge_kd=0.0, particle_radius=ex.contact_radius, label=label,
                custom_attributes_triangles={STRAIN_CAP: spec.strain_cap, COMPRESS_CAP: ex.compress_cap, AIR: ex.air,
                                             LIMIT_KE: E * h / (1.0 - nu * nu)},
                custom_attributes_edges={YIELD_ANGLE: ex.yield_angle},
                custom_attributes_particles={TEAR_FORCE: float(tear_force)})


def _film_hinges(b: newton.ModelBuilder, p0: int, e0: int, rest: np.ndarray, spec: FilmSpec,
                 creases: np.ndarray | None) -> int:
    """Per-hinge bending of the edges from ``e0`` (vertices from ``p0``, ``rest`` their rest positions): ke from D,
    crease hinges D/(crease_ratio·h) and crease_yield where set. Returns the crease hinge count."""
    ex = spec.extras
    local = np.asarray(b.edge_indices[e0:], dtype=np.int64)
    local[local >= 0] -= p0
    ke = bending_ke(np.asarray(rest, np.float64), local, spec.plate_rigidity)
    crease = crease_mask(local, creases)
    if ex.crease_ratio > 0.0:
        ke[crease] = spec.plate_rigidity / (ex.crease_ratio * spec.thickness)    # κ = D/(L*/h · h): Lechenault 2014
    for n, k in enumerate(ke):
        b.edge_bending_properties[e0 + n] = (float(k), float((spec.rayleigh_beta + ex.bend_damping) * k))
    if crease.any() and ex.crease_yield > 0.0:
        _set_edge_values(b, YIELD_ANGLE, e0 + np.flatnonzero(crease), ex.crease_yield)
    if spec.j2_yield > 0.0:
        _j2_hinges(b, e0, local, np.asarray(rest, np.float64), spec, crease)
    return int(crease.sum())


def hinge_length(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """[E] l_h = (A1 + A2)/|e|, the mean of the two triangles' heights over the edge (0 for a boundary edge): a bend of
    curvature κ normal to the edge turns the hinge by θ = κ·l_h on any mesh. (topology.bending_ke's centroid distance
    is 2/3 of it, which on a grid mesh reads 1.5 κ.)"""
    out = np.zeros(len(edges))
    for n, (o0, o1, a, c) in enumerate(edges):
        if o0 < 0 or o1 < 0:
            continue
        e = vertices[c] - vertices[a]
        le = float(np.linalg.norm(e))
        if le > 0.0:
            out[n] = sum(0.5 * float(np.linalg.norm(np.cross(e, vertices[o] - vertices[a]))) for o in (o0, o1)) / le
    return out


def _j2_hinges(b: newton.ModelBuilder, e0: int, local: np.ndarray, rest: np.ndarray, spec: FilmSpec,
               crease: np.ndarray) -> None:
    """A J2 card's section on every interior hinge from ``e0``: E', σ_y', H' (j2.plane_strain), t, l_h, and the
    hinge's ke = D/l_h with D = E' t³/12, so a bend reads its true curvature and the section's elastic moment is the
    hinge's own (sheet_j2.py)."""
    from alloy.solvers.plastic.j2 import plane_strain

    if spec.extras.yield_angle > 0.0:      # plastic_hinge's dead fold and the J2 rest write would both own a hinge
        raise ValueError("film: a J2 sheet's hinges are its own section; a dead-fold yield_angle is not supported on a "
                         "plastic.model j2 sheet")
    if crease.any():                        # a crease hinge's yield_angle and the J2 rest write would both own it
        raise ValueError("film: a J2 sheet's hinges are its own section; fold lines (creases) are not supported on a "
                         "plastic.model j2 sheet (route it PANELS with its creases, or drop them)")
    Ep, syp, hp = plane_strain(spec.youngs, spec.poisson, spec.j2_yield, spec.j2_hardening)
    D, beta = spec.plate_rigidity, spec.rayleigh_beta + spec.extras.bend_damping
    lh = hinge_length(rest, local)
    ids = np.flatnonzero(lh > 0.0)
    for name, value in ((J2_YIELD, syp), (J2_HARDENING, hp), (J2_MODULUS, Ep), (J2_THICKNESS, spec.thickness)):
        _set_edge_values(b, name, e0 + ids, float(value))
    attr = b.custom_attributes[HINGE_LENGTH]
    attr.values = attr.values or {}
    for n in ids:
        k = D / lh[n]
        attr.values[int(e0 + n)] = float(lh[n])
        b.edge_bending_properties[e0 + n] = (float(k), float(beta * k))


def add_film_mesh(b: newton.ModelBuilder, vertices: np.ndarray, tris: np.ndarray, spec: FilmSpec, *,
                  pos=(0.0, 0.0, 0.0), rot=None, vel=(0.0, 0.0, 0.0), label: str | None = None,
                  creases: np.ndarray | None = None) -> tuple[int, int]:
    """Adds one film mesh; returns (first particle id, first edge id). ``vertices`` [N, 3] rest positions (m).
    ``creases`` [K, 2] vertex pairs (fold lines): where the spec sets them (> 0), those hinges take the crease
    stiffness D/(crease_ratio·h) and ``crease_yield`` instead of the sheet's bending ke and yield angle."""
    register_custom_attributes(b)
    p0, e0 = b.particle_count, len(b.edge_indices)
    b.add_cloth_mesh(vertices=np.asarray(vertices, np.float32), indices=np.asarray(tris).reshape(-1).tolist(),
                     **_cloth_kwargs(spec, pos=pos, rot=rot, vel=vel, label=label))
    n_crease = _film_hinges(b, p0, e0, vertices, spec, creases)
    log.debug("%s: film %d particles, %d triangles, %d hinges (%d crease), cap %.3g, θ_y %.3g rad", label,
              b.particle_count - p0, len(tris), len(b.edge_indices) - e0, n_crease, spec.strain_cap,
              spec.extras.yield_angle)
    return p0, e0


def tear_force(obj, spec: FilmSpec) -> float:
    """A seam's break force at a crack tip or free edge, N: Γ·h/2 (a trouser tear: pulling the legs apart by δ tears
    δ/2, so F = Γ_length / 2 with Γ_length = tear_energy·h); inf without ``failure.tear_energy``."""
    f = getattr(obj.material, "failure", None)
    g = getattr(f, "tear_energy", None) if f is not None else None
    return 0.5 * float(g.value) * spec.thickness if g is not None else math.inf


def _seam_failure(failure):
    """``failure`` with the film's tear strain (config.SEAM_OVERLOAD_STRAIN) in place of the card's elongation."""
    if failure is None:
        return None
    keys = ("fracture_strain", "fracture_stress", "tear_energy", "cut_min_strain", "cut_min_force")
    out = {k: getattr(failure, k, None) for k in keys}
    out["fracture_strain"] = SEAM_OVERLOAD_STRAIN
    return SimpleNamespace(**out)


def add_film_presplit(b: newton.ModelBuilder, vertices: np.ndarray, tris: np.ndarray, spec: FilmSpec, plan, *,
                      failure, elastic, budget=None, tear: float = math.inf, pos=(0.0, 0.0, 0.0), rot=None,
                      label: str | None = None, creases: np.ndarray | None = None) -> tuple[int, int]:
    """A film that can tear: the topology pre-split along ``plan`` (``alloy.topology.presplit.SplitPlan``; twin
    CLOTH_EDGE bonds with ``failure``'s strength over each seam's length as break force, and the film's tear strain)
    with the film's attributes and hinges; seams.py holds the seams closed and reports their force. ``tear`` = a
    seam's break force at a crack tip or free edge, N."""
    from alloy.topology.emit import add_presplit_cloth

    register_custom_attributes(b)
    verts = np.asarray(vertices, np.float64).reshape(-1, 3)
    tris = np.asarray(tris).reshape(-1, 3)
    kw = _cloth_kwargs(spec, pos=pos, rot=rot, vel=(0.0, 0.0, 0.0), label=label, tear_force=tear)
    kw["edge_ke"] = 1.0                    # > 0 so the pre-split adds cross-seam hinges; _film_hinges sets every hinge
    p0, e0 = b.particle_count, len(b.edge_indices)
    rec = add_presplit_cloth(b, verts, tris, plan, kw, budget=budget, failure=_seam_failure(failure), elastic=elastic,
                             thickness=spec.thickness)
    rest = np.asarray(b.particle_q[p0:], np.float64)
    n_crease = _film_hinges(b, p0, e0, rest, spec, creases)
    log.info("%s: film pre-split, +%d vertices, %d seams, %d hinges (%d crease)", label, rec.vertices_added,
             len(rec.bonds), len(b.edge_indices) - e0, n_crease)
    return p0, e0


# MaterialCard.film field -> FilmExtras field (card values are SI: rad, m, -)
_CARD_FILM_FIELDS = {"dead_fold_yield": "yield_angle", "contact_radius": "contact_radius", "air_scale": "air",
                     "crease_ratio": "crease_ratio", "crease_yield": "crease_yield"}


def film_extras(mat) -> FilmExtras:
    """Film-only values of a material card: the fields its ``film`` block sets, over ``FILM_DEFAULTS[name]``
    (``FilmExtras()`` for an unlisted name). A card without the block gets exactly the table's values."""
    base = FILM_DEFAULTS.get(mat.name, FilmExtras())
    block = getattr(mat, "film", None)
    if block is None:
        return base
    over = {dst: float(getattr(block, src).value) for src, dst in _CARD_FILM_FIELDS.items()
            if getattr(block, src) is not None}
    return replace(base, **over) if over else base


def spec_from_card(obj) -> FilmSpec:
    """FilmSpec of a resolved card: E, ν, ρ, β, yield stress from its material, h from features.thickness, film-only
    values from the material's optional ``film`` block over ``FILM_DEFAULTS[material name]`` (``film_extras``)."""
    from alloy.core.errors import AssemblyError
    from alloy.scene.emitters.deformable import sheet_thickness

    mat = obj.material
    el = mat.elastic
    if el is None:
        raise AssemblyError(f"{obj.card.name}: route cloth_film needs material.elastic")
    ys = mat.plastic.yield_stress if mat.plastic is not None else None
    if ys is None:
        raise AssemblyError(f"{obj.card.name}: route cloth_film needs material.plastic.yield_stress (strain cap)")
    beta = float(el.rayleigh_beta.value) if el.rayleigh_beta is not None else 0.0
    j2y, j2h = 0.0, 0.0
    if mat.plastic.model == "j2":                       # a metal sheet: its hinges get the fibre section
        j2y = float(ys.value)
        j2h = float(mat.plastic.hardening_modulus.value) if mat.plastic.hardening_modulus is not None else 0.0
    return FilmSpec(youngs=float(el.youngs.value), poisson=float(el.poisson.value), density=float(mat.density.value),
                    thickness=sheet_thickness(obj), yield_stress=float(ys.value), rayleigh_beta=beta,
                    extras=film_extras(mat), j2_yield=j2y, j2_hardening=j2h)


def obj_creases(obj) -> np.ndarray | None:
    """[K, 2] vertex pairs of a card mesh's fold lines: its OBJ ``l`` (polyline) elements, 1-based positive indices
    into the ``v`` list, which the stock loader keeps in order. None for primitives and meshes without lines."""
    from alloy.cards.types import MeshSource
    from alloy.scene.emitters.common import _mesh_file      # the stock loader's path resolution

    shape = obj.card.shape
    if not isinstance(shape, MeshSource) or not str(shape.path).endswith(".obj"):
        return None
    pairs = []
    for line in _mesh_file(shape, None).read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "l":
            idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            pairs.extend(zip(idx[:-1], idx[1:]))
    return np.asarray(pairs, np.int64).reshape(-1, 2) if pairs else None


def emit_film_object(b: newton.ModelBuilder, obj, d) -> None:
    """Emitter for route ``cloth_film`` (card mesh as the stock cloth emitter builds it; OBJ lines are creases, OBJ
    ``vt`` lines the flat layout that places tear lines on a folded sheet, layout.py)."""
    from alloy.scene.emitters.deformable import cloth_mesh

    from alloy.scene.emitters.topology import wants_presplit

    mesh, rot = cloth_mesh(obj)
    tris, spec = mesh.indices.reshape(-1, 3), spec_from_card(obj)
    if wants_presplit(obj):
        from alloy.scene.emitters.topology import _plan, sheet_plane_axes

        from .layout import layout_tear_plan, obj_layout

        v = np.asarray(mesh.vertices, np.float64).reshape(-1, 3)
        layout, spacing = obj_layout(obj), obj.card.topology.split_spacing
        if layout is not None and spacing is not None:      # a folded sheet: tear lines on the paper as cut
            plan = layout_tear_plan(v, tris, layout, spacing)
        else:
            plan = _plan(v, tris, obj, sheet_plane_axes(v))
        add_film_presplit(b, v, tris, spec, plan, failure=obj.material.failure,
                          elastic=obj.material.elastic, budget=obj.card.topology, tear=tear_force(obj, spec), rot=rot,
                          label=obj.card.name, creases=obj_creases(obj))
    else:
        add_film_mesh(b, mesh.vertices, tris, spec, rot=rot, label=obj.card.name, creases=obj_creases(obj))


class FilmEmitter:
    """``Emitter`` protocol adapter (SolverCapability.emitter)."""

    def emit(self, b: newton.ModelBuilder, obj, d) -> None:
        emit_film_object(b, obj, d)
