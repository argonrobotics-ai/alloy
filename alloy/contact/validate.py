# SPDX-License-Identifier: Apache-2.0
"""Precision-pair validity rules (SPEC §3.5 "Rules", contact §4.4) and the mesh measurements they need.

``validate_pair`` raises ``PrecisionSpecError`` naming the rule and the numbers. Mesh measurements are numpy and
work on CPU: edge adjacency, watertightness, smooth-edge chord error and the smallest smooth feature radius. An edge
is *smooth* when its dihedral bend is below ``SMOOTH_BEND_RAD`` (a faceted curved surface); sharper edges are real
corners whose error the voxel budget (≤ h/2) covers, so they do not bound the voxel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.contact.types import PrecisionPairSpec
from alloy.core.defaults import RIGID_GAP_M
from alloy.core.errors import PrecisionSpecError
from alloy.core.log import get_logger

log = get_logger("contact.validate")

SMOOTH_BEND_RAD = math.radians(30.0)
MAX_VOXEL_M = 1.0e-4                # §3.5: 5e-5 default, 1e-4 only for features longer than 40 mm
COARSE_VOXEL_EXTENT_M = 0.04
MAX_EXTENT_VOXELS = 2048            # crop mandatory beyond this
SDF_PART_BUDGET_B = 64 * 2**20      # CT-10
CROSS_ENTRY_TOL_RATIO = 0.25        # interface_pos_tol <= 0.25 * max_penetration across entries


@dataclass(frozen=True)
class MeshInfo:
    """Measurements of one triangle mesh [m]."""

    watertight: bool
    extent: tuple[float, float, float]
    area: float
    max_chord_error: float           # over smooth edges; 0 if none
    min_feature_radius: float        # over smooth edges; inf if none


def _edges(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges (a->b) of every triangle and the owning triangle index."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    owner = np.concatenate([np.arange(len(tris))] * 3)
    return e, owner


def is_watertight(tris: np.ndarray) -> bool:
    """Every undirected edge has exactly two triangles, traversed in opposite directions (consistent winding)."""
    e, _ = _edges(tris)
    directed = {tuple(x) for x in e.tolist()}
    if len(directed) != len(e):
        return False
    return all((b, a) in directed for a, b in directed)


def _smooth_edges(verts: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(facet width across the edge, bend angle) of interior edges whose bend is in (0, SMOOTH_BEND_RAD).

    The width is the smaller height of the two triangles over the shared edge: on a faceted curved surface it is the
    chord ``w = 2R·sin(θ/2)`` perpendicular to the edge.
    """
    v = verts.astype(np.float64)
    n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    area2 = np.linalg.norm(n, axis=1)
    n = n / np.where(area2 > 0.0, area2, 1.0)[:, None]
    e, owner = _edges(tris)
    first: dict[tuple[int, int], int] = {}
    widths, bends = [], []
    for (a, b), t in zip(e.tolist(), owner.tolist()):
        key = (min(a, b), max(a, b))
        if key not in first:
            first[key] = t
            continue
        bend = math.acos(max(-1.0, min(1.0, float(np.dot(n[first[key]], n[t])))))
        if 1e-9 < bend < SMOOTH_BEND_RAD:
            length = float(np.linalg.norm(v[a] - v[b]))
            widths.append(min(area2[first[key]], area2[t]) / max(length, 1e-300))
            bends.append(bend)
    return np.asarray(widths), np.asarray(bends)


def mesh_info(verts: np.ndarray, tris: np.ndarray) -> MeshInfo:
    """Watertightness, extent, area, chord error ``R·(1-cos(θ/2))`` and feature radius ``R = w/(2·sin(θ/2))``."""
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    v = np.asarray(verts, dtype=np.float64)
    ext = tuple(float(x) for x in (v.max(axis=0) - v.min(axis=0)))
    area = 0.5 * float(np.linalg.norm(np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]]),
                                      axis=1).sum())
    widths, bends = _smooth_edges(v, tris)
    radii = widths / (2.0 * np.sin(bends / 2.0)) if len(bends) else np.zeros(0)
    chord = float(np.max(radii * (1.0 - np.cos(bends / 2.0)))) if len(bends) else 0.0
    radius = float(np.min(radii)) if len(bends) else math.inf
    return MeshInfo(is_watertight(tris), ext, area, chord, radius)


def sdf_bytes_estimate(area: float, band: float, voxel: float) -> float:
    """Sparse texture SDF size ``2·(9/8)³·A·(b + 8h)/h³`` bytes (uint16 subgrids, contact §4.7)."""
    return 2.0 * (9.0 / 8.0) ** 3 * area * (band + 8.0 * voxel) / voxel**3


def _mesh_arrays(mesh) -> tuple[np.ndarray, np.ndarray] | None:
    if mesh is None:
        return None
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)


def _fail(rule: str, msg: str) -> None:
    raise PrecisionSpecError(f"precision pair rule '{rule}': {msg}")


def _check_spec_numbers(spec: PrecisionPairSpec) -> None:
    h = spec.voxel_size
    if h <= 0.0 or h > MAX_VOXEL_M:
        _fail("voxel_range", f"voxel_size {h} m outside (0, {MAX_VOXEL_M}]")
    if spec.narrow_band[1] < spec.gap + spec.margin:
        _fail("band_covers_gap", f"narrow_band[1]={spec.narrow_band[1]} < gap + margin = {spec.gap + spec.margin}")
    if spec.gap <= 0.0 or spec.gap > RIGID_GAP_M:
        _fail("gap", f"gap {spec.gap} m must be in (0, {RIGID_GAP_M}] (never the builder's 0.1 m)")
    if spec.radial_clearance is not None and h > spec.radial_clearance:
        _fail("voxel_vs_clearance", f"voxel {h} m > clearance/2 = {spec.radial_clearance} m")
    if spec.wall_thickness is not None and h > 0.25 * spec.wall_thickness:
        _fail("voxel_vs_wall", f"voxel {h} m > wall/4 = {0.25 * spec.wall_thickness} m")
    if not 0.0 < spec.mj_imp < 1.0:
        _fail("mj_imp", f"mj_imp {spec.mj_imp} outside (0, 1)")


def _check_mesh(spec: PrecisionPairSpec, side: str, arrays: tuple[np.ndarray, np.ndarray],
                crop: object | None) -> MeshInfo:
    info = mesh_info(*arrays)
    h = spec.voxel_size
    if h > info.min_feature_radius:
        _fail("voxel_vs_feature", f"{side}: voxel {h} m > smallest feature radius {info.min_feature_radius:.4g} m")
    if info.max_chord_error > spec.mesh_chord_tol:
        _fail("chord", f"{side}: tessellation chord error {info.max_chord_error:.3g} m > {spec.mesh_chord_tol} m")
    if spec.mode == "hydroelastic" and not info.watertight:
        _fail("watertight", f"{side}: hydroelastic mode needs a watertight mesh (use mode='sdf' for open shells)")
    if h > 0.5 * MAX_VOXEL_M and max(info.extent) <= COARSE_VOXEL_EXTENT_M:
        _fail("voxel_range", f"{side}: voxel {h} m > 5e-5 m needs a feature longer than {COARSE_VOXEL_EXTENT_M} m")
    if crop is None and spec.analytic_id is None and max(info.extent) / h > MAX_EXTENT_VOXELS:
        _fail("crop", f"{side}: extent {max(info.extent):.4g} m / voxel {h} m > {MAX_EXTENT_VOXELS}; set crop_{side}")
    if spec.analytic_id is None and crop is None:
        size = sdf_bytes_estimate(info.area, spec.narrow_band[1] - spec.narrow_band[0], h)
        if size > SDF_PART_BUDGET_B:
            _fail("sdf_budget", f"{side}: SDF estimate {size / 2**20:.1f} MB > 64 MB; crop the feature")
    return info


def validate_pair(spec: PrecisionPairSpec, mesh_a, mesh_b, *, entry_a: str, entry_b: str,
                  interface_pos_tol: float) -> None:
    """Check every §3.5 rule; ``mesh_a``/``mesh_b`` are ``newton.Mesh`` (or None for primitives)."""
    _check_spec_numbers(spec)
    max_pen = spec.resolved_max_penetration()
    if entry_a != entry_b and interface_pos_tol > CROSS_ENTRY_TOL_RATIO * max_pen:
        _fail("cross_entry", f"entries {entry_a!r}/{entry_b!r}: interface_pos_tol {interface_pos_tol} m > "
                             f"0.25 x max_penetration = {CROSS_ENTRY_TOL_RATIO * max_pen} m; keep the pair in one entry")
    for side, mesh, crop in (("a", mesh_a, spec.crop_a), ("b", mesh_b, spec.crop_b)):
        arrays = _mesh_arrays(mesh)
        if arrays is not None:
            _check_mesh(spec, side, arrays, crop)
    log.debug("pair (%d, %d) valid: mode=%s voxel=%g max_pen=%g", spec.shape_a, spec.shape_b, spec.mode,
              spec.voxel_size, max_pen)
