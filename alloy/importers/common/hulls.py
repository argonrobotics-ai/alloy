# SPDX-License-Identifier: Apache-2.0
"""Convex decomposition of a physics mesh with CoACD (Wei et al., SIGGRAPH 2022; MIT licence).

Budget: at most ``max_parts`` hulls (the card's ``shape.max_parts``, <= ``budget.collision_parts``) and at most
``MAX_HULL_VERTS`` vertices per hull. CoACD only applies its per-hull vertex cap when ``decimate=True``; hulls that
still exceed it are re-hulled from a farthest-point subset of their vertices (shrinks the hull, never grows it).
``HullStats`` records the fit: total hull volume over mesh volume (1.0 = exact, overlap and filled concavity push it
up) so a reader can judge collision fidelity. CoACD can also emit a flat (zero-thickness) hull, which MuJoCo contact
generation rejects ("planar mesh collider"); hulls thinner than ``THIN_REL`` of their own diagonal cover no volume and
are dropped (``HullStats.dropped``). ``hullfit`` then makes the result a good collider: one hull for a near-convex
mesh (CoACD does not run), and hulls thinner than ``THIN_FLOOR`` merged into a neighbour or thickened into the object
(``HullStats.merged``, ``thickened``); a thin wall that allows neither stays (``HullStats.thin``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from alloy.importers.common.errors import ImporterError, need

log = logging.getLogger("alloy.importers")

MAX_HULL_VERTS = 64
MAX_PARTS = 32
THRESHOLD = 0.04             # CoACD concavity threshold (fraction of the bounding-box diagonal)
SEED = 0
THIN_REL = 1e-4              # thinner than this fraction of its own diagonal = flat; 100x Newton's planar tolerance
# hullfit thresholds (2026-11-08; why these values: hullfit's docstring):
THIN_FLOOR = 5e-3            # m: thinner hulls are merged or thickened (2 x GAP_MAX: a contact GAP_MAX deep stays
#                              on the near side of the hull's middle)
GAP_MAX = 2.5e-3             # m: a merged or single hull may reach this far outside the real surface (99th pct)
SOLID_MIN = 0.8              # single hull only when the mesh fills this much of its convex hull


@dataclass(frozen=True)
class HullParams:
    """CoACD settings recorded with every decomposition."""

    threshold: float = THRESHOLD
    max_parts: int = MAX_PARTS
    max_hull_verts: int = MAX_HULL_VERTS
    seed: int = SEED
    preprocess: str = "off"      # the input is already watertight (meshops.physics_mesh)
    solid_min: float = SOLID_MIN
    thin_floor: float = THIN_FLOOR
    gap_max: float = GAP_MAX


@dataclass(frozen=True)
class HullStats:
    """Result summary (JSON-ready via ``asdict``)."""

    n_hulls: int
    max_verts: int
    volume_ratio: float          # sum(hull volume) / mesh volume
    seconds: float
    params: dict
    dropped: int = 0             # flat CoACD hulls removed (``is_thin``) and thin ones inside others (hullfit)
    single: bool = False         # near-convex mesh: its convex hull (cut along a long one), no CoACD (hullfit)
    merged: int = 0              # thin hulls merged into a neighbour (hullfit.thicken)
    thickened: int = 0           # thin skins thickened into the object (hullfit.thicken)
    thin: int = 0                # hulls still thinner than ``params.thin_floor`` (the object is that thin there)
    min_thickness: float = 0.0   # m, the thinnest hull

    def to_json(self) -> dict:
        """Plain dict for ``json.dumps``.

        Example::

            json.dumps(result.to_json())
        """
        return asdict(self)


def _farthest_points(pts: np.ndarray, k: int) -> np.ndarray:
    """``k`` points spread over ``pts`` (greedy farthest-point order from the point farthest from the centroid)."""
    first = int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    chosen = [first]
    d = np.linalg.norm(pts - pts[first], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return pts[chosen]


def convex_hull(points: np.ndarray, max_verts: int = MAX_HULL_VERTS) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` of the convex hull of ``points`` with at most ``max_verts`` vertices, outward wound.

    Example::

        v, f = convex_hull(np.random.rand(500, 3), max_verts=64)
    """
    trimesh = need("trimesh")
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    hull = trimesh.convex.convex_hull(pts)
    if len(hull.vertices) > max_verts:
        hull = trimesh.convex.convex_hull(_farthest_points(np.asarray(hull.vertices), max_verts))
    trimesh.repair.fix_normals(hull)
    return np.asarray(hull.vertices, np.float64), np.asarray(hull.faces, np.int64)


def min_width(v: np.ndarray, f: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """``(width [m], unit direction)``: the smallest width of a point set along its principal axes and, given faces,
    along each face normal.

    Never below the true minimum width, and equal to it for a slab, so a small value means the set really is flat.

    Example::

        w, axis = min_width(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1e-9]]))   # ~1e-9, +-z
    """
    pts = np.asarray(v, np.float64)
    c = pts - pts.mean(axis=0)
    dirs = np.linalg.svd(c, full_matrices=False)[2]
    if f is not None and len(f):
        tri = pts[np.asarray(f)]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(n, axis=1)
        dirs = np.vstack([dirs, n[norm > 0] / norm[norm > 0, None]])
    proj = c @ dirs.T
    widths = proj.max(axis=0) - proj.min(axis=0)
    k = int(np.argmin(widths))
    return float(widths[k]), dirs[k]


def hull_thickness(v: np.ndarray, f: np.ndarray | None = None) -> float:
    """Smallest width [m] of a point set (``min_width``).

    Example::

        hull_thickness(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1e-9]]))   # ~1e-9
    """
    return min_width(v, f)[0]


def is_thin(v: np.ndarray, f: np.ndarray | None = None, rel: float = THIN_REL) -> bool:
    """True when a hull (or point set) is flat: thinner than ``rel`` of its bounding-box diagonal.

    Example::

        is_thin(np.random.rand(20, 3) * [1, 1, 0])   # True
    """
    pts = np.asarray(v, np.float64)
    if len(pts) < 4:
        return True
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return hull_thickness(pts, f) <= rel * diag


def _run_coacd(vertices: np.ndarray, faces: np.ndarray, p: HullParams) -> list:
    coacd = need("coacd")
    coacd.set_log_level("error")
    mesh = coacd.Mesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int64))
    return coacd.run_coacd(mesh, threshold=p.threshold, max_convex_hull=p.max_parts, preprocess_mode=p.preprocess,
                           decimate=True, max_ch_vertex=p.max_hull_verts, seed=p.seed)


def hull_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Volume [m^3] of one closed, outward-wound hull.

    Example::

        hull_volume(*convex_hull(np.random.rand(50, 3)))
    """
    tri = np.asarray(v, np.float64)[np.asarray(f)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def _coacd_hulls(vertices: np.ndarray, faces: np.ndarray, params: HullParams) -> tuple[list, int]:
    """CoACD's hulls, re-hulled and capped, flat ones (``is_thin``) dropped; also the number CoACD returned."""
    parts = _run_coacd(vertices, faces, params)
    if not parts:
        raise ImporterError("<mesh>", "hulls", "CoACD returned no hulls; check that the mesh is watertight")
    hulls = [convex_hull(np.asarray(v), params.max_hull_verts) for v, _ in parts if not is_thin(v)]
    hulls = [h for h in hulls if hull_volume(*h) > 0 and not is_thin(*h)]
    if not hulls:
        raise ImporterError("<mesh>", "hulls", f"all {len(parts)} CoACD hulls are flat; check the mesh scale")
    return hulls, len(parts)


def refit(vertices: np.ndarray, faces: np.ndarray, mesh_volume: float, hulls: list[tuple[np.ndarray, np.ndarray]],
          params: HullParams = HullParams(), n_parts: int | None = None, t0: float | None = None,
          surf=None) -> tuple[list[tuple[np.ndarray, np.ndarray]], HullStats]:
    """``hullfit``'s rules applied to hulls already made (CoACD's, or a shipped sidecar's): one hull when the mesh is
    near-convex, else thin hulls dropped, merged or thickened. ``decompose`` = CoACD + this, so re-fitting a sidecar
    written by an earlier importer gives what a fresh import gives.

    Example::

        hulls, stats = refit(phys.vertices, phys.faces, phys.volume, read_hulls_obj(path)[1])
    """
    from alloy.importers.common import hullfit  # noqa: PLC0415 - hullfit imports this module

    t0 = time.monotonic() if t0 is None else t0
    n_parts = len(hulls) if n_parts is None else n_parts
    surf = hullfit.Surface(vertices, faces) if surf is None else surf
    one = hullfit.single_hull(vertices, mesh_volume, surf, params)
    if one is not None:
        hulls, fit, single = one, hullfit.Fit(), True
    else:
        hulls, fit = hullfit.thicken(hulls, surf, params)
        single = False
    thick = [hull_thickness(*h) for h in hulls]
    stats = HullStats(n_hulls=len(hulls), max_verts=max(len(v) for v, _ in hulls),
                      volume_ratio=float(sum(hull_volume(*h) for h in hulls) / mesh_volume),
                      seconds=round(time.monotonic() - t0, 2), params=asdict(params),
                      dropped=0 if single else n_parts - len(hulls) - fit.merged, single=single, merged=fit.merged,
                      thickened=fit.thickened,
                      thin=sum(x < params.thin_floor for x in thick), min_thickness=float(min(thick)))
    log.debug("decompose: %d hulls (single %s, %d merged, %d thickened, %d dropped, %d thin), volume ratio %.3f, "
              "%.1f s", stats.n_hulls, single, stats.merged, stats.thickened, stats.dropped, stats.thin,
              stats.volume_ratio, stats.seconds)
    return hulls, stats


def decompose(vertices: np.ndarray, faces: np.ndarray, mesh_volume: float,
              params: HullParams = HullParams()) -> tuple[list[tuple[np.ndarray, np.ndarray]], HullStats]:
    """Convex hulls of a watertight mesh: its own convex hull when it is near-convex (``hullfit.single_hull``), else
    CoACD's hulls, each re-hulled and capped at ``params.max_hull_verts`` vertices, flat ones (``is_thin``) dropped and
    thin ones merged into a neighbour or thickened into the object (``hullfit.thicken``).

    Example::

        hulls, stats = decompose(phys.vertices, phys.faces, phys.volume)
        stats.n_hulls, stats.volume_ratio
    """
    from alloy.importers.common import hullfit  # noqa: PLC0415 - hullfit imports this module

    if mesh_volume <= 0:
        raise ImporterError("<mesh>", "hulls", f"mesh volume {mesh_volume:.3e} m^3 <= 0; repair the mesh first")
    t0 = time.monotonic()
    surf = hullfit.Surface(vertices, faces)
    one = hullfit.single_hull(vertices, mesh_volume, surf, params)
    hulls, n_parts = (one, len(one)) if one is not None else _coacd_hulls(vertices, faces, params)
    hulls, stats = refit(vertices, faces, mesh_volume, hulls, params, n_parts, t0, surf)
    if stats.dropped:
        log.info("decompose: dropped %d flat, empty or covered hull(s) of %d", stats.dropped, n_parts)
    return hulls, stats


def hull_volumes(hulls: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
    """Volume [m^3] of each hull (outward-wound, closed).

    Example::

        sum(hull_volumes(hulls))
    """
    return [hull_volume(v, f) for v, f in hulls]
