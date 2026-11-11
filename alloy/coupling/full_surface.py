# SPDX-License-Identifier: Apache-2.0
"""Full-surface proxy contacts for VBD cloth (SCR-SC4-4): which shapes, at what geometric detail.

Newton's per-vertex soft contact lets a rigid feature narrower than the cloth's vertex gap pass between vertices: C7's
towel (8.5 mm edges, r 1 mm) threads onto the YAM pad tips (6.4 mm thick, 2.1 mm ridges) and the open pads lift it.
Newton's full-surface contact adds edge and triangle records, but its mesh pass pairs every soft feature with every
rigid feature in the contact band: one YAM pad tip hull (1,423 vertices, 2,842 faces) against a 121-particle patch
gives 141 k records at Newton's 1 cm band, 7.9 k at 2 mm.

- ``moving_shapes``: shapes on bodies that can move get the edge/face records; static shapes and bodies welded to the
  world (SCR-SC8-2's kinematic proxies, a table or plywood sheet) keep per-vertex contact (a cloth lying on the
  plywood would put every one of its features in the band of one body);
- ``contact_view``: a destination entry's ``configure_view`` for those shapes, in that view only (the source,
  MuJoCo, keeps its geometry): a convex mesh (``is_convex``: a hull geom) with more than ``LOD_VERTICES`` vertices
  collides as a convex hull of ``LOD_VERTICES`` of them (farthest-point sampled), pushed out until it encloses the
  original (SCR-SC4-4b; the sampled hull alone sat up to 0.34 mm inside the YAM pad tip, where health measures);
  663 records instead of 7.9 k at a 2 mm band. A non-convex mesh (a beaker, a bag) keeps its
  triangles. Each record keeps VBD's stiffness, 0.5 (soft_contact_ke + shape ke): that already bounds the
  compression the ruling asks for (C7 pads rest at 2.6-2.7 mm on the double layer at 15 N, the CPU rig at 3.6 mm).
  A softer view value (the cloth's own 1e4, or 2e4-6e4) made the clips worse, not better: C7 latched PEN_FAIL with
  a finger kicked to 12-20 m/s in 1 of 4 worlds at 1e4 and at 3e4 (0 of 12 world-runs at the shape's own value);
- ``FULL_SURFACE_GAP``: the contact band of a full-surface pipeline (Newton's default 1 cm is per-vertex slack).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.coupling.kinematic import world_welded_bodies

log = logging.getLogger("alloy.coupling")

LOD_VERTICES = 64          # hull vertices a full-surface mesh keeps in a VBD view
CONVEX_TOL = 0.02          # a mesh within 2 % of its convex hull's volume counts as convex (reducible)
FULL_SURFACE_GAP = 0.003   # [m] detection band of a full-surface proxy pipeline (C7 moves < 0.5 mm per exchange)
ENCLOSE_PAD = 1.0e-6       # [m] a reduced hull clears every original vertex by this much (float32 rounding)
_MESH_TYPES = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))


def moving_shapes(model: Any) -> list[int]:
    """Shapes on bodies that can move: not static (body -1) and not welded to the world."""
    if not int(model.shape_count or 0):
        return []
    body = model.shape_body.numpy()
    welded = world_welded_bodies(model)
    moving = body >= 0
    moving[moving] &= ~welded[body[moving]]
    return np.flatnonzero(moving).tolist()


def reduced_hull(points: np.ndarray, k: int = LOD_VERTICES) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [k, 3] float32, outward triangle indices [3 F] int32): the convex hull of ``k`` of the hull vertices
    of ``points``, picked by farthest-point sampling from the point farthest from the centroid, then pushed out until
    it encloses every point (``_enclose``)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    pts = np.asarray(points, dtype=np.float64)
    hull_pts = pts[ConvexHull(pts).vertices]
    if len(hull_pts) > k:
        c = hull_pts.mean(0)
        pick = [int(np.argmax(np.linalg.norm(hull_pts - c, axis=1)))]
        d = np.linalg.norm(hull_pts - hull_pts[pick[0]], axis=1)
        while len(pick) < k:
            j = int(np.argmax(d))
            pick.append(j)
            d = np.minimum(d, np.linalg.norm(hull_pts - hull_pts[j], axis=1))
        hull_pts = _enclose(hull_pts[pick], hull_pts)
    hull = ConvexHull(hull_pts)
    tris = hull.simplices.copy()
    for t, eq in zip(tris, hull.equations, strict=True):          # wind each face with its outward normal
        n = np.cross(hull_pts[t[1]] - hull_pts[t[0]], hull_pts[t[2]] - hull_pts[t[0]])
        if np.dot(n, eq[:3]) < 0.0:
            t[1], t[2] = t[2], t[1]
    return hull_pts.astype(np.float32), tris.astype(np.int32).ravel()


def _outside(v: np.ndarray, pts: np.ndarray) -> tuple[Any, np.ndarray]:
    """(hull of ``v``, per hull face the largest distance [m] of a point of ``pts`` outside its plane)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    hull = ConvexHull(v)
    eq = hull.equations
    return hull, (pts @ eq[:, :3].T + eq[:, 3]).max(axis=0)


def _enclose(v: np.ndarray, pts: np.ndarray, iters: int = 8) -> np.ndarray:
    """Push the sampled hull vertices ``v`` out until their hull holds every point of ``pts`` (SCR-SC4-4b: health
    measures against the original mesh, so the view hull must not sit inside it). Each pass moves every vertex by
    the short displacement that lifts each of its faces past the farthest point outside it, plus ``ENCLOSE_PAD``
    (``_lift``). If the passes leave any point out, a scale about the centroid finishes it (always enough)."""
    v = v.copy()
    for _ in range(iters):
        hull, out = _outside(v, pts)
        if out.max() <= -ENCLOSE_PAD / 2:
            return v
        faces: list[list[int]] = [[] for _ in range(len(v))]
        for f, tri in enumerate(hull.simplices):
            for i in tri:
                faces[i].append(f)
        need = out + ENCLOSE_PAD
        # a vertex the last lift left inside the others' hull has no face: it stays (the hull does not need it)
        v = v + np.array([_lift(hull.equations[fs, :3], need[fs]) if fs else np.zeros(3) for fs in faces])
    hull, out = _outside(v, pts)
    if out.max() > -ENCLOSE_PAD / 2:
        c = v.mean(0)
        h = -(c @ hull.equations[:, :3].T + hull.equations[:, 3])              # centroid to each face plane
        v = c + (v - c) * (1.0 + (out.max() + ENCLOSE_PAD) / h.min())
    return v


def _lift(normals: np.ndarray, need: np.ndarray, sweeps: int = 20) -> np.ndarray:
    """A short u with ``normals @ u >= need`` (cyclic projection onto the half-spaces, from 0)."""
    u = np.zeros(3)
    for _ in range(sweeps):
        short = need - normals @ u
        if short.max() <= 1e-12:
            break
        for n, gap in zip(normals, need, strict=True):
            d = gap - float(n @ u)
            if d > 0.0:
                u = u + d * n
    return u


def is_convex(mesh: Any, tol: float = CONVEX_TOL) -> bool:
    """A closed mesh whose volume is within ``tol`` of its convex hull's (a hull geom); a container, bag or any
    other non-convex mesh keeps its own triangles (its cavity must stay open)."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.indices, dtype=np.int64).reshape(-1, 3)
    vol = abs(float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()) / 6.0)
    try:
        hull = ConvexHull(v).volume
    except Exception:                          # degenerate (flat) point set: not a solid
        return False
    return hull > 0.0 and vol >= (1.0 - tol) * hull


def contact_view(shapes: Sequence[int], k: int = LOD_VERTICES) -> Callable[[Any], None]:
    """A ``SolverCoupled.Entry.configure_view`` for the full-surface ``shapes`` (module docstring): reduced hulls of
    fine convex meshes (one per source mesh), in this view only."""
    shapes = np.asarray(sorted({int(s) for s in shapes}), dtype=np.int64)
    keep: list[newton.Mesh] = []                      # the views reference these meshes' Warp ids

    def configure(view: Any) -> None:
        n = int(view.shape_count or 0)
        own = shapes[shapes < n]
        if not own.size:
            return
        made = _reduced_meshes(view, own, k)
        keep.extend(m for m, _ in made.values())
        log.info("proxy contact view %s: %d full-surface shapes, %d meshes -> %d-vertex hulls",
                 getattr(view, "name", "?"), own.size, len(made), k)

    return configure


def _reduced_meshes(view: Any, shapes: np.ndarray, k: int) -> dict[int, tuple[newton.Mesh, int]]:
    """Point the view's fine mesh shapes at ``reduced_hull`` meshes; returns {id(source): (mesh, Warp id)}."""
    stype = view.shape_type.numpy()
    sources = list(view.shape_source)
    ptr = view.shape_source_ptr.numpy().copy()
    made: dict[int, tuple[newton.Mesh, int]] = {}
    for s in shapes:
        src = sources[s]
        if int(stype[s]) not in _MESH_TYPES or src is None or len(src.vertices) <= k or not is_convex(src):
            continue
        if id(src) not in made:
            v, f = reduced_hull(np.asarray(src.vertices), k)
            mesh = newton.Mesh(v, f, compute_inertia=False)
            made[id(src)] = (mesh, int(mesh.finalize(device=view.device)))
        sources[s], ptr[s] = made[id(src)]
    if made:
        view.shape_source = sources
        view.shape_source_ptr = wp.array(ptr, dtype=wp.uint64, device=view.device)
    return made
