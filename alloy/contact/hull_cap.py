# SPDX-License-Identifier: Apache-2.0
"""Convex-mesh collision hulls capped by geometric error instead of a fixed vertex count.

A ``collision="convex"`` mesh collides as a hull of at most ``maxhullvert`` vertices (qhull adds the farthest point
until it has that many). Newton's default is 64 (``newton.Mesh.MAX_HULL_VERTICES``), which can drop a small feature: a
brass top's 1,282-vertex lathe kept one tip vertex 0.3 mm off its axis and the next hull vertices 8 mm up the body, so
the top spun on that one point instead of its rounded tip. MuJoCo's soft friction does positive work on a contact
point that whirls with the spin (+14.6 mW at the contact), and the top gained spin and energy where tip friction should
slow it (CPU, rk4; the same at a 4x finer step, under Euler, and in plain MuJoCo with that hull).

``maxhullvert`` raises a mesh's cap along ``LADDER`` until its hull lies within ``tolerance`` of the full hull: no
vertex of the full hull is farther from the capped hull than min(0.1 mm, 1 % of the mesh's smallest extent), the exact
point-to-polytope distance (``deviation``). A mesh whose full hull has at most 64 vertices keeps Newton's default (its
model is bit for bit the same), as does a flat or degenerate point set (Newton and MuJoCo report those as before). A
mesh whose 64-vertex hull is already within the tolerance keeps the cap of 64 (over 50,000 vertices: MuJoCo's hull at
that cap, in place of Newton's, see below). At ``CEILING`` vertices the search stops; a hull still out of
tolerance there makes the build warn once per mesh, naming the deviation and the fix.

The hull measured is the one the mesh collides with: MuJoCo's (an ``MjSpec`` compile of all the vertices at each cap,
mapped back through the mesh frame it reports). A mesh the emitter reduces to its hull first (more than 50,000
vertices) gets that same hull from ``capped_mesh``, in place of Newton's scipy pass at the same cap, which left a patch
about 19 deg across unrefined on a 60,000-vertex sphere (2.16 mm out at 2,048 vertices, MuJoCo's 0.10 mm). The distance
runs over row chunks, so memory stays bounded on scans whose every vertex is on the hull. Choices are cached per vertex
array for the process. ``ALLOY_HULL_TOL=0`` keeps Newton's default for every mesh (A/B runs).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from scipy.spatial import cKDTree

from alloy.contact.hull_limits import HULL_VERTICES_MAX
from alloy.contact.hull_limits import hull_vertices_problem
from alloy.core.errors import AssemblyError

log = logging.getLogger(__name__)

ENV = "ALLOY_HULL_TOL"           # "0": Newton's default cap for every mesh
DEFAULT = 64                     # newton.Mesh.MAX_HULL_VERTICES (checked by tests/contact/test_hull_cap.py)
TOL_ABS = 1e-4                   # m
TOL_REL = 0.01                   # of the mesh's smallest extent
CEILING = HULL_VERTICES_MAX       # 2,048 vertices: 320 sampled Hub scans as convex meshes all fit (17 over at 1,024)
LADDER = (64, 128, 256, 512, 1024, 2048)
CHUNK = 1 << 20                  # point-face pairs per pass of ``deviation`` (memory: about 100 MB)
FIX = "use collision='decompose' (convex parts), or a simpler mesh"
FIX_CONVEX = "use a primitive shape (sphere, capsule, cylinder), or a mesh with fewer vertices"
_CACHE: dict[bytes, HullChoice] = {}
_WARNED: set[bytes] = set()
_RULED: dict[bytes, int] = {}    # vertex array -> the cap the rule gave it (the emitter's last call for that mesh)


@dataclass(frozen=True)
class HullChoice:
    """``cap``: the ``maxhullvert`` to give the mesh (``DEFAULT``: unchanged); ``deviation`` (m): how far the full
    hull lies from the hull at ``cap``; ``tolerance`` (m); ``full``: vertices of the full hull; ``convex``: the mesh
    is (about) its own hull, so decomposing it would not help."""

    cap: int
    deviation: float
    tolerance: float
    full: int
    convex: bool = False

    @property
    def within(self) -> bool:
        return self.deviation <= self.tolerance


def tolerance(vertices: np.ndarray) -> float:
    """min(TOL_ABS, TOL_REL x the smallest extent of the vertices' bounding box), in m."""
    return float(min(TOL_ABS, TOL_REL * float(np.ptp(np.asarray(vertices, float), axis=0).min())))


def mujoco_hull(vertices: np.ndarray, cap: int) -> np.ndarray:
    """The vertices of the hull MuJoCo builds from ``vertices`` with ``maxhullvert=cap`` (-1: the full hull), in the
    input frame (MuJoCo stores mesh vertices in the mesh's own frame, ``mesh_pos`` / ``mesh_quat``)."""
    spec = mujoco.MjSpec()
    spec.add_mesh(name="m", uservert=np.asarray(vertices, float).ravel().tolist(), maxhullvert=int(cap),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m")
    m = spec.compile()
    g, a = int(m.mesh_graphadr[0]), int(m.mesh_vertadr[0])
    nv = int(m.mesh_graph[g])
    local = m.mesh_vert[a:a + int(m.mesh_vertnum[0])].astype(float)[m.mesh_graph[g + 2 + nv: g + 2 + 2 * nv]]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.mesh_quat[0])
    return local @ rot.reshape(3, 3).T + m.mesh_pos[0]


def capped_mesh(mesh: Any) -> Any:
    """The large-mesh collider: ``mesh.compute_convex_hull()`` with MuJoCo's capped hull at ``mesh.maxhullvert`` in
    place of Newton's scipy pass, snapped onto the input vertices and wound outward, keeping the whole mesh's mass,
    centre and inertia as Newton's does. Newton's own hull when the rule did not set the cap (``ENV``, or the mesh's
    ``hull_vertices``) or the full hull has at most ``DEFAULT`` vertices (degenerate input included), so those stay as
    they were."""
    if _RULED.get(_key(mesh.vertices)) != mesh.maxhullvert or choose(mesh.vertices).full <= DEFAULT:
        return mesh.compute_convex_hull()
    from alloy._vendor import newton  # noqa: PLC0415

    v = np.asarray(mesh.vertices, float)
    hv = v[cKDTree(v).query(mujoco_hull(v, mesh.maxhullvert))[1]]        # the frame map is within 1e-9 m
    h = ConvexHull(hv)
    tri = h.simplices.copy()
    flip = _dot(np.cross(hv[tri[:, 1]] - hv[tri[:, 0]], hv[tri[:, 2]] - hv[tri[:, 0]]), h.equations[:, :3]) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    out = newton.Mesh(hv.astype(np.float32), tri.reshape(-1).astype(np.int32), compute_inertia=False)
    for k in ("maxhullvert", "is_solid", "has_inertia", "mass", "com", "inertia"):
        setattr(out, k, getattr(mesh, k))
    return out


def point_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from each point ``p[i]`` to triangle ``(a[i], b[i], c[i])`` (all (n, 3)); Ericson's closest-point
    regions, evaluated for every row and picked in order."""
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        q = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]                     # inside the face
        q = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[:, None],
                     b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None], q)      # edge BC
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], a + ac * (d2 / (d2 - d6))[:, None], q)  # AC
        q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, q)                               # vertex C
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], a + ab * (d1 / (d1 - d3))[:, None], q)  # AB
        q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, q)                               # vertex B
        q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, q)                                # vertex A
    return np.linalg.norm(p - q, axis=1)


def deviation(full: np.ndarray, hull: np.ndarray, stop: float | None = None) -> float:
    """The largest distance (m) from a ``full`` hull vertex to the convex hull of ``hull`` (0 inside). A point outside
    is nearest to a face whose plane it is above, so only those faces are measured; rows go in chunks of ``CHUNK``
    point-face pairs. With ``stop``, returns as soon as the answer is known to exceed it (a value above ``stop``)."""
    h = ConvexHull(hull)
    n, off, tri = h.equations[:, :3], h.equations[:, 3], h.points[h.simplices]
    rows = max(1, CHUNK // len(off))
    worst = 0.0
    for i in range(0, len(full), rows):
        p = full[i:i + rows]
        s = p @ n.T + off
        above = s > 0.0
        if not above.any():
            continue
        plane = s.max(axis=1)                                  # a plane distance never exceeds the true distance
        if stop is not None and float(plane.max()) > stop:
            return float(plane.max())
        pi, fi = np.nonzero(above)
        d = point_triangle_distance(p[pi], tri[fi, 0], tri[fi, 1], tri[fi, 2])
        best = np.full(len(p), np.inf)
        np.minimum.at(best, pi, np.where(np.isfinite(d), d, np.inf))   # a sliver face gives NaN: skip it
        out = plane > 0.0
        worst = max(worst, float(np.where(np.isfinite(best[out]), best[out], plane[out]).max()))
        if stop is not None and worst > stop:
            return worst
    return worst


def choose(vertices: np.ndarray) -> HullChoice:
    """The cap of ``vertices`` (as the collider gets them: float32) by the rule in the module docstring (cached)."""
    v = np.ascontiguousarray(np.asarray(vertices, np.float32), dtype=np.float64)
    key = _key(v)
    if key not in _CACHE:
        _CACHE[key] = _search(v)
    return _CACHE[key]


def _key(vertices: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(vertices, np.float32), dtype=np.float64).tobytes()).digest()


def _search(v: np.ndarray) -> HullChoice:
    tol = tolerance(v)
    try:
        if not _full_rank(v):
            return HullChoice(DEFAULT, 0.0, tol, 0)
        full = v[ConvexHull(v).vertices]
        if len(full) <= DEFAULT:
            return HullChoice(DEFAULT, 0.0, tol, len(full))
        for cap in LADDER:
            if cap >= len(full):
                return HullChoice(cap, 0.0, tol, len(full))
            last = cap == LADDER[-1]
            hull = mujoco_hull(v, cap)
            dev = deviation(full, hull, stop=None if last else tol)
            if dev <= tol:
                return HullChoice(cap, dev, tol, len(full))
            if last:                                             # the ceiling, out of tolerance
                return HullChoice(cap, dev, tol, len(full), convex(v, hull, max(tol, dev)))
    except (QhullError, ValueError) as e:                       # degenerate: Newton / MuJoCo report it as before
        log.debug("hull_cap: %s; keeping maxhullvert %d", e, DEFAULT)
    return HullChoice(DEFAULT, 0.0, tol, 0)


def convex(v: np.ndarray, hull: np.ndarray, near: float) -> bool:
    """Whether a mesh is about its own hull, so decomposing it would not help: 90 % of its vertices lie within
    2 x ``near`` of ``hull``'s face planes (a scanned ball's noisy surface counts; a cup's inside does not)."""
    eq = ConvexHull(hull).equations
    u = np.unique(v, axis=0)
    rows = max(1, CHUNK // len(eq))
    close = sum(int(((u[i:i + rows] @ eq[:, :3].T + eq[:, 3]).max(axis=1) >= -2.0 * near).sum())
                for i in range(0, len(u), rows))
    return close >= 0.9 * len(u)


def _full_rank(v: np.ndarray, eps: float = 1e-6) -> bool:
    """At least 4 points spanning 3-D (Newton's ``remesh_convex_hull`` threshold on the singular values)."""
    if len(v) < 4:
        return False
    s = np.linalg.svd(v - v.mean(axis=0), compute_uv=False)
    return bool(s[0] > 0.0 and s[2] > eps * s[0])


def maxhullvert(vertices: np.ndarray, label: str, fixed: int | None = None) -> int:
    """The ``maxhullvert`` for a convex collision mesh named ``label``; warns once per mesh when the hull at
    ``CEILING`` is still out of tolerance. ``fixed`` (the mesh's ``hull_vertices``, the user's opt-out) or
    ``ALLOY_HULL_TOL=0``: that cap or ``DEFAULT``, with Newton's own hull (``capped_mesh``), as before the rule."""
    key = _key(vertices)
    _RULED.pop(key, None)
    if problem := hull_vertices_problem(fixed, "convex"):
        raise AssemblyError(f"{label}: shape.hull_vertices: {problem}")
    if fixed is not None or os.environ.get(ENV, "1") == "0":
        return DEFAULT if fixed is None else int(fixed)
    c = choose(vertices)
    _RULED[key] = c.cap
    if c.cap != DEFAULT:
        log.info("%s: convex hull of %d vertices (full hull %d): within %.3g mm of the full hull (tolerance %.3g mm)",
                 label, c.cap, c.full, c.deviation * 1e3, c.tolerance * 1e3)
    if not c.within:
        _warn_once(label, c)
    return c.cap


def _warn_once(label: str, c: HullChoice) -> None:
    key = (label + "|" + repr((c.cap, c.full, c.deviation))).encode()
    if key in _WARNED:
        return
    _WARNED.add(key)
    from alloy.api import _say  # noqa: PLC0415 - the user-facing warning (alloy.api imports the scene layers)

    _say.warn("%s: its convex collision hull (%d of %d hull vertices, the most Alloy uses) lies up to %.2f mm inside "
              "the mesh's hull (tolerance %.2f mm), so contacts there are off by that much: %s", label, c.cap, c.full,
              c.deviation * 1e3, c.tolerance * 1e3, FIX_CONVEX if c.convex else FIX)


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", x, y)


__all__ = ["CEILING", "DEFAULT", "ENV", "FIX", "FIX_CONVEX", "LADDER", "HullChoice", "capped_mesh", "choose", "convex",
           "deviation", "maxhullvert", "mujoco_hull", "point_triangle_distance", "tolerance"]
