# SPDX-License-Identifier: Apache-2.0
"""Clean-up of a convex decomposition: a near-convex mesh as its own hull, and no needlessly thin hulls.

Hygiene, not a pile fix. A convex collider resolves a contact through its nearest face, so a contact deeper than half
a hull's width pushes out through the far side, into the body's next hull; CoACD cuts many scans into slivers a few
mm thick for no geometric reason (``ycb_040_large_marker``: 14 hulls 3.6-5 mm thick on a 21 x 121 x 19 mm bar).
A 100-object clutter pile (2026-11-08) that blew up on that marker went INVALID about as often with one hull, with
merged slivers or with the original card (3-4 of 8 worlds, CPU): its cause is the marker's small axial inertia
(radius of gyration 6.2 mm, spun to 300-4,000 rad/s by rim contacts), fixed engine-side, not by hulls.

The rules (thresholds in ``hulls.HullParams``):
1. ``single_hull``: no CoACD when the mesh fills at least ``solid_min`` of its convex hull and that hull stays within
   ``gap_max`` of the real surface (``Surface.gap``); the hull is cut across its long axis into pieces at most
   ``PIECE_ASPECT`` times as long as its middle extent (``split_long``; one piece for a compact body).
2. ``thicken`` (after CoACD and its flat-hull filter), thinnest hull first: a hull thinner than ``thin_floor`` is
   dropped when ``COVERED`` of it already lies inside the other hulls; else it is replaced by whichever fix adds the
   least volume: merged with a touching hull, if the surface the joint hull adds (outside both parts) stays within
   ``gap_max`` of the real surface and the air it encloses beyond the parts is at most ``AIR_MAX`` of their volume
   (no filled bores or dishes), or thickened to ``thin_floor`` along its thin axis into the object (a skin on a
   solid body), if the space that adds lies inside the mesh. A thin hull that allows neither is kept and counted
   (``Fit.thin``): the object really is that thin there (a plate, a cup wall, a blade, a can's rim).

Why these values:
- ``thin_floor`` 5 mm = 2 x ``gap_max``: a contact up to 2.5 mm deep resolves outward from every hull. Settled and
  sliding contacts in that pile mostly read 0.1-0.5 mm; impacts briefly reach 7-18 mm, which no floor short of the
  object's own width holds, so the floor does not try to. The engine's thin-hull check uses the same value.
- ``gap_max`` 2.5 mm, 99th percentile of the hull surface: a fix may move the collision surface out by no more than
  the contact depth the floor is sized for. The marker's own hull reaches 2.2 mm out (its clip); the tomato soup
  can's 5.8 mm (the recessed lid), so the can keeps CoACD's hulls.
- ``solid_min`` 0.8: the cheap test before the gap test. Below it a fifth of the hull is empty (a handle, a cavity, a
  hook); in YCB no mesh under 0.8 passes the gap test anyway, and the marker (0.84) is the lowest that does.
- ``PIECE_ASPECT`` 2: pieces no longer than twice the body's width spread a long body's contacts along its length
  (the marker: 4 pieces), as a capsule's two end contacts do; a modelling choice, not a measured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.importers.common import hulls as hl

SEED = 0
SPACING = 1e-3               # m between surface samples of the mesh (KD-tree for the gap)
MESH_SAMPLES = (20_000, 200_000)
HULL_SAMPLES = (256, 4096)
GAP_Q = 99.0                 # the gap of a hull: this percentile of its surface's distance outside the mesh
PIECE_ASPECT = 2.0           # a near-convex mesh's pieces are at most this x as long as its middle extent (below)
COVERED = 0.95               # a thin hull this much inside the other hulls adds no collision volume: drop it
EXTRA = 0.02                 # thicken to (1 + EXTRA) x thin_floor, so the vertex cap's re-hull stays above the floor
INWARD_TOL = 5e-4            # m: the space a thickening adds may lie this far outside the object on average
#                              (sampling noise; a skin on a solid body reads ~0, a free-standing wall ~step / 2)
INSIDE_TOL = 1e-6            # m: a point this close outside a hull face still counts as inside
NEW_TOL = 5e-4               # m: joint-hull surface this close to a part's own surface is not new (vertex-cap noise)
AIR_MAX = 0.1                # a merge may enclose air (outside the mesh and both parts) up to this x the parts' volume:
#                              mass the emitter spreads over hull volume then moves into air by at most a tenth
INTERIOR_SAMPLES = 2048


def _areas(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return cross, 0.5 * np.linalg.norm(cross, axis=1)


def sample_surface(v: np.ndarray, f: np.ndarray, n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """``n`` area-uniform points on a triangle mesh and the unit normal of the face each lies on (seeded).

    Example::

        pts, normals = sample_surface(v, f, 1000)
    """
    v, f = np.asarray(v, np.float64), np.asarray(f, np.int64)
    cross, area = _areas(v, f)
    rng = np.random.default_rng(seed)
    k = rng.choice(len(f), size=n, p=area / area.sum())
    s, r = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    tri = v[f[k]]
    pts = (1.0 - s) * tri[:, 0] + s * (1.0 - r) * tri[:, 1] + s * r * tri[:, 2]
    return pts, cross[k] / np.maximum(2.0 * area[k], 1e-300)[:, None]


class Surface:
    """Dense samples of a watertight mesh's surface: how far a hull reaches outside the real object.

    Example::

        surf = Surface(phys.vertices, phys.faces)
        surf.gap(hull)          # m, 99th percentile of the hull surface's distance outside the mesh
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, spacing: float = SPACING):
        from scipy.spatial import cKDTree  # noqa: PLC0415 - scipy comes with the import extra (trimesh hulls)

        v, f = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
        n = int(np.clip(_areas(v, f)[1].sum() / spacing ** 2, *MESH_SAMPLES))
        self.points, self.normals = sample_surface(v, f, n)
        self.tree = cKDTree(self.points)
        self.spacing = spacing

    def outside(self, pts: np.ndarray) -> np.ndarray:
        """Distance [m] of each point outside the mesh (0 inside), from the nearest sample and its face normal."""
        d, i = self.tree.query(np.asarray(pts, np.float64))
        out = np.einsum("ij,ij->i", pts - self.points[i], self.normals[i]) > 0.0
        return np.where(out, d, 0.0)

    def samples(self, hull: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Area-uniform points (and face normals) on a hull, about ``spacing`` apart (``HULL_SAMPLES`` bounds)."""
        v, f = np.asarray(hull[0], np.float64), np.asarray(hull[1])
        return sample_surface(v, f, int(np.clip(_areas(v, f)[1].sum() / self.spacing ** 2, *HULL_SAMPLES)))

    def gap(self, hull: tuple[np.ndarray, np.ndarray], q: float = GAP_Q) -> float:
        """How far [m] the hull's surface reaches outside the mesh: the ``q`` percentile over area-uniform samples."""
        return float(np.percentile(self.outside(self.samples(hull)[0]), q))

    def added_gap(self, joint: tuple[np.ndarray, np.ndarray], parts: list["_Part"], q: float = GAP_Q) -> float:
        """``gap`` over only the joint hull's surface that lies outside every one of ``parts`` (0 if none does)."""
        pts = self.samples(joint)[0]
        new = ~np.any([p.contains(pts, NEW_TOL) for p in parts], axis=0)
        return float(np.percentile(self.outside(pts[new]), q)) if new.any() else 0.0

    def swept_out(self, hull: tuple[np.ndarray, np.ndarray], shift: np.ndarray) -> float:
        """Mean distance [m] outside the mesh of the space a hull sweeps when moved by ``shift`` (0 = all inside)."""
        pts, normals = self.samples(hull)
        lead = normals @ shift > 0.0
        u = np.random.default_rng(SEED).random(int(lead.sum()))[:, None]
        return float(self.outside(pts[lead] + u * shift).mean()) if lead.any() else 0.0


def _long_axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """``(centre, unit long axis, length, middle extent)`` of a point set in its principal frame."""
    c = pts.mean(axis=0)
    axes = np.linalg.svd(pts - c, full_matrices=False)[2]
    ext = np.ptp((pts - c) @ axes.T, axis=0)
    order = np.argsort(ext)[::-1]
    return c, axes[order[0]], float(ext[order[0]]), float(ext[order[1]])


def _clip(pts: np.ndarray, axis: np.ndarray, lo: float, hi: float, max_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """The convex hull of ``pts`` cut to ``lo <= axis . x <= hi`` (exact: half-space intersection), capped."""
    from scipy.optimize import linprog  # noqa: PLC0415
    from scipy.spatial import ConvexHull, HalfspaceIntersection  # noqa: PLC0415

    hs = np.vstack([ConvexHull(pts).equations, [[*axis, -hi], [*(-axis), lo]]])   # rows n . x + d <= 0 inside
    norm = np.linalg.norm(hs[:, :3], axis=1)
    lp = linprog([0.0, 0.0, 0.0, -1.0], A_ub=np.hstack([hs[:, :3], norm[:, None]]), b_ub=-hs[:, 3],
                 bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")   # the slab's Chebyshev centre
    return hl.convex_hull(HalfspaceIntersection(hs, lp.x[:3]).intersections, max_verts)


def split_long(vertices: np.ndarray, params: hl.HullParams) -> list[tuple[np.ndarray, np.ndarray]]:
    """The convex hull of ``vertices``, cut across its long axis into equal pieces at most ``PIECE_ASPECT`` times as
    long as the hull's middle extent (one piece for a compact body), each capped at ``max_hull_verts``.

    Example::

        split_long(bar_vertices, HullParams())    # a 121 x 21 x 19 mm bar -> 3 pieces about 40 mm long
    """
    pts = np.unique(np.asarray(vertices, np.float64), axis=0)
    c, axis, length, middle = _long_axis(pts)
    k = int(min(params.max_parts, max(1, np.ceil(length / (PIECE_ASPECT * max(middle, 1e-12)) - 1e-9))))
    if k == 1:
        return [hl.convex_hull(pts, params.max_hull_verts)]
    s = pts @ axis
    edges = np.linspace(s.min(), s.max(), k + 1)
    return [_clip(pts, axis, a, b, params.max_hull_verts) for a, b in zip(edges[:-1], edges[1:])]


def single_hull(vertices: np.ndarray, mesh_volume: float, surf: Surface,
                params: hl.HullParams) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """The mesh's convex hull (capped at ``max_hull_verts``) when it is a good collider by itself, else None; a long
    one comes back cut into pieces along its length (``split_long``).

    Example::

        hulls = single_hull(phys.vertices, phys.volume, Surface(phys.vertices, phys.faces), HullParams())
    """
    hull = hl.convex_hull(np.asarray(vertices), params.max_hull_verts)
    if mesh_volume / hl.hull_volume(*hull) < params.solid_min or surf.gap(hull) > params.gap_max:
        return None
    return split_long(vertices, params)


@dataclass
class _Part:
    v: np.ndarray
    f: np.ndarray
    vol: float
    thick: float
    axis: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    stuck: bool = False

    @classmethod
    def of(cls, v: np.ndarray, f: np.ndarray) -> "_Part":
        v = np.asarray(v, np.float64)
        w, axis = hl.min_width(v, f)
        return cls(v, np.asarray(f, np.int64), hl.hull_volume(v, f), w, axis, v.min(0), v.max(0))

    def touches(self, other: "_Part", tol: float) -> bool:
        return bool(np.all(self.lo - tol <= other.hi) and np.all(other.lo - tol <= self.hi))

    def contains(self, pts: np.ndarray, tol: float = INSIDE_TOL) -> np.ndarray:
        tri = self.v[self.f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
        return np.all(pts @ n.T - np.einsum("ij,ij->i", n, tri[:, 0]) <= tol, axis=1)


@dataclass(frozen=True)
class Fit:
    """What ``thicken`` did: hulls merged into a neighbour, dropped as covered, and thin ones left (real thin walls)."""

    merged: int = 0
    covered: int = 0
    thickened: int = 0
    thin: int = 0


def interior(v: np.ndarray, f: np.ndarray, n: int = INTERIOR_SAMPLES, seed: int = SEED) -> np.ndarray:
    """``n`` points uniform inside a convex hull: tetrahedra from the centroid to each face, picked by volume, and
    uniform (Dirichlet(1, 1, 1, 1)) weights inside each.

    Example::

        pts = interior(*convex_hull(np.random.rand(50, 3)), n=500)
    """
    v, f = np.asarray(v, np.float64), np.asarray(f, np.int64)
    c = v.mean(axis=0)
    tri = v[f]
    vol = np.abs(np.einsum("ij,ij->i", tri[:, 0] - c, np.cross(tri[:, 1] - c, tri[:, 2] - c)))
    rng = np.random.default_rng(seed)
    k = rng.choice(len(f), size=n, p=vol / vol.sum())
    w = rng.dirichlet(np.ones(4), size=n)
    return w[:, :1] * c + np.einsum("ij,ijk->ik", w[:, 1:], tri[k])


def _covered(p: _Part, others: list[_Part]) -> float:
    pts = interior(p.v, p.f)
    inside = np.zeros(len(pts), bool)
    for o in others:
        if o.touches(p, 0.0):
            inside |= o.contains(pts)
    return float(inside.mean())


def _fixes(p: _Part, others: list[_Part], params: hl.HullParams):
    """Candidate replacements of thin part ``p``: ``(added volume, order, absorbed part or shift, new part)``."""
    out = []
    for o in others:
        if o.touches(p, params.thin_floor):
            m = _Part.of(*hl.convex_hull(np.vstack([p.v, o.v]), params.max_hull_verts))
            out.append((m.vol - p.vol - o.vol, len(out), o, m))
    step = (1.0 + EXTRA) * params.thin_floor - p.thick
    for sign in (1.0, -1.0):
        shift = sign * step * p.axis
        m = _Part.of(*hl.convex_hull(np.vstack([p.v, p.v + shift]), params.max_hull_verts))
        out.append((m.vol - p.vol, len(out), shift, m))
    return sorted(out, key=lambda c: c[:2])


def _ok(p: _Part, how: _Part | np.ndarray, m: _Part, surf: Surface, params: hl.HullParams) -> bool:
    """A merge's new surface (outside both parts) may reach ``gap_max`` outside the mesh and its new air is at most
    ``AIR_MAX`` of the parts' volume; a thickening must go into the object: the space it adds lies on average at most
    ``INWARD_TOL`` outside the mesh."""
    if isinstance(how, _Part):
        if surf.added_gap((m.v, m.f), [p, how]) > params.gap_max:
            return False
        pts = interior(m.v, m.f)
        air = (surf.outside(pts) > 0.0) & ~p.contains(pts) & ~how.contains(pts)
        return float(air.mean()) * m.vol <= AIR_MAX * (p.vol + how.vol)
    return surf.swept_out((p.v, p.f), how) <= INWARD_TOL


def _best_fix(p: _Part, others: list[_Part], surf: Surface,
              params: hl.HullParams) -> tuple[_Part | None, _Part] | None:
    """The fix of ``_fixes`` that adds the least volume and passes ``_ok``: ``(absorbed part or None, new part)``."""
    for _, _, how, m in _fixes(p, others, params):
        if _ok(p, how, m, surf, params):
            return (how if isinstance(how, _Part) else None), m
    return None


def thicken(hulls: list[tuple[np.ndarray, np.ndarray]], surf: Surface,
            params: hl.HullParams) -> tuple[list[tuple[np.ndarray, np.ndarray]], Fit]:
    """Drop, merge or thicken hulls thinner than ``params.thin_floor``, thinnest first (rule 2 of the module docstring).

    Example::

        hulls, fit = thicken(hulls, Surface(phys.vertices, phys.faces), HullParams())
        fit.merged, fit.thin
    """
    parts = [_Part.of(v, f) for v, f in hulls]
    n = {"merged": 0, "covered": 0, "thickened": 0}
    for _ in range(4 * len(parts)):             # each pass fixes or settles one hull; the bound only guards
        thin = [p for p in parts if p.thick < params.thin_floor and not p.stuck]
        if len(parts) < 2 or not thin:
            break
        p = min(thin, key=lambda q: q.thick)
        others = [q for q in parts if q is not p]
        if _covered(p, others) >= COVERED:
            parts, n["covered"] = others, n["covered"] + 1
            continue
        best = _best_fix(p, others, surf, params)
        if best is None:
            p.stuck = True
            continue
        o, m = best
        m.stuck = o is None and m.thick < params.thin_floor     # a capped re-hull can land just under the floor
        parts = [q for q in others if q is not o] + [m]
        n["merged" if o is not None else "thickened"] += 1
    fit = Fit(thin=sum(p.thick < params.thin_floor for p in parts), **n)
    return [(p.v, p.f) for p in parts], fit
