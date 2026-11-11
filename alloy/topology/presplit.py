# SPDX-License-Identifier: Apache-2.0
"""Pre-split meshes along candidate seams at assembly (SPEC D19, §3.8).

A seam is a set of mesh facets (triangle edges or tet faces) between elements in different regions. Every vertex on
a seam is duplicated once per region that meets at it, elements reference their region's copy, and each pair of
copies that touch across a seam facet becomes one twin bond. Topology is then fixed: tearing or cutting only
breaks twin bonds, so VBD adjacency and colouring never change (solver_vbd.py:1126-1130).

Pure numpy (float64); deterministic: the copy kept as the original is the one whose region holds the vertex's
lowest-index element, and new vertices are numbered by (vertex, lowest element of its region).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from alloy.core.errors import AssemblyError

log = logging.getLogger("alloy.topology")

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class SplitPlan:
    """Where seams go. Regions are cells of a regular pattern, sides of planes, or both; creases add seams."""

    spacing: float | None = None                      # regular pattern spacing [m]; None = no pattern
    axes: tuple[int, ...] = (0, 1, 2)                 # mesh-frame axes the pattern cuts across
    origin: Vec3 = (0.0, 0.0, 0.0)                    # pattern origin, mesh frame [m]
    planes: tuple[tuple[Vec3, float], ...] = ()       # (normal, offset): side = sign(n·x - d)
    creases: tuple[tuple[Vec3, Vec3], ...] = ()       # triangle meshes: edges lying on these segments are seams
    crease_tol: float = 1.0e-6                        # [m] distance of an edge's endpoints to the segment

    @property
    def empty(self) -> bool:
        return self.spacing is None and not self.planes and not self.creases


@dataclass(frozen=True)
class PresplitMesh:
    vertices: np.ndarray    # [V', 3] float64; first V rows are the input vertices
    elements: np.ndarray    # [E, k] int64; k = 3 (triangles) or 4 (tets)
    source: np.ndarray      # [V'] input vertex each vertex was copied from
    twins: np.ndarray       # [B, 2] int64 twin bonds (a, b), a < b
    normal: np.ndarray      # [B, 3] unit seam normal pointing from a's region to b's region
    measure: np.ndarray     # [B] tributary seam length (triangles, m) or area (tets, m^2)
    n_original: int         # input vertex count V
    seam_pairs: np.ndarray  # [S, 2] int64 element pairs (e1, e2) that shared a facet before the split

    @property
    def n_added(self) -> int:
        return int(self.vertices.shape[0] - self.n_original)


def element_labels(plan: SplitPlan, vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Region id per element from the pattern cell and plane sides of its centroid (int64, dense from 0)."""
    cen = vertices[elements].mean(axis=1)
    keys = [np.zeros(len(elements), dtype=np.int64)]
    if plan.spacing is not None:
        if plan.spacing <= 0.0:
            raise AssemblyError(f"SplitPlan.spacing must be > 0, got {plan.spacing}")
        rel = (cen - np.asarray(plan.origin)) / plan.spacing
        keys += [np.floor(rel[:, ax]).astype(np.int64) for ax in plan.axes]
    for normal, offset in plan.planes:
        keys.append((cen @ np.asarray(normal, dtype=np.float64) > offset).astype(np.int64))
    _, labels = np.unique(np.stack(keys, axis=1), axis=0, return_inverse=True)
    return labels.reshape(-1).astype(np.int64)


def _facet_slots(k: int) -> np.ndarray:
    """Local vertex indices of each facet of a simplex with k vertices (edges of a triangle, faces of a tet)."""
    if k == 3:
        return np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    if k == 4:
        return np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64)
    raise AssemblyError(f"elements must be triangles (3) or tets (4), got {k} vertices")


@dataclass(frozen=True)
class _FacetPairs:
    """Interior facets shared by two elements, with local corner indices aligned by global vertex id."""

    e1: np.ndarray          # [F] element on one side
    e2: np.ndarray          # [F] element on the other side
    l1: np.ndarray          # [F, k-1] local indices in e1, sorted by global vertex id
    l2: np.ndarray          # [F, k-1] local indices in e2, same order
    verts: np.ndarray       # [F, k-1] global vertex ids (sorted)


def _interior_facets(elements: np.ndarray) -> _FacetPairs:
    n_el, k = elements.shape
    slots = _facet_slots(k)
    local = np.broadcast_to(slots, (n_el,) + slots.shape)                       # [E, nf, k-1]
    glob = np.take_along_axis(elements[:, None, :].repeat(len(slots), 1), local, axis=2)
    order = np.argsort(glob, axis=2, kind="stable")
    glob = np.take_along_axis(glob, order, axis=2).reshape(-1, k - 1)
    local = np.take_along_axis(local, order, axis=2).reshape(-1, k - 1)
    owner = np.repeat(np.arange(n_el, dtype=np.int64), len(slots))
    perm = np.lexsort(glob.T[::-1])
    glob, local, owner = glob[perm], local[perm], owner[perm]
    same = np.all(glob[1:] == glob[:-1], axis=1)
    if np.any(same[1:] & same[:-1]):
        raise AssemblyError("non-manifold mesh: a facet is shared by more than two elements")
    i = np.flatnonzero(same)
    return _FacetPairs(owner[i], owner[i + 1], local[i], local[i + 1], glob[i])


def _crease_mask(plan: SplitPlan, vertices: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Facets (triangle edges) whose endpoints all lie within crease_tol of one crease segment."""
    mask = np.zeros(len(verts), dtype=bool)
    for a, b in plan.creases:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ab = b - a
        p = vertices[verts]                                                      # [F, 2, 3]
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
        d = np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)
        mask |= np.all(d <= plan.crease_tol, axis=1)
    return mask


def _corner_components(elements: np.ndarray, pairs: _FacetPairs, seam: np.ndarray) -> np.ndarray:
    """Component label per corner (element, local vertex) after joining corners across non-seam facets."""
    n_el, k = elements.shape
    keep = ~seam
    a = (pairs.e1[keep, None] * k + pairs.l1[keep]).reshape(-1)
    b = (pairs.e2[keep, None] * k + pairs.l2[keep]).reshape(-1)
    n = n_el * k
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    return comp.astype(np.int64)


def _assign_copies(elements: np.ndarray, comp: np.ndarray, n_vert: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex id per corner: the region holding a vertex's lowest corner keeps it; others get new ids in order."""
    n_el, k = elements.shape
    corner_vertex = elements.reshape(-1)
    corner_id = np.arange(n_el * k, dtype=np.int64)
    # one representative per (vertex, component): its lowest corner
    keys = np.stack([corner_vertex, comp], axis=1)
    uniq, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    rep_corner = corner_id[first]
    order = np.lexsort((rep_corner, uniq[:, 0]))                 # by vertex, then lowest corner
    rank = np.empty(len(uniq), dtype=np.int64)
    is_first = np.ones(len(order), dtype=bool)
    is_first[1:] = uniq[order[1:], 0] != uniq[order[:-1], 0]
    new_ids = n_vert + np.cumsum(~is_first) - 1
    rank[order] = np.where(is_first, uniq[order, 0], new_ids)
    source = np.concatenate([np.arange(n_vert, dtype=np.int64), uniq[order[~is_first], 0]])
    return rank[inverse].reshape(n_el, k), source


def _facet_measure_normal(vertices: np.ndarray, elements: np.ndarray, pairs: _FacetPairs, idx: np.ndarray):
    """Facet length/area and unit normal oriented from e1 to e2 for the facets in idx."""
    p = vertices[pairs.verts[idx]]
    c1 = vertices[elements[pairs.e1[idx]]].mean(axis=1)
    c2 = vertices[elements[pairs.e2[idx]]].mean(axis=1)
    d = c2 - c1
    if p.shape[1] == 2:                                           # edge: in-plane normal = d minus its edge part
        e = p[:, 1] - p[:, 0]
        size = np.linalg.norm(e, axis=1)
        t = e / np.maximum(size, 1e-300)[:, None]
        n = d - (d * t).sum(1, keepdims=True) * t
    else:                                                         # triangle face
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        size = 0.5 * np.linalg.norm(n, axis=1)
        n = n * np.sign((n * d).sum(1, keepdims=True))
    n = n / np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    return size, n


def _collect_bonds(vertices, elements, new_el, pairs: _FacetPairs, seam: np.ndarray):
    """Twin bonds over seam facets: one per pair of copies of a vertex that touch across a seam facet."""
    idx = np.flatnonzero(seam)
    k1 = pairs.verts.shape[1]
    size, normal = _facet_measure_normal(vertices, elements, pairs, idx)
    a = np.take_along_axis(new_el[pairs.e1[idx]], pairs.l1[idx], axis=1).reshape(-1)
    b = np.take_along_axis(new_el[pairs.e2[idx]], pairs.l2[idx], axis=1).reshape(-1)
    share = np.repeat(size / k1, k1)
    nrm = np.repeat(normal, k1, axis=0)
    split = a != b
    a, b, share, nrm = a[split], b[split], share[split], nrm[split]
    flip = a > b
    a, b = np.where(flip, b, a), np.where(flip, a, b)
    nrm = np.where(flip[:, None], -nrm, nrm)
    if len(a) == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 3)), np.zeros(0)
    uniq, inv = np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    meas = np.bincount(inv, weights=share, minlength=len(uniq))
    nsum = np.stack([np.bincount(inv, weights=nrm[:, j] * share, minlength=len(uniq)) for j in range(3)], 1)
    nsum /= np.maximum(np.linalg.norm(nsum, axis=1), 1e-300)[:, None]
    return uniq.astype(np.int64), nsum, meas


def presplit_mesh(vertices: np.ndarray, elements: np.ndarray, plan: SplitPlan) -> PresplitMesh:
    """Duplicate seam vertices per region; returns the new mesh, copy sources and twin bonds."""
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    elements = np.asarray(elements, dtype=np.int64)
    if elements.ndim != 2 or elements.min(initial=0) < 0 or elements.max(initial=-1) >= len(vertices):
        raise AssemblyError("elements must be [E, 3|4] indices into vertices")
    pairs = _interior_facets(elements)
    labels = element_labels(plan, vertices, elements)
    seam = labels[pairs.e1] != labels[pairs.e2]
    if plan.creases:
        if elements.shape[1] != 3:
            raise AssemblyError("creases apply to triangle meshes only")
        seam |= _crease_mask(plan, vertices, pairs.verts)
    comp = _corner_components(elements, pairs, seam)
    new_el, source = _assign_copies(elements, comp, len(vertices))
    twins, normal, measure = _collect_bonds(vertices, elements, new_el, pairs, seam)
    seam_pairs = np.stack([pairs.e1[seam], pairs.e2[seam]], axis=1)
    out = PresplitMesh(vertices[source], new_el, source, twins, normal, measure, len(vertices), seam_pairs)
    log.debug("presplit: %d elements, %d seam facets, %d vertices added, %d twin bonds",
              len(elements), int(seam.sum()), len(source) - len(vertices), len(twins))
    return out


def count_duplicates(vertices: np.ndarray, elements: np.ndarray, plan: SplitPlan) -> int:
    """Vertices a plan would add (for sizing a pattern against TopologyBudget.spare_vertices)."""
    return presplit_mesh(vertices, elements, plan).n_added


def spacing_for_budget(vertices: np.ndarray, elements: np.ndarray, base: SplitPlan, spare_vertices: int,
                       min_spacing: float, max_spacing: float, iters: int = 24) -> float | None:
    """Finest regular-pattern spacing whose duplicates fit spare_vertices (bisection on log spacing)."""
    def fits(s: float) -> bool:
        plan = SplitPlan(s, base.axes, base.origin, base.planes, base.creases, base.crease_tol)
        return count_duplicates(vertices, elements, plan) <= spare_vertices
    if not fits(max_spacing):
        return None
    lo, hi = np.log(min_spacing), np.log(max_spacing)
    if fits(min_spacing):
        return float(min_spacing)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if fits(float(np.exp(mid))) else (mid, hi)
    return float(np.exp(hi))


def plan_for_budget(vertices: np.ndarray, elements: np.ndarray, *, split_spacing: float | None, spare_vertices: int,
                    axes: tuple[int, ...], creases: tuple[tuple[Vec3, Vec3], ...] = (),
                    spacing_range: tuple[float, float] | None = None) -> SplitPlan:
    """The plan for an object card: creases, plus a regular pattern at ``TopologyBudget.split_spacing``. Without a
    spacing, it uses the finest spacing in ``spacing_range`` whose duplicates fit ``spare_vertices`` (default
    range: 1-20 mean edge lengths). With no fit, it returns the crease-only plan."""
    base = SplitPlan(axes=axes, creases=creases)
    if split_spacing is not None:
        return SplitPlan(split_spacing, axes, creases=creases)
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    el = np.asarray(elements, dtype=np.int64)
    edge = float(np.linalg.norm(v[el] - np.roll(v[el], 1, axis=1), axis=2).mean())
    lo, hi = spacing_range or (edge, 20.0 * edge)
    spare = spare_vertices - count_duplicates(v, el, base)
    s = spacing_for_budget(v, el, base, spare, lo, hi) if spare > 0 else None
    if s is None:
        log.info("presplit: no regular pattern fits %d spare vertices; creases only", spare_vertices)
        return base
    return SplitPlan(s, axes, creases=creases)
