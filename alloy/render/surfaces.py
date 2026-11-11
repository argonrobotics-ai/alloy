# SPDX-License-Identifier: Apache-2.0
"""Surface extraction for ARB bundles: triangle/tet components, tet boundaries, cut twins (numpy only)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

log = logging.getLogger("alloy.render")

TAG_OUTER, TAG_CUT = 0, 1


@dataclass
class SurfacePlan:
    """One exported surface: local faces, source particle ids, per-face tags and twin vertex pairs."""

    name: str
    kind: str
    world: int
    faces: np.ndarray                  # [F, 3] int32, local vertex ids
    src: np.ndarray                    # [V] int32, model particle ids
    face_tag: np.ndarray               # [F] uint8
    twins: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int32))


def element_components(elements: np.ndarray, n_particles: int,
                       links: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Label each particle with its connected component over ``elements`` [E, k] plus extra ``links`` [P, 2].

    Twin bonds are passed as links, so the pieces of a pre-split object (cut along twins) stay one component.
    """
    k = elements.shape[1]
    rows = np.repeat(elements[:, 0], k - 1)
    cols = elements[:, 1:].reshape(-1)
    if links is not None and len(links):
        rows = np.concatenate([rows, links[:, 0]])
        cols = np.concatenate([cols, links[:, 1]])
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_particles, n_particles))
    count, labels = connected_components(g, directed=False)
    return labels, count


def tet_boundary_faces(tets: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Outward-oriented faces [F, 3] that belong to exactly one tet."""
    combos = ((0, 2, 1, 3), (0, 1, 3, 2), (1, 2, 3, 0), (0, 3, 2, 1))   # face (a, b, c) and its opposite vertex
    faces = np.concatenate([tets[:, list(c[:3])] for c in combos])
    opposite = np.concatenate([tets[:, c[3]] for c in combos])
    key = np.sort(faces, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    boundary = counts[inv.reshape(-1)] == 1
    faces, opposite = faces[boundary], opposite[boundary]
    p = pos[faces].astype(np.float64)
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = ((pos[opposite] - p[:, 0]) * normal).sum(axis=1) > 0.0
    faces[inward] = faces[inward][:, ::-1]
    return faces.astype(np.int32)


def localize(faces_global: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (local faces, src particle ids) with vertices numbered in ascending particle id."""
    src, inv = np.unique(faces_global, return_inverse=True)
    return inv.reshape(faces_global.shape).astype(np.int32), src.astype(np.int32)


def twin_data(pairs: np.ndarray, src: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Twin pairs [P, 2] in local ids, and a per-face tag: 1 when every vertex of the face has a twin."""
    tags = np.zeros(len(faces), np.uint8)
    if len(pairs) == 0 or len(src) == 0:
        return np.zeros((0, 2), np.int32), tags
    idx = np.searchsorted(src, pairs)
    idx = np.clip(idx, 0, len(src) - 1)
    ok = (src[idx] == pairs).all(axis=1)
    twins = idx[ok].astype(np.int32)
    twinned = np.zeros(len(src), bool)
    twinned[twins.reshape(-1)] = True
    tags[twinned[faces].all(axis=1)] = TAG_CUT
    return twins.reshape(-1, 2), tags


def make_surface(name: str, kind: str, world: int, faces_global: np.ndarray, twin_pairs: np.ndarray) -> SurfacePlan:
    """Build a SurfacePlan from faces given in model particle ids."""
    faces, src = localize(np.asarray(faces_global, np.int64))
    twins, tags = twin_data(twin_pairs, src, faces)
    return SurfacePlan(name, kind, world, faces, src, tags, twins)


def read_twin_pairs(model) -> np.ndarray:
    """Particle pairs [P, 2] of CLOTH_EDGE / TET_CUT bonds, or empty for models without Alloy bonds."""
    empty = np.zeros((0, 2), np.int64)
    try:
        from alloy.topology.bonds import bond_column, has_bonds
        from alloy.topology.types import BondKind

        if not has_bonds(model):
            return empty
        kind = bond_column(model, "kind").numpy()
        keep = np.isin(kind, (int(BondKind.CLOTH_EDGE), int(BondKind.TET_CUT)))
        pa = bond_column(model, "particle_a").numpy()[keep]
        pb = bond_column(model, "particle_b").numpy()[keep]
    except Exception as exc:  # plain Newton models, or a topology layout we do not know
        log.debug("no twin bonds read: %s", exc)
        return empty
    valid = (pa >= 0) & (pb >= 0)
    return np.stack([pa[valid], pb[valid]], axis=1).astype(np.int64)


def detect_surfaces(model, world_of: np.ndarray, exported: set[int]) -> tuple[list[SurfacePlan], np.ndarray]:
    """Auto-detect cloth_<k> / soft_<k> surfaces. Returns (plans, covered-particle mask)."""
    n = len(world_of)
    covered = np.zeros(n, bool)
    plans: list[SurfacePlan] = []
    if n == 0:
        return plans, covered
    twin_pairs = read_twin_pairs(model)
    pos = model.particle_q.numpy()
    tets = _int_array(getattr(model, "tet_indices", None), 4)
    tris = _int_array(getattr(model, "tri_indices", None), 3)
    soft_nodes = np.zeros(n, bool)
    if len(tets):
        _add_components(plans, covered, soft_nodes, tets, "soft", world_of, exported, twin_pairs, pos)
    if len(tris):
        tris = tris[~soft_nodes[tris].all(axis=1)]
    if len(tris):
        _add_components(plans, covered, None, tris, "cloth", world_of, exported, twin_pairs, pos)
    return plans, covered


def _int_array(arr, width: int) -> np.ndarray:
    if arr is None:
        return np.zeros((0, width), np.int64)
    a = arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)
    return a.reshape(-1, width).astype(np.int64)


def _add_components(plans, covered, node_mask, elems, kind, world_of, exported, twin_pairs, pos) -> None:
    labels, count = element_components(elems, len(world_of), twin_pairs)
    elem_comp = labels[elems[:, 0]]
    k = 0
    for comp in range(count):
        sel = elems[elem_comp == comp]
        if len(sel) == 0:
            continue
        world = int(world_of[sel[0, 0]])
        if world >= 0 and world not in exported:
            continue
        faces = tet_boundary_faces(sel, pos) if kind == "soft" else sel
        covered[np.unique(sel)] = True
        if node_mask is not None:
            node_mask[np.unique(sel)] = True
        plans.append(make_surface(f"{kind}_{k}", kind, world, faces, twin_pairs))
        k += 1
