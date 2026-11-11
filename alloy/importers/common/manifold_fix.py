# SPDX-License-Identifier: Apache-2.0
"""Repair pinched (non-manifold) edges left by quadric simplification, in plain numpy.

Quadric edge collapse can fold two sheets onto one edge (an edge shared by four faces). ``fix_pinches`` removes the
faces around such edges and closes each resulting hole with a fan around its centroid, oriented against the hole's
boundary so the winding stays consistent. Repeated a few rounds, it turns a closed-but-pinched mesh into one that
passes Alloy's strict check (``alloy.cards.geometry.is_watertight``: every edge in exactly two faces, consistent
winding). It never touches a mesh that is already clean.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("alloy.importers")


def edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges ``(E, 2)`` and how many faces use each.

    Example::

        edges, counts = edge_counts(np.array([[0, 1, 2], [0, 2, 3]]))
    """
    und = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(und, axis=0, return_counts=True)


def is_clean(faces: np.ndarray) -> bool:
    """Every edge in exactly two faces and no directed edge used twice (closed, consistently wound).

    Example::

        is_clean(tetra_faces)  # True
    """
    if len(faces) == 0:
        return False
    _, counts = edge_counts(faces)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, dcounts = np.unique(directed, axis=0, return_counts=True)
    return bool((counts == 2).all() and (dcounts == 1).all())


def _boundary_loops(faces: np.ndarray) -> list[list[int]] | None:
    """Boundary loops as vertex lists in face winding order; None if a boundary vertex branches."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    _, inv, counts = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    border = directed[counts[inv.reshape(-1)] == 1]
    nxt: dict[int, int] = {}
    for a, b in border.tolist():
        if a in nxt:
            return None
        nxt[a] = b
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt.get(v, start)
        loops.append(loop)
    return loops


def _fill(vertices: np.ndarray, faces: np.ndarray, loops: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    new_v, new_f = [vertices], [faces]
    base = len(vertices)
    for loop in loops:
        c = base
        base += 1
        new_v.append(vertices[loop].mean(axis=0, keepdims=True))
        ring = np.asarray(loop + loop[:1])
        new_f.append(np.stack([ring[1:], ring[:-1], np.full(len(loop), c)], axis=1))
    return np.concatenate(new_v), np.concatenate(new_f)


def _cut_pinches(faces: np.ndarray) -> np.ndarray:
    edges, counts = edge_counts(faces)
    bad_v = np.unique(edges[counts > 2])
    return faces[~np.isin(faces, bad_v).any(axis=1)]


def fix_pinches(vertices: np.ndarray, faces: np.ndarray, rounds: int = 4) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(vertices, faces, ok)``: pinched edges cut out and holes fan-filled until ``is_clean`` (or rounds run out).

    Example::

        v, f, ok = fix_pinches(v, f)
    """
    v, f = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    for k in range(rounds):
        if is_clean(f):
            return v, f, True
        f = _cut_pinches(f)
        loops = _boundary_loops(f)
        if loops is None or not f.size:
            log.debug("fix_pinches: round %d left a branching boundary", k)
            return v, f, False
        v, f = _fill(v, f, loops)
    return v, f, is_clean(f)
