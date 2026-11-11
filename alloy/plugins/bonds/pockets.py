# SPDX-License-Identifier: Apache-2.0
"""Trapped-air pockets under a ``Bonds`` film (SCR-WP8-11 Part B2): labelling.

A pocket is a connected set of free (unbonded) film vertices, joined by mesh edges. One that reaches the label's edge
(a boundary vertex of the mesh) is open: it vents to the air. One wholly surrounded by bonds is enclosed: it holds
its air. GA scores the area of enclosed pockets (> 20 mm² fails). At the 2 mm cells the label stock needs (scr/WP8.md
"B2 resolution check") GA's 3 mm² barcode criterion is not scored: a pocket that small is under one cell.

**Exact labelling (WP13 must-fix 4).** Every free vertex starts as its own root. Each round hooks the larger of two
joined roots to the smaller (atomic min on the root's parent), then halves every path (parent <- parent's parent).
After ``rounds`` rounds a check counts the free edges whose ends still disagree; a non-zero count is reported, never
hidden (a pocket split in two could pass the 20 mm² criterion falsely). The root of a pocket is its smallest vertex
index, so a pocket keeps its root from step to step unless it splits or merges.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

FREE = wp.constant(0)
NONE = wp.constant(-1)


@wp.kernel
def label_init(state: wp.array[wp.int32], parent: wp.array[wp.int32]):
    i = wp.tid()
    parent[i] = NONE
    if state[i] == FREE:
        parent[i] = i


@wp.kernel
def label_hook(edges: wp.array[wp.vec2i], parent: wp.array[wp.int32]):
    """Joins the roots of an edge's two free ends: the larger root's parent becomes the smaller (atomic min)."""
    e = wp.tid()
    u = edges[e][0]
    v = edges[e][1]
    pu = parent[u]
    pv = parent[v]
    if pu < 0 or pv < 0 or pu == pv:
        return
    ru = parent[pu]
    rv = parent[pv]
    if ru < rv:
        wp.atomic_min(parent, rv, ru)
    elif rv < ru:
        wp.atomic_min(parent, ru, rv)


@wp.kernel
def label_jump(parent: wp.array[wp.int32]):
    """Path halving: every free vertex points to its parent's parent."""
    i = wp.tid()
    p = parent[i]
    if p >= 0:
        parent[i] = parent[p]


@wp.kernel
def label_check(edges: wp.array[wp.vec2i], world: wp.array[wp.int32], parent: wp.array[wp.int32],
                bad: wp.array[wp.int32]):
    """Counts, per world, free edges whose ends have different roots or do not point at their root (unfinished)."""
    e = wp.tid()
    u = edges[e][0]
    v = edges[e][1]
    pu = parent[u]
    pv = parent[v]
    if pu >= 0 and pv >= 0:
        if pu != pv or parent[pu] != pu:
            wp.atomic_add(bad, world[u], 1)


def label_rounds(vertices: int) -> int:
    """Hook-and-halve rounds that settle any pocket of a mesh with ``vertices`` vertices (2·log2(N) + 4)."""
    return 2 * int(math.ceil(math.log2(max(vertices, 2)))) + 4


def label(state: wp.array, edges: wp.array, parent: wp.array, bad: wp.array, rounds: int,
          world: wp.array | None = None) -> None:
    """Pocket roots into ``parent`` (-1 for bonded or held vertices); ``bad[w]`` world w's unsettled edge count
    (``world``: each edge's world, default all 0)."""
    dev = parent.device
    n, m = parent.shape[0], edges.shape[0]
    if world is None:
        world = wp.zeros(m, dtype=wp.int32, device=dev)
    wp.launch(label_init, dim=n, inputs=[state], outputs=[parent], device=dev)
    for _ in range(rounds):
        wp.launch(label_hook, dim=m, inputs=[edges], outputs=[parent], device=dev)
        wp.launch(label_jump, dim=n, inputs=[parent], device=dev)
        wp.launch(label_jump, dim=n, inputs=[parent], device=dev)
    bad.zero_()
    wp.launch(label_check, dim=m, inputs=[edges, world, parent], outputs=[bad], device=dev)


def throat_coefficients(x: np.ndarray, tris: np.ndarray, spacing: np.ndarray, viscosity: float) -> np.ndarray:
    """Per vertex, w_b/(12·mu·s_b) [1/(Pa·s)] on the label's edge, else 0: a pocket reaching the edge vents through
    a laminar slot of the throat's width w_b (half of each boundary edge at the vertex), one cell long (``spacing``
    s_b), and the gap h there: g = w_b·h³/(12·mu·s_b)."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    edge = uniq[count == 1]
    length = np.linalg.norm(x[edge[:, 0]] - x[edge[:, 1]], axis=1)
    w = np.zeros(len(x))
    np.add.at(w, edge[:, 0], 0.5 * length)
    np.add.at(w, edge[:, 1], 0.5 * length)
    return np.where(w > 0.0, w / (12.0 * viscosity * np.maximum(spacing, 1e-12)), 0.0)


def mesh_edges(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unique edges [M, 2], boundary flag per vertex): a boundary edge belongs to one triangle only."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    boundary = np.zeros(int(tris.max()) + 1 if tris.size else 0, bool)
    boundary[uniq[count == 1].ravel()] = True
    return uniq.astype(np.int32), boundary


__all__ = ["label", "label_rounds", "mesh_edges", "throat_coefficients"]
