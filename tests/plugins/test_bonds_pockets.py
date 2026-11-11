# SPDX-License-Identifier: Apache-2.0
"""Trapped-air pockets under a Bonds film (alloy/plugins/bonds/pockets.py, SCR-WP8-11 Part B2): exact labelling
(WP13 must-fix 4). CPU."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from alloy.plugins.bonds.pockets import label, label_rounds, mesh_edges
from alloy.solvers.film.rigs import grid_mesh


def _pockets(state: np.ndarray, edges: np.ndarray, rounds: int) -> tuple[np.ndarray, int]:
    n = len(state)
    parent = wp.zeros(n, dtype=wp.int32, device="cpu")
    bad = wp.zeros(1, dtype=wp.int32, device="cpu")
    label(wp.array(state.astype(np.int32), dtype=wp.int32, device="cpu"),
          wp.array(edges, dtype=wp.vec2i, device="cpu"), parent, bad, rounds)
    return parent.numpy(), int(bad.numpy()[0])


def _reference(state: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each free vertex's component, as its smallest member (scipy)."""
    n = len(state)
    free = state == 0
    keep = free[edges[:, 0]] & free[edges[:, 1]]
    e = edges[keep]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    root = np.full(n, -1)
    for c in np.unique(comp[free]):
        members = np.flatnonzero(free & (comp == c))
        root[members] = members.min()
    return root


@pytest.mark.parametrize("seed", range(6))
def test_pockets_are_labelled_exactly(seed):
    """Random bond masks on a 40 x 25 grid (and a serpentine pocket, the worst case for propagation): every free
    vertex's root is its component's smallest vertex, as scipy finds, and the check reports no unsettled edge."""
    v, t = grid_mesh(39, 24, 1.0)
    edges, _ = mesh_edges(t)
    rng = np.random.default_rng(seed)
    state = (rng.random(len(v)) < 0.45).astype(np.int32)       # 1 = bonded
    if seed == 0:                                              # a one-vertex-wide serpentine through the bonds
        state[:] = 1
        nx = 40
        for row in range(0, 25, 2):
            state[row * nx:(row + 1) * nx] = 0
            if row + 1 < 25:
                state[(row + 1) * nx + (nx - 1 if row % 4 == 0 else 0)] = 0
    got, bad = _pockets(state, edges, label_rounds(len(v)))
    np.testing.assert_array_equal(got, _reference(state, edges))
    assert bad == 0


def test_too_few_rounds_are_reported_not_hidden():
    """An unfinished labelling (here no hook round at all; on a GPU too few rounds) is counted by the check (non-zero)
    instead of passing a pocket split in pieces. (On the CPU one round settles the serpentine: the edges hook in
    order.)"""
    v, t = grid_mesh(39, 24, 1.0)
    edges, _ = mesh_edges(t)
    state = np.ones(len(v), np.int32)
    nx = 40
    for row in range(0, 25, 2):
        state[row * nx:(row + 1) * nx] = 0
        if row + 1 < 25:
            state[(row + 1) * nx + (nx - 1 if row % 4 == 0 else 0)] = 0
    _, bad = _pockets(state, edges, 0)
    assert bad > 0


def test_mesh_edges_and_boundary():
    v, t = grid_mesh(3, 2, 1.0)                                # 4 x 3 vertices
    edges, boundary = mesh_edges(t)
    assert len(edges) == len({tuple(e) for e in edges}) and boundary.sum() == 10   # all but the 2 inner vertices
