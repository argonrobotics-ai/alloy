# SPDX-License-Identifier: Apache-2.0
"""Tear lines for a folded film, placed on the paper as cut (docs/notes/paper-pack.md).

A sheet folded around something (a wrapped present) cannot take the topology pre-split's regular pattern: that cuts
along planes in the mesh frame, which on a folded sheet cross the layers. When a card's OBJ carries the sheet's flat
layout as ``vt`` lines (one per ``v``, in order: the stock reader keeps the vertex order), the tear lines are the mesh
edges that lie on the layout's grid lines at the card's ``topology.split_spacing``. They reach the pre-split as 3D
segments on the folded mesh (``SplitPlan.creases``: an edge with both ends on a segment becomes a seam), so the split
follows the paper's straight lines wherever they are folded. Layers stand at least a pitch apart, far beyond the
1 um match tolerance.
"""

from __future__ import annotations

import numpy as np


def obj_layout(obj) -> np.ndarray | None:
    """[N, 2] flat layout of a card mesh (its OBJ ``vt`` lines), or None for primitives and meshes without one."""
    from alloy.cards.types import MeshSource
    from alloy.scene.emitters.common import _mesh_file

    shape = obj.card.shape
    if not isinstance(shape, MeshSource) or not str(shape.path).endswith(".obj"):
        return None
    uv = [[float(p) for p in line.split()[1:3]] for line in _mesh_file(shape, None).read_text().splitlines()
          if line.startswith("vt ")]
    return np.asarray(uv, np.float64) if uv else None


def layout_tear_edges(tris: np.ndarray, layout: np.ndarray, spacing: float, tol: float = 1.0e-7) -> np.ndarray:
    """[K, 2] mesh edges on the layout's grid lines u = k·spacing or w = k·spacing more than half a spacing inside the
    layout's border: no piece on the paper's edge is a sliver narrower than that (on the wrapped present an 11 mm
    strip along an end flap's free edge, held only by its seams, peeled off its stack in some VBD colourings)."""
    t = np.asarray(tris, np.int64).reshape(-1, 3)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    lo, hi = layout.min(0) + 0.5 * spacing, layout.max(0) - 0.5 * spacing
    keep = np.zeros(len(e), bool)
    for k in range(2):
        a, b = layout[e[:, 0], k], layout[e[:, 1], k]
        q = a / spacing
        on = (np.abs(a - b) < tol) & (np.abs(q - np.round(q)) * spacing < tol)
        keep |= on & (a > lo[k] + tol) & (a < hi[k] - tol)
    return e[keep]


def layout_tear_plan(vertices: np.ndarray, tris: np.ndarray, layout: np.ndarray, spacing: float):
    """``SplitPlan`` whose seams are the layout's tear edges, as segments on the folded ``vertices``."""
    from alloy.topology.presplit import SplitPlan

    if len(layout) != len(vertices):
        raise ValueError(f"film layout: {len(layout)} vt lines for {len(vertices)} vertices (one per vertex, in order)")
    v = np.asarray(vertices, np.float64).reshape(-1, 3)
    edges = layout_tear_edges(tris, layout, spacing)
    segs = tuple((tuple(map(float, v[a])), tuple(map(float, v[b]))) for a, b in edges)
    return SplitPlan(spacing=None, axes=(0, 1, 2), creases=segs, crease_tol=1.0e-6)
