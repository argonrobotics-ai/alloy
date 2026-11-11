# SPDX-License-Identifier: Apache-2.0
"""Tear lines on a folded sheet's flat layout (alloy/solvers/film/layout.py). CPU, no build.

A sheet 4 x 2 cells folded in half across its long side: the tear lines at spacing two cells are the layout's interior
grid lines more than half a spacing inside its border (u = 2 cells; w none: the sheet is only 2 cells wide), handed
to the pre-split as segments at the folded positions. A layout of the wrong length is refused; primitives have none.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.cards.types import MeshSource, Primitive
from alloy.solvers.film.layout import layout_tear_edges, layout_tear_plan, obj_layout

C = 0.01


def _folded_sheet():
    """4 x 2 cells (u along 0..4, w along 0..2), folded 90 deg along u = 2: vertices, triangles, layout."""
    uv = np.array([(i, j) for i in range(5) for j in range(3)], float) * C
    v = np.stack([np.minimum(uv[:, 0], 2 * C), uv[:, 1], np.maximum(uv[:, 0] - 2 * C, 0.0)], 1)
    idx = lambda i, j: 3 * i + j  # noqa: E731
    t = np.array([tri for i in range(4) for j in range(2) for tri in ((idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)),
                                                                      (idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)))])
    return v, t, uv


def test_tear_edges_are_the_interior_grid_lines_of_the_layout():
    v, t, uv = _folded_sheet()
    e = layout_tear_edges(t, uv, 2 * C)
    assert len(e) == 2
    assert np.allclose(uv[e][..., 0], 2 * C)                          # on u = 2 cells, both ends
    assert len(layout_tear_edges(t, uv, C)) == 3 * 2 + 1 * 4          # u = 1, 2, 3 (2 edges each), w = 1 (4 edges)
    assert len(layout_tear_edges(t, uv, 3 * C)) == 0                  # u = 3 cells would leave a 1-cell sliver
                                                                      # (under half a spacing) on the edge: dropped


def test_plan_carries_the_folded_segments():
    v, t, uv = _folded_sheet()
    plan = layout_tear_plan(v, t, uv, 2 * C)
    assert plan.spacing is None and plan.crease_tol == pytest.approx(1e-6)
    seg = np.array(plan.creases)
    assert seg.shape == (2, 2, 3)
    assert np.allclose(seg[..., 0], 2 * C) and np.allclose(seg[..., 2], 0.0)   # the fold line, in 3D
    with pytest.raises(ValueError, match="vt lines"):
        layout_tear_plan(v, t, uv[:-1], 2 * C)


def test_obj_layout_reads_vt_lines(tmp_path):
    v, t, uv = _folded_sheet()
    obj = tmp_path / "s.obj"
    obj.write_text("\n".join([f"v {a} {b} {c}" for a, b, c in v] + [f"vt {a} {b}" for a, b in uv]
                             + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in t]) + "\n")
    card = lambda shape: SimpleNamespace(card=SimpleNamespace(shape=shape))  # noqa: E731
    assert np.allclose(obj_layout(card(MeshSource(path=str(obj), sha256=""))), uv)
    assert obj_layout(card(Primitive(kind="box", size=(0.1, 0.1, 1e-4)))) is None
