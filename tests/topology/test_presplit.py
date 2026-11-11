# SPDX-License-Identifier: Apache-2.0
"""presplit.py: seam duplication, twin bonds, tributary measures, determinism (pure numpy, no Newton)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import AssemblyError
from alloy.testing.topology import grid_tet_mesh, grid_tri_mesh, tet_volumes, tri_areas
from alloy.topology.presplit import SplitPlan, count_duplicates, presplit_mesh, spacing_for_budget


def _facet_sets(elements: np.ndarray) -> dict[tuple[int, ...], list[int]]:
    k = elements.shape[1]
    out: dict[tuple[int, ...], list[int]] = {}
    for e, el in enumerate(elements):
        for drop in range(k):
            key = tuple(sorted(np.delete(el, drop).tolist()))
            out.setdefault(key, []).append(e)
    return out


def test_vertical_seam_on_tri_grid():
    v, t = grid_tri_mesh(4, 2, 1.0)
    m = presplit_mesh(v, t, SplitPlan(spacing=2.0, axes=(0,)))
    # one column of 3 vertices at x = 2 is duplicated; tributary lengths 0.5 + 1 + 0.5 = seam length 2
    assert m.n_added == 3
    assert m.twins.shape == (3, 2)
    np.testing.assert_allclose(sorted(m.measure), [0.5, 0.5, 1.0])
    np.testing.assert_allclose(m.normal, np.tile([1.0, 0.0, 0.0], (3, 1)), atol=1e-12)
    np.testing.assert_array_equal(m.source[m.elements], t)          # every element keeps its original corners
    np.testing.assert_array_equal(m.source[m.twins[:, 0]], m.source[m.twins[:, 1]])


def test_grid_pattern_cross_seams_tri():
    v, t = grid_tri_mesh(6, 6, 1.0)
    m = presplit_mesh(v, t, SplitPlan(spacing=3.0, axes=(0, 1)))
    # seams x = 3 and y = 3: 7 + 7 vertices on the lines, centre vertex has 4 regions (3 extra) -> 6 + 6 + 3 = 15
    assert m.n_added == 15
    assert m.measure.sum() == pytest.approx(12.0)                    # two seams of length 6, each split in halves
    centre = 3 * 7 + 3
    copies = np.flatnonzero(m.source == centre)
    assert len(copies) == 4
    # bonds at the centre join adjacent regions only (4-cycle, no diagonals)
    at_centre = [tuple(b) for b in m.twins if m.source[b[0]] == centre]
    assert len(at_centre) == 4


def test_area_and_volume_preserved():
    v, t = grid_tri_mesh(5, 3, 0.01)
    m = presplit_mesh(v, t, SplitPlan(spacing=0.02, axes=(0, 1)))
    assert tri_areas(m.vertices, m.elements).sum() == pytest.approx(tri_areas(v, t).sum(), rel=1e-12)
    vt, tt = grid_tet_mesh(3, 3, 3, 0.01)
    mt = presplit_mesh(vt, tt, SplitPlan(spacing=0.015, axes=(0, 1, 2)))
    assert tet_volumes(mt.vertices, mt.elements).sum() == pytest.approx(tet_volumes(vt, tt).sum(), rel=1e-12)
    assert (tet_volumes(mt.vertices, mt.elements) > 0).all()


def test_tet_plane_seam_measure_and_topology():
    v, t = grid_tet_mesh(4, 2, 2, 0.01)
    m = presplit_mesh(v, t, SplitPlan(planes=(((1.0, 0.0, 0.0), 0.02),)))
    on_plane = np.isclose(v[:, 0], 0.02)
    assert m.n_added == int(on_plane.sum())                            # 3 x 3 vertices on x = 0.02
    assert m.measure.sum() == pytest.approx(0.02 * 0.02)              # seam area, faces split over 3 vertices
    np.testing.assert_allclose(m.normal[:, 0], 1.0, atol=1e-12)
    facets = _facet_sets(m.elements)
    # after the split no facet on the plane is shared: the two sides are disconnected
    shared_on_plane = [f for f, es in facets.items() if len(es) == 2 and np.allclose(m.vertices[list(f), 0], 0.02)]
    assert shared_on_plane == []


def test_staircase_seam_is_manifold_and_bonds_touch():
    v, t = grid_tet_mesh(3, 3, 3, 0.01)
    m = presplit_mesh(v, t, SplitPlan(planes=(((1.0, 1.0, 0.3), 0.025),)))
    assert m.n_added > 0
    for a, b in m.twins:
        assert m.source[a] == m.source[b]
        assert np.array_equal(m.vertices[a], m.vertices[b])
    facets = _facet_sets(m.elements)
    assert max(len(es) for es in facets.values()) <= 2


def test_crease_seam_and_end_vertex_not_duplicated():
    v, t = grid_tri_mesh(4, 4, 1.0)
    # crease along x = 2 from y = 0 to y = 2 (stops inside the mesh): the end vertex (2, 2) stays shared
    m = presplit_mesh(v, t, SplitPlan(creases=(((2.0, 0.0, 0.0), (2.0, 2.0, 0.0)),)))
    dup_sources = sorted(set(m.source[m.n_original:].tolist()))
    assert dup_sources == [2, 7]                                       # (2,0) and (2,1); (2,2) is the seam end
    assert m.measure.sum() == pytest.approx(1.5)                       # edges (2,0)-(2,1) and half of (2,1)-(2,2)


def test_deterministic_and_empty_plan():
    v, t = grid_tet_mesh(2, 2, 2, 0.01)
    plan = SplitPlan(spacing=0.01, axes=(0, 2))
    a, b = presplit_mesh(v, t, plan), presplit_mesh(v, t, plan)
    for f in ("vertices", "elements", "source", "twins", "normal", "measure"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    e = presplit_mesh(v, t, SplitPlan())
    assert e.n_added == 0 and e.twins.shape == (0, 2)
    np.testing.assert_array_equal(e.elements, t)


def test_budget_spacing_search():
    v, t = grid_tet_mesh(4, 4, 4, 0.01)
    base = SplitPlan(axes=(0,))
    s = spacing_for_budget(v, t, base, spare_vertices=50, min_spacing=0.005, max_spacing=0.03)
    assert s is not None
    assert count_duplicates(v, t, SplitPlan(spacing=s, axes=(0,))) <= 50
    assert spacing_for_budget(v, t, base, spare_vertices=0, min_spacing=0.005, max_spacing=0.03) is None


def test_rejects_bad_input():
    v, t = grid_tri_mesh(2, 2, 1.0)
    with pytest.raises(AssemblyError):
        presplit_mesh(v, t + 100, SplitPlan(spacing=1.0))
    with pytest.raises(AssemblyError):
        presplit_mesh(v, t, SplitPlan(spacing=-1.0))
    vt, tt = grid_tet_mesh(1, 1, 1, 1.0)
    with pytest.raises(AssemblyError):
        presplit_mesh(vt, tt, SplitPlan(creases=(((0, 0, 0), (1, 0, 0)),)))


def test_plan_for_budget_uses_card_spacing_or_fits_the_budget():
    from alloy.topology.presplit import plan_for_budget

    v, t = grid_tri_mesh(12, 6, 1.0)
    p = plan_for_budget(v, t, split_spacing=3.0, spare_vertices=0, axes=(0, 1))
    assert p.spacing == 3.0
    fit = plan_for_budget(v, t, split_spacing=None, spare_vertices=20, axes=(0,))
    assert fit.spacing is not None and count_duplicates(v, t, fit) <= 20
    none = plan_for_budget(v, t, split_spacing=None, spare_vertices=0, axes=(0,))
    assert none.spacing is None
