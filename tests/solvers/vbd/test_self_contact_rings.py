# SPDX-License-Identifier: Apache-2.0
"""``make_vbd`` with particle self contact skips only one ring of topological neighbours (laundry CT-13: a squeezed
cloth ridge crossed itself under Newton's two-ring default); an explicit kwarg wins; without self contact nothing
changes."""

from __future__ import annotations

from alloy.solvers.vbd import SELF_CONTACT_FILTER_RINGS, make_vbd
from alloy.solvers.vbd import scenes as S


def _solver(**kw):
    sc = S.build("stack")
    return make_vbd(sc.model, {**S.solver_kwargs(sc, 2), **kw})


def test_self_contact_filters_one_ring():
    s = _solver(particle_enable_self_contact=True)
    assert SELF_CONTACT_FILTER_RINGS == 1
    assert s.particle_topological_contact_filter_threshold == 1


def test_an_explicit_threshold_wins_and_no_self_contact_keeps_newtons_default():
    assert _solver(particle_enable_self_contact=True,
                   particle_topological_contact_filter_threshold=2).particle_topological_contact_filter_threshold == 2
    assert _solver(particle_enable_self_contact=False).particle_topological_contact_filter_threshold == 2


def test_c7_self_contact_excludes_nothing_at_rest():
    """C7's VBD settings (``fold_scene.VbdSettings``) filter one topological ring and no rest-shape distance: the
    1.5-cell exclusion they had hid every pair within 12.9 mm of the flat towel, the crease's among them (C7's towel
    card on the laundry pinch rig: 5 crossings with it, 0 without; docs/notes/laundry-results.md)."""
    from alloy.clips import fold_scene as fs

    kw = fs.VbdSettings().kwargs(0.0086, 1700)
    assert kw["particle_topological_contact_filter_threshold"] == 1
    assert kw.get("particle_rest_shape_contact_exclusion_radius", 0.0) == 0.0
