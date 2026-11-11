# SPDX-License-Identifier: Apache-2.0
"""Pad drag law (SCR-WP11-6's full-surface "against" drag, coordinator 2026-11-08 07:56Z; SCR-SC4-4's edge/face
records), on ``alloy.coupling.pad_drag``: a pad pressed onto cloth on a frictionless table drags the cloth its own
way. By Coulomb nothing but the cloth's inertia holds it back, so the cloth under a lone pad follows the pad.

Per-vertex and full-surface (edge/face) pad contacts, plain SolverVBD (fast) and the lagged MuJoCo -> VBD Scheduler
(slow), at the towel's 10 VBD iterations. Pass: the cloth under the pad moves >= 95 % (plain) / 90 % (coupled) of
the pad's travel, in the pad's direction; no cloth triangle inside a pad.

With two pads on the cloth the friction is not converged at 10 iterations and the whole cloth drifts toward one pad
(WP11's 3.9 mm "against" the pad); that is SCR-WP11-6 step 2's convergence law (WP11), not checked here.
Per-vertex contacts gather that cloth only by pushing cloth triangles into the pads, so they are no reference;
full-surface keeps it out (``test_two_pad_pinch_keeps_cloth_out_of_full_surface_pads``). Rows:
docs/notes/sc4-pad-drag.md.
"""

from __future__ import annotations

import pytest

from alloy.coupling import pad_drag as D


@pytest.mark.parametrize("full_surface", [False, True], ids=["per_vertex", "full_surface"])
def test_plain_pad_drags_cloth_its_way(full_surface):
    """Plain SolverVBD, a kinematic pad slid 7.5 mm at 37.5 mm/s (1 ms steps)."""
    r = D.run_plain_drag(full_surface)
    assert r.finite and r.pad_travel > 0.007, r
    assert r.under_travel >= 0.95 * r.pad_travel, r
    assert r.tris_in_pads == 0, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("full_surface", [False, True], ids=["per_vertex", "full_surface"])
def test_lone_pad_drags_cloth_its_way_coupled(full_surface):
    """WP11's pinch-from-flat rig with the jaw open 100 mm: pad b stays on the bare table, pad a closes 9.5 mm over
    the cloth through the lagged Scheduler. Nothing else touches the cloth, so the whole cloth follows pad a."""
    r = D.run_flat_pinch(full_surface, D.FlatPinch(lone=True))
    assert r.finite and r.pad_travel > 0.008, r
    assert r.under_travel >= 0.90 * r.pad_travel, r
    assert r.cloth_travel >= 0.90 * r.pad_travel, r
    assert r.tris_in_pads == 0, r


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
def test_two_pad_pinch_keeps_cloth_out_of_full_surface_pads():
    """Both pads on C7's towel (8.5 mm cells), centred on the jaw, closing 9 mm each: full-surface contacts keep
    every cloth triangle out of the pads. (Per-vertex contacts let 6 triangles in, 1.5 mm deep: the ridge they raise
    is cloth folded into the pads' edges; a row in docs/notes/sc4-pad-drag.md, not a law, WP4's review.)"""
    full = D.run_flat_pinch(True)
    assert full.finite and full.tris_in_pads == 0, full
