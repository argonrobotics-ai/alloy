# SPDX-License-Identifier: Apache-2.0
"""CT-13 (SPEC: cloth vs cloth and rigid under VBD DAT, 0 crossings) on folds, Gym-Anything laundry gap (coordinator
2026-11-06 18:40Z): ``alloy.tasks.laundry_crossings`` carries one long edge of a towel card (cloth route, features.
self_contact: WP2's VBD self contact) round a fold line onto the rest and lets go; self crossings are counted every
0.05 s (``fold_checks.self_intersections``). The bath towel's half and thirds folds through the fold task
(``tests/tasks/test_laundry_fold.py``) check the same at 0.70 x 0.45 m with two corner grasps."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.laundry_crossings import FoldRig, run

pytestmark = [pytest.mark.slow, pytest.mark.law]


@pytest.mark.timeout(7200)
def test_ct13_a_towel_folded_in_half_never_crosses_itself():
    """Rows (CPU, towel_cotton 0.30 x 0.20 m, h 0.6 mm, 1,700 particles, route dt 1.25 ms): 0 crossings in every
    counted frame, also at a 12 mm pinch inset and swing heights 0.3-0.95 of the fold radius, and on C7's towel card
    with the route's and C7's own self-contact settings ([internal note] notes/wp11-laundry/ct13_v*.jsonl)."""
    r = run(FoldRig(folds="half", apex=0.5))
    assert r.finite and r.crossings_max == 0, r.frames
    assert r.table_pen < 0.5 * 6.0e-4, r.table_pen              # the bottom layer stays on the table top
    assert r.y_span < 0.55 * r.y_span0, (r.y_span, r.y_span0)  # folded, not turned over


@pytest.mark.timeout(900)
def test_ct13_without_self_contact_the_count_sees_the_layers_cross():
    """The control: the same towel without features.self_contact folds through itself (901 crossings at 1 s)."""
    r = run(FoldRig(folds="half", self_contact=False, fold_s=0.6, settle_s=0.2))
    assert r.finite and r.crossings_max > 50, r.frames
    assert np.isfinite(r.crossings_end)
