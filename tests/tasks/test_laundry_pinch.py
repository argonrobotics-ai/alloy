# SPDX-License-Identifier: Apache-2.0
"""CT-13 under C7's grip (``alloy.tasks.laundry_pinch``): two kinematic pad pairs gather a ridge near a towel's long
edge, squeeze it and lift it round the fold line. With the route's self contact (``make_vbd``: one topological ring)
nothing crosses. The control turns self contact off on the same squeeze, so the crossing detector must fire."""

from __future__ import annotations

import pytest

from alloy.tasks.laundry_pinch import PinchRig, run

pytestmark = [pytest.mark.slow, pytest.mark.law]

FAST = (0.1, 0.2, 0.3, 0.1, 0.1, 0.1)          # s: down, close, swing, open, up, settle (a fast pinch)


@pytest.mark.timeout(3000)
def test_a_squeezed_ridge_does_not_cross_itself():
    """Rows (CPU, towel_cotton): fast pinch 0 crossings in every frame; at C7's speed on C7's towel card (route self
    contact, friction_epsilon 1e-3) 0 crossings and the ridge carried ([internal note] notes/wp11-laundry/pinch_v*.jsonl)."""
    r = run(PinchRig(phases=FAST))
    assert r.finite and r.crossings_max == 0, r.frames


@pytest.mark.timeout(1800)
def test_without_self_contact_the_ridge_crosses():
    """The control: the same fast pinch with self contact off crosses (1,508 crossings at most, CPU, on the Lame
    membrane map 4d27e9e83), so the law above can see a crossing. Until 2026-11-08 the control was Newton's two rings
    with the linear body-particle penalty: 3 crossings at the lift on the old membrane map (E h / (1 - nu^2), lambda 0),
    0 on the Lame map, which is 2.2x softer in stretch (WP2, coordinator ruling 08:40Z (a)). The ring filter itself is
    chosen by the self-contact ladder (ruling (b), [internal note] notes/wp11-sc/)."""
    r = run(PinchRig(phases=FAST, vbd={"particle_enable_self_contact": False}))
    assert r.finite and r.crossings_max > 0, r.frames
