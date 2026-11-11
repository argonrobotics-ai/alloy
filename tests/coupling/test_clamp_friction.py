# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-6 (CPU): a closed two-pad clamp on a cloth double layer (top and bottom edges fixed), then the carriage
ramps down (drag) or up (lift) by 0.5 mm or 3 mm in 0.15 s with the pads closed
(``alloy.coupling.pinch_rigs.run_drag``).
Coulomb limit: mu N = 2 faces x 0.42 x 15 N = 12.6 N. The clamp carries the cloth below it and slides at it, within
20 % (ruling 2026-11-04 20:23Z). The force is the cloth's on the pads (coupling f_used): absolute, not a change from
the hold, because the clamped cloth already pulls the pads ~5 N down at the hold (a drag/lift bias).

VBD's friction is smoothed Coulomb (mu f.n f1(v / friction_epsilon), never above mu f.n), so a held cloth creeps below
1 mm/s and a settled hold force decays; the ramps (3-20 mm/s) are well above that."""

from __future__ import annotations

import pytest

from alloy.coupling.pinch_rigs import run_drag
from alloy.coupling.types import ClockPolicy, SchedulerPolicy

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]

MU_N = 2.0 * 0.42 * 15.0
# The exchange step the numbers below hold at: 1/60 s / 32 = 0.52 ms from the start. Unpinned, the rig starts at
# 1.04 ms, where drag 3 mm peaks at 19.6-20.9 N (1.6 mu N) with the coordinate carry on or off. It used to pass only
# because float32 joint freezing booked a false interface-energy gain while the pads closed (windowed 0.44, limit 0.05)
# and the watchdog halved the step at ~0.23 s. With the carry in the step's FK that gain is 0.017, so no halving
# (WP3 trace 2026-11-06, [internal note] notes/wp3-infk).
CLOCK = SchedulerPolicy(clock=ClockPolicy(dt_x_max=6.0e-4))


def test_flat_edges_of_the_box_pads_are_found():
    """Newton's full-surface edge table holds every triangulation edge. The box pads' face diagonals (6 of 18 per
    pad) are flat; their edge-edge records are muted, while rim edges keep theirs."""
    from alloy.coupling.pinch_rigs import _cloth_scheduler, gripper_model

    sched = _cloth_scheduler(gripper_model(), True)
    mutes = [m for fac in sched._contact_factories() for m in fac.flat_edges if m is not None and m.flags is not None]
    assert len(mutes) == 1
    flags = mutes[0].flags.numpy()
    assert flags.size == 36 and int(flags.sum()) == 12


@pytest.mark.parametrize("travel", [0.0005, -0.0005, 0.003, -0.003], ids=["drag-0.5", "lift-0.5", "drag-3", "lift-3"])
def test_clamp_carries_below_mu_n_and_slides_at_it(travel):
    """Measured on Newton 65b7c92e (per exchange): drag 0.5 mm carries the cloth 0.43 mm at 5.5 N; lift 0.5 mm, drag
    3 mm and lift 3 mm slide at a 12.61-12.66 N peak (12.30-12.56 N while sliding). Before the corner Hessian fix and
    the flat-edge mute the clamp bounced (gap 3.2-5.3 mm) and slid at 2.5-3.1 N."""
    r = run_drag(travel, policy=CLOCK)
    assert r.finite and r.pass_through == 0
    assert r.gap_spread < 1.0e-4, r                          # a steady clamp (the bounce was mm)
    assert r.peak <= 1.2 * MU_N, r                           # never more than Coulomb
    carried = r.follow >= 0.8 * abs(travel)
    assert carried or r.peak >= 0.8 * MU_N, r                # it lets go only at mu N
    if abs(travel) >= 0.003:                                 # 3 mm pulls past mu N: it slides, at mu N
        assert not carried and 0.8 * MU_N <= r.sliding <= 1.2 * MU_N, r
