# SPDX-License-Identifier: Apache-2.0
"""Rungs follow a finer dt_x (speed-now S2, opt-in ``ClockPolicy.rungs_follow_dt_x``): when dt_x gets finer for any
reason, each adaptive rung goes to the larger of what its estimate wants and the rung that keeps its own step no
coarser than before, as it already does through a contact-gate jump.

The numbers are a settled drape (a Panda at rest with a draped towel, contact gate + graph-budget rungs, 8 worlds,
seeds 300-307, one L4 run that failed 0 of 8; [internal note] scr/WP10.md "S2 proposal"). The pair bound holds dt_x at
78 us. In period 9 VBD's stable step fell to 76 us (4 substeps at 78 us), the contact band then took dt_x to 19.5 us,
and the rung stayed at 4 for 9 periods: 4.9 us steps where the estimate asked for 19.5 us. Under Chebyshev rho 0.9,
measured at 0.625 ms steps, the towels were thrown.

These laws pin the clock rule only. The flag did NOT fix the drape: over 9 runs per arm it lost 11 of 72 towels
against the defaults' 3 (``ClockPolicy.rungs_follow_dt_x``'s comment), so the small steps were not shown to be the
cause of that run's failure."""

import pytest

from alloy.coupling.clocks import ClockController
from alloy.coupling.types import ClockPolicy

CONTROL = 0.01
PAIR = CONTROL / 128                                  # [s] the pair bound's dt_x (78 us)
BAND = 20.0e-6                                        # [s] the band's bound in the trip: the grid's 19.5 us
MUJOCO = 1.0e-3
# VBD's stable step per period [s]: periods 7, 8, the trip's first attempt, its band re-run, then periods 10-17
BEFORE = (111.0e-6, 111.0e-6, 76.0e-6)
TRIP = (76.0e-6, 68.0e-6, 56.0e-6, 52.0e-6, 54.0e-6, 63.0e-6, 60.0e-6, 61.0e-6, 68.0e-6, 46.0e-6)


def _clock(follow: bool) -> ClockController:
    c = ClockController(ClockPolicy(rungs_follow_dt_x=follow), {"mujoco": 1, "vbd": 1}, {"mujoco": True, "vbd": True})
    c.safety_for = {"mujoco": 0.5, "vbd": 0.5}
    return c


def _run(follow: bool) -> tuple[list, list]:
    c = _clock(follow)
    before = [c.decide(CONTROL, PAIR, {"mujoco": MUJOCO, "vbd": v}) for v in BEFORE]
    trip = [c.decide(CONTROL, BAND, {"mujoco": MUJOCO, "vbd": v}) for v in TRIP]
    return before, trip


def test_today_a_band_trip_leaves_the_rung_high():
    before, trip = _run(False)
    assert before[-1].dt_x == pytest.approx(PAIR) and before[-1].substeps["vbd"] == 4
    for d in trip[:9]:                                # the run's 9 periods at 4.9 us
        assert d.dt_x == pytest.approx(PAIR / 4) and d.substeps["vbd"] == 4


def test_with_the_flag_the_rung_keeps_the_entry_step_through_the_trip():
    before, trip = _run(True)
    assert before[-1].substeps["vbd"] == 4            # the same raise before the trip
    own = before[-1].dt_x / before[-1].substeps["vbd"]
    first = trip[0]
    assert first.dt_x == pytest.approx(PAIR / 4) and first.substeps["vbd"] == 1
    assert first.dt_x / first.substeps["vbd"] == pytest.approx(own)      # 19.5 us, not 4.9
    for d, stable in zip(trip, TRIP, strict=True):    # every period still meets VBD's own bound
        assert d.dt_x / d.substeps["vbd"] <= 0.5 * stable * (1.0 + 1.0e-9)
        assert d.substeps["vbd"] == 1


def test_with_the_flag_a_rung_the_estimate_needs_is_kept():
    c = _clock(True)
    c.decide(CONTROL, PAIR, {"mujoco": MUJOCO, "vbd": 76.0e-6})
    d = c.decide(CONTROL, BAND, {"mujoco": MUJOCO, "vbd": 12.0e-6})      # 19.5 us needs 3.3: rung 4 stays
    assert d.dt_x == pytest.approx(PAIR / 4) and d.substeps["vbd"] == 4


def test_with_the_flag_no_entry_step_coarsens_when_dt_x_gets_finer():
    c = _clock(True)
    first = c.decide(CONTROL, PAIR, {"mujoco": 30.0e-6, "vbd": 76.0e-6})  # MuJoCo at 8 substeps (9.8 us)
    assert first.substeps == {"mujoco": 8, "vbd": 4}
    d = c.decide(CONTROL, BAND, {"mujoco": 1.0, "vbd": 1.0})             # both estimates relax in the same period
    for name in d.substeps:
        assert d.dt_x / d.substeps[name] <= first.dt_x / first.substeps[name] * (1.0 + 1.0e-9)
    assert d.substeps == {"mujoco": 2, "vbd": 1}


def test_the_flag_changes_nothing_while_dt_x_holds_or_coarsens():
    stable = [{"mujoco": MUJOCO, "vbd": v} for v in (111.0e-6, 76.0e-6, 200.0e-6, 400.0e-6) + (1.0e-3,) * 40]
    bounds = [PAIR, PAIR, PAIR, PAIR] + [1.0e-3] * 40                    # the pair releases: dt_x coarsens
    runs = []
    for follow in (False, True):
        c = _clock(follow)
        runs.append([(d.dt_x, tuple(sorted(d.substeps.items())))
                     for d in (c.decide(CONTROL, b, s) for b, s in zip(bounds, stable, strict=True))])
    assert runs[0] == runs[1]


def _raised(follow: bool) -> tuple[ClockController, object]:
    c = _clock(follow)
    for v in BEFORE:
        last = c.decide(CONTROL, PAIR, {"mujoco": MUJOCO, "vbd": v})
    assert last.dt_x == pytest.approx(PAIR) and last.substeps["vbd"] == 4
    return c, last


@pytest.mark.parametrize("follow", [False, True])
def test_a_sticky_escalation_still_halves_every_entry_step(follow):
    """The watchdog's escalation raises the floor with the bound unchanged: the remedy is a finer step for every
    entry, so the rung holds (WP4's review)."""
    c, last = _raised(follow)
    assert c.escalate()
    d = c.decide(CONTROL, PAIR, {"mujoco": MUJOCO, "vbd": 76.0e-6})
    assert d.dt_x == pytest.approx(PAIR / 2) and d.substeps["vbd"] == 4
    assert d.dt_x / d.substeps["vbd"] == pytest.approx(0.5 * last.dt_x / last.substeps["vbd"])


@pytest.mark.parametrize("follow", [False, True])
def test_an_escalation_with_a_band_trip_in_one_decide_keeps_the_rungs(follow):
    """Floor and bound rise in one decide: the watchdog has just tripped, so the rungs hold as today and every
    entry's step gets finer (WP4's review). After it the floor is not new: the band holds dt_x, nothing refines, and
    the rung comes down on the usual hysteresis."""
    c, last = _raised(follow)
    assert c.escalate()
    d = c.decide(CONTROL, BAND, {"mujoco": MUJOCO, "vbd": 76.0e-6})
    assert d.dt_x == pytest.approx(PAIR / 4) and d.substeps["vbd"] == 4
    later = [c.decide(CONTROL, BAND, {"mujoco": MUJOCO, "vbd": 76.0e-6}) for _ in range(10)]
    assert all(x.dt_x == pytest.approx(PAIR / 4) for x in later)
    assert [x.substeps["vbd"] for x in later] == [4] * 8 + [2, 2]     # 10 quiet periods, the trip's decide the first


def test_with_the_flag_a_trip_after_an_earlier_escalation_follows():
    """The floor rose in an earlier period: a later band trip is the bound's, and the rung keeps the entry's step."""
    c, _ = _raised(True)
    assert c.escalate()
    held = c.decide(CONTROL, PAIR, {"mujoco": MUJOCO, "vbd": 76.0e-6})
    assert held.dt_x == pytest.approx(PAIR / 2) and held.substeps["vbd"] == 4
    d = c.decide(CONTROL, BAND, {"mujoco": MUJOCO, "vbd": 76.0e-6})
    assert d.dt_x == pytest.approx(PAIR / 4) and d.substeps["vbd"] == 2
    assert d.dt_x / d.substeps["vbd"] == pytest.approx(held.dt_x / held.substeps["vbd"])
