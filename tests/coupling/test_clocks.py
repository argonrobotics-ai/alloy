# SPDX-License-Identifier: Apache-2.0
"""Pure clock logic: interface bound, D4 grid snapping, rung choice, hysteresis, escalation (SPEC §3.6, D4)."""

from __future__ import annotations

import math

import pytest

from alloy.core.timebase import grid_dt, snap_to_grid
from alloy.coupling.clocks import ClockController, damping_bound, grid_level, interface_bound, pick_rung
from alloy.coupling.types import ClockPolicy

CONTROL_DT = 0.02


def test_interface_bound_formula():
    # safety·ω_max·sqrt(m/k) = 0.5·1.0·sqrt(0.2/1e5) = 7.0711e-4 s
    assert interface_bound(1e5, 0.2, 0.5, 1.0) == pytest.approx(0.5 * math.sqrt(0.2 / 1e5), rel=1e-15)
    assert interface_bound(0.0, 0.2, 0.5, 1.0) == math.inf          # no stiffness: no bound
    assert interface_bound(1e5, 0.0, 0.5, 1.0) == math.inf
    assert interface_bound(1e5, 0.2, 0.5, math.inf) == math.inf     # bound disabled


def test_damping_bound_formula():
    """safety x omega_dt_max x m_eff / c_d: the held contact damping force's rate c_d/m_eff times dt (WP4 P3 law)."""
    assert damping_bound(55.0, 1.2e-3, 0.5, 1.0) == pytest.approx(0.5 * 1.2e-3 / 55.0)
    assert math.isinf(damping_bound(0.0, 1.0, 0.5, 1.0))           # undamped contacts: no bound
    assert math.isinf(damping_bound(55.0, math.inf, 0.5, 1.0))      # unknown mass: no bound
    assert math.isinf(damping_bound(55.0, 1.0, 0.5, math.inf))


@pytest.mark.parametrize("cap, j, unmet", [
    (math.inf, 0, False),                 # no bound -> control_dt itself
    (2.0e-3, 4, False),                   # 0.02/16 = 1.25e-3 <= 2e-3 < 2.5e-3 (SPEC §4.2 grid)
    (1.25e-3, 4, False),                  # exactly on the grid
    (7.0711e-4, 5, False),                # 0.625e-3
    (1.0e-5, 10, True),                   # below the base tick 1.953e-5 -> finest, flagged
])
def test_grid_level(cap, j, unmet):
    assert grid_level(cap, CONTROL_DT, 10) == (j, unmet)
    if not unmet:
        assert grid_dt(CONTROL_DT, j) == snap_to_grid(min(cap, CONTROL_DT), CONTROL_DT)


def test_pick_rung():
    ladder = (1, 2, 4, 8, 16, 32, 64)
    # dt_x/n <= safety·stable: 1.25e-3/n <= 0.5·1e-3 -> n >= 2.5 -> 4
    assert pick_rung(1.25e-3, 1.0e-3, 0.5, ladder) == 4
    assert pick_rung(1.25e-3, math.inf, 0.5, ladder) == 1
    assert pick_rung(1.25e-3, 1.0e-9, 0.5, ladder) == 64            # nothing qualifies -> largest rung
    assert pick_rung(1.25e-3, 0.0, 0.5, ladder) == 64


def _controller(**kw):
    return ClockController(ClockPolicy(**kw), {"a": 1, "b": 1}, {"a": False, "b": True})


def test_raise_now_lower_after_quiet_periods():
    c = _controller(downshift_quiet_periods=3)
    d = c.decide(CONTROL_DT, math.inf, {"b": 1.0e-3})               # 1.25e-3/n <= 5e-4 -> n = 4
    assert (d.j, d.substeps) == (4, {"a": 1, "b": 4})
    for k in range(2):                                              # quiet, but not long enough
        d = c.decide(CONTROL_DT, math.inf, {"b": 1.0})
        assert d.substeps["b"] == 4 and d.rung_changes == 0
    d = c.decide(CONTROL_DT, math.inf, {"b": 1.0})                  # third quiet period: one rung down
    assert d.substeps["b"] == 2 and d.rung_changes == 1
    d = c.decide(CONTROL_DT, math.inf, {"b": 1.0e-4})               # raise is immediate (n = 32)
    assert d.substeps["b"] == 32


def test_non_adaptive_and_unestimated_entries_hold():
    c = _controller()
    d = c.decide(CONTROL_DT, math.inf, {"a": 1.0e-6})               # "a" is not adaptive
    assert d.substeps["a"] == 1
    d = c.decide(CONTROL_DT, math.inf, {})                          # no estimate for "b": hold
    assert d.substeps["b"] == 1


def test_interface_bound_sets_dt_x_and_relaxes_with_hysteresis():
    c = _controller(downshift_quiet_periods=2)
    assert c.decide(CONTROL_DT, 7.0e-4, {}).dt_x == grid_dt(CONTROL_DT, 5)
    assert c.decide(CONTROL_DT, math.inf, {}).dt_x == grid_dt(CONTROL_DT, 5)   # quiet 1
    assert c.decide(CONTROL_DT, math.inf, {}).dt_x == grid_dt(CONTROL_DT, 4)   # quiet 2: one level coarser
    assert c.decide(CONTROL_DT, math.inf, {}).dt_x == grid_dt(CONTROL_DT, 4)   # dt_x_max 2e-3 caps at j=4


def test_escalation_is_sticky_and_bounded():
    c = _controller(dt_x_max=1 / 240, omega_dt_max=math.inf, downshift_quiet_periods=1, j_max=4)
    control = 1 / 60
    assert c.decide(control, math.inf, {}).j == 2
    assert c.escalate()
    assert c.decide(control, math.inf, {}).j == 3
    for _ in range(5):                                              # never relaxes below the floor
        assert c.decide(control, math.inf, {}).j == 3
    assert c.escalate()
    d = c.decide(control, math.inf, {})
    assert d.j == 4 and d.escalation_exhausted
    assert not c.escalate()


def test_ladder_validation():
    with pytest.raises(ValueError):
        ClockController(ClockPolicy(substep_ladder=(1, 4, 2)), {"a": 1}, {"a": True})
    with pytest.raises(ValueError):
        ClockController(ClockPolicy(), {"a": 3}, {"a": True})
