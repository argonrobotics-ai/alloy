# SPDX-License-Identifier: Apache-2.0
"""Tick clock and grid snapping (SPEC D4, §3.1)."""

from __future__ import annotations

import pytest

from alloy.core.errors import ClockError
from alloy.core.timebase import TICK_EXP, SimClock, grid_dt, grid_index, snap_to_grid


@pytest.mark.parametrize(("dt", "control_dt", "expected"), [
    (2.0e-3, 0.02, 1.25e-3),          # dt_x bound 2 ms -> 1.25 ms (D4)
    (1.0 / 600.0, 0.02, 1.25e-3),     # rigid target
    (5.0e-4, 0.02, 3.125e-4),         # old precision dt snaps down
    (1.36e-4, 0.02, 7.8125e-5),       # L2 impact rule example (§3.5)
    (1.25e-3, 0.02, 1.25e-3),         # exactly on the grid
    (0.02, 0.02, 0.02),
    (1.0, 0.02, 0.02),                # above control_dt -> control_dt
    (0.02 / 1024, 0.02, 0.02 / 1024),  # base tick
    (2.0e-3, 0.01, 1.25e-3),
    (4.0e-3, 1 / 60, 1 / 60 / 8),
])
def test_snap_to_grid_table(dt: float, control_dt: float, expected: float) -> None:
    assert snap_to_grid(dt, control_dt) == expected


@pytest.mark.parametrize("dt", [0.02 / 1024 * 0.999, 0.0, -1e-3, float("nan")])
def test_snap_to_grid_rejects_below_tick(dt: float) -> None:
    with pytest.raises(ClockError):
        snap_to_grid(dt, 0.02)


def test_grid_index_exact_only() -> None:
    assert grid_index(1.25e-3, 0.02) == 4
    assert grid_dt(0.02, 4) == 1.25e-3
    with pytest.raises(ClockError):
        grid_index(1.0e-3, 0.02)
    with pytest.raises(ValueError):
        grid_dt(0.02, TICK_EXP + 1)


def test_clock_properties() -> None:
    c = SimClock(0.02, 4)
    assert c.tick == 0.02 / 1024 and c.dt == 1.25e-3
    assert c.ticks_per_step == 64 and c.steps_per_period == 16 and c.period_ticks == 1024
    c.advance(16)
    assert c.ticks == 1024 and c.period == 1 and c.at_period_boundary and c.t == 0.02


def test_set_dt_rejects_off_grid() -> None:
    c = SimClock(0.02, 4)
    with pytest.raises(ValueError):
        c.set_dt(1.0e-3)
    c.set_dt(3.125e-4)
    assert c.j == 6


def test_clock_exact_after_600k_periods_with_dt_changes() -> None:
    """t == ticks·Δ bitwise after 6e5 control periods, dt changing every period (WP0 acceptance)."""
    c = SimClock(0.02, 4)
    rungs = [4, 6, 3, 10, 0, 5]
    expected_ticks = 0
    accumulated = 0.0
    for period in range(600_000):
        c.j = rungs[period % len(rungs)]
        c.advance(c.steps_per_period)
        expected_ticks += 1024
        accumulated += 0.02
    assert c.ticks == expected_ticks == 600_000 * 1024
    assert c.t == float(c.ticks) * (0.02 / 1024)
    assert c.t == 600_000 * 1024 * (0.02 / 1024)
    assert accumulated != c.t            # float accumulation drifts; the tick clock does not


def test_clock_validation() -> None:
    for bad in [(0.0, 0), (0.02, -1), (0.02, 11), (float("inf"), 0)]:
        with pytest.raises(ClockError):
            SimClock(*bad)
    c = SimClock(0.02, 0)
    with pytest.raises(ClockError):
        c.advance(-1)
