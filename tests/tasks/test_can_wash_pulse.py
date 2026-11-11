# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can's robot stage: the valve's pulses give every band stage 1's dose per metre (fast, no
simulation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.can_wash_pulse import Pulse, yaw_of
from alloy.tasks.can_wash_strokes import STROKE
from alloy.tasks.can_wash_turn import eased
from alloy.tasks.rinse_flush_policy import CONTROL_DT

V = 0.25                                     # m/s, stage 1's aim speed along a band (``Rinse.speed``)


def run(pulse: Pulse, angles: np.ndarray, on: bool = True) -> np.ndarray:
    """The openings for a can turning through ``angles`` [rad, one per period] from 0, one world."""
    pulse.opening(np.zeros(1), on)
    return np.array([pulse.opening(np.array([a]), on)[0] for a in angles])


def test_a_steady_turn_opens_the_valve_for_stage_1s_time_over_the_band():
    """1 rad/s under the floor's innermost ring (0.09 m): open 36 % of the periods, stage 1's 2.26 s a revolution."""
    p = Pulse(0.09, V, 1)
    angles = np.arange(1, int(round(2 * math.pi / CONTROL_DT)) + 1) * CONTROL_DT
    out = run(p, angles)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert p.open_s[0] == pytest.approx(2 * math.pi * 0.09 / V, abs=CONTROL_DT)
    assert p.share(angles[-1])[0] == pytest.approx(1.0, abs=0.01)
    assert p.duty()[0] == pytest.approx(0.09 / V, abs=0.01)           # open 36 % of the turning periods


def test_a_strokes_ramps_take_no_more_than_stage_1s_dose_per_metre():
    """A 60° stroke with its cosine ramps at r 0.17 m: the valve's open time is the arc over 0.25 m/s, within a
    period, and no ten periods running hold more than their arc's share plus one opening."""
    p = Pulse(0.17, V, 1)
    angles = eased(STROKE, 1.0, CONTROL_DT)
    out = run(p, angles)
    assert p.open_s[0] == pytest.approx(STROKE * 0.17 / V, abs=CONTROL_DT)
    arc = np.diff(np.r_[0.0, angles]) * 0.17 / V / CONTROL_DT
    window = np.convolve(out, np.ones(10), "valid") - np.convolve(arc, np.ones(10), "valid")
    assert window.max() <= 1.0 + 1e-9


def test_a_band_faster_than_stage_1s_aim_keeps_the_valve_open_and_books_its_share():
    """The 0.26 m rim band at 1 rad/s passes at 0.26 m/s: open every period, 0.25/0.26 of stage 1's dose."""
    p = Pulse(0.26, V, 1)
    angles = np.arange(1, 101) * CONTROL_DT
    assert run(p, angles).all()
    assert p.share(angles[-1])[0] == pytest.approx(V / 0.26, rel=1e-6)


def test_the_valve_stays_shut_unless_the_ur_is_turning_and_reads_the_turn_across_pi():
    """Shut while ``on`` is false (the regrip), the yaw read across ±π unwrapped; each world counts its own turn."""
    p = Pulse(0.09, V, 2)
    assert not run(p, np.arange(1, 50) * 0.05, on=False).any() and p.open_s[0] == 0.0
    p = Pulse(0.2, V, 2)
    yaw = np.array([math.pi - 0.004, 0.0])
    p.opening(yaw, True)
    opened = [p.opening(np.array([(math.pi - 0.004 + 0.01 * k + math.pi) % (2 * math.pi) - math.pi, 0.0]), True)
              for k in range(1, 101)]
    assert np.asarray(opened)[:, 0].sum() * CONTROL_DT == pytest.approx(1.0 * 0.2 / V, abs=CONTROL_DT)
    assert np.asarray(opened)[:, 1].sum() == 0.0


def test_yaw_of_reads_a_turn_about_the_vertical():
    for a in (0.3, -2.0, 3.0):
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.sin(a / 2), math.cos(a / 2)])
        assert yaw_of(q) == pytest.approx(a, abs=1e-12)
    with pytest.raises(ValueError):
        Pulse(0.0, V, 1)
