# SPDX-License-Identifier: Apache-2.0
"""alloy.sysid.excite: trajectory limits, hold, speed bound, sequential / together modes, errors, save / load."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.sysid.errors import SysidError
from alloy.sysid.excitation import F1, HOLD, MARGIN, SLIDE, STEP_TIME, load, plan, trajectory

INF = float("inf")
RATE = 100.0
LOWER = [-2.0, -3.07, -INF, 0.0]
UPPER = [2.0, -0.07, INF, 0.1]
HINGE = [True, True, True, False]
NAMES = ["h0", "h1", "free", "slide"]


def make(**kwargs):
    base = {"rate": RATE, "seconds": 8.0, "amplitude": 0.15, "mode": "sequential"}
    return trajectory(NAMES, LOWER, UPPER, HINGE, **{**base, **kwargs})


def raises_field(field: str, fn, *args, **kwargs) -> SysidError:
    with pytest.raises(SysidError) as info:
        fn(*args, **kwargs)
    assert info.value.field == field
    return info.value


def test_every_command_stays_inside_the_range_minus_margin():
    exc = make()
    margin = np.where(HINGE, MARGIN, MARGIN * SLIDE)
    low = np.asarray(LOWER) + margin
    high = np.asarray(UPPER) - margin
    for mode in ("sequential", "together"):
        cmd = make(mode=mode).command
        assert np.all(cmd >= low - 1e-12) and np.all(cmd <= high + 1e-12)
    assert exc.command.shape == (len(exc.t), 4)


def test_range_excluding_zero_centres_inside_the_range():
    exc = make()
    assert -3.07 < exc.centre[1] < -0.07
    assert exc.centre[1] == pytest.approx(-0.07 - MARGIN - exc.amplitude[1], abs=1e-12)
    assert exc.centre[0] == pytest.approx(0.0, abs=1e-12)
    assert exc.centre[2] == pytest.approx(0.0, abs=1e-12)


def test_large_amplitude_is_cut_to_fit_a_narrow_range():
    exc = trajectory(["j"], [-0.2], [0.2], [True], amplitude=5.0)
    assert exc.amplitude[0] == pytest.approx(0.8 * (0.2 - MARGIN), abs=1e-12)
    assert np.all(np.abs(exc.command) <= 0.2 - MARGIN + 1e-12)


def test_slide_joints_get_the_smaller_amplitude():
    exc = make()
    assert exc.amplitude[0] == pytest.approx(0.15, abs=1e-12)
    assert exc.amplitude[3] == pytest.approx(0.15 * SLIDE, abs=1e-12)


def test_first_and_last_second_hold_the_centre():
    for mode in ("sequential", "together"):
        exc = make(mode=mode)
        start, end = exc.t <= HOLD, exc.t >= exc.t[-1] - HOLD
        for part in (start, end):
            np.testing.assert_allclose(exc.command[part], np.tile(exc.centre, (part.sum(), 1)), atol=1e-12)


def test_time_axis_length_and_rate():
    exc = make(seconds=4.0)
    assert exc.t[0] == 0.0
    assert exc.t[-1] == pytest.approx(2 * HOLD + 4 * 4.0, abs=1e-9)
    assert np.diff(exc.t) == pytest.approx(1.0 / RATE, abs=1e-12)
    assert make(seconds=4.0, mode="together").t[-1] == pytest.approx(2 * HOLD + 4.0, abs=1e-9)


@pytest.mark.parametrize("mode", ["sequential", "together"])
def test_peak_joint_speed_is_bounded(mode):
    exc = make(mode=mode)
    speed = np.abs(np.diff(exc.command, axis=0)) * RATE
    shift = 1.4 if mode == "together" else 1.0           # together: chirp band scaled by 0.6 ... 1.4
    chirp = 2.0 * math.pi * F1 * shift * 1.5             # 1.5: the window's own slope
    quick = 1.875 * 0.5 / STEP_TIME                      # minimum-jerk peak speed of a half-amplitude move
    bound = max(chirp, quick) * exc.amplitude
    assert np.all(speed.max(axis=0) <= bound + 1e-9)
    assert np.all(speed.max(axis=0) > 0.0)


def test_sequential_moves_one_joint_at_a_time():
    seconds = 6.0
    exc = make(seconds=seconds)
    for j in range(4):
        window = (exc.t >= HOLD + j * seconds) & (exc.t <= HOLD + (j + 1) * seconds)
        moving = exc.command[window, j]
        assert np.ptp(moving) > 0.5 * exc.amplitude[j]
        others = np.delete(np.arange(4), j)
        np.testing.assert_allclose(exc.command[window][:, others], np.tile(exc.centre[others], (window.sum(), 1)),
                                   atol=1e-12)
        outside = ~window
        np.testing.assert_allclose(exc.command[outside, j], exc.centre[j], atol=1e-12)


def test_together_moves_every_joint_in_the_same_window():
    seconds = 6.0
    exc = make(seconds=seconds, mode="together")
    window = (exc.t >= HOLD) & (exc.t <= HOLD + seconds)
    for j in range(4):
        assert np.ptp(exc.command[window, j]) > 0.5 * exc.amplitude[j]
        np.testing.assert_allclose(exc.command[~window, j], exc.centre[j], atol=1e-12)


def test_together_gives_each_joint_its_own_band():
    exc = make(mode="together")
    assert not np.allclose(exc.command[:, 0] / exc.amplitude[0], exc.command[:, 1] / exc.amplitude[1] -
                           (exc.centre[1] / exc.amplitude[1]), atol=1e-3)


def test_trajectory_is_deterministic():
    a, b = make(), make()
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.command, b.command)


def test_bad_mode():
    err = raises_field("mode", make, mode="parallel")
    assert "sequential" in err.fix


@pytest.mark.parametrize("kwargs", [{"rate": 10.0}, {"rate": 2000.0}, {"seconds": 1.9}, {"amplitude": 0.0}])
def test_bad_rate_seconds_or_amplitude(kwargs):
    raises_field("rate", make, **kwargs)


def test_range_narrower_than_the_margin():
    err = raises_field("joints", trajectory, ["j"], [0.0], [0.08], [True])
    assert "--joints" in err.fix
    raises_field("joints", plan, np.array([0.0]), np.array([0.08]), np.array([True]), 0.15)


def test_save_load_round_trip(tmp_path):
    exc = make(seconds=3.0)
    path = exc.save(tmp_path / "out" / "excite.csv")
    back = load(path)
    assert back.joints == NAMES
    np.testing.assert_allclose(back.t, exc.t, atol=1e-6)
    np.testing.assert_allclose(back.command, exc.command, atol=1e-7)
    np.testing.assert_allclose(back.centre, exc.centre, atol=1e-7)
    assert path.read_text().splitlines()[0] == "t,cmd.h0,cmd.h1,cmd.free,cmd.slide"

