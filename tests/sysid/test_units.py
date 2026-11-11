# SPDX-License-Identifier: Apache-2.0
"""alloy.sysid.units: convert (deg -> rad on hinges, ms -> s) and check (degrees / milliseconds refused with a fix)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.sysid.errors import SysidError
from alloy.sysid.recording import Recording, make_episode
from alloy.sysid.units import check, convert

INF = float("inf")
RANGED = (np.array([-2.0]), np.array([2.0]), np.array([True]))
UNRANGED = (np.array([-INF]), np.array([INF]), np.array([True]))


def rec_from(pos: np.ndarray, dt: float = 0.01, joints: list[str] | None = None) -> Recording:
    pos = np.asarray(pos, float).reshape(len(pos), -1)
    t = dt * np.arange(len(pos))
    names = joints or [f"j{i}" for i in range(pos.shape[1])]
    return Recording(names, [make_episode(t, pos, pos, pos * 2.0, pos * 3.0)])


def ramp(peak: float, n: int = 50) -> np.ndarray:
    return np.linspace(-peak, peak, n)


def refusal(rec: Recording, bounds) -> SysidError:
    with pytest.raises(SysidError) as info:
        check(rec, *bounds)
    return info.value


def test_convert_degrees_only_on_hinges():
    rec = rec_from(np.stack([ramp(90.0), ramp(0.5)], axis=1))
    out = convert(rec, angle="deg", hinge=np.array([True, False]))
    ep, src = out.episodes[0], rec.episodes[0]
    np.testing.assert_allclose(ep.position[:, 0], src.position[:, 0] * math.pi / 180.0, atol=1e-12)
    np.testing.assert_allclose(ep.command[:, 0], src.command[:, 0] * math.pi / 180.0, atol=1e-12)
    np.testing.assert_allclose(ep.velocity[:, 0], src.velocity[:, 0] * math.pi / 180.0, atol=1e-12)
    np.testing.assert_array_equal(ep.position[:, 1], src.position[:, 1])
    np.testing.assert_array_equal(ep.velocity[:, 1], src.velocity[:, 1])
    np.testing.assert_array_equal(ep.torque, src.torque)
    assert any("deg" in n for n in out.notes)


def test_convert_without_hinge_mask_converts_every_joint():
    out = convert(rec_from(ramp(180.0)), angle="deg")
    assert out.episodes[0].position.max() == pytest.approx(math.pi, abs=1e-12)


def test_convert_milliseconds_scales_time_and_velocity():
    rec = rec_from(ramp(1.0), dt=10.0)
    out = convert(rec, time="ms")
    np.testing.assert_allclose(out.episodes[0].t, rec.episodes[0].t * 1e-3, atol=1e-12)
    np.testing.assert_allclose(out.episodes[0].velocity, rec.episodes[0].velocity * 1e3, atol=1e-9)
    np.testing.assert_array_equal(out.episodes[0].position, rec.episodes[0].position)
    assert out.dt == pytest.approx(0.01, abs=1e-12)


def test_convert_radians_seconds_is_identity():
    rec = rec_from(ramp(1.0))
    assert convert(rec) is rec


def test_convert_unknown_unit():
    for kwargs in ({"angle": "grad"}, {"time": "us"}):
        with pytest.raises(SysidError) as info:
            convert(rec_from(ramp(1.0)), **kwargs)
        assert info.value.field == "units"


def test_check_degrees_recording_is_refused_naming_the_option():
    err = refusal(rec_from(ramp(90.0)), RANGED)
    assert err.field == "angle_unit"
    assert "--angle-unit deg" in err.fix


def test_check_radians_inside_range_passes():
    check(rec_from(ramp(1.9)), *RANGED)


def test_check_small_overshoot_within_slack_passes():
    check(rec_from(ramp(2.05)), *RANGED)


def test_check_positions_that_fit_neither_way_are_data_errors():
    err = refusal(rec_from(ramp(500.0)), RANGED)       # 500 deg = 8.7 rad: outside +-2 as degrees too
    assert err.field == "data"
    assert "--angle-unit" not in err.fix


def test_check_slide_outside_range_is_data_not_degrees():
    bounds = (np.array([-0.1]), np.array([0.1]), np.array([False]))
    assert refusal(rec_from(ramp(3.0)), bounds).field == "data"


def test_check_unranged_joint_passes_below_four_pi():
    check(rec_from(ramp(12.0)), *UNRANGED)


def test_check_unranged_joint_beyond_four_pi_is_refused_as_degrees():
    err = refusal(rec_from(ramp(90.0)), UNRANGED)
    assert err.field == "angle_unit"
    assert "--angle-unit deg" in err.fix


def test_check_unranged_joint_far_beyond_degrees_is_data():
    assert refusal(rec_from(ramp(5000.0)), UNRANGED).field == "data"


def test_check_checks_the_command_of_an_unranged_joint_too():
    t = 0.01 * np.arange(50)
    cmd = ramp(90.0).reshape(-1, 1)
    rec = Recording(["j"], [make_episode(t, cmd, np.zeros((50, 1)))])
    assert refusal(rec, UNRANGED).field == "angle_unit"


def test_check_time_in_milliseconds_is_refused_naming_the_option():
    err = refusal(rec_from(ramp(1.0), dt=10.0), RANGED)
    assert err.field == "t"
    assert "--time-unit ms" in err.fix


def test_check_slow_sample_period_is_refused_without_the_ms_hint():
    err = refusal(rec_from(ramp(1.0), dt=0.7), RANGED)
    assert err.field == "t"
    assert "--time-unit" not in err.fix


def test_check_too_fast_sample_period_is_refused():
    err = refusal(rec_from(ramp(1.0), dt=1e-6), RANGED)
    assert err.field == "t"


def test_check_known_limit_degrees_that_fit_the_radian_range_are_not_detected():
    """Known limit: +-1.5 'degrees' on a +-2 rad joint cannot be told from radians, so check() lets it through."""
    check(rec_from(ramp(1.5)), *RANGED)
