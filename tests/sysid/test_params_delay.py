# SPDX-License-Identifier: Apache-2.0
"""alloy.sysid.params (variables, Box bounds / encode / decode / start, to_fields) and alloy.sysid.delay.DelayLine."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.sysid.delay import DelayLine, period_steps
from alloy.sysid.errors import SysidError
from alloy.sysid.params import MAX_DELAY, PARAMETERS, Box, to_fields, variables


def nominal(**over) -> dict[str, np.ndarray]:
    base = {"kp": [100.0, 50.0], "kd": [10.0, 0.0], "damping": [2.0, 0.0], "friction": [0.5, 0.0],
            "armature": [0.0, 0.01], "torque_limit": [30.0, np.inf]}
    base.update(over)
    return {k: np.asarray(v, float) for k, v in base.items()}


def test_variables_default():
    assert variables(None) == ["kp", "vel", "friction", "delay"]
    assert variables([]) == ["kp", "vel", "friction", "delay"]
    assert variables(["default"]) == ["kp", "vel", "friction", "delay"]


def test_variables_gains_group_and_kd_alone():
    assert variables(["gains"]) == ["kp", "kd"]
    assert variables(["kd"]) == ["kd"]
    assert variables(["damping", "kp"]) == ["kp", "damping"]


def test_variables_kd_with_damping_become_vel():
    assert variables(["kd", "damping", "delay"]) == ["vel", "delay"]
    assert variables(["gains", "damping"]) == ["kp", "vel"]


def test_variables_all():
    got = variables(["all"])
    assert got == ["kp", "vel", "friction", "armature", "torque_limit", "delay"]
    assert set(PARAMETERS) >= set(got) - {"vel"}


def test_variables_unknown_name():
    with pytest.raises(SysidError) as info:
        variables(["kp", "stiction"])
    assert info.value.field == "fit" and "stiction" in info.value.message


def test_box_bounds_log_for_positive_and_linear_from_zero():
    box = Box.around(["kp", "vel"], nominal())
    np.testing.assert_allclose(box.lo[0], [100 / 8, 50 / 8], atol=1e-12)
    np.testing.assert_allclose(box.hi[0], [800, 400], atol=1e-12)
    assert box.log[0].all()
    np.testing.assert_allclose(box.lo[1], [12 / 8, 0.0], atol=1e-12)
    np.testing.assert_allclose(box.hi[1], [96.0, 0.2 * 50.0], atol=1e-12)   # zero in the file: 0 ... 0.2 kp
    assert list(box.log[1]) == [True, False]


def test_box_friction_upper_is_five_times_nominal_or_a_tenth_of_torque():
    box = Box.around(["friction"], nominal())
    # joint 0: max(5 * 0.5, 0.1 * 30) = 3; joint 1: torque limit inf -> 0.2 kp = 10, max(0, 1.0) = 1
    np.testing.assert_allclose(box.hi[0], [3.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(box.lo[0], 0.0, atol=1e-12)
    assert not box.log.any()
    big = Box.around(["friction"], nominal(friction=[10.0, 0.0]))
    assert big.hi[0, 0] == pytest.approx(50.0, abs=1e-12)


def test_box_delay_is_zero_to_max_delay():
    box = Box.around(["delay"], nominal())
    np.testing.assert_allclose(box.lo[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(box.hi[0], MAX_DELAY, atol=1e-12)
    np.testing.assert_allclose(Box.around(["delay"], nominal(), max_delay=0.02).hi[0], 0.02, atol=1e-12)


def test_box_torque_limit_bounds():
    box = Box.around(["torque_limit"], nominal())
    np.testing.assert_allclose(box.hi[0], [30.0, 10.0], atol=1e-12)
    np.testing.assert_allclose(box.lo[0], [6.0, 2.0], atol=1e-12)
    assert box.log.all()


def test_decode_encode_round_trip_inside_the_box():
    box = Box.around(["kp", "vel", "friction", "delay"], nominal())
    u = np.random.default_rng(0).uniform(0.0, 1.0, size=(5, 4, 2))
    x = box.decode(u)
    assert np.all(x >= box.lo - 1e-12) and np.all(x <= box.hi + 1e-12)
    np.testing.assert_allclose(box.encode(x), u, atol=1e-9)
    np.testing.assert_allclose(box.decode(box.encode(x)), x, rtol=1e-9, atol=1e-12)


def test_decode_corners_hit_the_bounds():
    box = Box.around(["kp", "friction"], nominal())
    np.testing.assert_allclose(box.decode(np.zeros((2, 2))), box.lo, atol=1e-12)
    np.testing.assert_allclose(box.decode(np.ones((2, 2))), box.hi, atol=1e-9)


def test_encode_clips_outside_the_box():
    box = Box.around(["kp", "friction"], nominal())
    assert np.all(box.encode(box.hi * 10.0 + 1.0) == 1.0)
    assert np.all(box.encode(box.lo - 1.0) == 0.0)
    assert np.all(box.encode(np.zeros((2, 2))) == 0.0)


def test_box_start_is_the_files_values_and_no_delay():
    nom = nominal()
    box = Box.around(["kp", "vel", "friction", "armature", "torque_limit", "delay"], nom)
    start = box.start(nom)
    np.testing.assert_allclose(start[0], nom["kp"], atol=1e-12)
    np.testing.assert_allclose(start[1], [12.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(start[2], nom["friction"], atol=1e-12)
    np.testing.assert_allclose(start[3], nom["armature"], atol=1e-12)
    np.testing.assert_allclose(start[4], [30.0, box.hi[4, 1]], atol=1e-12)   # no limit in the file: the box's top
    np.testing.assert_allclose(start[5], 0.0, atol=1e-12)


def test_to_fields_splits_vel_in_the_files_ratio():
    nom = nominal()
    x = np.array([[100.0, 50.0], [24.0, 6.0]])                  # kp, vel
    out = to_fields(["kp", "vel"], x, nom)
    np.testing.assert_allclose(out["kp"], [100.0, 50.0], atol=1e-12)
    np.testing.assert_allclose(out["kd"][0], 24.0 * 10.0 / 12.0, atol=1e-12)
    np.testing.assert_allclose(out["damping"][0], 24.0 * 2.0 / 12.0, atol=1e-12)
    np.testing.assert_allclose(out["kd"] + out["damping"], x[1], atol=1e-12)
    assert "vel" not in out


def test_to_fields_sends_vel_to_damping_when_file_has_neither():
    out = to_fields(["vel"], np.array([[3.0, 4.0]]), nominal(kd=[0.0, 0.0], damping=[0.0, 0.0]))
    np.testing.assert_allclose(out["kd"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["damping"], [3.0, 4.0], atol=1e-12)


def test_to_fields_keeps_leading_world_axis():
    x = np.ones((3, 2, 2))
    out = to_fields(["kp", "vel"], x, nominal())
    assert out["kp"].shape == out["kd"].shape == out["damping"].shape == (3, 2)


def push_all(line: DelayLine, values) -> list:
    return [np.array(line.push(np.asarray(v, float))) for v in values]


def test_zero_delay_returns_the_input():
    line = DelayLine(np.zeros(2))
    for v in ([1.0, 2.0], [3.0, 4.0], [5.0, 6.0]):
        np.testing.assert_allclose(line.push(np.array(v)), v, atol=1e-12)


def test_whole_delay_returns_the_value_pushed_k_pushes_ago():
    line = DelayLine(np.array([3.0]))
    out = push_all(line, [[float(i)] for i in range(1, 9)])
    got = [float(o[0]) for o in out]
    assert got == [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]      # first value fills the history until 3 periods passed


def test_fractional_delay_interpolates():
    line = DelayLine(np.array([1.5]))
    got = [float(line.push(np.array([float(i)]))[0]) for i in range(0, 8)]
    # pushes 0..7; output at push n is the value 1.5 pushes ago: n - 1.5 once it exists
    assert got[:2] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert got[2:] == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], abs=1e-12)


def test_per_column_delays_differ():
    line = DelayLine(np.array([0.0, 1.0, 2.0]))
    out = push_all(line, [[i, i, i] for i in range(1, 6)])
    np.testing.assert_allclose(out[-1], [5.0, 4.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(out[0], [1.0, 1.0, 1.0], atol=1e-12)


def test_negative_steps_are_treated_as_zero():
    line = DelayLine(np.array([-2.0]))
    np.testing.assert_allclose(line.push(np.array([7.0])), [7.0], atol=1e-12)


def test_reset_restarts_the_history():
    line = DelayLine(np.array([2.0]))
    push_all(line, [[1.0], [2.0], [3.0], [4.0]])
    line.reset()
    np.testing.assert_allclose(line.push(np.array([9.0])), [9.0], atol=1e-12)
    np.testing.assert_allclose(line.push(np.array([10.0])), [9.0], atol=1e-12)


def test_masked_reset_restarts_only_the_masked_columns():
    line = DelayLine(np.array([2.0, 2.0]))
    push_all(line, [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    line.reset(np.array([True, False]))
    out = line.push(np.array([10.0, 10.0]))
    np.testing.assert_allclose(out, [10.0, 3.0], atol=1e-12)         # column 1 keeps its history: 2 pushes ago was 3
    out = line.push(np.array([11.0, 11.0]))
    np.testing.assert_allclose(out, [10.0, 4.0], atol=1e-12)


def test_masked_reset_on_a_two_dimensional_line():
    line = DelayLine(np.full((3, 2), 1.0))
    push_all(line, [np.ones((3, 2)), 2 * np.ones((3, 2))])
    line.reset(np.array([[True, True], [False, False], [False, False]]))
    out = line.push(np.full((3, 2), 5.0))
    np.testing.assert_allclose(out[0], 5.0, atol=1e-12)
    np.testing.assert_allclose(out[1:], 2.0, atol=1e-12)


def test_period_steps():
    assert period_steps(0.015, 0.01) == pytest.approx(1.5, abs=1e-12)
    assert period_steps(-1.0, 0.01) == 0.0
    assert period_steps(float("inf"), 0.01) == 0.0
