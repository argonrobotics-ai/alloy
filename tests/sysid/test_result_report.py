# SPDX-License-Identifier: Apache-2.0
"""alloy.sysid.result (SysidResult, file, load checks), alloy.sysid.report (text, randomize lines, from_file, plots)
and alloy.sysid.identify (moved, nudged, unit_sigma, determined), all on hand-made data."""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from alloy.sysid import identify, result
from alloy.sysid.errors import SysidError
from alloy.sysid.reporting import from_file, plots, randomize_lines, text
from alloy.sysid.result import SCHEMA, SysidResult


def make_result(**over) -> SysidResult:
    base = dict(
        robot="arm", joints=["shoulder", "slide"], control_dt=0.01,
        values={"kp": np.array([120.0, 80.0]), "kd": np.array([8.0, 4.0]), "damping": np.array([0.5, 0.25]),
                "friction": np.array([0.3, 0.0]), "delay": np.array([0.012, 0.02])},
        variables=["kp", "vel", "friction", "delay"],
        nominal={"kp": np.array([100.0, 100.0]), "kd": np.array([10.0, 10.0])},
        rms_before=np.array([0.1, 0.01]), rms_after=np.array([0.02, 0.002]),
        moved=np.array([True, False]),
        sigma={"kp": np.array([0.01, 0.02]), "vel": np.array([0.2, 0.3])},
        determined={"kp": np.array([True, True]), "vel": np.array([False, True]),
                    "friction": np.array([True, False]), "delay": np.array([True, True])},
        hinge=np.array([True, False]), notes=["a note"], meta={"seed": 0})
    base.update(over)
    return SysidResult(**base)


def raises_sysid(fn, *args) -> SysidError:
    with pytest.raises(SysidError) as info:
        fn(*args)
    assert info.value.field in ("sysid", "result")
    assert info.value.message and info.value.fix
    return info.value


def test_to_dict_schema_and_values():
    d = make_result().to_dict()
    assert d["schema"] == SCHEMA == "alloy.sysid/1"
    assert d["robot"] == "arm" and d["control_dt"] == pytest.approx(0.01, abs=1e-12)
    assert d["joints"]["shoulder"] == {"kp": 120.0, "kd": 8.0, "damping": 0.5, "friction": 0.3, "delay": 0.012}
    assert d["fit"]["variables"] == ["kp", "vel", "friction", "delay"]
    assert d["fit"]["moved"] == {"shoulder": True, "slide": False}
    assert d["fit"]["seed"] == 0 and d["fit"]["notes"] == ["a note"]
    assert d["fit"]["nominal"]["shoulder"] == {"kp": 100.0, "kd": 10.0}
    assert d["fit"]["sigma"]["kp"] == {"shoulder": 0.01, "slide": 0.02}


def test_to_dict_rms_is_degrees_for_hinges_only():
    fit = make_result().to_dict()["fit"]
    assert fit["rms_before_deg"]["shoulder"] == pytest.approx(0.1 * 180.0 / math.pi, abs=1e-9)
    assert fit["rms_after_deg"]["shoulder"] == pytest.approx(0.02 * 180.0 / math.pi, abs=1e-9)
    assert fit["rms_before_deg"]["slide"] == pytest.approx(0.01, abs=1e-12)


def test_to_dict_non_finite_becomes_null_and_json_serialises():
    r = make_result(values={"kp": np.array([120.0, 80.0]), "torque_limit": np.array([np.inf, 5.0])})
    d = r.to_dict()
    assert d["joints"]["shoulder"]["torque_limit"] is None
    json.dumps(d, allow_nan=False)


def test_poorly_determined_lists_weak_variables_per_joint():
    assert make_result().poorly_determined() == {"shoulder": ["vel"], "slide": ["friction"]}
    assert make_result(determined={}).poorly_determined() == {}


def test_save_then_load_keeps_only_schema_robot_and_joints(tmp_path):
    path = make_result().save(tmp_path / "x" / "arm.sysid.json")
    data = result.load(path)
    assert set(data) == {"schema", "robot", "joints"}
    assert data["robot"] == "arm"
    assert data["joints"]["slide"] == {"kp": 80.0, "kd": 4.0, "damping": 0.25, "friction": 0.0, "delay": 0.02}
    assert all(isinstance(v, float) for vals in data["joints"].values() for v in vals.values())


def test_load_accepts_a_dict_a_string_path_and_a_result(tmp_path):
    r = make_result()
    assert result.load(r) == result.load(r.to_dict())
    path = r.save(tmp_path / "arm.sysid.json")
    assert result.load(str(path)) == result.load(r)


def test_load_hand_written_minimal_file():
    data = result.load({"schema": SCHEMA, "joints": {"j": {"kp": 5}}})
    assert data == {"schema": SCHEMA, "robot": "", "joints": {"j": {"kp": 5.0}}}


def test_load_skips_null_values():
    data = result.load({"schema": SCHEMA, "joints": {"j": {"kp": 5, "torque_limit": None}}})
    assert data["joints"]["j"] == {"kp": 5.0}


def test_load_missing_file(tmp_path):
    assert "does not exist" in raises_sysid(result.load, tmp_path / "none.json").message


def test_load_not_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{nope")
    assert "not JSON" in raises_sysid(result.load, path).message


@pytest.mark.parametrize("data", [
    {"schema": "other/1", "joints": {}},
    {"joints": {}},
    {"schema": SCHEMA},
    {"schema": SCHEMA, "joints": []},
    [1, 2],
])
def test_load_wrong_schema_or_layout(data):
    raises_sysid(result.load, data)


@pytest.mark.parametrize("vals", [
    {"stiffness": 1.0},          # unknown parameter
    {"kp": -1.0},                # negative
    {"kp": 0.0},                 # kp must be > 0
    {"kd": -0.1},
    {"delay": 0.6},              # above the 0.5 s limit
    {"friction": float("inf")},
    {"kd": float("nan")},
])
def test_load_rejects_bad_values(vals):
    raises_sysid(result.load, {"schema": SCHEMA, "joints": {"j": vals}})


def test_load_joint_entry_must_be_a_mapping():
    raises_sysid(result.load, {"schema": SCHEMA, "joints": {"j": [1.0]}})


def test_load_accepts_boundary_values():
    data = result.load({"schema": SCHEMA, "joints": {"j": {"kd": 0.0, "delay": 0.5, "friction": 0.0}}})
    assert data["joints"]["j"] == {"kd": 0.0, "delay": 0.5, "friction": 0.0}


def test_report_text_lists_joints_weak_variables_and_unmoved_joints():
    out = text(make_result(), robot_name="arm")
    assert "shoulder" in out and "slide" in out
    assert "POORLY DETERMINED" in out
    assert "kd + damping" in out
    assert re.search(r"- shoulder: kd \+ damping", out)
    assert re.search(r"- slide: friction .*never moved", out)
    assert "- a note" in out


def test_report_text_all_determined_message():
    r = make_result(determined={"kp": np.array([True, True])}, moved=np.array([True, True]))
    out = text(r)
    assert "POORLY DETERMINED" not in out
    assert "Every fitted parameter is determined" in out


def test_report_text_has_one_randomize_line_per_scale_with_lo_below_one_below_hi():
    out = text(make_result(), robot_name="arm")
    found = re.findall(r'scene\.randomize\("arm", "(\w+)", ([\d.]+), ([\d.]+)\)', out)
    assert {f[0] for f in found} == {"kp_scale", "kd_scale"}
    for _, lo, hi in found:
        assert float(lo) < 1.0 < float(hi)


def test_randomize_spread_is_three_sigma_of_the_worst_joint():
    data = {"fit": {"sigma": {"kp": {"a": 0.02, "b": 0.05}}}}
    line = randomize_lines(data, "r")[1]
    assert line == 'scene.randomize("r", "kp_scale", 0.85, 1.15)'


def test_randomize_spread_is_clipped_to_five_and_fifty_percent():
    small = randomize_lines({"fit": {"sigma": {"kp": {"a": 0.001}}}}, "r")[1]
    large = randomize_lines({"fit": {"sigma": {"vel": {"a": 0.9}}}}, "r")[1]
    assert small == 'scene.randomize("r", "kp_scale", 0.95, 1.05)'
    assert large == 'scene.randomize("r", "kd_scale", 0.50, 1.50)'


def test_randomize_with_unknown_sigma_uses_the_widest_spread_and_without_sigma_a_default():
    unknown = randomize_lines({"fit": {"sigma": {"kp": {"a": None}}}}, "r")[1]
    assert unknown == 'scene.randomize("r", "kp_scale", 0.50, 1.50)'
    lines = randomize_lines({"fit": {}}, "r")
    assert 'scene.randomize("r", "kp_scale", 0.90, 1.10)' in lines
    assert 'scene.randomize("r", "kd_scale", 0.90, 1.10)' in lines


def test_randomize_ignores_variables_without_a_scale_field():
    lines = randomize_lines({"fit": {"sigma": {"friction": {"a": 0.1}, "delay": {"a": 0.1}}}}, "r")
    assert sum(line.startswith("scene.randomize") for line in lines) == 2     # the default pair


def test_report_from_file_round_trip_and_errors(tmp_path):
    path = make_result().save(tmp_path / "arm.sysid.json")
    data = from_file(path)
    assert data["fit"]["moved"]["slide"] is False
    assert "POORLY DETERMINED" in text(data)
    raises_sysid(from_file, tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    raises_sysid(from_file, bad)
    bad.write_text(json.dumps({"schema": "x"}))
    raises_sysid(from_file, bad)


def test_plots_without_curves_raises(tmp_path):
    err = raises_sysid(plots, make_result(), tmp_path)
    assert err.field == "result" and "no curves" in err.message


def test_moved_thresholds_for_hinge_and_slide():
    hinge = np.array([True, False])
    flat = np.zeros((10, 2))

    def span(a: float, b: float) -> np.ndarray:
        pos = flat.copy()
        pos[0, 0], pos[0, 1] = a, b
        return identify.moved(flat, pos, hinge)

    assert list(span(0.019, 0.0019)) == [False, False]
    assert list(span(0.021, 0.0021)) == [True, True]
    assert list(span(0.021, 0.0019)) == [True, False]
    assert list(span(0.0019, 0.021)) == [False, True]      # a hinge threshold does not apply to a slide


def test_moved_looks_at_the_command_too():
    cmd, pos = np.zeros((10, 1)), np.zeros((10, 1))
    cmd[5, 0] = 0.5
    assert identify.moved(cmd, pos, np.array([True]))[0]


def test_nudged_shapes_and_directions():
    u = np.array([[0.5, 0.2], [0.7, 0.9]])
    points, steps = identify.nudged(u)
    assert points.shape == (4, 2, 2) and steps.shape == (2, 2)
    np.testing.assert_allclose(steps, identify.NUDGE, atol=1e-12)
    np.testing.assert_allclose(points[0, 0], u[0] + identify.NUDGE, atol=1e-12)
    np.testing.assert_allclose(points[1, 0], u[0] - identify.NUDGE, atol=1e-12)
    np.testing.assert_allclose(points[0, 1], u[1], atol=1e-12)        # the other variable stays put
    np.testing.assert_allclose(points[3, 1], u[1] - identify.NUDGE, atol=1e-12)


def test_nudged_clips_at_the_bounds_and_halves_the_step():
    u = np.array([[1.0, 0.0]])
    points, steps = identify.nudged(u)
    np.testing.assert_allclose(points[0, 0], [1.0, 0.04], atol=1e-12)
    np.testing.assert_allclose(points[1, 0], [0.96, 0.0], atol=1e-12)
    np.testing.assert_allclose(steps, [[0.02, 0.02]], atol=1e-12)
    assert points.min() >= 0.0 and points.max() <= 1.0


def synthetic(dt: float, count: int = 2000, noise: float = 0.01, zero_second: bool = False):
    rng = np.random.default_rng(7)
    t = dt * np.arange(count)
    slopes = np.stack([np.sin(2 * math.pi * 0.7 * t), 2.0 * np.cos(2 * math.pi * 1.3 * t)], axis=1)   # (N, V)
    if zero_second:
        slopes[:, 1] = 0.0
    steps = np.full((2, 1), identify.NUDGE)
    base = np.zeros(count)
    response = np.zeros((4, count, 1))
    for v in range(2):
        response[2 * v, :, 0] = base + slopes[:, v] * steps[v, 0]
        response[2 * v + 1, :, 0] = base - slopes[:, v] * steps[v, 0]
    residual = rng.normal(0.0, noise, size=(count, 1))
    return residual, response, steps, slopes


def test_unit_sigma_matches_least_squares_when_samples_are_independent():
    residual, response, steps, slopes = synthetic(dt=identify.CORRELATION)
    sigma = identify.unit_sigma(residual, response, steps, identify.CORRELATION)
    analytic = np.sqrt(np.mean(residual ** 2) * np.diag(np.linalg.inv(slopes.T @ slopes)))
    assert sigma.shape == (2, 1)
    assert np.all(sigma > analytic[:, None] / 2.0) and np.all(sigma < analytic[:, None] * 2.0)


def test_unit_sigma_grows_with_the_oversampling_correlation():
    residual, response, steps, slopes = synthetic(dt=0.01)           # 10 samples per correlation time
    sigma = identify.unit_sigma(residual, response, steps, 0.01)
    analytic = np.sqrt(np.mean(residual ** 2) * np.diag(np.linalg.inv(slopes.T @ slopes)))
    np.testing.assert_allclose(sigma[:, 0] / analytic, math.sqrt(10.0), rtol=0.2)


def test_unit_sigma_is_infinite_for_a_variable_with_no_response():
    residual, response, steps, _ = synthetic(dt=identify.CORRELATION, zero_second=True)
    sigma = identify.unit_sigma(residual, response, steps, identify.CORRELATION)
    assert math.isinf(sigma[1, 0])
    assert math.isfinite(sigma[0, 0]) and sigma[0, 0] > 0.0


def test_unit_sigma_non_finite_response_stays_infinite():
    residual, response, steps, _ = synthetic(dt=identify.CORRELATION)
    response[0, 3, 0] = np.nan
    assert np.all(np.isinf(identify.unit_sigma(residual, response, steps, identify.CORRELATION)))


def test_determined_needs_small_sigma_and_a_joint_that_moved():
    sigma = np.array([[0.01, 0.01, 0.079], [0.08, 0.5, np.inf]])
    got = identify.determined(sigma, np.array([True, False, True]))
    assert got.tolist() == [[True, False, True], [False, False, False]]
