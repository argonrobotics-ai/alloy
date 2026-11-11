# SPDX-License-Identifier: Apache-2.0
"""The fit against a known answer: a simulated robot with hidden joint parameters, a delay and sensor noise is
recorded, and the fit must find the hidden values within stated tolerances. Fast checks run a short fit of the
two-joint arm on the CPU; the laws (arm on the CPU, Panda and Go2 on a GPU) are marked slow."""

from __future__ import annotations

import json

import numpy as np
import pytest

import alloy
import alloy.sysid as S
from alloy.sysid import synth
from alloy.sysid.excitation import trajectory

RANGE = ([-2.0, -2.0], [2.0, 2.0], [True, True])
#: law tolerances, relative to the hidden value (delay: absolute, seconds)
TOL = {"kp": 0.12, "vel": 0.15, "friction": 0.30, "delay": 0.003}


@pytest.fixture(scope="module", autouse=True)
def _default_cpu_route():
    """These fits run on the CPU's default route (MuJoCo C), as a user's do; the suite's Warp route is about 10x
    slower per replay step for this arm."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALLOY_CPU_ROUTE", "mjc")
        yield


def _recording(arm_file, hidden, seconds=3.0, rate=50.0, noise=2e-4, still=()):
    exc = trajectory(["shoulder", "elbow"], *RANGE, seconds=seconds, rate=rate, mode="together")
    for j in still:
        exc.command[:, j] = exc.centre[j]
    return synth.record(arm_file, exc, hidden, noise=noise, seed=1, device="cpu")


@pytest.fixture(scope="module")
def quick(arm_file, arm_kit):
    """A short fit (16 worlds x 6 generations) of a 3 s recording, shared by the fast checks."""
    _, hidden, _, _ = arm_kit
    rec = _recording(arm_file, hidden)
    return rec, S.fit(arm_file, rec, worlds=16, generations=6, seed=0, device="cpu")


@pytest.mark.timeout(900)
def test_quick_fit_cuts_the_error_and_moves_toward_the_hidden_values(quick, arm_kit):
    _, hidden, _, _ = arm_kit
    _, result = quick
    assert np.all(result.rms_after < 0.5 * result.rms_before)
    assert result.values["kp"][0] == pytest.approx(hidden["kp"][0], rel=0.3, abs=0.0)      # the laws are tighter
    assert result.moved.all() and result.meta["generations"] == 6
    assert set(result.values) == {"kp", "kd", "damping", "friction", "delay"}
    ratio = result.values["kd"] / result.values["damping"]
    assert ratio == pytest.approx([10.0 / 2.0, 4.0 / 1.0], rel=1e-9, abs=0.0)      # the robot file's kd : damping


@pytest.mark.timeout(900)
def test_fitted_file_applies_and_tracks_better_than_the_robot_file(quick, arm_file, arm_kit, tmp_path):
    _, _, _, replay = arm_kit
    rec, result = quick
    path = result.save(tmp_path / "arm.sysid.json")
    assert json.loads(path.read_text())["schema"] == "alloy.sysid/1"
    episode = rec.episodes[0]
    errors = []
    for sysid in (None, str(path)):
        scene = alloy.Scene(dt=0.02)
        scene.add_robot("arm", arm_file, sysid=sysid)
        got = replay(alloy.build(scene, device="cpu"), "arm", episode.command[:-1])
        errors.append(float(np.sqrt(np.mean((got - episode.position[1:]) ** 2))))
    assert errors[1] < 0.5 * errors[0]
    text = S.report(result, robot_name="arm")
    assert 'scene.randomize("arm", "kp_scale"' in text and "shoulder" in text


@pytest.mark.timeout(900)
def test_same_seed_same_file_and_unmoved_joint_is_flagged(arm_file, arm_kit):
    _, hidden, _, _ = arm_kit
    rec = _recording(arm_file, hidden, seconds=2.0, still=(1,))
    runs = [S.fit(arm_file, rec, worlds=16, generations=2, seed=5, device="cpu") for _ in range(2)]
    for f in runs[0].values:
        assert np.array_equal(runs[0].values[f], runs[1].values[f]), f
    result = runs[0]
    assert list(result.moved) == [True, False]
    weak = result.poorly_determined()
    assert set(weak["elbow"]) == {"kp", "vel", "friction", "delay"}
    assert result.values["kp"][1] == pytest.approx(80.0, rel=1e-6, abs=0.0)      # the robot file's value is kept
    assert result.values["delay"][1] == 0.0
    assert any("elbow: never moved" in n for n in result.notes)
    assert "the joint never moved" in S.report(result)


@pytest.mark.timeout(300)
def test_degrees_and_empty_motion_are_refused(arm_file, arm_kit):
    _, hidden, _, _ = arm_kit
    rec = _recording(arm_file, hidden, seconds=2.0, noise=0.0)
    e = rec.episodes[0]
    degrees = S.Recording(rec.joints, [S.make_episode(e.t, np.degrees(e.command) * 8, np.degrees(e.position) * 8)])
    with pytest.raises(S.SysidError) as err:
        S.fit(arm_file, degrees, worlds=16, generations=1, device="cpu")
    assert err.value.field == "angle_unit" and "--angle-unit deg" in err.value.fix
    still = S.Recording(rec.joints, [S.make_episode(e.t, e.command * 0.0, e.position * 0.0)])
    with pytest.raises(S.SysidError) as err:
        S.fit(arm_file, still, worlds=16, generations=1, device="cpu")
    assert "no joint moves" in err.value.message
    with pytest.raises(S.SysidError):
        S.fit(arm_file, rec, worlds=4, device="cpu")
    with pytest.raises(S.SysidError) as err:
        S.fit("no_such_robot", rec, worlds=16, device="cpu")
    assert err.value.field == "robot"


def _law(robot, joints, device, worlds, generations, seconds):
    rec, hidden = S.demo(robot, joints, seconds=seconds, noise=2e-4, seed=0, device=device)
    result = S.fit(robot, rec, worlds=worlds, generations=generations, seed=0, device=device)
    got = result.values
    vel, vel_true = got["kd"] + got["damping"], hidden["kd"] + hidden["damping"]
    report = {"kp": np.abs(got["kp"] / hidden["kp"] - 1.0), "vel": np.abs(vel / vel_true - 1.0),
              "friction": np.abs(got["friction"] / hidden["friction"] - 1.0),
              "delay": np.abs(got["delay"] - hidden["delay"])}
    print({k: np.round(v, 4).tolist() for k, v in report.items()}, result.meta["wall_s"])
    for name, err in report.items():
        assert np.all(err <= TOL[name]), (name, err.tolist())
    assert np.all(result.rms_after < 0.2 * result.rms_before)
    assert not result.poorly_determined()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(3600)
def test_law_arm_recovers_hidden_parameters_cpu(arm_file):
    _law(arm_file, None, "cpu", worlds=32, generations=30, seconds=6.0)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("robot,joints", [
    ("panda", ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]),
    ("go2", None)])
def test_law_recovers_hidden_parameters_gpu(robot, joints):
    _law(robot, joints, "cuda:0", worlds=256, generations=30, seconds=6.0)
