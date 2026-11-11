# SPDX-License-Identifier: Apache-2.0
"""``scene.add_robot(sysid=)``: the fitted values reach every world's model on both CPU routes, the action delay
acts on every command path and clears on reset, ``scene.randomize`` scales the fitted values, and a scene without
``sysid=`` is untouched."""

from __future__ import annotations

import json

import numpy as np
import pytest

import alloy
from alloy.sysid import synth
from alloy.sysid.excitation import trajectory
from alloy.sysid.model import RobotJoints

pytestmark = pytest.mark.timeout(600)
RANGE = ([-2.0, -2.0], [2.0, 2.0], [True, True])


def _build(arm_file, sysid=None, worlds=1, rows=(), seed=0):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_robot("arm", arm_file, sysid=sysid)
    for row in rows:
        scene.randomize(*row)
    return alloy.build(scene, device="cpu", seed=seed)


def _commands(seconds=2.0):
    return trajectory(["shoulder", "elbow"], *RANGE, seconds=seconds, mode="together")


@pytest.fixture(scope="module", params=[pytest.param("mjc", marks=pytest.mark.cpu_route), "warp"])
def route(request):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALLOY_CPU_ROUTE", request.param)
        yield request.param


def test_values_reach_every_world_on_both_routes(arm_file, arm_kit, route):
    joints, hidden, hidden_file, _ = arm_kit
    sim = _build(arm_file, hidden_file(), worlds=3)
    got = RobotJoints.read(sim.built, "arm").nominal
    for f in ("kp", "kd", "damping", "friction"):
        assert got[f] == pytest.approx(hidden[f], rel=1e-6, abs=0.0), f
    model = sim.built.model
    assert model.joint_friction.numpy().reshape(3, -1)[:, :2] == pytest.approx(np.tile(hidden["friction"], (3, 1)),
                                                                              rel=1e-6, abs=0.0)


def test_applied_robot_reproduces_the_recording(arm_file, arm_kit, route):
    """The path the fit simulates (candidate values + delay line) and the path a user gets (sysid=) agree."""
    _, hidden, hidden_file, replay = arm_kit
    exc = _commands(seconds=2.0)
    exc.command, exc.t = exc.command[100:220], exc.t[:120]            # 1.2 s of the motion
    rec = synth.record(arm_file, exc, hidden, noise=0.0, device="cpu")
    sim = _build(arm_file, hidden_file(), worlds=2)
    got = replay(sim, "arm", exc.command[:-1])
    assert np.abs(got - rec.episodes[0].position[1:]).max() < 1e-6
    assert sim.check()["ok"]


def test_no_sysid_is_untouched_and_nominal_file_is_identity(arm_file, arm_kit):
    _, _, _, replay = arm_kit
    exc = _commands(seconds=2.0)
    plain = _build(arm_file)
    assert "step_period" not in vars(plain.engine) and not hasattr(plain.engine, "sysid_delay")
    assert "sysid" not in plain.scene.items["arm"]
    base = replay(plain, "arm", exc.command[:100])
    nominal = RobotJoints.read(plain.built, "arm").nominal
    same = {"schema": "alloy.sysid/1", "joints": {
        j: {f: float(nominal[f][i]) for f in ("kp", "kd", "damping", "friction", "armature")}
        for i, j in enumerate(("shoulder", "elbow"))}}
    again = replay(_build(arm_file, same), "arm", exc.command[:100])
    assert np.array_equal(base, again)                      # the file's own values, no delay: bit for bit


def test_delay_holds_commands_back_and_reads_back_as_commanded(arm_file, arm_kit):
    _, _, _, replay = arm_kit
    late = {"schema": "alloy.sysid/1", "joints": {"shoulder": {"delay": 0.03}, "elbow": {"delay": 0.03}}}
    step = np.zeros((40, 2))
    step[5:] = 0.3
    shifted = np.zeros((40, 2))
    shifted[8:] = 0.3                                         # the same step, commanded 3 control periods later
    plain = replay(_build(arm_file), "arm", shifted)
    sim = _build(arm_file, late)
    delayed = replay(sim, "arm", step)
    assert np.abs(delayed - plain).max() < 1e-7               # 30 ms = 3 control periods
    assert np.abs(delayed[10] - replay(_build(arm_file), "arm", step)[10]).max() > 1e-3
    state = sim.state(["arm/shoulder"], world=0)
    assert float(state["arm/shoulder.q"]) == pytest.approx(delayed[-1, 0], abs=1e-9)
    assert float(sim.engine.control.joint_target_q.numpy()[0]) == pytest.approx(0.3, abs=1e-6)   # as commanded


def test_run_ramp_is_delayed_too(arm_file):
    late = {"schema": "alloy.sysid/1", "joints": {"shoulder": {"delay": 0.05}}}
    plain, sim = _build(arm_file), _build(arm_file, late)
    a = plain.run(0.3, record=["arm/shoulder"], targets={"shoulder": (0.0, 0.4)})["arm/shoulder.q"][:, 0]
    b = sim.run(0.3, record=["arm/shoulder"], targets={"shoulder": (0.0, 0.4)})["arm/shoulder.q"][:, 0]
    assert b[-1] < a[-1] - 0.02                               # the delayed arm is behind the ramp


def test_reset_clears_the_delay_history(arm_file, arm_kit):
    _, _, hidden_file, replay = arm_kit
    exc = _commands()
    sim = _build(arm_file, hidden_file())
    first = replay(sim, "arm", exc.command[150:250])
    sim.reset()
    assert np.array_equal(first, replay(sim, "arm", exc.command[150:250]))


def test_randomize_scales_the_fitted_gains(arm_file, arm_kit):
    _, hidden, hidden_file, _ = arm_kit
    sim = _build(arm_file, hidden_file(), worlds=4, rows=[("arm", "kp_scale", 0.5, 1.5)], seed=3)
    draw = sim.randomized()["arm.kp_scale"]
    ke = sim.built.model.joint_target_ke.numpy().reshape(4, -1)[:, 0]
    gain = sim.built.model.mujoco.actuator_gainprm.numpy().reshape(4, -1, 10)[:, :, 0]
    assert np.ptp(draw) > 0.05
    assert ke == pytest.approx(hidden["kp"][0] * draw, rel=1e-5, abs=0.0)
    assert np.any(np.isclose(gain, (hidden["kp"][1] * draw)[:, None], rtol=1e-5, atol=0.0), axis=1).all()


def test_scene_json_carries_the_values(arm_file, arm_kit):
    _, hidden, hidden_file, _ = arm_kit
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", arm_file, sysid=hidden_file())
    back = alloy.Scene.from_dict(json.loads(scene.to_json()))
    assert back.items["arm"]["sysid"]["joints"]["elbow"]["kp"] == hidden["kp"][1]
    sim = alloy.build(back, device="cpu")
    assert RobotJoints.read(sim.built, "arm").nominal["kp"] == pytest.approx(hidden["kp"], rel=1e-6, abs=0.0)


def test_scene_save_warns_that_cards_drop_the_values(arm_file, arm_kit, tmp_path, caplog):
    for sysid, expected in ((arm_kit[2](), 1), (None, 0)):
        scene = alloy.Scene(dt=0.01)
        scene.add_robot("arm", arm_file, sysid=sysid)
        caplog.clear()
        with caplog.at_level("WARNING", logger="alloy.api"):
            scene.save(tmp_path / f"cards{expected}")
        assert sum("do not keep sysid= for arm" in r.getMessage() for r in caplog.records) == expected


def test_errors_name_the_fix(arm_file, tmp_path):
    scene = alloy.Scene(dt=0.01)
    with pytest.raises(alloy.SceneError) as err:
        scene.add_robot("arm", arm_file, sysid=str(tmp_path / "missing.sysid.json"))
    assert err.value.field == "sysid" and "alloy.sysid fit" in err.value.fix
    with pytest.raises(alloy.SceneError) as err:
        scene.add_robot("arm", arm_file, sysid={"schema": "alloy.sysid/1", "joints": {"elbow": {"kp": -1.0}}})
    assert "out of range" in err.value.message
    wrong = alloy.Scene(dt=0.01)
    wrong.add_robot("arm", arm_file, sysid={"schema": "alloy.sysid/1", "joints": {"wrist": {"kp": 5.0}}})
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(wrong, device="cpu")
    assert "wrist" in err.value.message and "shoulder" in err.value.fix
    ok = alloy.Scene(worlds=16, dt=0.01)
    ok.add_robot("arm", arm_file, sysid={"schema": "alloy.sysid/1", "joints": {"elbow": {"kp": 90.0}}})
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(ok, device="cpu", rerun_worlds=1)
    assert err.value.field == "rerun_worlds"


@pytest.mark.gpu
def test_applied_robot_reproduces_the_recording_on_a_gpu(arm_file, arm_kit):
    """The same check on CUDA (float32, captured graphs): the delay swap runs outside the graph."""
    _, hidden, hidden_file, replay = arm_kit
    exc = _commands()
    rec = synth.record(arm_file, exc, hidden, noise=0.0, device="cuda:0")
    scene = alloy.Scene(worlds=4, dt=0.01)
    scene.add_robot("arm", arm_file, sysid=hidden_file())
    sim = alloy.build(scene, device="cuda:0")
    got = replay(sim, "arm", exc.command[:-1])
    assert np.abs(got - rec.episodes[0].position[1:]).max() < 1e-4
    sim.reset()
    assert np.abs(replay(sim, "arm", exc.command[:50]) - got[:50]).max() < 1e-4
