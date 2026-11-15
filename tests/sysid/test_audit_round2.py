# SPDX-License-Identifier: Apache-2.0
"""Regressions from an outside audit of 0.1.2: the first command after a build or reset is delayed, a torque
limit can be read, raised and fitted on both servo kinds, a fit of some joints replays the others, and robot names
and files that cannot be fitted say so."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
import alloy.sysid as sysid
from alloy.sysid import params, synth
from alloy.sysid.delay import DelayLine
from alloy.sysid.errors import SysidError
from alloy.sysid.excitation import Excitation, trajectory
from alloy.sysid.model import RobotJoints
from alloy.sysid.replay import Bench, resample
from alloy.sysid.robot import NAME, bench_scene, resolve, shipped

pytestmark = pytest.mark.timeout(900)

ONE = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.2">
  <joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="1.0"/>
  <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.5"/></body></worldbody>
  <actuator><position name="a1" joint="j1" kp="80" kv="8"/></actuator></mujoco>"""
DELAYED = {"schema": "alloy.sysid/1", "joints": {"j1": {"delay": 0.04}}}          # 2 periods at dt = 0.02 s


def _steps(sim, target, count=4):
    out = []
    for _ in range(count):
        sim.set_targets({"arm/j1": target})
        sim.step(1)
        out.append(float(sim.state(["arm/j1"], world=0)["arm/j1.q"]))
    return np.asarray(out)


def _one(sysid_values=None, worlds=1):
    scene = alloy.Scene(worlds=worlds, dt=0.02)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ONE, sysid=sysid_values)
    return scene


# --- the first command after a build or reset waits out the delay ---------------------------------------------

def test_delay_line_starts_from_the_rest_values():
    line = DelayLine(np.array([2.0]))
    got = [float(line.push(np.array([v]), np.array([-1.0]))[0]) for v in (5.0, 6.0, 7.0)]
    assert got == [-1.0, -1.0, 5.0]
    line.reset(np.array([True]))
    assert float(line.push(np.array([9.0]), np.array([3.0]))[0]) == 3.0


def test_first_command_after_build_and_reset_is_delayed():
    plain, late = alloy.build(_one(), device="cpu"), alloy.build(_one(DELAYED), device="cpu")
    for target in (0.8, -0.6):
        now, then = _steps(plain, target), _steps(late, target)
        assert abs(now[0]) > 0.05
        np.testing.assert_allclose(then[:2], 0.0, atol=1e-6)            # held at rest for the two delayed periods
        np.testing.assert_allclose(then[2:], now[:2], atol=1e-5)        # then the same motion, two periods late
        plain.reset()
        late.reset()


@pytest.mark.parametrize("mode", ["absolute", "delta"])
def test_env_actions_are_delayed_from_the_first_step_and_after_a_masked_reset(mode):
    from alloy.envs import ActionSpec, Task, VectorEnv, jitter_joint  # noqa: PLC0415

    runs = {}
    for label, values in (("plain", None), ("late", DELAYED)):
        spec = ActionSpec(mode="delta", delta=0.2) if mode == "delta" else ActionSpec(mode="absolute")
        task = Task(obs=["arm"], actions=spec, randomize=[jitter_joint("arm/j1", by=0.0, at=0.5)], max_steps=20)
        env = VectorEnv(_one(values, worlds=2), task, worlds=2, device="cpu", backend="numpy")
        env.reset(seed=0)
        act = np.full((2, 1), 1.0 if mode == "delta" else -0.8, np.float32)
        first = []
        for _ in range(4):
            env.step(act)
            first.append(float(env.state["arm/j1.q"][0]))
        env.reset_worlds(np.asarray([True, False]))
        env.step(act)
        runs[label] = (np.asarray(first), float(env.state["arm/j1.q"][0]))
        env.close()
    (now, _), (then, after_reset) = runs["plain"], runs["late"]
    assert abs(now[0] - 0.5) > 0.01
    np.testing.assert_allclose(then[:2], 0.5, atol=1e-5)                # the start pose is held, not the action
    np.testing.assert_allclose(then[2], now[0], atol=1e-4)
    assert abs(after_reset - 0.5) < 1e-5                                # the reset world holds its start again


# --- torque limit ------------------------------------------------------------------------------------------------

def _servo(kind, limit):
    act = (f'<position name="a1" joint="j1" kp="200" kv="10" forcelimited="true" forcerange="-{limit} {limit}"/>'
           if kind == "position" else
           f'<general name="a1" joint="j1" biastype="affine" gainprm="200 0 0" biasprm="0 -200 -10" '
           f'forcelimited="true" forcerange="-{limit} {limit}"/>')
    return f"""<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" range="-2.5 2.5" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.04" mass="2.0"/></body></worldbody>
      <actuator>{act}</actuator></mujoco>"""


def _reach(path, torque=None):
    scene = alloy.Scene(dt=0.01)
    values = None if torque is None else {"schema": "alloy.sysid/1", "joints": {"j1": {"torque_limit": torque}}}
    scene.add_robot("arm", path, sysid=values)
    sim = alloy.build(scene, device="cpu")
    return _steps(sim, 1.5, 20)


@pytest.mark.parametrize("kind", ["position", "general"])
def test_torque_limit_is_read_and_can_be_raised(kind, tmp_path):
    weak, strong = tmp_path / "weak.xml", tmp_path / "strong.xml"
    weak.write_text(_servo(kind, 8.0))
    strong.write_text(_servo(kind, 25.0))
    scene, _ = bench_scene(str(weak), worlds=1, dt=0.01)
    nominal = RobotJoints.read(alloy.build(scene, device="cpu").built, NAME).nominal
    assert nominal["torque_limit"].tolist() == [8.0]
    box = params.Box.around(["torque_limit"], nominal)
    assert box.lo[0, 0] < 8.0 < box.hi[0, 0]
    as_file, as_strong = _reach(str(weak)), _reach(str(strong))
    assert np.max(np.abs(as_strong - as_file)) > 0.2                    # the limit matters in this move
    np.testing.assert_allclose(_reach(str(weak), 8.0), as_file, atol=1e-6)     # the file's own value: no change
    np.testing.assert_allclose(_reach(str(weak), 25.0), as_strong, atol=1e-6)  # raised: as a file with that limit
    # in a running simulation too (the fit's replay): a hidden 25 N·m on the 8 N·m file is the 25 N·m file
    exc = sysid.excite(str(strong), rate=50, seconds=2.0, amplitude=0.8)
    truth = synth.record(str(strong), exc).episodes[0].position
    hidden = synth.record(str(weak), exc, {"torque_limit": 25.0}).episodes[0].position
    capped = synth.record(str(weak), exc).episodes[0].position
    assert np.max(np.abs(truth - capped)) > 0.2
    np.testing.assert_allclose(hidden, truth, atol=1e-6)


# --- a fit of some joints replays the recording's other joints -------------------------------------------------

TWO = """<mujoco><compiler angle="radian"/><worldbody><body name="upper" pos="0 0 0.8">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.8 1.8" damping="2.0"/>
  <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.025" mass="1.0"/>
  <body name="forearm" pos="0.30 0 0">
    <joint name="elbow" type="hinge" axis="0 1 0" range="-2.8 -0.4" damping="1.0"/>
    <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.025" mass="2.0"/></body></body></worldbody>
  <actuator>
    <general name="a_sh" joint="shoulder" biastype="affine" gainprm="120 0 0" biasprm="0 -120 -8" forcerange="-80 80"/>
    <general name="a_el" joint="elbow" biastype="affine" gainprm="120 0 0" biasprm="0 -120 -8" forcerange="-80 80"/>
  </actuator></mujoco>"""


def test_fit_of_one_joint_replays_the_other_recorded_joint(tmp_path):
    arm = tmp_path / "arm2.xml"
    arm.write_text(TWO)
    moving = trajectory(["shoulder", "elbow"], [-1.8, -2.8], [1.8, -0.4], [True, True], rate=50.0, seconds=2.0,
                        amplitude=0.3, mode="together")
    centre = np.array([0.2, -2.2])                         # a folded elbow: far from the file's start pose
    exc = Excitation(moving.joints, moving.t, moving.command - moving.centre + centre, centre, moving.amplitude)
    truth = {"kp": [84.0, 120.0], "kd": [11.2, 8.0], "damping": [2.8, 1.0]}       # only the shoulder differs
    rec = synth.record(str(arm), exc, truth)
    track = resample(rec.episodes[0], 0.02).split(1)
    shoulder = {k: np.asarray(v[:1]) for k, v in truth.items()}
    whole = Bench(str(arm), ["shoulder"], worlds=1, dt=0.02, device="cpu", held=["elbow"])
    np.testing.assert_allclose(whole.rollout(shoulder, [track])[0][:, 0], track.position, atol=1e-6)
    alone = Bench(str(arm), ["shoulder"], worlds=1, dt=0.02, device="cpu")
    assert any("elbow" in note for note in alone.notes)                  # said: the elbow is not replayed
    lone = resample(rec.select(["shoulder"]).episodes[0], 0.02)
    assert np.sqrt(np.mean((alone.rollout(shoulder, [lone])[0][:, 0] - lone.position) ** 2)) > 0.02
    fit = sysid.fit(str(arm), rec, joints=("shoulder",), parameters=("kp", "kd", "damping"), worlds=16,
                    generations=10, seed=0, device="cpu")
    assert fit.joints == ["shoulder"]
    assert abs(float(fit.values["kp"][0]) - 84.0) < 0.05 * 84.0          # 76 % off when the elbow was dropped


# --- robot names and files ---------------------------------------------------------------------------------------

def test_every_listed_robot_name_resolves_to_a_robot():
    names = shipped()
    assert {"panda", "go2", "fr3", "yam_bench"} <= set(names)
    assert not [n for n in names if n.startswith(("task", "bin", "marble")) or n in ("c3", "c6", "face", "wedge")]
    with pytest.raises(SysidError) as err:
        resolve("aloha")
    assert str(sorted(names)) in err.value.fix
    for name in names:
        assert resolve(name) == names[name] and names[name].is_file()
    assert resolve("emb_e2_panda") == names["panda"]                     # a folder's full name still works


URDF = """<?xml version="1.0"?><robot name="bot">
  <link name="base"><inertial><mass value="1.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial></link>
  <link name="arm"><inertial><origin xyz="0 0 0.2"/><mass value="0.5"/>
    <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/></inertial>
    <collision><geometry><box size="0.05 0.05 0.4"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="50" velocity="5"/></joint></robot>"""


@pytest.mark.parametrize("call", ["excite", "demo"])
def test_a_robot_file_without_servos_is_refused_with_the_fix(call, tmp_path):
    bot = tmp_path / "bot.urdf"
    bot.write_text(URDF)
    with pytest.raises(SysidError) as err:
        getattr(sysid, call)(str(bot), seconds=2.0)
    assert "no position-servoed joint" in err.value.message and "MJCF" in err.value.fix
