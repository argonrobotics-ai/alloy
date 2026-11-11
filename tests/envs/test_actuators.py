# SPDX-License-Identifier: Apache-2.0
"""Actuator models (alloy.envs.actuators) on one hinge of known inertia: PD law against gravity, effort limit, DC-motor
torque-speed curve, joint ownership (motorized files), delay in control periods, the refusal of file servos, the
stable gain range, and an untouched default ActionSpec. CPU, NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy
from alloy.envs import ActionSpec, EnvError, Task, VectorEnv
from alloy.envs.actions import ActionWriter
from alloy.envs.actuators import ActuatorSpec, motorized

INERTIA = 1e-3 + 1.0 * 0.25**2          # kg m^2 about the pivot: a 1 kg point mass 0.25 m out, plus a little
LOAD_ARM = 1.0 * 9.81 * 0.25            # N m: gravity torque of the horizontal arm at q = 0
BODY = ('<body name="link" pos="0 0 1"><inertial pos="0.25 0 0" mass="1" diaginertia="1e-3 1e-3 1e-3"/>'
        '<joint name="j" type="hinge" axis="{axis}" range="-1000 1000"/></body>')
SERVOS = {
    "general": '<actuator><general joint="j" gainprm="50" biastype="affine" biasprm="0 -50 0"/></actuator>',
    "position": '<actuator><position joint="j" kp="50"/></actuator>',
    "motor": '<actuator><motor joint="j"/></actuator>',
    "none": "",
}


def robot(axis: str = "0 0 1", actuator: str = "none") -> str:
    body = BODY.format(axis=axis)
    return f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody>{SERVOS[actuator]}</mujoco>'


def make_env(xml: str, spec: ActuatorSpec, worlds: int = 2) -> VectorEnv:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_robot("arm", mjcf=xml, pos=(0, 0, 0))
    sim = alloy.build(scene, device="cpu", plugins=[spec.plugin()])
    env = VectorEnv(sim, Task(actions=spec, max_steps=10**6), backend="numpy")
    env.reset(seed=0)
    return env


def absolute(**kw) -> ActuatorSpec:
    return ActuatorSpec(["arm/j"], mode="absolute", low=[-1000.0], high=[1000.0], **kw)


def drive(env: VectorEnv, targets, steps: int) -> None:
    for _ in range(steps):
        env.step(np.asarray(targets, np.float32).reshape(-1, 1))


@pytest.mark.compile_heavy
def test_law1_pd_static_error_is_load_over_kp() -> None:
    kp = 40.0
    env = make_env(robot("0 1 0", "motor"), ActuatorSpec(["arm/j"], kp=kp, kd=2.0, mode="absolute",
                                                          low=[-1.0], high=[1.0]))
    drive(env, [0.3, 0.3], 300)
    q, tau = env.state["arm/j.q"], env.writer.torque[:, 0]
    load = LOAD_ARM * np.cos(q)                                  # gravity torque at the settled angle
    error = np.abs(q - 0.3)
    print("LAW1 error*kp / load =", error * kp / load, " torque / load =", np.abs(tau) / load)
    assert np.allclose(error * kp, load, rtol=0.02)
    assert np.allclose(np.abs(tau), load, rtol=0.02)


@pytest.mark.compile_heavy
def test_law2_effort_limit_saturates_and_sets_the_acceleration() -> None:
    limit = 2.0
    env = make_env(robot(), absolute(kp=100.0, kd=0.0, effort_limit=limit))
    drive(env, [3.0, -3.0], 1)
    tau, qd = env.writer.torque[:, 0], env.state["arm/j.qd"]
    accel = qd / env.sim.dt                                      # from rest over one control period
    print("LAW2 torque", tau, " accel / (limit / I) =", np.abs(accel) / (limit / INERTIA))
    assert np.array_equal(np.abs(tau), np.full(2, limit, np.float32)) and tau[0] > 0 > tau[1]
    assert np.allclose(np.abs(accel), limit / INERTIA, rtol=0.03)


@pytest.mark.compile_heavy
def test_law3_dc_motor_curve_and_no_load_speed() -> None:
    sat, vmax = 2.0, 10.0
    env = make_env(robot(), absolute(kp=1000.0, kd=0.0, model="dc_motor", effort_limit=100.0,
                                     saturation_effort=sat, velocity_limit=vmax))
    worst = -np.inf
    for _ in range(300):
        drive(env, [1000.0, -1000.0], 1)
        w, tau = env.state["arm/j.qd"], env.writer.torque[:, 0]
        worst = max(worst, float(np.max(np.abs(tau) - sat * (1.0 - np.abs(w) / vmax))))
    print("LAW3 top speed", env.state["arm/j.qd"], " worst excess over the curve (N m)", worst)
    assert worst < 0.02                                           # one physics step of speed gain at most
    assert np.allclose(np.abs(env.state["arm/j.qd"]), vmax, rtol=0.05)


@pytest.mark.compile_heavy
def test_law3_effort_limit_clips_below_the_stall_torque() -> None:
    env = make_env(robot(), absolute(kp=1000.0, kd=0.0, model="dc_motor", effort_limit=1.0,
                                     saturation_effort=2.0, velocity_limit=10.0))
    drive(env, [1000.0, 1000.0], 1)
    assert np.allclose(env.writer.torque[:, 0], 1.0)


def _free_swing(actuator: str, steps: int) -> float:
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", mjcf=robot("0 1 0", actuator), pos=(0, 0, 0))
    sim = alloy.build(scene, device="cpu")
    sim.step(steps)
    return float(sim.state(world=0)["arm/j.q"])


@pytest.mark.compile_heavy
def test_law4_motorized_joint_with_zero_gains_moves_like_a_free_joint() -> None:
    """A swinging arm: the motorized file with a zero-gain model follows a robot with no actuator (and with a plain
    motor at zero command) exactly. The robot with no actuator at all is compared over 20 periods only: a passive
    world's health check freezes it after about 30."""
    spec = ActuatorSpec(["arm/j"], kp=0.0, kd=0.0, mode="absolute", low=[-1.0], high=[1.0])
    env = make_env(motorized(robot("0 1 0", "general")), spec, worlds=1)
    at = {}
    for k in range(1, 101):
        drive(env, [0.5], 1)
        at[k] = float(env.state["arm/j.q"][0])
    print("LAW4 20 periods: owned", at[20], "none", _free_swing("none", 20), "| 100 periods: owned", at[100],
          "motor", _free_swing("motor", 100))
    assert abs(at[20]) > 0.5 and np.isclose(at[20], _free_swing("none", 20), rtol=0, atol=1e-5)
    assert abs(at[100]) > 0.5 and np.isclose(at[100], _free_swing("motor", 100), rtol=0, atol=1e-5)


def test_motorized_rewrites_every_servo_in_order() -> None:
    xml = ('<mujoco><worldbody><body name="b"><joint name="a"/><joint name="c"/><geom size="1"/></body></worldbody>'
           '<actuator><position name="pa" joint="a" kp="9"/><general name="pc" joint="c" gear="2"/>'
           '<motor name="m" joint="a"/></actuator></mujoco>')
    out = motorized(xml)
    assert out.count("<motor") == 3 and "<position" not in out and "<general" not in out and "kp" not in out
    assert out.index('name="pa"') < out.index('name="pc"') < out.index('name="m"') and 'gear="2"' in out


def test_motorized_follows_includes_and_makes_paths_absolute(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    (tmp_path / "meshes").mkdir()
    (tmp_path / "arm.xml").write_text('<mujoco><worldbody><body name="b"><joint name="a"/><geom size="1"/></body>'
                                      '</worldbody><actuator><position joint="a"/></actuator></mujoco>')
    (tmp_path / "scene.xml").write_text('<mujoco><compiler meshdir="meshes"/><include file="arm.xml"/></mujoco>')
    out = motorized(tmp_path / "scene.xml")
    text = out.read_text()
    inc = text.split('file="')[1].split('"')[0]
    assert out.is_absolute() and inc.startswith(str(tmp_path / "cache")) and "<motor" in open(inc).read()


@pytest.mark.compile_heavy
def test_delay_responds_k_control_periods_late_per_world() -> None:
    env = make_env(robot(), absolute(kp=10.0, kd=0.0, delay_steps=(1, 3)), worlds=4)
    delays = np.asarray(env.writer._delays)
    first = np.full(4, -1)
    for k in range(8):
        env.step(np.full((4, 1), 0.0 if k < 2 else 0.5, np.float32))
        hit = (np.abs(env.writer.torque[:, 0]) > 1.0) & (first < 0)
        first[hit] = k
    print("delay per world", delays, " first torque at step", first)
    assert delays.min() >= 1 and delays.max() <= 3 and np.array_equal(first, 2 + delays)


@pytest.mark.compile_heavy
def test_no_delay_responds_in_the_same_period() -> None:
    env = make_env(robot(), absolute(kp=10.0, kd=0.0), worlds=2)
    drive(env, [0.0, 0.0], 1)
    assert np.all(np.abs(env.writer.torque) < 1e-3)
    drive(env, [0.5, 0.5], 1)
    assert np.all(env.writer.torque[:, 0] > 1.0)


@pytest.mark.compile_heavy
@pytest.mark.parametrize("kind", ["general", "position"])
def test_file_servo_is_refused_naming_motorized(kind: str) -> None:
    spec = ActuatorSpec(["arm/j"], mode="absolute", low=[-1.0], high=[1.0])
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", mjcf=robot("0 1 0", kind), pos=(0, 0, 0))
    sim = alloy.build(scene, device="cpu", plugins=[spec.plugin()])
    with pytest.raises(EnvError, match=r"motorized\(") as err:
        VectorEnv(sim, Task(actions=spec), backend="numpy")
    assert "servo" in err.value.message


@pytest.mark.compile_heavy
def test_missing_plugin_is_refused_with_the_build_recipe() -> None:
    spec = ActuatorSpec(["arm/j"], mode="absolute", low=[-1.0], high=[1.0])
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", mjcf=robot("0 1 0", "motor"), pos=(0, 0, 0))
    with pytest.raises(EnvError, match="plugins="):
        VectorEnv(alloy.build(scene, device="cpu"), Task(actions=spec), backend="numpy")


@pytest.mark.parametrize("bad", [dict(model="spring"), dict(mode="speed"), dict(delay_steps=(3, 1)),
                                 dict(model="dc_motor"), dict(kp=-1.0), dict(effort_limit=0.0)])
def test_spec_validation_names_the_field(bad: dict) -> None:
    with pytest.raises(EnvError):
        ActuatorSpec(["arm/j"], **bad).validate()


@pytest.mark.compile_heavy
def test_default_action_spec_keeps_the_plain_writer() -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", mjcf=robot("0 1 0", "position"), pos=(0, 0, 0))
    env = VectorEnv(alloy.build(scene, device="cpu"), Task(actions=ActionSpec(["arm/j"], mode="absolute",
                                                                              low=[-1.0], high=[1.0])), backend="numpy")
    assert type(env.writer) is ActionWriter


@pytest.mark.compile_heavy
def test_stable_gain_range_is_the_physics_step_bound() -> None:
    """Explicit PD at physics step h is stable when kp h^2 + 2 kd h < 4 I (and not beyond); at the control period
    (8 times h here) the same law would need 64 times softer kp, which is why the torque is recomputed every step."""
    probe = make_env(robot(), absolute(kp=1.0, kd=0.0), worlds=1)
    before = probe.sim.physics_steps
    drive(probe, [0.0], 1)
    h = probe.sim.dt / (probe.sim.physics_steps - before)
    for margin, stable in ((0.8, True), (1.25, False)):
        kd = 0.4 * margin * 4 * INERTIA / (2 * h)
        kp = 0.6 * margin * 4 * INERTIA / h**2
        env = make_env(robot(), absolute(kp=kp, kd=kd), worlds=1)
        drive(env, [0.005], 60)
        ratio = float(env.state["arm/j.q"][0]) / 0.005
        print(f"STABLE margin {margin}: h={h:.5f} kp={kp:.0f} kd={kd:.1f} -> q/target {ratio:.3f}")
        assert (abs(ratio - 1.0) < 0.01) == stable
