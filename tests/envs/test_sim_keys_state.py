# SPDX-License-Identifier: Apache-2.0
"""The env state has every key ``sim.state()`` has (audit T2-016).

``env.state`` raised "no state key" for a heated liquid's ``"<name>.temperature"``, a metal sheet's or yielding wire's
``"<name>.plastic_strain"`` and a rod's ``"<name>.twist"``. A twist is now computed from the segments' poses on the
device; the others are read through the Sim (``alloy.envs.sim_keys``), and ``state.start`` holds their values at
each world's last reset.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv

ARM = """<mujoco><worldbody><body name="b" pos="0 0 0.05">
  <joint name="x" type="slide" axis="1 0 0" range="-0.2 0.2" damping="20"/>
  <geom type="box" size="0.02 0.02 0.02" mass="1.0"/></body></worldbody>
  <actuator><position joint="x" kp="1000" kv="50"/></actuator></mujoco>"""


def _env(scene: alloy.Scene) -> VectorEnv:
    scene.add_robot("arm", mjcf=ARM, pos=(-0.2, 0.0, 0.05))
    return VectorEnv(scene, Task(obs=["arm"], actions=ActionSpec(["arm/x"]), max_steps=50), worlds=2, device="cpu",
                     backend="numpy")


def _same_as_sim(env: VectorEnv, key: str, atol: float = 1e-6) -> np.ndarray:
    ours, sims = np.asarray(env.state[key]), env.sim.state(key.rpartition(".")[0])[key]
    assert key in env.state.keys() and ours.shape == sims.shape and np.allclose(ours, sims, atol=atol), key
    return ours


def _step(env: VectorEnv, n: int) -> None:
    for _ in range(n):
        env.step(np.zeros((2, 1), np.float32))


@pytest.mark.timeout(900)
def test_a_heated_liquids_temperature_and_its_start():
    sc = alloy.Scene(worlds=2, dt=0.01)
    sc.add_ground()
    sc.add_open_box("pan", inner=(0.06, 0.06, 0.03), wall=0.005, pos=(0.0, 0.0, 0.015))
    sc.add_liquid("water", size=(0.04, 0.04, 0.015), pos=(0.0, 0.0, 0.028), temperature=293.15)
    sc.set_heater("pan_floor", temperature=363.15)
    sc.set_heat(time_scale=60.0)
    env = _env(sc)
    env.reset(seed=0)
    _step(env, 10)
    now = _same_as_sim(env, "water.temperature")
    start = np.asarray(env.state.start["water.temperature"])
    assert np.allclose(start, 293.15, atol=1e-3) and now.mean() > start.mean() + 0.1   # heated since the reset
    env.close()


@pytest.mark.timeout(900)
def test_a_metal_sheets_plastic_strain():
    sc = alloy.Scene(worlds=2, dt=0.005, cloth_accuracy="fast")
    sc.add_ground()
    sc.add_cloth("foil", size=(0.06, 0.06), pos=(0.0, 0.0, 0.02), resolution=5, thickness=0.00025,
                 material="aluminium_6061_t6")
    env = _env(sc)
    env.reset(seed=0)
    _step(env, 3)
    _same_as_sim(env, "foil.plastic_strain")
    env.close()


@pytest.mark.timeout(900)
def test_a_yielding_wires_plastic_strain_and_twist():
    sc = alloy.Scene(worlds=2, dt=0.005)
    sc.add_ground()
    sc.add_rod("wire", length=0.12, radius=0.002, pos=(0.0, 0.0, 0.04), direction=(0, 1, 0),
               material="copper_c110", clamp="start", segments=6)
    env = _env(sc)
    env.reset(seed=0)
    _step(env, 3)
    _same_as_sim(env, "wire.plastic_strain")
    _same_as_sim(env, "wire.twist", atol=1e-5)
    env.close()


@pytest.mark.timeout(600)
def test_a_twisted_cable_reads_its_twist_on_the_device_and_a_rigid_scene_reads_nothing_through_the_sim():
    sc = alloy.Scene(worlds=2, dt=0.01)
    sc.add_ground()
    sc.add_rod("cable", length=0.3, radius=0.004, pos=(0.0, 0.0, 0.05), direction=(1, 0, 0), segments=8)
    env = _env(sc)
    assert env._sim_keys.names == {}                  # every key from the engine's arrays: no per-step device copy
    env.reset(seed=0)
    env.sim.apply_wrench("cable", torque=(0.004, 0.0, 0.0), at="end")
    _step(env, 10)
    twist = _same_as_sim(env, "cable.twist", atol=1e-5)
    assert twist.shape == (2, 7) and np.abs(twist).max() > 0.01                     # it really turned
    assert np.allclose(np.asarray(env.state.start["cable.twist"]), 0.0, atol=1e-5)
    env.close()
