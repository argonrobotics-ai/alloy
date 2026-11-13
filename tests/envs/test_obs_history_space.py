# SPDX-License-Identifier: Apache-2.0
"""An env's observation spaces hold what it returns, also with ``ObsHistory(flatten=False)`` (audit T1-010).

``ObsHistory(3, flatten=False)`` returns (W, 3, D) observations, but the spaces were built from ``shape[1]`` alone:
a (3,) single-world Box for (3, D) observations, which Gymnasium's wrappers, Stable-Baselines3 and skrl then reject.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs.realism import ObsHistory

SLIDER = """<mujoco><worldbody><body name="cart">
  <joint name="x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="20"/>
  <geom type="box" size="0.02 0.02 0.02" mass="0.5"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="100"/></actuator></mujoco>"""


@pytest.mark.timeout(300)
@pytest.mark.parametrize("flatten", [True, False])
def test_the_spaces_hold_the_observations(flatten):
    sc = alloy.Scene(dt=0.02)
    sc.add_ground()
    sc.add_robot("slider", mjcf=SLIDER, pos=(0.0, 0.0, 0.1))
    task = Task(obs=["slider"], actions=ActionSpec(["slider/x"], mode="delta"), max_steps=20,
                layers=[ObsHistory(3, flatten=flatten)])
    env = VectorEnv(sc, task, worlds=2, device="cpu", backend="numpy")
    obs, _ = env.reset(seed=0)
    obs = np.asarray(obs)
    env.close()
    assert tuple(env.single_observation_space.shape) == obs.shape[1:]
    assert tuple(env.observation_space.shape) == obs.shape
    assert obs.shape[1:] == ((3, obs.shape[-1]) if not flatten else (obs.shape[-1],))
