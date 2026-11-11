# SPDX-License-Identifier: Apache-2.0
"""E06 reference environment (Alloy): a cart-pole with a force on the cart.

A 1 kg cart on a frictionless rail (+-2 m) carries a 0.5 m, 0.1 kg uniform pole on a free hinge. The pole starts
within 0.05 rad of upright. Action: force on the cart, 10 N x action. Reward 1 per step while upright. The episode
ends when |pole angle| > 0.4 rad or the cart reaches the rail's end.

pole_angle is measured from upright, positive when the tip leans towards +x.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs import checks as C
from alloy.envs import randomize as R

DT = 0.01
EPISODE_S = 6.0
RAIL = 2.0
MAX_ANGLE = 0.4
START_ANGLE = 0.05

# The pole's inertia is written out (uniform rod about its centre: m L^2 / 12) so no capsule end caps change it.
CARTPOLE = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="cart">
    <joint name="slide" type="slide" axis="1 0 0" range="-2 2"/>
    <geom type="box" size="0.1 0.05 0.03" mass="1" contype="0" conaffinity="0"/>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0.25" mass="0.1" diaginertia="0.00208333 0.00208333 0.00001"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.01" mass="0" contype="0" conaffinity="0"/>
    </body>
  </body></worldbody>
  <actuator><motor name="push" joint="slide" gear="10" ctrlrange="-1 1"/></actuator></mujoco>"""


def _fallen(s: Any) -> Any:
    return (abs(s["cp/hinge.q"]) > MAX_ANGLE) | (abs(s["cp/slide.q"]) > RAIL - 0.01)


def _task() -> Task:
    fallen = C.custom("fallen", _fallen)
    return Task(
        name="CartPole",
        obs=[C.custom("x", lambda s: s["cp/slide.q"]), C.custom("angle", lambda s: s["cp/hinge.q"]),
             C.custom("v", lambda s: s["cp/slide.qd"]), C.custom("rate", lambda s: s["cp/hinge.qd"])],
        actions=ActionSpec(["cp/push"], kind="ctrl", mode="normalized", low=[-1.0], high=[1.0]),
        reward={"upright": (1.0, 1.0 - fallen)},
        failure=fallen,
        max_steps=int(round(EPISODE_S / DT)),
        randomize=[R.jitter_joint("cp/hinge", by=START_ANGLE)])


class CartPoleEnv:
    """The row-7 contract over an Alloy VectorEnv."""

    obs_dim = 4
    act_dim = 1
    dt = DT
    episode_steps = int(round(EPISODE_S / DT))

    def __init__(self, num_envs: int, seed: int) -> None:
        scene = alloy.Scene(dt=DT)
        scene.add_robot("cp", mjcf=CARTPOLE, pos=(0.0, 0.0, 1.0))
        self.num_envs = int(num_envs)
        self._seed: int | None = int(seed)
        self._env = VectorEnv(scene, _task(), worlds=self.num_envs, seed=int(seed), backend="numpy", device="cpu")

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None                                   # later resets draw new starts
        return np.asarray(obs, np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        act = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, self.act_dim), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = self._env.step(act)
        return np.asarray(obs, np.float32), np.asarray(reward, np.float32), np.asarray(terminated | truncated)

    def probe(self) -> dict[str, np.ndarray]:
        s = self._env.state
        return {"cart_x": np.array(s["cp/slide.q"]), "pole_angle": np.array(s["cp/hinge.q"]),
                "cart_v": np.array(s["cp/slide.qd"]), "pole_rate": np.array(s["cp/hinge.qd"])}

    def success(self) -> np.ndarray:
        return ~np.asarray(_fallen(self._env.state), bool)


def make_env(num_envs: int, seed: int) -> CartPoleEnv:
    return CartPoleEnv(num_envs, seed)
