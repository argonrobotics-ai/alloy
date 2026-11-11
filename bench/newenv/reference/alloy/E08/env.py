# SPDX-License-Identifier: Apache-2.0
"""E08 reference environment (Alloy): a sprung, damped door with a torque on it.

A 0.8 m wide, 2 m tall, 10 kg door turns on a vertical hinge (0 to 1.5 rad) with a closing spring of 2 N m/rad and
damping of 1 N m s/rad. Action: torque on the door, 3 N m x action. Reward: minus the distance of the angle from
1.0 rad. Starts closed (within 0.005 rad).
"""
from __future__ import annotations

from typing import Any

import numpy as np

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs import checks as C
from alloy.envs import randomize as R

DT = 0.01
EPISODE_S = 5.0
GOAL = 1.0
GOAL_TOL = 0.05

# A 4 cm thick slab hinged on one vertical edge: I = m (w^2 + t^2) / 12 + m (w / 2)^2 = 2.135 kg m^2.
DOOR = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="leaf">
    <joint name="hinge" type="hinge" axis="0 0 1" limited="true" range="0 1.5" stiffness="2" springref="0"
           damping="1"/>
    <geom type="box" size="0.4 0.02 1.0" pos="0.4 0 1.0" mass="10" contype="0" conaffinity="0"/>
  </body></worldbody>
  <actuator><motor name="push" joint="hinge" gear="3" ctrlrange="-1 1"/></actuator></mujoco>"""


def _error(s: Any) -> Any:
    return abs(s["door/hinge.q"] - GOAL)


def _task() -> Task:
    return Task(
        name="SpringDoor",
        obs=[C.custom("angle", lambda s: s["door/hinge.q"]), C.custom("rate", lambda s: s["door/hinge.qd"])],
        actions=ActionSpec(["door/push"], kind="ctrl", mode="normalized", low=[-1.0], high=[1.0]),
        reward={"at_goal": (-1.0, C.custom("error", _error))},
        max_steps=int(round(EPISODE_S / DT)),
        randomize=[R.jitter_joint("door/hinge", at=0.0025, by=0.0025)])


class SpringDoorEnv:
    """The row-7 contract over an Alloy VectorEnv."""

    obs_dim = 2
    act_dim = 1
    dt = DT
    episode_steps = int(round(EPISODE_S / DT))

    def __init__(self, num_envs: int, seed: int) -> None:
        scene = alloy.Scene(dt=DT)
        scene.add_robot("door", mjcf=DOOR, pos=(0.0, 0.0, 0.0))
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
        return {"door_angle": np.array(s["door/hinge.q"]), "door_rate": np.array(s["door/hinge.qd"])}

    def success(self) -> np.ndarray:
        return np.asarray(_error(self._env.state) < GOAL_TOL, bool)


def make_env(num_envs: int, seed: int) -> SpringDoorEnv:
    return SpringDoorEnv(num_envs, seed)
