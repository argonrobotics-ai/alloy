# SPDX-License-Identifier: Apache-2.0
"""E07 reference environment (Alloy): a double pendulum with a torque at the elbow.

Two 0.3 m, 0.5 kg uniform rods hang from a fixed pivot on frictionless hinges and start at random joint angles
within +-0.5 rad. Action: elbow torque, 2 N m x action. Reward: tip height above the pivot / 0.6 m (1 = straight up).

joint_pos = (shoulder, elbow): shoulder from straight down, elbow relative to the upper rod, both positive towards
+x (rotation about -y). The pendulum swings in the x-z plane.
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
ROD = 0.3
PIVOT = (0.0, 0.0, 1.0)
START = 0.5

# Rod inertia written out (uniform rod about its centre: m L^2 / 12 = 0.5 * 0.09 / 12).
PENDULUM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="upper">
    <joint name="shoulder" type="hinge" axis="0 -1 0"/>
    <inertial pos="0 0 -0.15" mass="0.5" diaginertia="0.00375 0.00375 0.00001"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.01" mass="0" contype="0" conaffinity="0"/>
    <body name="lower" pos="0 0 -0.3">
      <joint name="elbow" type="hinge" axis="0 -1 0"/>
      <inertial pos="0 0 -0.15" mass="0.5" diaginertia="0.00375 0.00375 0.00001"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.01" mass="0" contype="0" conaffinity="0"/>
    </body>
  </body></worldbody>
  <actuator><motor name="elbow_motor" joint="elbow" gear="2" ctrlrange="-1 1"/></actuator></mujoco>"""


def _rotate(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q_wxyz[:, :1], q_wxyz[:, 1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _tip(s: Any) -> np.ndarray:
    """World position of the lower rod's free end (its body frame sits at the elbow)."""
    pos, quat = np.asarray(s["dp/lower.pos"]), np.asarray(s["dp/lower.quat_wxyz"])
    return pos + _rotate(quat, np.broadcast_to(np.array([0.0, 0.0, -ROD], np.float32), pos.shape))


def _height(s: Any) -> np.ndarray:
    return (_tip(s)[:, 2] - PIVOT[2]) / (2.0 * ROD)


def _task() -> Task:
    joints = ["dp/shoulder", "dp/elbow"]
    obs = [C.custom(f"{j}.{f}", lambda s, k=f"{j}.{f}": s[k]) for j in joints for f in ("q", "qd")]
    return Task(
        name="DoublePendulum", obs=obs,
        actions=ActionSpec(["dp/elbow_motor"], kind="ctrl", mode="normalized", low=[-1.0], high=[1.0]),
        reward={"tip_height": (1.0, C.custom("tip_height", _height))},
        max_steps=int(round(EPISODE_S / DT)),
        randomize=[R.jitter_joint(j, by=START) for j in joints])


class DoublePendulumEnv:
    """The row-7 contract over an Alloy VectorEnv."""

    obs_dim = 4
    act_dim = 1
    dt = DT
    episode_steps = int(round(EPISODE_S / DT))

    def __init__(self, num_envs: int, seed: int) -> None:
        scene = alloy.Scene(dt=DT)
        scene.add_robot("dp", mjcf=PENDULUM, pos=PIVOT)
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
        return {"joint_pos": np.stack([s["dp/shoulder.q"], s["dp/elbow.q"]], axis=1),
                "joint_vel": np.stack([s["dp/shoulder.qd"], s["dp/elbow.qd"]], axis=1),
                "tip_pos": _tip(s)}

    def success(self) -> np.ndarray:
        return _height(self._env.state) > 0.0


def make_env(num_envs: int, seed: int) -> DoublePendulumEnv:
    return DoublePendulumEnv(num_envs, seed)
