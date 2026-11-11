# SPDX-License-Identifier: Apache-2.0
"""E10 reference environment (Alloy): a four-wheeled, skid-steered cart driven by wheel torques.

A 30 x 20 x 5 cm, 2 kg body rides on four wheels of 4 cm radius (0.1 kg each; the request gives no wheel mass).
Action: (left, right) wheel torque, 0.2 N m x action on each wheel of that side. Each episode starts at a random
pose (within +-0.5 m of the origin, any heading) with a random goal on the ground within 2 m of it. Reward: minus
the distance to the goal; success within 10 cm.

cart_yaw is the heading of the body's long (+x) axis from world +x, about +z; "left" is the body's +y side.
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
RIDE_Z = 0.05                    # body centre above the ground: wheel radius 0.04 + axle 0.01 below the centre
START_XY = 0.5
GOAL_XY = 0.9                    # start +-0.5 and goal +-0.9 per axis: at most 1.98 m apart
SUCCESS_M = 0.10
WHEELS = ("fl", "rl", "fr", "rr")                  # left pair first

_WHEEL = """<body name="{n}" pos="{x} {y} -0.01">
      <joint name="{n}_spin" type="hinge" axis="0 1 0"/>
      <geom type="cylinder" size="0.04 0.015" euler="1.5707963 0 0" mass="0.1" friction="1.0 0.005 0.0001"/>
    </body>"""
CART = ("""<mujoco><compiler angle="radian"/><worldbody>
  <body name="body"><freejoint/>
    <geom type="box" size="0.15 0.10 0.025" mass="2"/>
    """ + "\n    ".join(_WHEEL.format(n=n, x=x, y=y) for n, x, y in
                        (("fl", 0.1, 0.125), ("rl", -0.1, 0.125), ("fr", 0.1, -0.125), ("rr", -0.1, -0.125)))
        + """
  </body></worldbody>
  <actuator>""" + "".join(f'<motor name="{n}_motor" joint="{n}_spin" gear="0.2" ctrlrange="-1 1"/>' for n in WHEELS)
        + "</actuator></mujoco>")


def _distance(s: Any) -> np.ndarray:
    return np.linalg.norm(np.asarray(s["cart/body.pos"])[:, :2] - np.asarray(s["goal.pos"])[:, :2], axis=1)


def _yaw(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (q_wxyz[:, i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _task() -> Task:
    obs = ["cart", C.custom("goal_x", lambda s: s["goal.pos"][:, 0]), C.custom("goal_y", lambda s: s["goal.pos"][:, 1])]
    return Task(
        name="WheeledCart", obs=obs,
        actions=ActionSpec([f"cart/{n}_motor" for n in WHEELS], kind="ctrl", mode="normalized",
                           low=[-1.0] * 4, high=[1.0] * 4),
        reward={"distance": (-1.0, C.custom("distance", _distance))},
        success=C.custom("at_goal", lambda s: _distance(s) < SUCCESS_M), end_on_success=False,
        max_steps=int(round(EPISODE_S / DT)),
        goals={"goal": ((-GOAL_XY, -GOAL_XY, 0.0), (GOAL_XY, GOAL_XY, 0.0))},
        randomize=[R.jitter_body("cart/body", pos=(START_XY, START_XY, 0.0), yaw_deg=180.0)])


class WheeledCartEnv:
    """The row-7 contract over an Alloy VectorEnv."""

    act_dim = 2
    dt = DT
    episode_steps = int(round(EPISODE_S / DT))

    def __init__(self, num_envs: int, seed: int) -> None:
        scene = alloy.Scene(dt=DT)
        scene.add_ground()
        scene.add_robot("cart", mjcf=CART, pos=(0.0, 0.0, RIDE_Z))
        self.num_envs = int(num_envs)
        self._seed: int | None = int(seed)
        self._env = VectorEnv(scene, _task(), worlds=self.num_envs, seed=int(seed), backend="numpy", device="cpu")
        self.obs_dim = int(self._env.single_observation_space.shape[0])

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None                                   # later resets draw new starts
        return np.asarray(obs, np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        act = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, self.act_dim), -1.0, 1.0)
        wheels = act[:, [0, 0, 1, 1]]                       # left pair, right pair
        obs, reward, terminated, truncated, _ = self._env.step(wheels)
        return np.asarray(obs, np.float32), np.asarray(reward, np.float32), np.asarray(terminated | truncated)

    def probe(self) -> dict[str, np.ndarray]:
        s = self._env.state
        return {"cart_pos": np.array(s["cart/body.pos"]), "cart_yaw": _yaw(np.asarray(s["cart/body.quat_wxyz"])),
                "goal_pos": np.array(s["goal.pos"])}

    def success(self) -> np.ndarray:
        return _distance(self._env.state) < SUCCESS_M


def make_env(num_envs: int, seed: int) -> WheeledCartEnv:
    return WheeledCartEnv(num_envs, seed)
