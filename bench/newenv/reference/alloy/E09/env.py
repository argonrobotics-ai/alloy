# SPDX-License-Identifier: Apache-2.0
"""E09 reference environment (Alloy): a Unitree Go2 standing on flat ground, joint-target offsets as the action.

Shipped Go2 (`emb_e8_go2`, Menagerie's model): position servos hold the standing pose at target 0. Action (12-D, in
[-1, 1]): offset of each joint target from the standing pose, 0.3 rad x action, order FL, FR, RL, RR x hip, thigh,
calf. Reward: forward (+x) speed of the trunk; success: standing and above 0.2 m/s forward. The `<general>` servos read
`ctrl` (a target angle in the file's joint frame, 0 = standing), so the action is written with kind="ctrl". The episode ends when the trunk is below 0.18 m or tilts past 1 rad
(57.3 deg), or after 10 s. Starts: the standing pose, shifted up to 5 cm and turned up to 10 deg about z.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import alloy
from alloy.cards.library import default_root
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs import checks as C
from alloy.envs import randomize as R

DT = 0.02
EPISODE_S = 10.0
OFFSET = 0.3
MIN_HEIGHT = 0.18
FORWARD_MPS = 0.2                 # success: standing and moving forward at least this fast
MAX_TILT_DEG = float(np.degrees(1.0))
LEGS = ("FL", "FR", "RL", "RR")
PARTS = ("hip", "thigh", "calf")
JOINTS = [f"go2/{leg}_{part}_joint" for leg in LEGS for part in PARTS]
SERVOS = [f"go2/{leg}_{part}" for leg in LEGS for part in PARTS]      # the file's <general> position servos


def _task() -> Task:
    fallen = C.height("go2/base") < MIN_HEIGHT
    tipped = C.tilt("go2/base") > MAX_TILT_DEG
    return Task(
        name="Go2Stand",
        obs=["go2"],
        actions=ActionSpec(SERVOS, kind="ctrl", mode="normalized", low=[-OFFSET] * 12, high=[OFFSET] * 12),
        reward={"forward": (1.0, C.custom("vx", lambda s: s["go2/base.vel"][:, 0]))},
        failure=fallen | tipped,
        max_steps=int(round(EPISODE_S / DT)),
        randomize=[R.jitter_body("go2/base", pos=(0.05, 0.05, 0.0), yaw_deg=10.0)])


class Go2StandEnv:
    """The row-7 contract over an Alloy VectorEnv."""

    act_dim = 12
    dt = DT
    episode_steps = int(round(EPISODE_S / DT))

    def __init__(self, num_envs: int, seed: int) -> None:
        scene = alloy.Scene(dt=DT)
        scene.add_robot("go2", default_root() / "robots" / "emb_e8_go2" / "scene.xml")
        self.num_envs = int(num_envs)
        self._seed: int | None = int(seed)
        self._env = VectorEnv(scene, _task(), worlds=self.num_envs, seed=int(seed), backend="numpy", device="cpu")
        self.obs_dim = int(self._env.single_observation_space.shape[0])

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset(seed=self._seed)
        self._seed = None
        return np.asarray(obs, np.float32)

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        act = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, self.act_dim), -1.0, 1.0)
        obs, reward, terminated, truncated, _ = self._env.step(act)
        return np.asarray(obs, np.float32), np.asarray(reward, np.float32), np.asarray(terminated | truncated)

    def probe(self) -> dict[str, np.ndarray]:
        s = self._env.state
        return {"base_pos": np.array(s["go2/base.pos"]), "base_quat": np.array(s["go2/base.quat_wxyz"]),
                "joint_pos": np.stack([np.array(s[f"{j}.q"]) for j in JOINTS], axis=1)}

    def success(self) -> np.ndarray:
        s = self._env.state
        standing = s["go2/base.pos"][:, 2] > MIN_HEIGHT
        return np.asarray(standing & (s["go2/base.vel"][:, 0] > FORWARD_MPS), bool)


def make_env(num_envs: int, seed: int) -> Go2StandEnv:
    return Go2StandEnv(num_envs, seed)
