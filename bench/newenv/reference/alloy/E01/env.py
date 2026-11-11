# SPDX-License-Identifier: Apache-2.0
"""E01 reference (Alloy): a 5 cm, 100 g cube pushed by a horizontal force on a 1 m table whose top is at z = 0.5 m.

Contract: ``make_env(num_envs, seed)``; action (N, 2) in [-1, 1] = force on the cube, 1 N per axis at 1.
Episodes of all worlds share one clock (done at ``episode_steps``); a cube that leaves the table stays on the floor.
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
TABLE_TOP = 0.5
CUBE = 0.05
MASS = 0.1
FORCE_N = 1.0
SPAWN_HALF = 0.4          # starts and targets within +-0.4 m of the table centre (table half-width 0.5 m)
SUCCESS_M = 0.03
FRICTION = 0.3


def _spots(rng: np.random.Generator, n: int, z: float) -> np.ndarray:
    xy = rng.uniform(-SPAWN_HALF, SPAWN_HALF, (n, 2))
    return np.concatenate([xy, np.full((n, 1), z)], axis=1).astype(np.float32)


class PushCubeEnv:
    obs_dim, act_dim, dt = 8, 2, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self._start = _spots(self._rng, num_envs, TABLE_TOP + CUBE / 2)
        self._target = _spots(self._rng, num_envs, TABLE_TOP)
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground()
        scene.add_box("table", size=(1.0, 1.0, TABLE_TOP), pos=(0.0, 0.0, TABLE_TOP / 2), static=True,
                      friction=FRICTION)
        scene.add_box("cube", size=(CUBE,) * 3, pos=self._start, mass=MASS, friction=FRICTION)
        self._sim = alloy.build(scene, seed=seed, device="cpu")
        self._k = 0
        self._fresh = True

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._start = _spots(self._rng, self.num_envs, TABLE_TOP + CUBE / 2)
            self._target = _spots(self._rng, self.num_envs, TABLE_TOP)
            self._sim.reset()
            self._sim.set_pose("cube", pos=self._start, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        self._fresh = False
        self._k = 0
        return self._obs()

    def _state(self) -> dict[str, np.ndarray]:
        return self._sim.state(["cube"])

    def _obs(self) -> np.ndarray:
        s = self._state()
        return np.concatenate([s["cube.pos"], s["cube.vel"][:, :3], self._target[:, :2]], axis=1).astype(np.float32)

    def _distance(self) -> np.ndarray:
        return np.linalg.norm(self._state()["cube.pos"][:, :2] - self._target[:, :2], axis=1)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, 2), -1.0, 1.0)
        force = np.concatenate([a * FORCE_N, np.zeros((self.num_envs, 1), np.float32)], axis=1)
        self._sim.apply_wrench("cube", force=force)
        self._sim.step(1)
        self._k += 1
        dist = self._distance()
        reward = (-dist + (dist < SUCCESS_M)).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        return {"cube_pos": self._state()["cube.pos"].astype(np.float32), "target_pos": self._target.copy()}

    def success(self) -> np.ndarray:
        return self._distance() < SUCCESS_M


def make_env(num_envs: int, seed: int) -> PushCubeEnv:
    return PushCubeEnv(num_envs, seed)
