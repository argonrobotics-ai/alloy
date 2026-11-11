# SPDX-License-Identifier: Apache-2.0
"""E12 reference (Alloy): a 50 cm, 100 g cloth dropped flat from 10 cm above a 10 cm cube on a table.

The cube is kinematic and moved with ``set_pose`` every control period (``sim.drive`` takes one velocity for all
worlds; the action is per world). Reward: how far the cloth's edge reaches past the cube's nearest side, scaled; success
when that margin is at least 2 cm on all four sides.
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
TABLE_TOP = 0.5
CUBE = 0.1
CLOTH = 0.5
DROP = 0.1
SPEED = 0.1
SPAWN = 0.1                 # the cube's start is within +-0.1 m of the table centre
LIMIT = 0.25                # and it never leaves +-0.25 m
SUCCESS_M = 0.02
RES = 13


class ClothCoverEnv:
    obs_dim, act_dim, dt = 7, 2, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self._cube0 = self._draw()
        scene = alloy.Scene(dt=DT, worlds=num_envs)
        scene.add_ground()
        scene.add_box("table", size=(1.0, 1.0, TABLE_TOP), pos=(0, 0, TABLE_TOP / 2), static=True, friction=0.5)
        scene.add_box("cube", size=(CUBE,) * 3, pos=self._cube0, kinematic=True, friction=0.5)
        scene.add_cloth("cloth", size=(CLOTH, CLOTH), pos=self._cloth_start(self._cube0), resolution=RES, mass=0.1)
        self._sim = _build(scene, seed)
        self._cube = self._cube0.copy()
        self._k = 0
        self._fresh = True

    def _draw(self) -> np.ndarray:
        xy = self._rng.uniform(-SPAWN, SPAWN, (self.num_envs, 2))
        return np.concatenate([xy, np.full((self.num_envs, 1), TABLE_TOP + CUBE / 2)], axis=1)

    @staticmethod
    def _cloth_start(cube: np.ndarray) -> np.ndarray:
        return cube * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, TABLE_TOP + CUBE + DROP])

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._sim.reset()                    # a cloth cannot be teleported: every episode of a world starts alike
            self._sim.set_pose("cube", pos=self._cube0, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        self._fresh = False
        self._cube = self._cube0.copy()
        self._k = 0
        return self._obs()

    def _points(self) -> np.ndarray:
        return self._sim.state(["cloth"])["cloth.points"]

    def _margin(self) -> np.ndarray:
        """Smallest reach (m) of the cloth past the cube's four sides; negative: a side is not covered."""
        p = self._points()
        lo = self._cube[:, None, :2] - CUBE / 2
        hi = self._cube[:, None, :2] + CUBE / 2
        past_lo = (lo - p[..., :2]).max(axis=1)
        past_hi = (p[..., :2] - hi).max(axis=1)
        return np.minimum(past_lo, past_hi).min(axis=1)

    def _obs(self) -> np.ndarray:
        p = self._points()
        return np.concatenate([self._cube, p.mean(axis=1)[:, :2], self._margin()[:, None], p[:, :, 2].max(axis=1)[:, None]],
                              axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, 2), -1.0, 1.0)
        self._cube[:, :2] = np.clip(self._cube[:, :2] + a * SPEED * DT, -LIMIT, LIMIT)
        self._sim.set_pose("cube", pos=self._cube)
        self._sim.step(1)
        self._k += 1
        margin = self._margin()
        reward = np.clip(margin / SUCCESS_M, -1.0, 1.0).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        return {"cloth_points": self._points().astype(np.float32), "cube_pos": self._cube.astype(np.float32),
                "table_z": np.full(self.num_envs, TABLE_TOP, np.float32)}

    def success(self) -> np.ndarray:
        return self._margin() >= SUCCESS_M


def _build(scene: alloy.Scene, seed: int) -> alloy.Sim:
    """One L4 when there is one; else the CPU, one process per core for many worlds (AGENTS.md section 1)."""
    import warp as wp  # noqa: PLC0415

    if wp.get_cuda_device_count() > 0:
        return alloy.build(scene, seed=seed)
    return alloy.build(scene, seed=seed, device="cpu", processes="auto" if scene.worlds > 1 else None)


def make_env(num_envs: int, seed: int) -> ClothCoverEnv:
    return ClothCoverEnv(num_envs, seed)
