# SPDX-License-Identifier: Apache-2.0
"""E05 reference (Alloy): ten dominoes (8 x 24 x 48 mm, 10 g) in a row along +x, 25 mm between faces (33 mm pitch).

Action (N, 1) in [-1, 1] = horizontal force on the top of the first domino along +x, 0.5 N at 1. The force is applied
at the top point of that domino (re-computed from its pose every step, so it turns with it) as a force plus the torque
about its centre. Success: all ten tilted more than 60 degrees. Done at ``episode_steps`` for all worlds.
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
THICK, WIDTH, HEIGHT = 0.008, 0.024, 0.048
MASS = 0.01
GAP = 0.025
COUNT = 10
FORCE_N = 0.5
FRICTION = 0.5
TOPPLED = np.radians(60.0)
PITCH = THICK + GAP
NAMES = [f"d{i}" for i in range(COUNT)]


def _tilt(quat_wxyz: np.ndarray) -> np.ndarray:
    q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    return np.arccos(np.clip(1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0))


def _top_offset(q: np.ndarray) -> np.ndarray:
    """World vector from a domino's centre to its top face centre, for quaternions (N, 4) wxyz."""
    w, x, y, z = q.T
    axis_z = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)
    return axis_z * (HEIGHT / 2)


class DominoEnv:
    obs_dim, act_dim, dt = COUNT * 4 + 1, 1, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground(friction=FRICTION)
        for i, name in enumerate(NAMES):
            scene.add_box(name, size=(THICK, WIDTH, HEIGHT), pos=(i * PITCH, 0.0, HEIGHT / 2), mass=MASS,
                          friction=FRICTION)
        self._sim = alloy.build(scene, seed=seed, device="cpu")
        self._k = 0
        self._fresh = True

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._sim.reset()
        self._fresh = False
        self._sim.apply_wrench(NAMES[0])
        self._k = 0
        return self._obs()

    def _state(self) -> dict[str, np.ndarray]:
        return self._sim.state(NAMES)

    def _pos_quat(self) -> tuple[np.ndarray, np.ndarray]:
        s = self._state()
        pos = np.stack([s[f"{n}.pos"] for n in NAMES], axis=1)
        quat = np.stack([s[f"{n}.quat_wxyz"] for n in NAMES], axis=1)
        return pos, quat

    def _obs(self) -> np.ndarray:
        pos, quat = self._pos_quat()
        return np.concatenate([pos[:, :, 0], pos[:, :, 2], _tilt(quat), np.zeros((self.num_envs, COUNT)),
                               np.zeros((self.num_envs, 1))], axis=1)[:, :self.obs_dim].astype(np.float32)

    def _toppled(self) -> np.ndarray:
        return _tilt(self._pos_quat()[1]) > TOPPLED

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 1), -1.0, 1.0)
        force = np.concatenate([a * FORCE_N, np.zeros((self.num_envs, 2))], axis=1)
        _, quat = self._pos_quat()
        torque = np.cross(_top_offset(quat[:, 0]), force)
        self._sim.apply_wrench(NAMES[0], force=force.astype(np.float32), torque=torque.astype(np.float32))
        self._sim.step(1)
        self._k += 1
        reward = self._toppled().mean(axis=1).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        pos, quat = self._pos_quat()
        return {"domino_pos": pos.astype(np.float32), "domino_quat": quat.astype(np.float32)}

    def success(self) -> np.ndarray:
        return self._toppled().all(axis=1)


def make_env(num_envs: int, seed: int) -> DominoEnv:
    return DominoEnv(num_envs, seed)
