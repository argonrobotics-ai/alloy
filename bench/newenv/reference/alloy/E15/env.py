# SPDX-License-Identifier: Apache-2.0
"""E15 reference (Alloy): a cup of 200 ml of water held above a bowl and tilted about its lip.

Action (1,): tilt rate of the cup about its +x lip, up to 1 rad/s at +-1 (positive tips the +x side down, toward the
bowl). Reward: the fraction of the water inside the bowl's radius and above the ground. All worlds share one clock
(Alloy cannot reset a liquid world by itself); the cup is kinematic and moved by ``set_pose`` once per control
period, a pose per world (``sim.drive`` takes one angular velocity for every world).
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 10.0
CUP_R, CUP_DEPTH, WALL = 0.04, 0.10, 0.004          # inner radius, inner depth, wall (m)
CUP_Z = 0.22                                        # cup floor height (m)
FILL_M3 = 200e-6 * 1.18                       # the box is filled with whole particles: about 15 % under its volume
BOWL_R, BOWL_DEPTH, BOWL_X = 0.10, 0.08, 0.17       # inner radius, depth, bowl axis x (m); the cup lip is at x = 0.044
RATE_MAX = 1.0
TILT_MAX = 2.8
SUCCESS_FRACTION = 0.7
LIP = np.array([CUP_R + WALL, 0.0, CUP_Z + WALL + CUP_DEPTH])
CUP_POS = np.array([0.0, 0.0, CUP_Z])


def rotate_about_lip(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cup floor position (W, 3) and quaternion wxyz (W, 4) after tipping by theta about +y through the lip."""
    c, s = np.cos(theta), np.sin(theta)
    d = CUP_POS - LIP
    pos = np.stack([LIP[0] + c * d[0] + s * d[2], np.full_like(theta, d[1]), LIP[2] - s * d[0] + c * d[2]], axis=1)
    pos[:, 1] = LIP[1] + d[1]
    quat = np.stack([np.cos(theta / 2), np.zeros_like(theta), np.sin(theta / 2), np.zeros_like(theta)], axis=1)
    return pos.astype(np.float32), quat.astype(np.float32)


class CupPourEnv:
    obs_dim, act_dim, dt = 3, 1, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground()
        scene.add_open_cylinder("cup", inner_radius=CUP_R, depth=CUP_DEPTH, wall=WALL, pos=tuple(CUP_POS), kinematic=True)
        h = FILL_M3 / (np.pi * CUP_R ** 2)
        scene.add_liquid("water", size=(2 * CUP_R - 0.002, 2 * CUP_R - 0.002, h), shape="cylinder",
                         pos=(0.0, 0.0, CUP_Z + WALL + h / 2 + 0.001))
        scene.add_open_cylinder("bowl", inner_radius=BOWL_R, depth=BOWL_DEPTH, wall=WALL, pos=(BOWL_X, 0.0, 0.0))
        self._sim = alloy.build(scene, seed=seed)
        self._sim.settle(0.3)
        self._theta = np.zeros(num_envs)
        self._k = 0
        self._pv = float(self._sim.particle_volume("water"))

    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._theta[:] = 0.0
        self._k = 0
        return self._obs()

    def _points(self) -> np.ndarray:
        return self._sim.state(["water"])["water.points"]

    def _fractions(self) -> tuple[np.ndarray, np.ndarray]:
        p = self._points()
        in_bowl = (np.hypot(p[..., 0] - BOWL_X, p[..., 1]) < BOWL_R) & (p[..., 2] >= 0.0)
        in_cup = (np.hypot(p[..., 0], p[..., 1]) < CUP_R + WALL) & (p[..., 2] > CUP_Z) & (self._theta[:, None] < 0.3)
        return in_bowl.mean(axis=1), in_cup.mean(axis=1)

    def _obs(self) -> np.ndarray:
        bowl, cup = self._fractions()
        k = np.full(self.num_envs, self._k / self.episode_steps)
        return np.stack([self._theta, bowl, k], axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 1), -1.0, 1.0)[:, 0]
        self._theta = np.clip(self._theta + a * RATE_MAX * DT, 0.0, TILT_MAX)
        pos, quat = rotate_about_lip(self._theta)
        self._sim.set_pose("cup", pos=pos, quat_wxyz=quat)
        self._sim.step(1)
        self._k += 1
        bowl, _ = self._fractions()
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        reward = bowl.astype(np.float32)
        self._last_success = bowl >= SUCCESS_FRACTION
        if done[0]:
            obs = self.reset()
            return obs, reward, done
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        return {"liquid_points": self._points().astype(np.float32), "particle_volume": np.full(n, self._pv, np.float32),
                "cup_tilt": self._theta.astype(np.float32),
                "bowl_pos": np.tile(np.array([BOWL_X, 0.0, 0.0], np.float32), (n, 1)),
                "bowl_radius": np.full(n, BOWL_R, np.float32)}

    def success(self) -> np.ndarray:
        return getattr(self, "_last_success", np.zeros(self.num_envs, bool))


def make_env(num_envs: int, seed: int) -> CupPourEnv:
    return CupPourEnv(num_envs, seed)
