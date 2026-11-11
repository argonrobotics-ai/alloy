# SPDX-License-Identifier: Apache-2.0
"""Harness test double for request E06 (cart-pole), integrated in numpy. Not a reference environment.

Options through ``FAKE_ENV_FAULT`` (for the driver's tests): "nan" (NaN obs after 30 steps), "crash" (raises at step
30), "shape" (reward (N, 1)), "noseed" (starts ignore the seed and differ on every make).
"""
from __future__ import annotations

import os

import numpy as np

G, M_CART, M_POLE, LENGTH, FORCE, RAIL, ANGLE_END = 9.81, 1.0, 0.1, 0.5, 10.0, 2.0, 0.4
SUBSTEPS = 10


class CartPole:
    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs, self.obs_dim, self.act_dim = num_envs, 4, 1
        self.dt, self.episode_steps = 0.01, 500
        self.fault = os.environ.get("FAKE_ENV_FAULT", "")
        self.rng = np.random.default_rng(None if self.fault == "noseed" else seed)
        self.state = np.zeros((num_envs, 4))  # x, angle, v, rate
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.total = 0
        self.last_success = np.zeros(num_envs, dtype=bool)

    def _draw(self, count: int) -> np.ndarray:
        start = np.zeros((count, 4))
        start[:, 0] = self.rng.uniform(-0.1, 0.1, count)
        start[:, 1] = self.rng.uniform(-0.05, 0.05, count)
        return start

    def _obs(self) -> np.ndarray:
        return self.state.astype(np.float32)

    def reset(self) -> np.ndarray:
        self.state = self._draw(self.num_envs)
        self.steps[:] = 0
        self.last_success[:] = False
        return self._obs()

    def _integrate(self, force: np.ndarray) -> None:
        h, half, total = self.dt / SUBSTEPS, 0.5 * LENGTH, M_CART + M_POLE
        x, th, v, w = self.state.T.copy()
        for _ in range(SUBSTEPS):
            sin, cos = np.sin(th), np.cos(th)
            tmp = (force + M_POLE * half * w * w * sin) / total
            th_acc = (G * sin - cos * tmp) / (half * (4.0 / 3.0 - M_POLE * cos * cos / total))
            x_acc = tmp - M_POLE * half * th_acc * cos / total
            v, w = v + h * x_acc, w + h * th_acc
            x, th = x + h * v, th + h * w
        self.state = np.stack([x, th, v, w], axis=1)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.total += 1
        if self.fault == "crash" and self.total >= 30:
            raise RuntimeError("fake engine crash")
        self._integrate(FORCE * np.clip(np.asarray(action, np.float64)[:, 0], -1.0, 1.0))
        self.steps += 1
        fell = (np.abs(self.state[:, 1]) > ANGLE_END) | (np.abs(self.state[:, 0]) >= RAIL)
        timeout = self.steps >= self.episode_steps
        reward = (1.0 - (self.state[:, 1] / ANGLE_END) ** 2).astype(np.float32)
        self.last_success = timeout & ~fell
        done = fell | timeout
        if done.any():
            self.state[done] = self._draw(int(done.sum()))
            self.steps[done] = 0
        obs = self._obs()
        if self.fault == "nan" and self.total >= 30:
            obs[0, 0] = np.nan
        return obs, (reward[:, None] if self.fault == "shape" else reward), done

    def probe(self) -> dict[str, np.ndarray]:
        x, th, v, w = self.state.T
        return {"cart_x": x.copy(), "pole_angle": th.copy(), "cart_v": v.copy(), "pole_rate": w.copy()}

    def success(self) -> np.ndarray:
        return self.last_success.copy()


def make_env(num_envs: int, seed: int) -> CartPole:
    return CartPole(num_envs, seed)


def scripted_policy(env: CartPole, obs: np.ndarray) -> np.ndarray:
    """State feedback that balances the pole (used when a test marks the request as a policy request)."""
    gains = np.array([1.0, 30.0, 2.0, 6.0])
    return np.clip(np.asarray(obs) @ gains / FORCE, -1.0, 1.0)[:, None]
