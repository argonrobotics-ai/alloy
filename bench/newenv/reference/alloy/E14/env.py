# SPDX-License-Identifier: Apache-2.0
"""E14 reference (Alloy): a 1 m, 50 g rope hangs from a fixed point 1.5 m up; a gripper holds its free end.

The rope is ``add_rod(clamp="start", route="chain")`` (the anchor). Alloy has no call that attaches a rod's end to a
kinematic body or a robot hand (friction grips only), so the gripper is a force servo on the free end segment
(``sim.apply_wrench(..., at="end")``, a PD law with velocity feed-forward on a target that moves at the commanded speed).
Reward: minus the distance from the rope's midpoint to a random target (reachable: within 0.5 m of the anchor).
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.01                       # the servo's period: its end segment is ~4 g (the whole rope 50 g), so a coarser clock is unstable
EPISODE_S = 6.0
ANCHOR = np.array([0.0, 0.0, 1.5])
ROPE_LENGTH = 1.0
MASS = 0.05
RADIUS = 0.004
SPEED = 0.3
KP, KD, KI = 5.0, 0.3, 3.0      # N/m, N s/m, N/(m s): stable at 0.01 s and 0.02 s in a sweep; 10 / 0.4 only up to 0.01 s
SEGMENTS = 12
FMAX = 6.0
SUCCESS_M = 0.03
REACH = ROPE_LENGTH           # the gripper never pulls the rope past its length


class RopeHangEnv:
    obs_dim, act_dim, dt = 12, 3, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        scene = alloy.Scene(dt=DT, worlds=num_envs)
        scene.add_ground()
        scene.add_rod("rope", length=ROPE_LENGTH, radius=RADIUS, pos=tuple(ANCHOR), direction=(0, 0, -1), mass=MASS,
                      clamp="start", route="chain", segments=SEGMENTS)
        self._sim = alloy.build(scene, seed=seed, device="cpu")
        self._goal = np.zeros((num_envs, 3))
        self._prev = np.zeros((num_envs, 3))
        self._int = np.zeros((num_envs, 3))
        self._target = self._draw()
        self._k = 0
        self._fresh = True

    def _draw(self) -> np.ndarray:
        d = self._rng.normal(size=(self.num_envs, 3))
        d[:, 2] = -np.abs(d[:, 2])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return (ANCHOR + d * self._rng.uniform(0.2, 0.4, (self.num_envs, 1))).astype(np.float32)

    def _points(self) -> np.ndarray:
        return self._sim.state(["rope"])["rope.points"]

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._sim.reset()
            self._target = self._draw()
        self._fresh = False
        self._k = 0
        end = self._points()[:, -1]
        self._goal, self._prev, self._int = end.copy(), end.copy(), np.zeros_like(end)
        self._sim.apply_wrench("rope", force=np.zeros((self.num_envs, 3)), at="end")
        return self._obs()

    def _mid_dist(self) -> np.ndarray:
        p = self._points()
        return np.linalg.norm(p[:, p.shape[1] // 2] - self._target, axis=1)

    def _obs(self) -> np.ndarray:
        p = self._points()
        return np.concatenate([p[:, -1], p[:, p.shape[1] // 2], self._target, self._goal], axis=1).astype(np.float32)

    def _servo(self, a: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Force on the free end (N): PD with velocity feed-forward and a small integral term, on the moving goal."""
        vel = (end - self._prev) / DT
        self._prev = end.copy()
        goal = self._goal + a * SPEED * DT
        rel = goal - ANCHOR
        norm = np.linalg.norm(rel, axis=1, keepdims=True)
        self._goal = ANCHOR + rel * np.minimum(1.0, REACH / np.maximum(norm, 1e-9))
        err = self._goal - end
        self._int = np.clip(self._int + err * DT, -0.5, 0.5)
        return np.clip(KP * err + KD * (a * SPEED - vel) + KI * self._int, -FMAX, FMAX)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, 3), -1.0, 1.0)
        self._sim.apply_wrench("rope", force=self._servo(a, self._points()[:, -1]), at="end")
        self._sim.step(1)
        self._k += 1
        dist = self._mid_dist()
        reward = (-dist + (dist < SUCCESS_M)).astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        p = self._points()
        return {"rope_points": p.astype(np.float32), "end_pos": p[:, -1].astype(np.float32),
                "target_pos": self._target.copy()}

    def success(self) -> np.ndarray:
        return self._mid_dist() < SUCCESS_M


def make_env(num_envs: int, seed: int) -> RopeHangEnv:
    return RopeHangEnv(num_envs, seed)
