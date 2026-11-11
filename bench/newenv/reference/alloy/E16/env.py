# SPDX-License-Identifier: Apache-2.0
"""E16 reference (Alloy): an open tank 20 x 10 cm in plan, 15 cm tall, half full of water, on a horizontal rail.

Action (1,): the tank's acceleration along +x, up to 3 m/s2 at +-1 (the velocity is the integral, held through the
period by ``sim.drive``; the tank is kinematic, so the rail is the drive). Reward: progress toward 1 m minus the
share of the water outside the tank. The episode ends after EPISODE_S; all worlds share one clock (Alloy cannot reset
one liquid world).
"""
from __future__ import annotations

import numpy as np

import alloy
from liquid_kin import KinematicMover

DT = 0.02
EPISODE_S = 4.0
INNER = (0.20, 0.10, 0.15)                         # cavity x, y, depth (m)
WALL = 0.004
A_MAX = 3.0
FILL = (INNER[0] - 0.002, INNER[1] - 0.002, 0.0765)    # liquid box; whole particles come out near 1.5 l (measured)
TANK_POS = np.array([0.0, 0.0, 0.0])               # centre of the tank's bottom face
TRAVEL = 1.0


class TankSloshEnv:
    obs_dim, act_dim, dt = 3, 1, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground()
        scene.add_open_box("tank", inner=INNER, wall=WALL, pos=tuple(TANK_POS), kinematic=True)
        scene.add_liquid("water", size=FILL, pos=(0.0, 0.0, WALL + FILL[2] / 2 + 0.001))
        self._sim = alloy.build(scene, seed=seed)
        self._sim.settle(0.5)
        self._mover = KinematicMover(self._sim, "tank", TANK_POS)
        self._pv = float(self._sim.particle_volume("water"))
        self._v = np.zeros(num_envs)
        self._k = 0
        self._ok = np.zeros(num_envs, bool)

    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._mover.reset()
        self._v[:] = 0.0
        self._k = 0
        return self._obs()

    def _points(self) -> np.ndarray:
        return self._sim.state(["water"])["water.points"]

    def _spill(self) -> np.ndarray:
        """Share of the water outside the tank's cavity (tank frame, 1 cm margin)."""
        p = self._points() - (self._mover.pos - self._mover.start)[:, None, :]
        lo = np.array([-INNER[0] / 2 - 0.01, -INNER[1] / 2 - 0.01, WALL - 0.005])
        hi = np.array([INNER[0] / 2 + 0.01, INNER[1] / 2 + 0.01, WALL + INNER[2] + 0.005])
        inside = ((p >= lo) & (p <= hi)).all(axis=-1)
        return 1.0 - inside.mean(axis=1)

    def _obs(self) -> np.ndarray:
        x = self._mover.pos[:, 0] - self._mover.start[:, 0]
        return np.stack([x, self._v, np.full(self.num_envs, self._k / self.episode_steps)], axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 1), -1.0, 1.0)[:, 0] * A_MAX
        v_mid = self._v + 0.5 * a * DT                    # mean velocity of the period: the position is then exact
        vel = np.stack([v_mid, np.zeros_like(v_mid), np.zeros_like(v_mid)], axis=1)
        self._mover.advance(vel, DT)
        self._v = self._v + a * DT
        self._sim.step(1)
        self._k += 1
        x = self._mover.pos[:, 0] - self._mover.start[:, 0]
        spill = self._spill()
        reward = (np.clip(x / TRAVEL, -1, 1) - 5.0 * spill).astype(np.float32)
        self._ok = (x >= 0.95 * TRAVEL) & (spill < 0.01)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            return self.reset(), reward, done
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        return {"liquid_points": self._points().astype(np.float32), "particle_volume": np.full(n, self._pv, np.float32),
                "tank_pos": self._mover.pos.astype(np.float32)}

    def success(self) -> np.ndarray:
        return self._ok.copy()


def make_env(num_envs: int, seed: int) -> TankSloshEnv:
    return TankSloshEnv(num_envs, seed)
