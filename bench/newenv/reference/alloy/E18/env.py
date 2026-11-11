# SPDX-License-Identifier: Apache-2.0
"""E18 reference (Alloy): a 30 cm square tray with a 5 cm bed of dry sand (granular MPM) and an 8 cm wide scoop.

Action (4,): scoop velocity x, y, z up to 0.1 m/s and pitch rate about y up to 1 rad/s, each at +-1. The scoop is a
kinematic blade (2 cm thick, 8 cm wide, 8 cm tall) that ``sim.drive`` moves; ``scoop_pos`` is the middle of its bottom
edge. Reward: the mass of sand more than 10 cm above the tray's floor, per step. All worlds share one clock.
The 5 cm bed is two boxes of sand: one box of 4.5 l is 288,000 particles, over the 200,000 an object may hold.
"""
from __future__ import annotations

import numpy as np

import alloy
from liquid_kin import KinematicMover

DT = 0.02
EPISODE_S = 10.0
TRAY_INNER, TRAY_DEPTH, TRAY_WALL = 0.30, 0.12, 0.01
FLOOR_Z = TRAY_WALL                                 # top of the tray's floor
BED_H = 0.05
BULK = 1555.0                                       # kg/m3, sand_silica_dry's card note (funnel-pour bed)
BLADE = (0.02, 0.08, 0.08)                          # x thickness, y width, z height
START_TIP = np.array([-0.10, 0.0, 0.14])            # bottom-edge centre of the blade
V_MAX, W_MAX = 0.1, 1.0
LIFT_M = 0.10
MASS_GOAL_KG = 0.05


class SandScoopEnv:
    obs_dim, act_dim, dt = 6, 4, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground()
        scene.add_open_box("tray", inner=(TRAY_INNER, TRAY_INNER, TRAY_DEPTH), wall=TRAY_WALL, static=True)
        for name, y in (("sand_a", -TRAY_INNER / 4), ("sand_b", TRAY_INNER / 4)):
            scene.add_box(name, size=(TRAY_INNER, TRAY_INNER / 2, BED_H), pos=(0.0, y, FLOOR_Z + BED_H / 2 + 0.001),
                          material="sand_silica_dry")
        centre = START_TIP + np.array([0.0, 0.0, BLADE[2] / 2])
        scene.add_box("scoop", size=BLADE, pos=tuple(centre), kinematic=True)
        self._sim = alloy.build(scene, seed=seed)
        self._sim.settle(0.5)
        self._mover = KinematicMover(self._sim, "scoop", centre)
        self._mass = BULK * TRAY_INNER ** 2 * BED_H / self._count()
        self._k = 0
        self._lifted = np.zeros(num_envs)

    def _points(self) -> np.ndarray:
        s = self._sim.state(["sand_a", "sand_b"])
        return np.concatenate([s["sand_a.points"], s["sand_b.points"]], axis=1)

    def _count(self) -> int:
        return len(self._points()[0])

    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._mover.reset()
        self._k = 0
        self._lifted[:] = 0.0
        return self._obs()

    def _tip(self) -> np.ndarray:
        return self._mover.pos - np.array([0.0, 0.0, BLADE[2] / 2])

    def _lifted_kg(self) -> np.ndarray:
        return (self._points()[..., 2] > FLOOR_Z + LIFT_M).sum(axis=1) * self._mass

    def _obs(self) -> np.ndarray:
        k = np.full(self.num_envs, self._k / self.episode_steps)
        return np.concatenate([self._tip(), self._mover.pitch[:, None], self._lifted[:, None], k[:, None]], axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 4), -1.0, 1.0)
        self._mover.advance(a[:, :3] * V_MAX, DT, a[:, 3] * W_MAX)
        self._sim.step(1)
        self._k += 1
        self._lifted = self._lifted_kg()
        reward = self._lifted.astype(np.float32)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            return self.reset(), reward, done
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        return {"sand_points": self._points().astype(np.float32), "particle_mass": np.full(n, self._mass, np.float32),
                "scoop_pos": self._tip().astype(np.float32), "scoop_pitch": self._mover.pitch.astype(np.float32)}

    def success(self) -> np.ndarray:
        return self._lifted >= MASS_GOAL_KG


def make_env(num_envs: int, seed: int) -> SandScoopEnv:
    return SandScoopEnv(num_envs, seed)
