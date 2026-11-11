# SPDX-License-Identifier: Apache-2.0
"""E17 reference (Alloy): a tap above a cylindrical glass (inner radius 3 cm, 12 cm tall); the action sets the flow.

Alloy's pour (``add_pour``) streams at one rate from the first step and cannot change or stop at run time (no public
call). The nearest honest valve: the nozzle always emits 50 ml/s, and a kinematic shelf under it slides its edge
across the 8 mm stream: the part of the stream left of the edge falls into the glass, the rest lands on the tilted
shelf and runs off to the side. The action a in [-1, 1] (negative = closed) puts the edge so that clip(a, 0, 1) of the
stream's width goes to the glass, i.e. a flow of 50 ml/s x clip(a, 0, 1) reaches the glass. Water behind the valve is
the tap's supply, so ``liquid_points`` lists all water emitted by the nozzle, the diverted part included.
"""
from __future__ import annotations

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 8.0
GLASS_R, GLASS_H, WALL = 0.03, 0.12, 0.004
NOZZLE = (0.0, 0.0, 0.30)
POUR_ML = 1500.0                                   # more than any episode emits: sim.reset() after a pour emptied its whole reservoir leaves it empty (FRICTION)
STREAM_HALF = 0.0045                               # half the stream's width (m)
SHELF_SIZE = (0.30, 0.10, 0.01)
SHELF_TILT = np.radians(20.0)                      # +x end down: runs the diverted water off to the side
SHELF_EDGE_Z = 0.20                                # height of the shelf's upper face at its edge
TARGET_LEVEL = 0.08
Q_MAX = 50e-6                                      # m^3/s at action 1
GLASS_POS = np.array([0.0, 0.0, 0.0])              # centre of the glass's bottom face


def shelf_pose(edge_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shelf body positions (W, 3) and quaternion wxyz (W, 4) for the shelf's edge at ``edge_x`` (W,)."""
    c, s = np.cos(SHELF_TILT), np.sin(SHELF_TILT)
    off = np.array([c * SHELF_SIZE[0] / 2 + s * SHELF_SIZE[2] / 2, 0.0, -s * SHELF_SIZE[0] / 2 - c * SHELF_SIZE[2] / 2])
    pos = np.stack([edge_x + off[0], np.zeros_like(edge_x), np.full_like(edge_x, SHELF_EDGE_Z + off[2])], axis=1)
    quat = np.tile(np.array([np.cos(SHELF_TILT / 2), 0.0, np.sin(SHELF_TILT / 2), 0.0]), (len(edge_x), 1))
    return pos.astype(np.float32), quat.astype(np.float32)


def edge_for(valve: np.ndarray) -> np.ndarray:
    """Shelf edge x for a valve opening in [0, 1]: closed covers the stream (edge at -w/2 - 1.5 mm), open clears it."""
    w = STREAM_HALF + 0.0015
    return -w + 2.0 * w * valve


class TapEnv:
    obs_dim, act_dim, dt = 3, 1, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        scene = alloy.Scene(worlds=num_envs, dt=DT)
        scene.add_ground()
        scene.add_open_cylinder("glass", inner_radius=GLASS_R, depth=GLASS_H, wall=WALL, pos=tuple(GLASS_POS))
        pos, quat = shelf_pose(edge_for(np.zeros(1)))
        scene.add_box("shelf", size=SHELF_SIZE, pos=tuple(pos[0]), quat_xyzw=(quat[0, 1], quat[0, 2], quat[0, 3], quat[0, 0]),
                      kinematic=True)
        scene.add_pour("water", POUR_ML * 1e-6, pos=NOZZLE, rate=Q_MAX, speed=0.5)
        self._sim = alloy.build(scene, seed=seed)
        self._pv = float(self._sim.particle_volume("water"))
        self._valve = np.zeros(num_envs)
        self._k = 0
        self._ok = np.zeros(num_envs, bool)

    def reset(self) -> np.ndarray:
        self._sim.reset()
        self._valve[:] = 0.0
        self._k = 0
        self._set_shelf()
        return self._obs()

    def _set_shelf(self) -> None:
        pos, quat = shelf_pose(edge_for(self._valve))
        self._sim.set_pose("shelf", pos=pos, quat_wxyz=quat)

    def _water(self) -> tuple[np.ndarray, np.ndarray]:
        """Points (W, P, 3) with NaN rows for particles not emitted yet, and the mask of emitted ones."""
        s = self._sim.state(["water"])
        pts, live = s["water.points"].astype(np.float32), s["water.active"].astype(bool)
        return np.where(live[..., None], pts, np.nan), live

    def _in_glass(self, pts: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return (np.hypot(pts[..., 0] - GLASS_POS[0], pts[..., 1] - GLASS_POS[1]) < GLASS_R + 0.002) & \
                   (pts[..., 2] < GLASS_POS[2] + WALL + GLASS_H + 0.002)

    def _level(self) -> np.ndarray:
        pts, _ = self._water()
        n = self._in_glass(pts).sum(axis=1)
        return n * self._pv / (np.pi * GLASS_R ** 2)

    def _obs(self) -> np.ndarray:
        k = np.full(self.num_envs, self._k / self.episode_steps)
        return np.stack([self._level(), self._valve, k], axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float64).reshape(self.num_envs, 1), -1.0, 1.0)[:, 0]
        self._valve = np.clip(a, 0.0, 1.0)
        self._set_shelf()
        self._sim.step(1)
        self._k += 1
        level = self._level()
        over = level > GLASS_H
        reward = (1.0 - np.abs(level - TARGET_LEVEL) / TARGET_LEVEL - 2.0 * over).astype(np.float32)
        self._ok = (np.abs(level - TARGET_LEVEL) < 0.005) & (self._valve < 0.05)
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            return self.reset(), reward, done
        return self._obs(), reward, done

    def probe(self) -> dict[str, np.ndarray]:
        pts, _ = self._water()
        n = self.num_envs
        return {"liquid_points": pts, "particle_volume": np.full(n, self._pv, np.float32),
                "glass_pos": np.tile(GLASS_POS.astype(np.float32), (n, 1))}

    def success(self) -> np.ndarray:
        return self._ok.copy()


def make_env(num_envs: int, seed: int) -> TapEnv:
    return TapEnv(num_envs, seed)
