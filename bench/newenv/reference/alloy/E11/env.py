# SPDX-License-Identifier: Apache-2.0
"""E11 reference (Alloy): a 40 cm, 100 g cotton cloth held by two adjacent corners 1 m up; the corners move at <= 0.2 m/s.

Alloy has no pin / attach call for cloth, so each held corner is pinched between two thin kinematic plates (friction
only), moved with ``set_pose`` every control period (``sim.drive`` takes one velocity for all worlds; the action is per
world). ``corner_pos`` reads the two cloth vertices that started at the held corners (where the cloth really is).
Reward: the share of cloth points lying on the table top (z = 0.5 m, within 1.5 cm).
"""
from __future__ import annotations

import math

import numpy as np

import alloy

DT = 0.02
EPISODE_S = 5.0
SIDE = 0.4
MASS = 0.1
TOP = 1.0
TABLE_TOP = 0.5
SPEED = 0.2
RES = 11                         # 121 points; 21 (441 points) costs 3.4 s per control step per world on the CPU
JAW = (0.08, 0.004, 0.08)        # one pinch plate: along the edge, thickness, down the cloth
JAW_GAP = 0.0001                 # plate face 0.1 mm off the cloth: 0 slides 4 mm in 2 s, <= -0.1 mm penetrates (health INVALID, world frozen: corners then look perfectly held), 0.2 mm slips in the move
ON_TABLE_M = 0.015
SUCCESS_FRAC = 0.9
Z_LIM = (0.55, 1.4)              # corners stay in this height band
XY_LIM = 0.7


class ClothHangEnv:
    obs_dim, act_dim, dt = 13, 6, DT
    episode_steps = round(EPISODE_S / DT)

    def __init__(self, num_envs: int, seed: int) -> None:
        self.num_envs = num_envs
        self._home = np.array([[-SIDE / 2, 0.0, TOP], [SIDE / 2, 0.0, TOP]])        # the held corners at the start
        scene = alloy.Scene(dt=DT, worlds=num_envs)
        scene.add_ground()
        scene.add_box("table", size=(1.0, 1.0, TABLE_TOP), pos=(0, 0, TABLE_TOP / 2), static=True, friction=0.5)
        scene.add_cloth("cloth", size=(SIDE, SIDE), pos=(0, 0, TOP - SIDE / 2), rpy=(math.pi / 2, 0, 0),
                        resolution=RES, mass=MASS, self_contact=False)
        self._jaws = self._add_jaws(scene)
        self._sim = _build(scene, seed)
        start = self._sim.state(["cloth"])["cloth.points"][0]
        self._ix = [int(np.argmin(np.linalg.norm(start - c, axis=1))) for c in self._home]
        self._offset = np.zeros((num_envs, 2, 3))
        self._k = 0
        self._fresh = True

    @staticmethod
    def _add_jaws(scene: alloy.Scene) -> list[tuple[str, int, float]]:
        """Four kinematic plates (name, corner index, side of the cloth); returns their specs."""
        jaws = []
        for c, sx in enumerate((-1.0, 1.0)):
            for side in (-1.0, 1.0):
                name = f"jaw{c}{'p' if side > 0 else 'm'}"
                pos = _jaw_pos(c, side)
                scene.add_box(name, size=JAW, pos=pos, kinematic=True, friction=1.5)
                jaws.append((name, c, side))
        return jaws

    def reset(self) -> np.ndarray:
        if not self._fresh:
            self._sim.reset()
        self._fresh = False
        self._k = 0
        self._offset[:] = 0.0
        self._place()
        return self._obs()

    def _place(self) -> None:
        for name, c, side in self._jaws:
            pos = _jaw_pos(c, side)[None] + self._offset[:, c]
            self._sim.set_pose(name, pos=pos)

    def _points(self) -> np.ndarray:
        return self._sim.state(["cloth"])["cloth.points"]

    def _corners(self) -> np.ndarray:
        return self._points()[:, self._ix, :]

    def _on_table(self, pts: np.ndarray) -> np.ndarray:
        near = (pts[..., 2] < TABLE_TOP + ON_TABLE_M) & (np.abs(pts[..., 0]) < 0.5) & (np.abs(pts[..., 1]) < 0.5)
        return near.mean(axis=1)

    def _obs(self) -> np.ndarray:
        pts = self._points()
        extra = np.stack([pts[..., 2].min(axis=1), self._on_table(pts)], axis=1)
        return np.concatenate([self._corners().reshape(self.num_envs, 6), pts.mean(axis=1), extra,
                               self._offset[:, 0, :2]], axis=1).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(action, np.float32).reshape(self.num_envs, 2, 3), -1.0, 1.0)
        self._offset = self._offset + a * SPEED * DT
        self._offset[..., 2] = np.clip(self._offset[..., 2], Z_LIM[0] - TOP, Z_LIM[1] - TOP)
        self._offset[..., :2] = np.clip(self._offset[..., :2], -XY_LIM, XY_LIM)
        self._place()
        self._sim.step(1)
        self._k += 1
        frac = self._on_table(self._points())
        done = np.full(self.num_envs, self._k >= self.episode_steps)
        if done[0]:
            self.reset()
        return self._obs(), frac.astype(np.float32), done

    def probe(self) -> dict[str, np.ndarray]:
        return {"cloth_points": self._points().astype(np.float32), "corner_pos": self._corners().astype(np.float32)}

    def success(self) -> np.ndarray:
        return self._on_table(self._points()) >= SUCCESS_FRAC


def _jaw_pos(corner: int, side: float) -> np.ndarray:
    sx = -1.0 if corner == 0 else 1.0
    x = sx * (SIDE / 2 - JAW[0] / 2)
    return np.array([x, side * (JAW_GAP + JAW[1] / 2), TOP - JAW[2] / 2 + 0.002])


def _build(scene: alloy.Scene, seed: int) -> alloy.Sim:
    """One L4 when there is one; else the CPU, one process per core for many worlds (AGENTS.md section 1)."""
    import warp as wp  # noqa: PLC0415

    if wp.get_cuda_device_count() > 0:
        return alloy.build(scene, seed=seed)
    return alloy.build(scene, seed=seed, device="cpu", processes="auto" if scene.worlds > 1 else None)


def make_env(num_envs: int, seed: int) -> ClothHangEnv:
    return ClothHangEnv(num_envs, seed)
