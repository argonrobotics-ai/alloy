# SPDX-License-Identifier: Apache-2.0
"""E20 reference (Alloy): a Franka Panda pushes a 6 cm, 200 g cube to a target 15-25 cm away on its table.

Actions (7): joint position target offsets, 0.05 rad per step at +-1; the fingers stay closed (a pusher). The cube
starts in a random spot in front of the arm (yawed up to 10 degrees); the target is drawn at every reset 15-25 cm from
the cube, within 40 degrees of straight ahead. Success: the cube within 3 cm of the target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(d) for d in (_HERE, _HERE.parent) if (d / "panda_env.py").is_file()][:1]

import panda_kin as kin  # noqa: E402
from alloy.envs.randomize import jitter_body  # noqa: E402
from panda_env import TABLE_Z, PandaConfig, PandaEnv  # noqa: E402

CUBE = 0.06                          # edge (m)
CENTRE = (0.45, 0.0)                 # middle of the cube start range
START_HALF = (0.05, 0.08)            # cube start: centre +- this (m)
DIST = (0.15, 0.25)                  # cube -> target distance range (m)
SPREAD_DEG = 40.0                    # target direction: straight ahead (+x) +- this
YAW_DEG = 10.0                      # cube start yaw: +- this (the pusher is 2 cm wide: steeper faces deflect it)
SUCCESS_M = 0.03
HOVER, PUSH_Z = 0.10, 0.025          # policy: grasp point height above the table when approaching / pushing
BEHIND = 0.075                       # policy: approach point distance behind the cube centre (m)


class PushEnv(PandaEnv):
    def __init__(self, num_envs: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed + 20)
        self.target = np.zeros((num_envs, 3))
        start = kin.solve_ik(np.array([*CENTRE, TABLE_Z + HOVER]))
        cfg = PandaConfig(start_q=start, gripper_action=False, finger_start=0.0, finger_target=0.0, episode_s=8.0,
                          obs=["panda", "cube"],
                          jitters=[jitter_body("cube", pos=(*START_HALF, 0.0), yaw_deg=YAW_DEG)])
        super().__init__(num_envs, seed, cfg)

    def add_objects(self, scene: Any) -> None:
        scene.add_box("cube", size=(CUBE,) * 3, pos=(*CENTRE, TABLE_Z + CUBE / 2), mass=0.2)

    # --- target and observations ---------------------------------------------------------------------------------

    def _clear(self, mask: np.ndarray) -> None:
        super()._clear(mask)
        self._draw_targets(mask)

    def _draw_targets(self, mask: np.ndarray) -> None:
        """New targets for the worlds in ``mask``, 15-25 cm from the cube's (new) start."""
        n = self.num_envs
        dist = self.rng.uniform(*DIST, n)
        angle = np.radians(self.rng.uniform(-SPREAD_DEG, SPREAD_DEG, n))
        cube = self.state("cube.pos")
        goal = cube + np.stack([dist * np.cos(angle), dist * np.sin(angle), np.zeros(n)], axis=1)
        goal[:, 2] = TABLE_Z + CUBE / 2
        self.target[mask] = goal[mask]

    def _with_goal(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self.target - self.state("cube.pos")], axis=1).astype(np.float32)

    def reset(self) -> np.ndarray:
        return self._with_goal(super().reset())

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs, reward, done = super().step(action)
        return self._with_goal(obs), reward, done

    # --- task ----------------------------------------------------------------------------------------------------

    def miss(self) -> np.ndarray:
        return np.linalg.norm((self.state("cube.pos") - self.target)[:, :2], axis=1)

    def probe(self) -> dict[str, np.ndarray]:
        return {"cube_pos": self.state("cube.pos"), "target_pos": self.target.copy(), "ee_pos": self.ee_pos(),
                "joint_pos": self.joint_pos(), "table_z": np.full(self.num_envs, TABLE_Z)}

    def reward(self) -> np.ndarray:
        reach = np.linalg.norm(self.state("cube.pos") - self.ee_pos(), axis=1)
        return -self.miss() - 0.1 * reach + 1.0 * (self.miss() < SUCCESS_M)

    def success_now(self) -> np.ndarray:
        return self.miss() < SUCCESS_M


def make_env(num_envs: int, seed: int) -> PushEnv:
    return PushEnv(num_envs, seed)


PUSH_SPEED = 0.08                    # policy: m/s along the line to the target
ARRIVED_M = 0.008                    # policy: stop pushing when the cube is this close to the target
CLEAR_M = 0.09                       # policy: retreat until the grasp point is this far from the cube


def push_target(env: PushEnv, cube: np.ndarray, grasp: np.ndarray) -> np.ndarray:
    """(N, 3) grasp-point goal per stage: behind the cube (0), down beside it (1), a small step along the line through
    the cube and the target with the sideways error to that line removed (2), straight back from the cube (3), hold (4)."""
    gap = env.target[:, :2] - cube[:, :2]
    direction = gap / np.maximum(np.linalg.norm(gap, axis=1, keepdims=True), 1e-9)
    away = grasp[:, :2] - cube[:, :2]
    away = away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9)
    side = cube[:, :2] - grasp[:, :2]
    side = side - direction * np.sum(side * direction, axis=1, keepdims=True)
    xy = np.stack([cube[:, :2] - direction * BEHIND] * 2 + [grasp[:, :2] + direction * 0.05 + 0.5 * side,
                                                  grasp[:, :2] + away * 0.05, grasp[:, :2]])
    goal = np.zeros((env.num_envs, 3))
    goal[:, :2] = xy[env.stage, np.arange(env.num_envs)]
    goal[:, 2] = TABLE_Z + np.where(env.stage == 0, HOVER, PUSH_Z)
    return goal


def scripted_policy(env: PushEnv, obs: Any) -> np.ndarray:
    """Hover behind the cube (stage 0), descend (1), push toward the target (2), back away once the cube is within
    8 mm (3; releasing the contact lets the stored servo load go without shoving the cube on), hold still (4)."""
    cube, grasp = env.state("cube.pos"), env.ee_pos()
    goal = push_target(env, cube, grasp)
    miss = goal - grasp
    over = (np.linalg.norm(miss[:, :2], axis=1) < 0.006) & (np.abs(miss[:, 2]) < 0.02)
    down = np.abs(miss[:, 2]) < 0.005
    clear = np.linalg.norm((grasp - cube)[:, :2], axis=1) > CLEAR_M
    env.advance(((env.stage == 0) & over) | ((env.stage == 1) & down) | ((env.stage == 2) & (env.miss() < ARRIVED_M))
                | ((env.stage == 3) & clear))
    slow = (env.stage >= 2).reshape(-1, 1)
    return np.where(slow, env.arm_action(goal, speed=PUSH_SPEED), env.arm_action(goal, speed=0.3)).astype(np.float32)
