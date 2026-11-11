# SPDX-License-Identifier: Apache-2.0
"""E19 reference (Alloy): a Franka Panda lifts a 4 cm, 100 g cube 15 cm above its table and holds it for 1 s.

Actions (8): joint position target offsets (0.05 rad per step at +-1) and the gripper (+1 opens, -1 closes, 0 holds).
The cube starts at a random spot in a 10 cm square in front of the arm at every reset.
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

CUBE = 0.04                         # edge (m)
CENTRE = (0.5, 0.0)                 # middle of the 10 cm start square (m, robot base at the origin)
LIFT = 0.15                         # success: cube this far above its rest height, for HOLD_S
HOLD_S = 1.0
HOVER, CARRY = 0.10, 0.21           # policy: grasp point above the rest height before the descent / when lifted


class LiftEnv(PandaEnv):
    success_hold_s = HOLD_S

    def __init__(self, num_envs: int, seed: int) -> None:
        start = kin.solve_ik(np.array([*CENTRE, TABLE_Z + CUBE / 2 + HOVER]))
        cfg = PandaConfig(start_q=start, obs=["panda", "cube"],
                          jitters=[jitter_body("cube", pos=(0.05, 0.05, 0.0))])
        super().__init__(num_envs, seed, cfg)

    def add_objects(self, scene: Any) -> None:
        scene.add_box("cube", size=(CUBE, CUBE, CUBE), pos=(*CENTRE, TABLE_Z + CUBE / 2), mass=0.1)

    def rise(self) -> np.ndarray:
        return self.state("cube.pos")[:, 2] - (TABLE_Z + CUBE / 2)

    def probe(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        return {"cube_pos": self.state("cube.pos"), "ee_pos": self.ee_pos(), "joint_pos": self.joint_pos(),
                "gripper_width": self.gripper_width(), "table_z": np.full(n, TABLE_Z)}

    def reward(self) -> np.ndarray:
        reach = np.linalg.norm(self.state("cube.pos") - self.ee_pos(), axis=1)
        return -reach + 10.0 * np.clip(self.rise(), 0.0, LIFT) + 2.0 * self.success()

    def success_now(self) -> np.ndarray:
        return self.rise() >= LIFT


def make_env(num_envs: int, seed: int) -> LiftEnv:
    return LiftEnv(num_envs, seed)


def scripted_policy(env: LiftEnv, obs: Any) -> np.ndarray:
    """Hover over the cube, descend, close, lift: stages 0-3 per world, switched on measured positions."""
    cube, grasp = env.state("cube.pos"), env.ee_pos()
    rest_z = TABLE_Z + CUBE / 2
    target = cube.copy()
    target[:, 2] = np.select([env.stage == 0, env.stage == 3], [rest_z + HOVER, rest_z + CARRY], rest_z + 0.004)
    miss = target - grasp
    over = (np.linalg.norm(miss[:, :2], axis=1) < 0.004) & (np.abs(miss[:, 2]) < 0.02)
    down = np.linalg.norm(miss, axis=1) < 0.003
    env.advance(((env.stage == 0) & over) | ((env.stage == 1) & down) | ((env.stage == 2) & (env.clock >= 20)))
    lifting = env.stage == 3
    target[lifting, :2] = grasp[lifting, :2]                       # lift straight up from where the hand is
    action = np.zeros((env.num_envs, 8), np.float32)
    action[:, :7] = env.arm_action(target, speed=0.3)
    action[:, 7] = np.where(env.stage >= 2, -1.0, 1.0)
    return action
