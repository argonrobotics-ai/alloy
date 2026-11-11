# SPDX-License-Identifier: Apache-2.0
"""E24 reference (Alloy): a Franka Panda pulls a 30 cm, 50 g cloth off a 6 cm cube it covers on the table.

Actions (8): joint position target offsets (0.05 rad per step at +-1) and the gripper (+1 opens, -1 closes). The cloth
is dropped on the cube at build and settles into a drape (``Task.settle``), which every reset returns to. The scripted
policy closes the fingers, presses them on the cloth's table-flat part beside the cube and drags the sheet sideways
(-y) until it has slid off: a friction drag, not a pinch (docs/known_gaps/cloth-robot-grasp-carry.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in (_HERE, _HERE.parent):
    if (_d / "panda_env.py").is_file() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import panda_kin as kin  # noqa: E402
from alloy.envs.randomize import jitter_joint  # noqa: E402
from panda_env import ARM_JOINTS, FINGER_OPEN, TABLE_Z, PandaConfig  # noqa: E402
from panda_gpu import PandaGpuEnv  # noqa: E402

CUBE, CUBE_MASS = 0.06, float(os.environ.get("E24_CUBE_MASS", 0.58))     # edge (m), kg (aluminium)
CUBE_XY = np.array([0.45, 0.0])
CLOTH_SIDE, CLOTH_MASS = 0.30, 0.05
CLOTH_RES = int(os.environ.get("E24_RES", 14))            # vertices per side
CLOTH_ACCURACY = os.environ.get("E24_ACC", "fast")
SELF_CONTACT = os.environ.get("E24_SELF", "0") == "1"
FOOTPRINT_MARGIN = 0.0                        # success: no point over the cube's footprint, above its top - 1 cm
MOVED_MAX = 0.02
PRESS_Y, PRESS_Z = float(os.environ.get("E24_PRESS_Y", -0.065)), TABLE_Z + float(os.environ.get("E24_PZ", 0.022))
END_Y, DRAG_SPEED = float(os.environ.get("E24_END_Y", -0.42)), float(os.environ.get("E24_SPEED", 0.15))      # drag: press point, grasp z, end y, m/s
HOVER_Z = 0.07
START = np.array([CUBE_XY[0], PRESS_Y, 0.12])


class ClothEnv(PandaGpuEnv):
    def __init__(self, num_envs: int, seed: int) -> None:
        start = kin.solve_ik(START)
        cfg = PandaConfig(start_q=start, obs=["panda", "cube"], dt=0.02, episode_s=4.2)
        jit = [jitter_joint(j, by=0.03, at=float(v)) for j, v in zip(ARM_JOINTS, start)]
        jit += [jitter_joint("panda/finger_joint1", at=FINGER_OPEN), jitter_joint("panda/finger_joint2", at=FINGER_OPEN)]
        super().__init__(num_envs, seed, cfg, settle=1.0, jitters=jit, scene_kwargs={"cloth_accuracy": CLOTH_ACCURACY})
        self.cube0 = self.state("cube.pos")[0].copy()

    def add_objects(self, scene: Any) -> None:
        scene.add_box("cube", size=(CUBE,) * 3, pos=(*CUBE_XY, TABLE_Z + CUBE / 2), mass=CUBE_MASS,
                      friction=float(os.environ.get("E24_CUBE_MU", 0.15)))
        scene.add_cloth("cloth", size=(CLOTH_SIDE, CLOTH_SIDE), pos=(*CUBE_XY, TABLE_Z + CUBE + 0.04),
                        resolution=CLOTH_RES, mass=CLOTH_MASS, self_contact=SELF_CONTACT)

    def points(self) -> np.ndarray:
        return self.state("cloth.points")

    def over_cube(self) -> np.ndarray:
        """(N,) number of cloth points above the cube's footprint (within the margin) and above its top face - 1 cm."""
        p, c = self.points(), self.state("cube.pos")[:, None, :]
        half = CUBE / 2 + FOOTPRINT_MARGIN
        over = (np.abs(p[..., 0] - c[..., 0]) < half) & (np.abs(p[..., 1] - c[..., 1]) < half)
        return (over & (p[..., 2] > c[..., 2] + CUBE / 2 - 0.01)).sum(axis=1)

    def moved(self) -> np.ndarray:
        return np.linalg.norm(self.state("cube.pos") - self.cube0, axis=1)

    def probe(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        return {"cloth_points": self.points().astype(np.float32), "cube_pos": self.state("cube.pos"),
                "ee_pos": self.ee_pos(), "joint_pos": self.joint_pos(), "gripper_width": self.gripper_width(),
                "table_z": np.full(n, TABLE_Z)}

    def reward(self) -> np.ndarray:
        return -self.over_cube() / self.points().shape[1] - self.moved()

    def success_now(self) -> np.ndarray:
        return (self.over_cube() == 0) & (self.moved() < MOVED_MAX)


def make_env(num_envs: int, seed: int) -> ClothEnv:
    return ClothEnv(num_envs, seed)


def scripted_policy(env: ClothEnv, obs: Any) -> np.ndarray:
    """Stage 0 hover over the press point, 1 descend with the fingers closing, 2 drag toward -y, 3 hold."""
    grasp = env.ee_pos()
    target = np.zeros_like(grasp)
    target[:, 0] = CUBE_XY[0]
    drag_y = np.where(env.stage == 2, np.maximum(END_Y, PRESS_Y - DRAG_SPEED * env.clock * env.dt), END_Y)
    target[:, 1] = np.where(env.stage >= 2, drag_y, PRESS_Y)
    target[:, 2] = np.where(env.stage == 0, HOVER_Z, PRESS_Z)
    miss = target - grasp
    over = np.linalg.norm(miss[:, :2], axis=1) < 0.006
    dragged = grasp[:, 1] <= END_Y + 0.01
    env.advance(((env.stage == 0) & over & (np.abs(miss[:, 2]) < 0.01)) | ((env.stage == 1) & (env.clock >= 30))
                | ((env.stage == 2) & dragged))
    action = np.zeros((env.num_envs, 8), np.float32)
    action[:, :7] = env.arm_action(target, speed=0.3)
    action[:, 7] = np.where(env.stage >= 1, -1.0, 1.0)
    return action
