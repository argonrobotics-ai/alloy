# SPDX-License-Identifier: Apache-2.0
"""E23 reference (Alloy): a Franka Panda holds a cup with 150 ml of water and pours it into a bowl 40 cm in front of it.

The cup is welded into the hand (a body under the hand body in the robot's MJCF, docs/guide/robots.md "Grasping"): the
fingers close on its rim, so it moves exactly with the hand and cannot be put down. Actions (7): joint position target
offsets, 0.05 rad per step at +-1. The start posture is baked into the robot file (cup_hand.py), so joints read the
displacement from it; ``joint_pos`` adds it back. All worlds share one episode clock (SPH cannot reset one world).
"""

from __future__ import annotations

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
from cup_hand import GRASP_Z, Cup, panda_with_cup  # noqa: E402
from panda_env import ARM_JOINTS, TABLE_Z, PandaConfig, quat_rotate_wxyz  # noqa: E402
from panda_gpu import PandaGpuEnv  # noqa: E402

CUP = Cup()
BOWL_X, BOWL_R, BOWL_DEPTH, BOWL_WALL = 0.40, 0.09, 0.06, 0.004       # axis x (m), inner radius, inner depth, wall
FILL_M3 = 150e-6 * 1.125       # add_liquid fills whole particles in a box narrower than the cup: 150 ml comes from 1.125x
LIP_X = BOWL_X - 0.03            # world x of the cup's pouring edge (outer wall), held fixed while tipping
GRASP_START = np.array([LIP_X - CUP.outer_radius, 0.0, 0.215])      # grasp point (pad centre) at the start
TILT_RATE, TILT_MAX, HOLD_S, SETTLE_S = 0.5, 2.0, 2.5, 0.4         # policy: rad/s, rad, s held at the end, s before
FRACTION_GOAL = 0.8
R_START = kin.DOWN               # hand z down, fingers close along world y; tipping about +y pours toward +x
LIP_HAND = np.array([CUP.outer_radius, 0.0, CUP.rim_z - GRASP_Z])   # the pouring edge in the hand frame, from the grasp


def ry(theta: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation about world +y."""
    c, s, z, o = np.cos(theta), np.sin(theta), np.zeros_like(theta), np.ones_like(theta)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


class PourEnv(PandaGpuEnv):
    def __init__(self, num_envs: int, seed: int) -> None:
        self.start_q = kin.solve_ik(GRASP_START, R_START)
        cfg = PandaConfig(start_q=np.zeros(7), gripper_action=False, dt=0.02, episode_s=9.0, obs=["panda"],
                          robot_file=panda_with_cup(self.start_q, CUP))
        super().__init__(num_envs, seed, cfg, jitters=[jitter_joint(j, at=0.0) for j in ARM_JOINTS])   # 0 = baked start
        self.pv = float(self.venv.sim.particle_volume("water"))

    def add_objects(self, scene: Any) -> None:
        scene.add_open_cylinder("bowl", inner_radius=BOWL_R, depth=BOWL_DEPTH, wall=BOWL_WALL, pos=(BOWL_X, 0.0, TABLE_Z))
        h = FILL_M3 / (np.pi * CUP.inner_radius**2)
        centre = np.array([GRASP_START[0], 0.0, 0.0])
        floor_z = GRASP_START[2] + (GRASP_Z - CUP.floor_top_z)               # the hand points down: z_hand = -z_world
        scene.add_liquid("water", size=(2 * CUP.inner_radius - 0.002, 2 * CUP.inner_radius - 0.002, h), shape="cylinder",
                         pos=(centre[0], 0.0, floor_z + h / 2 + 0.001))

    def joint_pos(self) -> np.ndarray:
        return super().joint_pos() + self.start_q

    def points(self) -> np.ndarray:
        return self.state("water.points")

    def in_bowl_fraction(self) -> np.ndarray:
        p = self.points()
        horizontal = np.hypot(p[..., 0] - BOWL_X, p[..., 1]) < BOWL_R
        inside = horizontal & (p[..., 2] >= TABLE_Z - 0.005) & (p[..., 2] <= TABLE_Z + BOWL_DEPTH + 0.01)
        return inside.mean(axis=1)

    def probe(self) -> dict[str, np.ndarray]:
        n = self.num_envs
        hand, quat = self.state("panda/hand.pos"), self.state("panda/hand.quat_wxyz")
        cup = hand + quat_rotate_wxyz(quat, np.array([0.0, 0.0, CUP.centre_z]))
        return {"liquid_points": self.points().astype(np.float32), "particle_volume": np.full(n, self.pv, np.float32),
                "cup_pos": cup, "bowl_pos": np.tile([BOWL_X, 0.0, TABLE_Z], (n, 1)),
                "bowl_radius": np.full(n, BOWL_R), "ee_pos": self.ee_pos(), "joint_pos": self.joint_pos()}

    def reward(self) -> np.ndarray:
        return self.in_bowl_fraction()

    def success_now(self) -> np.ndarray:
        return self.in_bowl_fraction() >= FRACTION_GOAL


def make_env(num_envs: int, seed: int) -> PourEnv:
    return PourEnv(num_envs, seed)


def scripted_policy(env: PourEnv, obs: Any) -> np.ndarray:
    """Wait, then tip the hand about +y at TILT_RATE up to TILT_MAX while the pouring edge stays put, then hold."""
    n = env.num_envs
    t = env.clock * env.dt
    theta = np.clip(TILT_RATE * (t - SETTLE_S), 0.0, TILT_MAX)
    rot = ry(theta) @ R_START
    lip_now = R_START @ LIP_HAND + GRASP_START                             # the edge at the start (world)
    target = lip_now[None, :] - np.einsum("nij,j->ni", rot, LIP_HAND)
    env.clock = env.clock + 1
    action = np.zeros((n, 7), np.float32)
    action[:] = env.arm_action(target, rot, speed=0.5, gain=0.2)
    return action
