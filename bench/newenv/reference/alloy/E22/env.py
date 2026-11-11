# SPDX-License-Identifier: Apache-2.0
"""E22 reference (Alloy): a Franka Panda holding a 16 mm steel peg (8 cm long) inserts it into a 17 mm hole.

The hole is a 5 cm deep round hole in a 20 kg block (a free-base MJCF robot of 24 wall slabs tangent to the 8.5 mm
radius on a 3 cm floor, about 8 cm across), put per world at a random spot within 2 cm of (0.5, 0) at every reset;
it rests on the table and is too heavy for the peg to move (a kinematic block, the closer 'fixed', slows the CPU
to 1 step/s for 16 worlds).
The peg starts gripped near its top, pointing down, the grasp servo stiffened (the library Panda's squeezes with
about 1 N). Actions (7): joint position target offsets (0.02 rad per step at +-1); the fingers stay closed on the peg.
Success: the peg tip 3 cm below the hole's opening with the peg on the hole axis (3 mm).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(d) for d in (_HERE, _HERE.parent) if (d / "panda_env.py").is_file()][:1]

import panda_kin as kin  # noqa: E402
from panda_env import TABLE_Z, PandaConfig, PandaEnv, quat_rotate_wxyz  # noqa: E402
from robota_common import solve_start, strong_grip_file  # noqa: E402

PEG_R, PEG_HALF = 0.008, 0.04        # 16 mm round peg, 8 cm long
HOLE_R = 0.0085                      # 17 mm hole
WALL, DEPTH = 0.03, 0.05             # block: floor and wall thickness, hole depth
TOP_Z = TABLE_Z + WALL + DEPTH       # the hole's opening
CENTRE = np.array([0.5, 0.0])        # middle of the hole's random spot
SPOT = 0.02                          # the hole is within this of CENTRE, per axis (m)
PEG_DROP = 0.01                      # peg centre below the grasp point (m): the pads hold its top 3 cm
INSERT, AXIS_TOL = 0.03, 0.003
ABOVE = 0.06                         # start: peg tip this far above the opening
FACING = kin.DOWN
GAIN, DESCENT_SPEED, ALIGNED_M, GO_DEEP = 0.3, 0.03, 0.0005, 0.035


def block_mjcf(segments: int = 24, mass: float = 20.0) -> str:
    """A free body: a floor plate and ``segments`` wall slabs whose inner faces are tangent to the hole radius."""
    outer = (HOLE_R + WALL) / np.cos(np.pi / segments)
    chord = 2.0 * outer * np.tan(np.pi / segments)
    geoms = [f'<geom type="box" pos="0 0 {WALL / 2}" size="{outer} {outer} {WALL / 2}"/>']
    for k in range(segments):
        a, c = 2.0 * np.pi * k / segments, HOLE_R + WALL / 2
        geoms.append(f'<geom type="box" pos="{c * np.cos(a)} {c * np.sin(a)} {WALL + DEPTH / 2}" euler="0 0 {a}" '
                     f'size="{WALL / 2} {chord / 2} {DEPTH / 2}"/>')
    inertia = mass * (2 * outer) ** 2 / 6.0
    return (f'<mujoco><compiler angle="radian"/><worldbody><body name="blk"><freejoint/>'
            f'<inertial pos="0 0 {(WALL + DEPTH) / 2}" mass="{mass}" diaginertia="{inertia} {inertia} {inertia}"/>'
            + "".join(geoms) + "</body></worldbody></mujoco>")


class PegEnv(PandaEnv):
    def __init__(self, num_envs: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed + 22)
        self.hole = np.tile([*CENTRE, TOP_Z], (num_envs, 1))
        tcp_z = TOP_Z + ABOVE + PEG_DROP + PEG_HALF
        start = solve_start(np.array([*CENTRE, tcp_z]), FACING)
        cfg = PandaConfig(start_q=start, gripper_action=False, finger_start=PEG_R, finger_target=0.0, joint_step=0.02,
                          episode_s=12.0, obs=["panda"], robot_file=strong_grip_file())
        super().__init__(num_envs, seed, cfg)

    def add_objects(self, scene: Any) -> None:
        scene.add_robot("block", mjcf=block_mjcf(), pos=np.tile([*CENTRE, TABLE_Z], (self.num_envs, 1)))
        scene.add_cylinder("peg", radius=PEG_R, height=2 * PEG_HALF, pos=(0.5, 0.4, TABLE_Z + PEG_HALF), density=7850.0)

    # --- resets: random hole, peg back in the hand --------------------------------------------------------------

    def _clear(self, mask: np.ndarray) -> None:
        super()._clear(mask)
        sim = self.venv.sim
        spot = self.rng.uniform(-SPOT, SPOT, (self.num_envs, 2))
        grasp = self.ee_pos()
        for w in np.flatnonzero(mask):
            self.hole[w] = [*(CENTRE + spot[w]), TOP_Z]
            sim.set_pose("block", pos=[*self.hole[w, :2], TABLE_Z], quat_wxyz=[1.0, 0.0, 0.0, 0.0], world=int(w))
            sim.set_pose("peg", pos=grasp[w] - np.array([0.0, 0.0, PEG_DROP]), quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                         world=int(w))

    def _obs(self) -> np.ndarray:
        parts = [self.joint_pos(), self.ee_pos(), self.state("peg.pos"), self.state("peg.quat_wxyz"), self.hole,
                 self.hole - self.state("peg.pos")]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def reset(self) -> np.ndarray:
        super().reset()
        return self._obs()

    def step(self, action: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _obs, reward, done = super().step(action)
        return self._obs(), reward, done

    # --- geometry --------------------------------------------------------------------------------------------------

    def tip(self) -> np.ndarray:
        """The peg's lower end: centre minus half a length along the (upward) peg axis; the peg's z is its length."""
        axis = quat_rotate_wxyz(self.state("peg.quat_wxyz"), np.array([0.0, 0.0, 1.0]))
        axis = axis * np.where(axis[:, 2:3] < 0.0, -1.0, 1.0)
        return self.state("peg.pos") - PEG_HALF * axis

    def depth(self) -> np.ndarray:
        return TOP_Z - self.tip()[:, 2]

    def lateral(self) -> np.ndarray:
        return (self.hole - self.tip())[:, :2]

    def probe(self) -> dict[str, np.ndarray]:
        return {"peg_pos": self.state("peg.pos"), "peg_quat": self.state("peg.quat_wxyz"), "hole_pos": self.hole.copy(),
                "ee_pos": self.ee_pos(), "joint_pos": self.joint_pos()}

    def reward(self) -> np.ndarray:
        miss = np.linalg.norm(self.lateral(), axis=1)
        return -miss - 0.5 * np.clip(INSERT - self.depth(), 0.0, None) + 1.0 * self.success_now()

    def success_now(self) -> np.ndarray:
        centre = np.linalg.norm((self.hole - self.state("peg.pos"))[:, :2], axis=1)
        return (self.depth() >= INSERT) & (np.linalg.norm(self.lateral(), axis=1) <= AXIS_TOL) & (centre <= AXIS_TOL)


def make_env(num_envs: int, seed: int) -> PegEnv:
    return PegEnv(num_envs, seed)


def scripted_policy(env: PegEnv, obs: Any) -> np.ndarray:
    """Slide the peg over the hole (stage 0: peg-tip error to the hole axis under 0.5 mm), lower it (1) at 3 cm/s, 1 cm/s
    from 8 mm above the opening, only while the sideways error is under 0.6 mm (it is fed back at every step), and hold
    it at 3.5 cm deep (2: the goal stays at the depth, a goal re-read from the hand would let the hand drift up)."""
    grasp, err = env.ee_pos(), env.lateral()
    miss = np.linalg.norm(err, axis=1)
    env.advance(((env.stage == 0) & (miss < ALIGNED_M)) | ((env.stage == 1) & (env.depth() >= GO_DEEP)))
    goal = grasp.copy()
    goal[:, :2] += err
    deep_z = TOP_Z + PEG_HALF + PEG_DROP - GO_DEEP               # grasp height with the tip GO_DEEP below the opening
    lowering = (env.stage == 1) & (miss < 0.0006)
    goal[:, 2] = np.where(env.stage == 2, deep_z, np.where(lowering, deep_z, grasp[:, 2]))
    pace = np.where(env.stage == 1, np.where(env.depth() > -0.008, 2, 1), 0)             # 0: 10 cm/s, 1: 3, 2: 1
    acts = np.stack([env.arm_action(goal, FACING, speed=v, gain=GAIN) for v in (0.1, DESCENT_SPEED, 0.01)])
    return acts[pace, np.arange(env.num_envs)].astype(np.float32)
