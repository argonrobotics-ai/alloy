# SPDX-License-Identifier: Apache-2.0
"""E21 reference (Alloy): a Franka Panda opens a drawer by its handle in a cabinet 60 cm in front of it.

The cabinet (static carcass, a drawer on a 30 cm slide with 2 N of rail friction, a horizontal bar handle on 12 cm stand-offs (the 0.2 m wide hand needs the room) 4 cm proud of
the drawer front) is one MJCF robot with a slide joint. Actions (8): joint position target offsets (0.05 rad per
step at +-1) and the gripper (+1 opens, -1 closes). The arm starts with the hand pointing down above the handle bar,
fingers apart; each reset jitters the arm's start joints by up to 0.03 rad. Success: the drawer open by 20 cm.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(d) for d in (_HERE, _HERE.parent) if (d / "panda_env.py").is_file()][:1]

import panda_kin as kin  # noqa: E402
from panda_env import ARM_JOINTS, TABLE_Z, PandaConfig, PandaEnv  # noqa: E402
from robota_common import solve_start, strong_grip_file  # noqa: E402

FRONT_X = 0.60                       # drawer front face, 60 cm in front of the robot base (m)
STANDOFF, STAND = 0.12, 0.14        # the handle bar is this far in front of the drawer front; the cabinet stands on a 14 cm plinth
BAR_Z = STAND + 0.10                 # bar height above the table (a lower bar needs a wrist posture the Panda cannot hold at the pull's end)
HANDLE = np.array([FRONT_X - STANDOFF, 0.0, TABLE_Z + BAR_Z])      # handle bar centre when closed
TILT = 0.0                           # hand pointing straight down (tilting it forward cost reach in the IK survey)
FACING = np.column_stack([[0.0, 1.0, 0.0], [np.cos(TILT), 0.0, np.sin(TILT)], [np.sin(TILT), 0.0, -np.cos(TILT)]])
GOAL_OPEN, TRAVEL = 0.20, 0.30
ABOVE = 0.10                         # policy: grasp point this far above the bar before the descent
GAIN = 0.15                         # policy: resolved-rate gain (the default 0.15 stalls a few mm short on the float32 servos)
START_JITTER = 0.03                  # rad, each arm joint at every reset

def _cabinet_mjcf() -> str:
    """Carcass (plinth, floor, roof, two walls, back) and the drawer: front panel, tray, handle (bar on two posts)."""
    x, zo = FRONT_X, STAND
    return f"""<mujoco model="cabinet"><compiler angle="radian"/>
<worldbody>
 <body name="carcass">
  <geom type="box" pos="{x+0.15} 0 {zo/2}" size="0.15 0.30 {zo/2}"/>
  <geom type="box" pos="{x+0.15} 0 {zo+0.01}" size="0.15 0.30 0.01"/>
  <geom type="box" pos="{x+0.15} 0 {zo+0.19}" size="0.15 0.30 0.01"/>
  <geom type="box" pos="{x+0.15} 0.29 {zo+0.10}" size="0.15 0.01 0.10"/>
  <geom type="box" pos="{x+0.15} -0.29 {zo+0.10}" size="0.15 0.01 0.10"/>
  <geom type="box" pos="{x+0.29} 0 {zo+0.10}" size="0.01 0.30 0.10"/>
 </body>
 <body name="drawer">
  <inertial pos="{x+0.14} 0 {zo+0.09}" mass="1.0" diaginertia="0.02 0.02 0.02"/>
  <joint name="slide" type="slide" axis="-1 0 0" range="0 {TRAVEL}" frictionloss="2.0" damping="0.5"/>
  <geom type="box" pos="{x+0.01} 0 {zo+0.10}" size="0.01 0.27 0.07"/>
  <geom type="box" pos="{x+0.14} 0 {zo+0.04}" size="0.12 0.25 0.01"/>
  <geom type="box" pos="{x+0.14} 0.25 {zo+0.08}" size="0.12 0.01 0.05"/>
  <geom type="box" pos="{x+0.14} -0.25 {zo+0.08}" size="0.12 0.01 0.05"/>
  <geom type="box" pos="{x+0.26} 0 {zo+0.08}" size="0.01 0.25 0.05"/>
  <geom type="cylinder" fromto="{x} 0.1 {BAR_Z} {x-STANDOFF} 0.1 {BAR_Z}" size="0.008"/>
  <geom type="cylinder" fromto="{x} -0.1 {BAR_Z} {x-STANDOFF} -0.1 {BAR_Z}" size="0.008"/>
  <geom type="cylinder" fromto="{x-STANDOFF} -0.1 {BAR_Z} {x-STANDOFF} 0.1 {BAR_Z}" size="0.008"
        friction="1.5 0.005 0.0001"/>
 </body>
</worldbody></mujoco>"""


CABINET = _cabinet_mjcf()


class DrawerEnv(PandaEnv):
    def __init__(self, num_envs: int, seed: int) -> None:
        start = solve_start(HANDLE + np.array([0.0, 0.0, ABOVE]), FACING)
        cfg = PandaConfig(start_q=start, obs=["panda", "cabinet"], episode_s=8.0, robot_file=strong_grip_file())
        self.handle_closed = HANDLE.copy()
        super().__init__(num_envs, seed, cfg)

    def add_objects(self, scene: Any) -> None:
        scene.add_robot("cabinet", mjcf=CABINET)

    def _task(self):  # noqa: ANN202
        """The base task plus a small random start for every arm joint."""
        import dataclasses  # noqa: PLC0415

        from alloy.envs.randomize import jitter_joint  # noqa: PLC0415

        task = super()._task()
        arm = [jitter_joint(j, at=float(v), by=START_JITTER) for j, v in zip(ARM_JOINTS, self.cfg.start_q)]
        rest = [j for j in task.randomize if j.name not in ARM_JOINTS]
        return dataclasses.replace(task, randomize=arm + rest)

    # --- state ---------------------------------------------------------------------------------------------------

    def drawer_open(self) -> np.ndarray:
        return self.state("cabinet/slide.q")

    def handle_pos(self) -> np.ndarray:
        """Handle bar centre: its closed position moved by the drawer body's pose (the body frame starts at the world
        origin with no rotation, and the drawer only slides)."""
        return self.handle_closed + self.state("cabinet/drawer.pos")

    def probe(self) -> dict[str, np.ndarray]:
        return {"drawer_open": self.drawer_open(), "handle_pos": self.handle_pos(), "ee_pos": self.ee_pos(),
                "joint_pos": self.joint_pos(), "gripper_width": self.gripper_width()}

    def reward(self) -> np.ndarray:
        reach = np.linalg.norm(self.handle_pos() - self.ee_pos(), axis=1)
        return -reach + 10.0 * np.clip(self.drawer_open(), 0.0, GOAL_OPEN) + 2.0 * self.success()

    def success_now(self) -> np.ndarray:
        return self.drawer_open() >= GOAL_OPEN


def make_env(num_envs: int, seed: int) -> DrawerEnv:
    return DrawerEnv(num_envs, seed)


def scripted_policy(env: DrawerEnv, obs: Any) -> np.ndarray:
    """Hover over the bar (stage 0), descend with the fingers apart (1), close (2), pull back along the rail (3) until
    the drawer is open 0.22 m, let go (4: fingers open; the rail friction holds it) and lift away (5); stages switch per world on measured positions."""
    handle, grasp = env.handle_pos(), env.ee_pos()
    above = handle + np.array([0.0, 0.0, ABOVE])
    pulling = handle + np.array([-0.05, 0.0, 0.0])
    stage = env.stage[:, None]
    lift = np.stack([grasp[:, 0], grasp[:, 1], handle[:, 2] + ABOVE], axis=1)
    goal = np.select([stage == 0, stage == 1, stage == 2, stage == 3, stage == 5], [above, handle, handle, pulling, lift], grasp)
    over = np.linalg.norm((above - grasp)[:, :2], axis=1) < 0.006
    on = np.linalg.norm(handle - grasp, axis=1) < 0.006
    env.advance(((env.stage == 0) & over) | ((env.stage == 1) & on) | ((env.stage == 2) & (env.clock >= 25))
                | ((env.stage == 3) & (env.drawer_open() >= 0.22)) | ((env.stage == 4) & (env.clock >= 15)))
    action = np.zeros((env.num_envs, 8), np.float32)
    slow = (env.stage == 3).reshape(-1, 1)
    action[:, :7] = np.where(slow, env.arm_action(goal, FACING, speed=0.12, gain=GAIN),
                             env.arm_action(goal, FACING, speed=0.3, gain=GAIN))
    action[:, 7] = np.where((env.stage >= 2) & (env.stage <= 3), -1.0, 1.0)
    return action
