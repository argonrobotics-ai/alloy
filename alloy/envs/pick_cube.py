# SPDX-License-Identifier: Apache-2.0
"""PickCube: a Franka Panda picks a 4 cm cube off its table and carries it to a goal point in the air.

The arm starts at a hand-pointing-down home pose (jittered), the cube at (0.5, 0) m (jittered +-10 cm and +-45 deg
yaw), and a goal is drawn at every reset 4 to 30 cm above the table top. Actions are joint deltas for the 7 arm servos
plus the gripper servo. Reward is dense (reach, grasp, place, success bonus, small effort penalty); success is the
cube centre within 2.5 cm of the goal. Covered: the scene, the task and a factory. Not covered: a published training
result, camera observations, and cube sizes or masses other than the one here. Example::

    from alloy.envs.pick_cube import make_pick_cube
    env = make_pick_cube(num_envs=64)
"""

from __future__ import annotations

from typing import Any

from alloy.envs import checks as C
from alloy.envs.actions import ActionSpec
from alloy.envs.randomize import jitter_body, jitter_joint
from alloy.envs.state import rotate
from alloy.envs.task import Task

ARM = [f"panda/actuator{k}" for k in range(1, 8)]
GRIPPER = "panda/hand_finger"         # one servo drives finger_joint1, an equality ties finger_joint2 (0-0.04 m)
HOME = {"joint1": 0.0, "joint2": 0.393, "joint3": 0.0, "joint4": -1.963, "joint5": 0.0, "joint6": 2.356,
        "joint7": -0.7853}            # hand pointing down over the table
CUBE_SIZE, CUBE_START = 0.04, (0.5, 0.0, 0.02)   # full edge (m); the table top is z = 0, so the centre rests at 2 cm
GOAL_BOX = ((0.4, -0.1, 0.04), (0.6, 0.1, 0.30))  # xy within 10 cm of the cube's start, 4 to 30 cm above the table
GOAL_TOL = 0.025
# The finger bodies sit 0.0584 m along the hand's z axis (measured in a built sim); the Franka TCP, the pad centre, is
# 0.1034 m from the hand origin (Menagerie / Franka convention: finger origin + 0.045 m to the pad centre).
TCP_OFFSET = 0.1034
FINGER_OPEN, FINGER_DELTA = 0.04, 0.04           # open width of a finger joint (m); one step opens or closes
ARM_WINDOW = 0.05            # rad: a joint target stays within one step's delta of the measured joint (for RL;
#                              the default 5 steps let a blocked arm wind up 0.25 rad)
HOLD_LO, HOLD_HI = 0.010, 0.024                  # a finger joint (m) closed onto the 4 cm cube sits near 0.020
# A held cube, in the hand frame from the TCP (m): within 1 cm of the midline between the pads (y, the closing
# axis), and within 2.5 cm along the pads (x) and the fingers (z): a held cube slides ~2.3 cm (scripted check).
IN_HAND = (0.025, 0.01, 0.025)
LIFT = 0.01                  # m above the cube's resting height: held, not resting


def pick_cube_scene(worlds: int = 1, seed: int = 0) -> Any:
    """The Panda on its table (library robot, no props) and a 4 cm, 50 g cube resting at (0.5, 0).

    Example: ``alloy.build(pick_cube_scene(), worlds=64)``."""
    from alloy.envs.tasks import panda_scene  # noqa: PLC0415

    scene = panda_scene(worlds, seed, props=False)
    scene.add_box("cube", size=(CUBE_SIZE,) * 3, pos=CUBE_START, mass=0.05, friction=1.0)
    return scene


def tcp(s: Any) -> Any:
    """(W, 3) point between the fingertips: the hand origin plus ``TCP_OFFSET`` along the hand's z axis."""
    quat = s["panda/hand.quat_xyzw"]
    offset = s.xp.asarray([0.0, 0.0, TCP_OFFSET]).reshape(1, 3) + quat[:, :3] * 0.0
    return s["panda/hand.pos"] + rotate(s.xp, quat, offset)


def _cube_to_tcp(s: Any) -> Any:
    return s.xp.norm(tcp(s) - s["cube.pos"])


def cube_in_hand(s: Any) -> Any:
    """(W, 3) cube centre in the hand frame, from the TCP: x along the pads, y the closing axis, z along the fingers."""
    quat = s["panda/hand.quat_xyzw"]
    inverse = quat * s.xp.asarray([-1.0, -1.0, -1.0, 1.0]).reshape(1, 4)       # the conjugate
    return rotate(s.xp, inverse, s["cube.pos"] - tcp(s))


def enclosed(s: Any) -> Any:
    """(W,) bool: the fingers closed onto the cube (1 to 2.4 cm open each) with the cube between the pads (``IN_HAND``),
    so fingers closed beside or above the cube do not count. It does not say the fingers squeeze: see ``grasped``."""
    width, v = s["panda/finger_joint1.q"], abs(cube_in_hand(s))
    inside = (v[:, 0] < IN_HAND[0]) & (v[:, 1] < IN_HAND[1]) & (v[:, 2] < IN_HAND[2])
    return inside & (width > HOLD_LO) & (width < HOLD_HI)


def grasped(s: Any) -> Any:
    """(W,) bool: the cube enclosed by the fingers and off the table by ``LIFT`` or more, so it is held, not resting
    between open-handed fingers (a policy rewarded for enclosing alone learned to hold the cube on the table)."""
    return enclosed(s) & (s["cube.pos"][:, 2] > CUBE_START[2] + LIFT)


def pick_cube_task(max_steps: int = 100) -> Task:
    """PickCube-v0: reach, grasp and carry the cube to a goal drawn in ``GOAL_BOX``.

    Example: ``VectorEnv(pick_cube_scene(8), pick_cube_task(), worlds=8)``."""
    xp_tanh = lambda s, d: 1.0 - s.xp.tanh(5.0 * d)  # noqa: E731
    reach = C.custom("reach", lambda s: xp_tanh(s, _cube_to_tcp(s)))
    around = C.custom("enclosed", lambda s: s.xp.float(enclosed(s)))
    held = C.custom("grasped", lambda s: s.xp.float(grasped(s)))
    place = C.custom("place", lambda s: s.xp.float(grasped(s)) * xp_tanh(s, s.xp.norm(s["cube.pos"] - s["goal.pos"])))
    success = C.near("cube", "goal", GOAL_TOL)
    obs = ["panda", "cube", "goal", ("tcp", tcp), ("tcp_to_cube", lambda s: s["cube.pos"] - tcp(s)),
           ("tcp_to_goal", lambda s: s["goal.pos"] - tcp(s)), ("cube_to_goal", lambda s: s["goal.pos"] - s["cube.pos"]),
           around, held, "action"]
    starts = [jitter_joint(f"panda/{j}", by=0.02, at=v) for j, v in HOME.items()]
    starts += [jitter_joint(f"panda/finger_joint{k}", at=FINGER_OPEN) for k in (1, 2)]
    starts.append(jitter_body("cube", pos=(0.1, 0.1, 0.0), yaw_deg=45.0))
    return Task(name="PickCube-v0", obs=obs, max_steps=max_steps, randomize=starts, goals={"goal": GOAL_BOX},
                actions=ActionSpec([*ARM, GRIPPER], mode="delta", delta=[0.05] * 7 + [FINGER_DELTA],
                                    window=[ARM_WINDOW] * 7 + [FINGER_OPEN]),
                reward={"reach": (1.0, reach), "enclosed": (0.5, around), "grasped": (1.0, held), "place": (2.0, place),
                        "success": (3.0, success), "effort": (-0.01, C.action_size())},
                success=success, end_on_success=False, failure=C.height("cube") < -0.05)


def make_pick_cube(num_envs: int = 1, device: str | None = None, seed: int = 0, max_steps: int = 100,
                   **env_kw: Any) -> Any:
    """A ``VectorEnv`` of PickCube-v0; ``max_steps`` goes to the task, the other keywords to ``VectorEnv``.

    Example: ``make_pick_cube(num_envs=64, device="cuda:0")``."""
    from alloy.envs.vector import VectorEnv  # noqa: PLC0415

    return VectorEnv(pick_cube_scene(num_envs, seed), pick_cube_task(max_steps), worlds=num_envs, device=device,
                     seed=seed, **env_kw)


__all__ = ["GOAL_BOX", "enclosed", "grasped", "make_pick_cube", "pick_cube_scene", "pick_cube_task", "tcp"]
