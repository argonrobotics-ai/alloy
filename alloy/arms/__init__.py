# SPDX-License-Identifier: Apache-2.0
"""Robot arm kit: Franka FR3 / Panda + Franka Hand and UR5e / UR10e + Robotiq 2F-85, for task scripts and clips.

Specs (``FR3``, ``UR10E``, ...), a robot-only MJCF of each arm (``robot_xml``), batched IK (``ChainIK``) and ``Rig``:
one or two arms at their bases, with their bindings, gravity compensation and servo targets.

    from alloy.arms import FR3, UR5E, Rig
    rig = Rig.single(UR5E, base=(-0.15, -0.20, 0.0))       # or Rig.pair(FR3, left_base, right_base)
    rig.add_to(builder)                                    # robot-only MJCF + render meshes, MJCF actuators
    ik = rig.make_ik(W, device)                            # batched IK: TCP poses -> joint angles
"""

from alloy.arms.binding import ArmBinding, BindingError
from alloy.arms.gripper import FrankaHand, Robotiq2F85, robotiq_table
from alloy.arms.ik import ChainIK, IKError
from alloy.arms.mjcf import ArmModelError, add_arm, robot_xml
from alloy.arms.reach import ReachReport, place_base, reach
from alloy.arms.rig import Rig
from alloy.arms.specs import FR3, PANDA, SPECS, UR5E, UR10E, ArmSpec, arm_spec

__all__ = [
    "FR3", "PANDA", "SPECS", "UR10E", "UR5E", "ArmBinding", "ArmModelError", "ArmSpec", "BindingError", "ChainIK",
    "FrankaHand", "IKError", "ReachReport", "Rig", "Robotiq2F85", "add_arm", "arm_spec", "place_base", "reach",
    "robot_xml", "robotiq_table",
]
