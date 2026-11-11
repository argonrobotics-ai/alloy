# SPDX-License-Identifier: Apache-2.0
"""Arm specs: Franka FR3 / Panda + Franka Hand, UR5e / UR10e + Robotiq 2F-85.

Physics comes from the validated embodiment packs, read in place (``library/robots/emb_e1_fr3``, ``emb_e2_panda``,
``emb_e3_ur10e_2f85``). The UR5e pack is not in the library: it is read from an assets checkout (``$ALLOY_ASSETS``),
as are mujoco_menagerie's render meshes (same pinned commit) for a pack without a visual layer of its own.

TCP convention: centre of the pads' gripping faces, z = approach (out of the
fingers), x = jaw axis (the direction the fingers close along).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from alloy.cards.library import default_root

ROBOTS = default_root() / "robots"
MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"   # the packs' pin (alloy.tasks.emb_specs)
SQRT_HALF = 0.7071067811865476
JAW_X_FROM_Y = (0.0, 0.0, SQRT_HALF, SQRT_HALF)                  # Rz(+90 deg): TCP x = flange y (fingers close on y)


def assets_dir() -> Path:
    """The arm files of the assets checkout (``$ALLOY_ASSETS`` or a sibling checkout): the UR5e pack, render meshes."""
    from alloy.cards.roots import assets_root

    root = assets_root()
    if root is None:
        raise FileNotFoundError("this arm needs files that are not in the library (the UR5e pack, or render meshes for "
                                "a pack without a visual layer): set ALLOY_ASSETS to an assets checkout")
    return Path(root) / "mjcf" / "clips" / "sc6"


@dataclass(frozen=True)
class GripperSpec:
    kind: str                         # "franka_hand" (linear, coupled) | "robotiq_2f85" (4-bar linkage)
    driver: str                       # actuated joint
    actuator: str                     # its MJCF actuator
    servo: dict = field(default_factory=dict)   # actuator overrides {kp, kv, ctrl, force} (empty = as in the pack)
    # opening, force cap and the width maps live in alloy.arms.gripper (FrankaHand, Robotiq2F85)


@dataclass(frozen=True)
class ArmSpec:
    key: str
    title: str
    pack: str                         # "library:<dir>" (Alloy library/robots) or "assets:<dir>" (the assets checkout)
    root: str                         # robot root body in the pack
    joints: tuple[str, ...]           # arm joints, base to wrist
    flange: str                       # body the TCP is fixed to
    tcp: tuple[float, ...]            # TCP in the flange frame (x y z qx qy qz qw)
    gripper: GripperSpec
    home: tuple[float, ...]           # ready joint angles (the pack's start pose unless noted)
    visuals: tuple[tuple[str, str], ...]  # (menagerie model xml, body-name prefix in the pack)
    reach_m: float                    # maker's nominal reach (for base placement)
    licence: str
    notes: tuple[str, ...] = ()

    def pack_xml(self) -> Path:
        where, name = self.pack.split(":", 1)
        return (ROBOTS if where == "library" else assets_dir()) / name / "scene.xml"


# Franka Hand servo: the pack keeps menagerie's law (kp 100 N/m, ctrl 0-0.04 m), which can squeeze at most ~4 N. The
# real hand holds 70 N continuous (Franka's data sheet), so the kit stiffens the position servo and
# let its target run below zero (a force-limited jaw closing on an object; the joint stops at 0 itself).
FRANKA_HAND_SERVO = {"kp": 2000.0, "kv": 40.0, "ctrl": (-0.04, 0.04), "force": (-70.0, 70.0)}
FRANKA_HAND = GripperSpec("franka_hand", "finger_joint1", "hand_finger", FRANKA_HAND_SERVO)
# 2F-85 servo: the pack's kp / ctrl / torque range, damping 10 -> 2 N m s/rad. Under a force cap the servo stays
# unsaturated, so a free close runs at cap / kv: 30 N (kit units) closed at ~17 mm/s with kv 10, below the real
# gripper's slowest 20 mm/s (20-150 mm/s, Robotiq's manual); kv 2 gives ~87 mm/s, step overshoot 6 %.
ROBOTIQ_SERVO = {"kp": 100.0, "kv": 2.0, "ctrl": (0.0, 0.8), "force": (-5.0, 5.0)}
ROBOTIQ = GripperSpec("robotiq_2f85", "gripper_right_driver_joint", "gripper_fingers_actuator", ROBOTIQ_SERVO)

# Franka Hand: pads (finger boxes at y 0..4 mm, z 36-53 mm in the finger, finger origin 58.4 mm along the hand z):
# gripping faces meet at y = 0 when closed, so width = 2 q; pad centre 102.9 mm along the hand z (Franka's TCP 103.4).
FRANKA_TCP = (0.0, 0.0, 0.1029, *JAW_X_FROM_Y)

FR3 = ArmSpec(
    key="fr3", title="Franka FR3 + Franka Hand", pack="library:emb_e1_fr3", root="base",
    joints=tuple(f"fr3_joint{i}" for i in range(1, 8)), flange="hand", tcp=FRANKA_TCP, gripper=FRANKA_HAND,
    home=(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853),
    visuals=(("franka_fr3/fr3.xml", ""), ("franka_emika_panda/hand.xml", "")), reach_m=0.855,
    licence="Apache-2.0 (mujoco_menagerie franka_fr3, franka_emika_panda hand)")

PANDA = ArmSpec(
    key="panda", title="Franka Panda + Franka Hand", pack="library:emb_e2_panda", root="link0",
    joints=tuple(f"joint{i}" for i in range(1, 8)), flange="hand", tcp=FRANKA_TCP, gripper=FRANKA_HAND,
    home=(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853), visuals=(), reach_m=0.855,
    licence="Apache-2.0 (mujoco_menagerie franka_emika_panda)",
    notes=("no render meshes vendored yet (the collision hulls render); add when a clip uses the Panda",))

_UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
              "wrist_3_joint")
_UR_HOME = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
# 2F-85 TCP: z from the gripper's pad table (alloy.arms.gripper.robotiq_table) at its reference width; the
# fingers close along the gripper_base y axis.
ROBOTIQ_TCP_REF_WIDTH = 0.030

UR10E = ArmSpec(
    key="ur10e", title="Universal Robots UR10e + Robotiq 2F-85", pack="library:emb_e3_ur10e_2f85", root="base",
    joints=_UR_JOINTS, flange="gripper_base", tcp=(0.0, 0.0, 0.0, *JAW_X_FROM_Y), gripper=ROBOTIQ, home=_UR_HOME,
    visuals=(("universal_robots_ur10e/ur10e.xml", ""), ("robotiq_2f85/2f85.xml", "gripper_")), reach_m=1.30,
    licence="BSD-3-Clause (mujoco_menagerie universal_robots_ur10e, robotiq_2f85)")

UR5E = ArmSpec(
    key="ur5e", title="Universal Robots UR5e + Robotiq 2F-85", pack="assets:emb_sc6_ur5e_2f85", root="base",
    joints=_UR_JOINTS, flange="gripper_base", tcp=(0.0, 0.0, 0.0, *JAW_X_FROM_Y), gripper=ROBOTIQ, home=_UR_HOME,
    visuals=(("universal_robots_ur5e/ur5e.xml", ""), ("robotiq_2f85/2f85.xml", "gripper_")), reach_m=0.85,
    licence="BSD-3-Clause (mujoco_menagerie universal_robots_ur5e, robotiq_2f85)")

SPECS: dict[str, ArmSpec] = {s.key: s for s in (FR3, PANDA, UR10E, UR5E)}


def arm_spec(key: str) -> ArmSpec:
    if key not in SPECS:
        raise KeyError(f"unknown arm {key!r}; known: {sorted(SPECS)}")
    return SPECS[key]


__all__ = ["FR3", "PANDA", "SPECS", "UR10E", "UR5E", "ArmSpec", "GripperSpec", "arm_spec", "assets_dir"]
