# SPDX-License-Identifier: Apache-2.0
"""SceneCards for the task scenes (``library/scenes/task_*.yaml``).

A converted scene's card is the replay tier: one ``RobotRef`` to the vendored full-precision MJCF (arms, bench,
fixtures and objects, with every recorded contact pair, margin and solver option), ``control_dt`` = the recorded
solve dt, so the D4 grid holds at j = 0 and one trace row is one control period. Objects are not split into
routed cards here, because card materials and Alloy's pair table would change the contact parameters the recorded
trajectory depends on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np

from alloy.cards.loader import dump_card
from alloy.cards.types import RobotRef, SceneCard
from alloy.tasks.newton_map import quat_mul, rotate, wxyz_to_xyzw

log = logging.getLogger("alloy.tasks")

IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def converted_scene_card(scene: str, robots_root: Path, worlds: int = 1) -> SceneCard:
    """Replay-tier SceneCard for ``library/robots/<scene>/scene.xml`` (path relative to the library root)."""
    xml = robots_root / scene / "scene.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))  # offline read of the option block only
    robot = RobotRef(path=f"robots/{scene}/scene.xml", kind="mjcf", base_pose=IDENTITY_POSE, name=scene)
    return SceneCard(robots=(robot,), objects=(), worlds=worlds, control_dt=float(m.opt.timestep))


def write_scene_card(card: SceneCard, scenes_dir: Path, name: str, header: str) -> Path:
    path = scenes_dir / f"{name}.yaml"
    path.write_text(dump_card(card, header=header))
    log.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------------------------- authored scenes
# Poses are (x, y, z, qx, qy, qz, qw) on the bench top (plywood_sheet top at z = 0, x 0.05-0.65, y ±0.6; arm bases at
# (0, ±0.31), as in every reference-task recording). Object frames are WP1's meshes: plug +z = mating direction (shell
# 0..11.75 mm, overmold back at -17 mm), receptacle opening at +z = 12 mm, bolt shank along -z from the head.
ROT_Y_M90 = (0.0, -0.7071067811865476, 0.0, 0.7071067811865476)   # +z -> -x, +x -> +z
CYCLE_XYZ = (0.5, 0.5, 0.5, 0.5)                                   # x -> y, y -> z, z -> x (120° about (1,1,1))
ROT_Z_90 = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)      # +x -> +y
ROT_X_180 = (1.0, 0.0, 0.0, 0.0)                                   # +z -> -z


def _robot(name: str) -> RobotRef:
    return RobotRef(path=f"robots/{name}/robot.xml", kind="mjcf", base_pose=IDENTITY_POSE, name=name)


def t09_card() -> SceneCard:
    """T09 USB-A plug into a horizontal port with a 1 m cable ([internal reference] layout, simplified).

    Receptacle on its side, opening facing -x at x = 0.448 m, port axis 10 mm above the bench (the hub case has no
    card; the real port centre is 5.0 mm high, image estimate). Plug lying flat 0.136 m in front of the port, axis
    along +x; cable straight along +y from 3 mm behind the plug, welded to it. The C end is left free (the task
    fixes it at (0.625, -0.55)). Robot: the arms and bench tuned at 1/1200 s (from r2_t12). The task ran at 1e-4 s,
    but the assembler picks dt from the D4 targets (6.25e-4 s here, `discrete` because of the CHAIN cable), and the
    1e-4-tuned bench (sheet solref time constant 2.5e-4 s with solmix 1e4) sends every free body on it to NaN within
    3 periods at that dt (a 30 mm cube too; measured 2026-11-01), while the 1/1200-tuned bench steps 20 periods.
    """
    from alloy.cards.types import Attachment, Placement

    objects = (
        Placement("usb_a_receptacle", (0.46, -0.31, 0.010, *ROT_Y_M90)),
        Placement("usb_a_plug", (0.30, -0.31, 0.00415, *CYCLE_XYZ)),
        Placement("usb_cable_1m", (0.280, -0.31, 0.00225, *ROT_Z_90)),
    )
    weld = Attachment(a="usb_a_plug", b="usb_cable_1m", kind="weld", anchor=(0.0, -0.0019, -0.020, 0.0, 0.0, 0.0, 1.0))
    return SceneCard(robots=(_robot("yam_bench"),), objects=objects, attachments=(weld,))


def t02_card() -> SceneCard:
    """T02 M10 nut on bolt ([internal reference] / bench P02): DIN 933 M10x60 bolt standing head-down on a
    plywood base and welded to it, DIN 934 nut lying on the base 60 mm beside the bolt; arms and bench tuned at
    1/1200 s (from r2_t12), the task's dt. The nut does not start on the bolt tip: a scene built from cards has no
    thread between the two (the analytic thread pair is wired only by alloy.contact.thread_twins and the clips), so
    a nut on the tip fell down the shank and hit the head at 1 m/s (audit LIB-4: PAIR_PEN_FAIL at 0.11 s)."""
    from alloy.cards.types import Attachment, Placement

    base_top = 0.0127
    bolt_z = base_top + 0.0064            # head (6.4 mm) on the base, shank up to bolt_z + 0.06
    objects = (
        Placement("pegboard_base_plywood", (0.40, 0.0, 0.00635, 0.0, 0.0, 0.0, 1.0)),
        Placement("bolt_m10x60", (0.40, 0.0, bolt_z, *ROT_X_180)),
        Placement("nut_m10", (0.46, 0.0, base_top + 0.004, 0.0, 0.0, 0.0, 1.0)),    # flat on the base (8 mm high)
    )
    weld = Attachment(a="pegboard_base_plywood", b="bolt_m10x60", kind="weld")
    return SceneCard(robots=(_robot("yam_bench"),), objects=objects, attachments=(weld,))


# ---------------------------------------------------------------------------------------------- card tier
# Card tier: the recorded layout rebuilt from WP1's object cards (routed, card materials, Alloy pair table) around the
# arms + bench robot, for packages that need routed objects (WP8 ladder, WP7 bully, WP3 precision pairs). Its
# trajectory is not gated against the recording: contact parameters come from the cards, not the recorded MJCF.
# Each entry: recorded body -> [(object card, offset pose (x, y, z, qx, qy, qz, qw) in that body's frame)].
_ID = (0.0, 0.0, 0.0, 1.0)
_RZ90 = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
T12_CARDS = {f"block_{i}": [("block_alder_75x25x15", (0.0, 0.0, 0.0, *_ID))] for i in range(1, 9)} | {
    "ledge": [("ledge_plywood", (0.0, 0.0, 0.04, *_ID))],    # recorded ledge_box centre is 40 mm above the origin
}
# reference-task tight_pegs (reference-task/src/reference-task/assets/scenes/tight_pegs/params.yaml:1033-1120): 12.7 mm ply plate,
# 9 mm acrylic on top (recorded board z 0..21.7 mm), hole centres (±22.5, ±22.5) mm in the board frame, the rect hole's
# long axis along board y (WP1's block has it along x, hence the 90° turn); WP1 tiles it as four 40 mm bore blocks.
_BORE_Z = 0.0127 + 0.0045
T11_CARDS = {
    "peg_board": [("pegboard_base_plywood", (0.0, 0.0, 0.00635, *_ID)),
                  ("block_bore_d12_2", (-0.0225, 0.0225, _BORE_Z, *_ID)),
                  ("block_rect_hole_12x8", (0.0225, 0.0225, _BORE_Z, *_RZ90)),
                  ("block_bore_d8_1", (-0.0225, -0.0225, _BORE_Z, *_ID)),
                  ("block_bore_d16_5", (0.0225, -0.0225, _BORE_Z, *_ID))],
    "round12_peg": [("peg_d12", (0.0, 0.0, 0.0, *_ID))], "rect_peg": [("peg_rect_12x8", (0.0, 0.0, 0.0, *_ID))],
    "round8_peg": [("peg_d8", (0.0, 0.0, 0.0, *_ID))], "round16_peg": [("peg_d16", (0.0, 0.0, 0.0, *_ID))],
}  # fmt: skip


def _initial_body_pose(m: mujoco.MjModel, qpos0: np.ndarray, body: str) -> tuple[np.ndarray, np.ndarray]:
    """World pose (p, q_xyzw) of ``body``: the trace's initial qpos for a free body, else the model's body pose."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    free = [k for k in range(m.njnt) if m.jnt_bodyid[k] == b and m.jnt_type[k] == int(mujoco.mjtJoint.mjJNT_FREE)]
    if free:
        q = np.asarray(qpos0[int(m.jnt_qposadr[free[0]]) : int(m.jnt_qposadr[free[0]]) + 7], dtype=np.float64)
        return q[:3], wxyz_to_xyzw(q[3:7])
    if int(m.body_parentid[b]) != 0:
        raise ValueError(f"{body}: only top-level static bodies are placed")
    return np.asarray(m.body_pos[b], dtype=np.float64), wxyz_to_xyzw(np.asarray(m.body_quat[b], dtype=np.float64))


def card_tier_scene(scene_xml: Path, qpos0: np.ndarray, cards: dict, robot: str) -> SceneCard:
    """Placements of ``cards`` ({recorded body: [(object card, offset pose in the body frame)]}) at the trace start."""
    from alloy.cards.types import Placement

    m = mujoco.MjModel.from_xml_path(str(scene_xml))  # offline read of body poses
    placements = []
    for body, items in cards.items():
        p, q = _initial_body_pose(m, qpos0, body)
        for card, off in items:
            pos = p + rotate(q, np.asarray(off[:3]))
            quat = quat_mul(q, np.asarray(off[3:]))
            placements.append(Placement(card, tuple(float(v) for v in (*pos, *quat))))
    return SceneCard(robots=(_robot(robot),), objects=tuple(placements), control_dt=float(m.opt.timestep))
