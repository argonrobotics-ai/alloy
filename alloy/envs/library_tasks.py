# SPDX-License-Identifier: Apache-2.0
"""Ready-made manipulation tasks built from the object library, with a Cartesian two-finger gripper.

- ``PickPlace-v0``: pick a YCB object off the table and put it in a bin.
- ``PegInsert-v0``: the gripper holds a square peg above a block; insert it into a hole 0.5 mm wider (a precise fit:
  rigid contacts step at 0.25 ms or finer, so it is several times slower per step than the others).
- ``DrawerOpen-v0``: grasp the handle of a RoboCasa drawer cabinet's middle drawer and pull it 15 cm open.

Each scene goes through ``alloy.vary`` (``vary=True``, seeded): placement and textures differ per environment
instance. Each reset also moves the objects and the gripper per world (``Task.randomize``). ``camera=True`` adds a
64x64 image observation (``obs["front"]``). The objects come from the object library: on a fresh install the first
build fetches them (``alloy fetch <name>``). Example::

    from alloy.envs import make
    env = make("PickPlace-v0", num_envs=64)
    env = make("PickPlace-v0", num_envs=64, obj="ycb_014_lemon", camera=True)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from alloy.envs import checks as C
from alloy.envs.errors import EnvError, install_fix
from alloy.envs.placement import rest_on_floor
from alloy.envs.randomize import jitter_body, jitter_joint
from alloy.envs.robots import WALL_SAFE, fingers, gantry_actions, gantry_gripper
from alloy.envs.state import BatchState
from alloy.envs.task import Task

log = logging.getLogger("alloy.envs")

GRIP = "gripper"
PICK_OBJECTS = ("ycb_077_rubiks_cube", "ycb_014_lemon", "ycb_018_plum", "ycb_056_tennis_ball", "ycb_009_gelatin_box")
BIN_POS, BIN_INNER = (0.18, 0.0, 0.0), (0.16, 0.16, 0.06)
HOLE_AT, BLOCK, PEG = (0.0, 0.0, 0.0), (0.08, 0.08, 0.04), (0.01, 0.01, 0.06)
PEG_OUT = 0.025                 # m, peg below the fingertips: the 2 cm insertion plus 5 mm
CABINET, DRAWER = "robocasa_cabinet_drawers_001_001", "drawer1"
HANDLE_OFFSET = (0.0, -0.019, 0.0)        # the handle bar's centre from its drawer body (m)
GRASP_OFFSET = (0.0, -0.041, 0.0)         # finger midpoint target: fingertips at the drawer front


def maybe_vary(scene: Any, vary: bool, seed: int, pos: float = 0.0, yaw_deg: float = 0.0) -> Any:
    """``alloy.vary`` one variant of ``scene`` (placement and textures) when ``vary``; the scene itself otherwise."""
    if not vary:
        return scene
    from alloy.api._vary import vary as alloy_vary  # noqa: PLC0415

    try:
        return alloy_vary(scene, 1, pos=pos, yaw_deg=yaw_deg, seed=seed)[0]
    except Exception as exc:  # noqa: BLE001 - a look library may be missing: the plain scene still works
        log.warning("alloy.vary failed (%s); using the scene as written", exc)
        return scene


def front_camera(eye: tuple[float, float, float], target: tuple[float, float, float], size: int = 64) -> Any:
    """A square camera at ``eye`` looking at ``target``. Example: ``front_camera((0.4, -0.4, 0.4), (0, 0, 0))``."""
    from alloy.camera import Camera  # noqa: PLC0415

    return Camera("front", look_at=(eye, target), width=size, height=size, fov=45, spp=4)


def grasp_point(s: BatchState) -> Any:
    """(W, 3) midpoint between the gripper's fingers."""
    left, right = fingers(GRIP)
    return 0.5 * (s[f"{left}.pos"] + s[f"{right}.pos"])


def _scene(worlds: int, ground: bool = True) -> Any:
    from alloy.api.scene import Scene  # noqa: PLC0415

    scene = Scene(worlds=worlds, dt=0.02)
    if ground:
        scene.add_ground()
    return scene


# --- PickPlace ---------------------------------------------------------------------------------------------------

def add_library_object(scene: Any, name: str, card: str, env: str, **kwargs: Any) -> None:
    """``scene.add_object`` for a ready-made env's library object, naming the install fix when the card comes from
    the Hugging Face object library and this install cannot fetch it (no ``huggingface_hub``)."""
    import importlib.util  # noqa: PLC0415

    from alloy.api import catalog  # noqa: PLC0415

    if importlib.util.find_spec("huggingface_hub") is None and not catalog.can_place(card, scene.roots):
        raise EnvError(f"{env} needs the object {card!r} from the Hugging Face object library, and huggingface_hub "
                       "is not installed", f"{install_fix('huggingface_hub')}; the first build then fetches it",
                       field="obj")
    scene.add_object(name, card, **kwargs)


def pick_place_scene(worlds: int = 1, seed: int = 0, obj: str = PICK_OBJECTS[0], vary: bool = True) -> Any:
    """A YCB object (``obj``, a library card) on the table, an open bin 18 cm away and the gripper 20 cm up.

    Example: ``alloy.build(pick_place_scene(64), worlds=64)``."""
    scene = _scene(worlds)
    scene.add_open_box("bin", inner=BIN_INNER, wall=0.006, pos=BIN_POS)
    add_library_object(scene, "obj", obj, "PickPlace-v0", pos=(-0.12, 0.0, 0.001), anchor="origin")
    scene.add_robot(GRIP, mjcf=gantry_gripper(**WALL_SAFE), pos=(0.0, 0.0, 0.2))
    return maybe_vary(scene, vary, seed, pos=0.02, yaw_deg=30)


def pick_place_task(camera: bool = False, max_steps: int = 250) -> Task:
    """Reward: reach the object, hold it between the fingers, lift it, carry it over the bin; success: its centre of
    mass inside the bin. Example: ``VectorEnv(pick_place_scene(16), pick_place_task(), worlds=16)``."""
    lo = (BIN_POS[0] - BIN_INNER[0] / 2, BIN_POS[1] - BIN_INNER[1] / 2, 0.0)
    hi = (BIN_POS[0] + BIN_INNER[0] / 2, BIN_POS[1] + BIN_INNER[1] / 2, BIN_INNER[2] + 0.01)
    held = C.between("obj.com", *fingers(GRIP), max_offset=0.04, max_gap=0.1)
    in_bin = C.inside("obj.com", lo, hi)
    reach = C.custom("reach", lambda s: s.xp.norm(grasp_point(s) - s["obj.com"]))
    above_bin = (BIN_POS[0], BIN_POS[1], 0.12)
    carry = C.custom("carry", lambda s: s.xp.float(held(s)) * C.distance("obj.com", above_bin)(s))
    lift = C.custom("lift", lambda s: s.xp.float(held(s)) * s.xp.clip(C.rise("obj.com")(s), 0.0, 0.1))
    cams = {"front": front_camera((0.45, -0.45, 0.4), (0.03, 0.0, 0.03))} if camera else {}
    return Task(
        name="PickPlace-v0", obs=[GRIP, "obj", "obj.com", "action"], actions=gantry_actions(GRIP, yaw=False),
        reward={"reach": (-1.0, reach), "held": (0.5, held), "lift": (5.0, lift), "carry": (-1.0, carry),
                "in_bin": (5.0, in_bin)},
        success=in_bin, end_on_success=False, max_steps=max_steps, cameras=cams, settle=0.3,
        on_build=lambda sim: rest_on_floor(sim, ["obj"]),
        randomize=[jitter_body("obj", pos=(0.03, 0.04, 0.0), yaw_deg=45.0),
                   jitter_joint(f"{GRIP}/x", by=0.03), jitter_joint(f"{GRIP}/y", by=0.03)])


# --- PegInsert ---------------------------------------------------------------------------------------------------

def peg_insert_scene(worlds: int = 1, seed: int = 0, clearance: float = 0.0005, precise: bool = True,
                     vary: bool = True) -> Any:
    """A block with a square hole (``PEG`` width + ``clearance``), a 10 mm square peg between the open fingers 2 cm
    above it (the first reset closes the fingers on it). The peg sticks out ``PEG_OUT`` below the fingertips, so it
    goes 2 cm in without sliding in the grip. Example: ``alloy.build(peg_insert_scene(8), worlds=8)``."""
    scene = _scene(worlds)
    hole = PEG[0] + clearance
    scene.add_block_with_hole("block", BLOCK, hole=(hole, hole), depth=0.03, pos=HOLE_AT, precise=precise)
    tip_z = BLOCK[2] + 0.02
    scene.add_box("peg", size=PEG, pos=(HOLE_AT[0], HOLE_AT[1], tip_z + PEG[2] / 2), mass=0.02, friction=0.8)
    palm_z = tip_z + PEG_OUT + 0.07                      # the palm sits 7 cm above the fingertips
    scene.add_robot(GRIP, mjcf=gantry_gripper(reach_z=0.12, **WALL_SAFE), pos=(HOLE_AT[0], HOLE_AT[1], palm_z))
    return maybe_vary(scene, vary, seed)


def peg_tip(s: BatchState) -> Any:
    """(W, 3) centre of the peg's bottom face."""
    from alloy.envs.state import rotate  # noqa: PLC0415

    down = s.xp.asarray([0.0, 0.0, -PEG[2] / 2]).reshape(1, 3) + s["peg.com"] * 0.0
    return s["peg.com"] + rotate(s.xp, s["peg.quat_xyzw"], down)


def peg_insert_task(camera: bool = False, max_steps: int = 150, start_offset: float = 0.015) -> Task:
    """Reward: bring the peg's tip over the hole and down into it; success: the tip 2 cm below the block top, inside
    the hole. Each reset moves the gripper and the held peg up to ``start_offset`` m in x and y.

    Example: ``VectorEnv(peg_insert_scene(8), peg_insert_task(), worlds=8)``."""
    bottom = (HOLE_AT[0], HOLE_AT[1], BLOCK[2] - 0.03)
    tip = ("peg_tip", peg_tip)
    to_hole = C.custom("to_hole", lambda s: s.xp.norm(peg_tip(s) - s.xp.asarray(bottom).reshape(1, 3)))
    depth = C.custom("depth", lambda s: s.xp.clip(BLOCK[2] - peg_tip(s)[:, 2], 0.0, 0.03))
    inserted = C.custom("inserted", lambda s: (depth(s) >= 0.02) & (s.xp.norm(peg_tip(s)[:, :2] - s.xp.asarray(
        HOLE_AT[:2]).reshape(1, 2)) < 0.004))
    grip = PEG[0] / 2
    touch = 0.04 - grip                                     # finger travel at which the fingers touch the peg
    cams = {"front": front_camera((0.25, -0.25, 0.2), (0.0, 0.0, 0.05))} if camera else {}
    return Task(
        name="PegInsert-v0", obs=[GRIP, "peg", tip, "action"], actions=gantry_actions(GRIP, step_m=0.004, yaw=False),
        reward={"to_hole": (-2.0, to_hole), "depth": (30.0, depth), "inserted": (5.0, inserted)},
        success=inserted, end_on_success=False, max_steps=max_steps, cameras=cams,
        failure=C.height("peg.com") < BLOCK[2] / 2 - 0.03,  # dropped beside the block
        randomize=[jitter_joint(f"{GRIP}/close", at=touch - 0.0005, target=touch + 0.003),
                   jitter_joint(f"{GRIP}/x", by=start_offset, carry=["peg"]),
                   jitter_joint(f"{GRIP}/y", by=start_offset, carry=["peg"])])


# --- DrawerOpen --------------------------------------------------------------------------------------------------

def drawer_open_scene(worlds: int = 1, seed: int = 0, cabinet: str = CABINET, vary: bool = True) -> Any:
    """A RoboCasa drawer cabinet (0.6 x 0.58 x 0.72 m) standing on the floor, the gripper 30 cm in front of its
    middle drawer, turned so its fingers point at the cabinet and close vertically.

    Example: ``alloy.build(drawer_open_scene(16), worlds=16)``."""
    scene = _scene(worlds)
    add_library_object(scene, "cabinet", cabinet, "DrawerOpen-v0", pos=(0.0, 0.3, 0.36), anchor="origin")
    scene.add_robot(GRIP, mjcf=gantry_gripper(reach_z=0.3, **WALL_SAFE), pos=(0.0, -0.3, 0.36),
                    rpy=(math.pi / 2, 0.0, 0.0))
    return maybe_vary(scene, vary, seed)


def handle_point(s: BatchState) -> Any:
    """(W, 3) centre of the middle drawer's handle bar."""
    return s[f"cabinet/{DRAWER}.pos"] + s.xp.asarray(HANDLE_OFFSET).reshape(1, 3)


def grasp_target(s: BatchState) -> Any:
    """(W, 3) where the finger midpoint should be to close on the handle bar (near the fingertips)."""
    return s[f"cabinet/{DRAWER}.pos"] + s.xp.asarray(GRASP_OFFSET).reshape(1, 3)


def drawer_open_task(camera: bool = False, max_steps: int = 200, opening: float = 0.15) -> Task:
    """Reward: bring the fingers to the handle, then pull the drawer open; success: open ``opening`` m.
    The gripper's slides are turned with it: its "z" moves toward / away from the cabinet, "y" up / down.

    Example: ``VectorEnv(drawer_open_scene(16), drawer_open_task(), worlds=16)``."""
    slide = f"cabinet/{DRAWER}_slide"
    reach = C.custom("reach", lambda s: s.xp.norm(grasp_point(s) - grasp_target(s)))
    handle = ("handle", handle_point)
    cams = {"front": front_camera((0.5, -0.7, 0.7), (0.0, 0.0, 0.35))} if camera else {}
    return Task(
        name="DrawerOpen-v0", obs=[GRIP, slide, handle, "action"], actions=gantry_actions(GRIP, yaw=False),
        reward={"reach": (-1.0, reach), "open": (10.0, C.joint(slide))},
        success=C.joint(slide) >= opening, end_on_success=False, max_steps=max_steps, cameras=cams,
        randomize=[jitter_joint(f"{GRIP}/x", by=0.04), jitter_joint(f"{GRIP}/y", by=0.04)])


__all__ = ["BIN_POS", "CABINET", "GRIP", "PICK_OBJECTS", "drawer_open_scene", "drawer_open_task", "front_camera",
           "grasp_point", "grasp_target", "handle_point", "maybe_vary", "peg_insert_scene", "peg_insert_task",
           "peg_tip", "pick_place_scene", "pick_place_task"]
