# SPDX-License-Identifier: Apache-2.0
"""The test run's task: GripperLift (alloy.envs.tasks) with a cable on the ground in view and a 128 px policy camera.

The cable is the soft object: a short rod that Alloy steps as a MuJoCo chain beside the robot (fast). A towel was
tried first: cloth beside the gripper stepped at about 4 s per control step for 64 worlds on an L4, too slow to
train (docs/notes/live-test.md). Success is "the cube is held and up ``level[0]`` metres"; ``level`` is the
curriculum setting, read every step.
"""
from __future__ import annotations

from typing import Any

from alloy.camera import Camera
from alloy.envs import checks as C
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task
from alloy.envs.vector import VectorEnv

CABLE = dict(length=0.3, radius=0.005, pos=(-0.15, -0.09, 0.006), direction=(1.0, 0.0, 0.0))


def build_env(worlds: int, camera: int, level: list[float], device: str | None = None, seed: int = 0) -> Any:
    scene = gripper_lift_scene(worlds, seed=seed)
    scene.add_rod("cable", **CABLE)
    task = gripper_lift_task(camera=bool(camera))
    held = C.between("cube", "gripper/finger_left", "gripper/finger_right", max_offset=0.03, max_gap=0.1)
    rise = C.rise("cube")
    task.success = C.custom("lifted_level", lambda s: rise(s) >= level[0]) & held
    task.reward = {**task.reward, "success": (task.reward["success"][0], task.success)}
    if camera:
        task.cameras = {"front": Camera("front", look_at=((0.26, -0.28, 0.2), (0.0, -0.03, 0.04)), width=camera,
                                        height=camera, fov=45, spp=4)}
    return VectorEnv(scene, task, worlds=worlds, device=device, seed=seed)
