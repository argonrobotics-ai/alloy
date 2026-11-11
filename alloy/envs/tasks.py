# SPDX-License-Identifier: Apache-2.0
"""Ready-made environments, by name, and their Gymnasium registration.

``GripperLift-v0``: a two-finger gripper above a 6 cm cube (each world's cube a little off-centre); squeeze, lift it
8 cm and hold it there until the episode ends (150 steps, 1.5 s); ``info["episode"]["success"]`` says whether the cube
was up and held at the end. Every reset moves the cube up to 1 cm and turns it up to 15 degrees. Actions move the
lift and finger targets (delta mode). ``GripperLiftCamera-v0`` adds a 64x64 camera image (``obs["front"]``).

``PandaReach-v0``: a Franka Panda arm (MuJoCo Menagerie, Apache-2.0) moves its hand to a target point drawn at every
reset; actions are the 7 arm servos' targets. Example::

    from alloy.envs import make
    env = make("GripperLift-v0", num_envs=256)               # GPU if there is one
    obs, info = env.reset(seed=0)

    import gymnasium as gym
    from alloy.envs import register_gymnasium
    register_gymnasium()
    env = gym.make_vec("alloy/GripperLift-v0", num_envs=256, vectorization_mode="vector_entry_point")
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from alloy.envs import checks as C
from alloy.envs import library_tasks as LT
from alloy.envs import pick_cube as PC
from alloy.envs import soft_tasks as ST
from alloy.envs.actions import ActionSpec
from alloy.envs.errors import EnvError
from alloy.envs.randomize import jitter_body, jitter_joint
from alloy.envs.task import Task

GRIPPER_MJCF = """<mujoco>
  <worldbody>
    <body name="palm">
      <joint name="lift" type="slide" axis="0 0 1" range="-0.2 0.5" damping="40"/>
      <geom type="box" size="0.04 0.06 0.01" mass="0.3" friction="1.0"/>
      <body name="finger_left" pos="0 0.055 -0.04">
        <joint name="close_left" type="slide" axis="0 -1 0" range="0 0.05" damping="5"/>
        <geom type="box" size="0.012 0.005 0.025" mass="0.03" friction="1.0"/>
      </body>
      <body name="finger_right" pos="0 -0.055 -0.04">
        <joint name="close_right" type="slide" axis="0 1 0" range="0 0.05" damping="5"/>
        <geom type="box" size="0.012 0.005 0.025" mass="0.03" friction="1.0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="lift" kp="3000" kv="150"/>
    <position joint="close_left" kp="400" kv="10"/>
    <position joint="close_right" kp="400" kv="10"/>
  </actuator>
</mujoco>"""

FINGERS = ("gripper/finger_left", "gripper/finger_right")


def gripper_lift_scene(worlds: int = 1, seed: int = 0, offset_m: float = 0.01) -> Any:
    """The GripperLift scene: ground, a 6 cm cube (y offset up to ``offset_m`` per world) and the gripper.

    Example: ``alloy.build(gripper_lift_scene(64), worlds=64)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    dy = rng.uniform(-offset_m, offset_m, size=worlds) if worlds > 1 else np.zeros(1)
    scene = Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    pos = [[0.0, float(y), 0.03] for y in dy] if worlds > 1 else (0.0, 0.0, 0.03)
    scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=pos, mass=0.15, friction=1.0)
    scene.add_robot("gripper", mjcf=GRIPPER_MJCF, pos=(0, 0, 0.075))
    return scene


def gripper_lift_task(camera: bool = False, max_steps: int = 150) -> Task:
    """The GripperLift task: reward for holding the cube between the fingers and raising it; success = up 8 cm and
    held. Success does not end the episode (a task that ends on success needs a bonus larger than the per-step
    rewards it gives up, or the policy learns to hover just below the bar).

    Example: ``VectorEnv(gripper_lift_scene(16), gripper_lift_task(), worlds=16)``."""
    held = C.between("cube", *FINGERS, max_offset=0.03, max_gap=0.1)
    success = C.lifted("cube", by=0.08) & held
    cameras = {}
    if camera:
        from alloy.camera import Camera  # noqa: PLC0415

        cameras["front"] = Camera("front", look_at=((0.26, -0.26, 0.2), (0.0, 0.0, 0.09)), width=64, height=64,
                                  fov=45, spp=4)
    return Task(
        name="GripperLift-v0" if not camera else "GripperLiftCamera-v0",
        obs=["gripper", "cube", "action"],
        actions=ActionSpec(["gripper/lift", "gripper/close_left", "gripper/close_right"], mode="delta",
                           low=[-0.03, 0.0, 0.0], high=[0.2, 0.05, 0.05], delta=[0.004, 0.002, 0.002]),
        reward={"held": (0.2, held), "rise": (10.0, C.custom("rise_held", _rise_held(held))),
                "success": (5.0, success), "effort": (-0.01, C.action_size())},
        success=success, failure=C.height("cube") < -0.1, max_steps=max_steps, cameras=cameras,
        end_on_success=False, randomize=[jitter_body("cube", pos=(0.01, 0.01, 0.0), yaw_deg=15.0)])


def _rise_held(held: C.Check) -> Callable[[Any], Any]:
    rise = C.rise("cube")
    return lambda s: s.xp.float(held(s)) * s.xp.clip(rise(s), 0.0, 0.1)


PANDA_HOME = {"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": -1.5708, "joint5": 0.0, "joint6": 1.5708,
              "joint7": -0.7854}
REACH_BOX = ((0.35, -0.25, 0.35), (0.65, 0.25, 0.75))   # where PandaReach targets are drawn (m, robot base at 0)


def panda_scene(worlds: int = 1, seed: int = 0, props: bool = True) -> Any:
    """The Franka Panda + Franka Hand from the library (MuJoCo Menagerie, Apache-2.0) on its table, with a box, a
    cylinder and a mug (``props=False`` leaves those out); robot parts are "panda/<body or joint>".

    Example: ``alloy.build(panda_scene(), worlds=64)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415

    path = default_root() / "robots" / "emb_e2_panda" / "scene.xml"
    if not path.is_file():
        raise EnvError(f"the Panda robot file is missing ({path.name})", "install Alloy with its library "
                       "(a source checkout or the full wheel)", field="scene")
    if not props:
        from alloy.envs.mjcf import without_bodies  # noqa: PLC0415

        path = without_bodies(path, ["box", "cylinder", "mug"])
    scene = Scene(worlds=worlds, dt=0.02)
    scene.add_robot("panda", path)
    return scene


def panda_reach_task(max_steps: int = 100) -> Task:
    """PandaReach: move the hand to a target point drawn in ``REACH_BOX`` at every reset (arm joints start at the
    home pose +-0.1 rad). Actions: the 7 arm servos' targets (delta mode, up to 0.05 rad per 20 ms step).
    Success: the hand within 3 cm of the target. Example: ``VectorEnv(panda_scene(256), panda_reach_task())``."""
    hand = "panda/hand"
    to_target = ("to_target", lambda s: s["target.pos"] - s[f"{hand}.pos"])
    close = C.custom("close", lambda s: 1.0 - s.xp.tanh(C.distance(hand, "target")(s) / 0.1))
    return Task(
        name="PandaReach-v0",
        obs=["panda", f"{hand}.pos", "target", to_target, "action"],
        actions=ActionSpec([f"panda/actuator{k}" for k in range(1, 8)], mode="delta", delta=0.05),
        reward={"distance": (-1.0, C.distance(hand, "target")), "close": (1.0, close),
                "effort": (-0.005, C.action_size())},
        success=C.near(hand, "target", 0.03), end_on_success=False, max_steps=max_steps,
        randomize=[jitter_joint(f"panda/{j}", by=0.1, at=v) for j, v in PANDA_HOME.items()],
        goals={"target": REACH_BOX})


#: name -> (scene(worlds, seed), task())
REGISTRY: dict[str, tuple[Callable[..., Any], Callable[[], Task]]] = {
    "GripperLift-v0": (gripper_lift_scene, gripper_lift_task),
    "GripperLiftCamera-v0": (gripper_lift_scene, partial(gripper_lift_task, camera=True)),
    "PandaReach-v0": (partial(panda_scene, props=False), panda_reach_task),
    "PickPlace-v0": (LT.pick_place_scene, LT.pick_place_task),
    "PickCube-v0": (PC.pick_cube_scene, PC.pick_cube_task),
    "PegInsert-v0": (LT.peg_insert_scene, LT.peg_insert_task),
    "DrawerOpen-v0": (LT.drawer_open_scene, LT.drawer_open_task),
    "Pour-v0": (ST.pour_scene, ST.pour_task),
    "ClothFold-v0": (ST.cloth_fold_scene, ST.cloth_fold_task),
}


def make(name: str, num_envs: int = 1, device: str | None = None, seed: int = 0, **kwargs: Any) -> Any:
    """A ready-made ``VectorEnv`` by name (see ``REGISTRY``). Keyword options go to the scene (e.g. ``obj=``,
    ``vary=``), the task (``camera=``, ``max_steps=``) or the env (``backend=``, ``autoreset=``), by name.

    Example: ``make("PickPlace-v0", num_envs=64, obj="ycb_014_lemon", camera=True)``."""
    import inspect  # noqa: PLC0415

    from alloy.envs.vector import VectorEnv  # noqa: PLC0415

    from alloy.envs.ports import PORTS, make_port  # noqa: PLC0415 - the task ports (Gym-Anything)

    if name.startswith("alloy/"):
        name = name.split("/", 1)[1]
    if name in PORTS:
        return make_port(name, num_envs=num_envs, device=device, seed=seed, **kwargs)
    if name not in REGISTRY:
        raise EnvError(f"no environment named {name!r}", f"use one of {sorted([*REGISTRY, *PORTS])}", field="name")
    scene_fn, task_fn = REGISTRY[name]
    split: dict[str, dict[str, Any]] = {"scene": {}, "task": {}, "env": {}}
    for key, value in kwargs.items():
        for part, fn in (("scene", scene_fn), ("task", task_fn)):
            if key in inspect.signature(fn).parameters and key not in ("worlds", "seed"):
                split[part][key] = value
                break
        else:
            split["env"][key] = value
    scene = scene_fn(num_envs, seed, **split["scene"])
    return VectorEnv(scene, task_fn(**split["task"]), worlds=num_envs, device=device, seed=seed, **split["env"])


def register_gymnasium() -> list[str]:
    """Register every ready-made environment with Gymnasium as "alloy/<name>" (for ``gym.make_vec``).

    Example: ``register_gymnasium()`` -> ["alloy/GripperLift-v0", ...]."""
    try:
        import gymnasium  # noqa: PLC0415
    except ImportError as exc:
        raise EnvError("Gymnasium is not installed", "pip install gymnasium") from exc
    from alloy.envs.ports import PORTS  # noqa: PLC0415

    ids = []
    for name in [*REGISTRY, *PORTS]:
        env_id = f"alloy/{name}"
        if env_id not in gymnasium.registry:
            gymnasium.register(id=env_id, vector_entry_point=f"{__name__}:_vector_entry_{_slug(name)}")
        ids.append(env_id)
    return ids


def _slug(name: str) -> str:
    return name.replace("-", "_")


def _vector_entry(name: str) -> Callable[..., Any]:
    def entry(num_envs: int = 1, **kwargs: Any) -> Any:
        return make(name, num_envs=num_envs, **kwargs)
    entry.__doc__ = f"Gymnasium vector entry point of {name}."
    return entry


from alloy.envs.ports import PORTS as _PORTS  # noqa: E402 - the port envs' Gymnasium entry points too

for _name in [*REGISTRY, *_PORTS]:
    globals()[f"_vector_entry_{_slug(_name)}"] = _vector_entry(_name)

__all__ = ["GRIPPER_MJCF", "PANDA_HOME", "REACH_BOX", "REGISTRY", "gripper_lift_scene", "gripper_lift_task",
           "make", "panda_reach_task", "panda_scene", "register_gymnasium"]
