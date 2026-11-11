# SPDX-License-Identifier: Apache-2.0
"""GPU golden cases for the routes the rc1-1110 FEAT slices touch (coordinator 2026-11-10: one L4 golden row per
route): robots with per-world randomization, a VectorEnv with episode ends and masked resets, and the fast camera.

Each case runs through the public API (``alloy.build`` / ``alloy.envs.VectorEnv`` / ``alloy.Camera``) and gives one
uint64 per world per period, like :func:`alloy.precision.repeat.run_digests`. The scenes are kept to motion a GPU
repeats bit for bit (robots touching nothing, one sphere on the ground; measured on an L4), so a
mismatch means a change moved bits, not contact chaos.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  <body name="link2" pos="0.3 0 0"><joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.015" mass="0.5"/></body>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/><position joint="elbow" kp="100" kv="5"/></actuator>
</mujoco>"""


HEALTH_KEYS = ("tunnel.",)    # health bookkeeping in state_arrays: a health-only change must not move a golden


def _is_health(key: str) -> bool:
    return any(key.split("/")[-1].startswith(h) for h in HEALTH_KEYS)


def _engine_digests(engine: Any) -> np.ndarray:
    """World digests of the physics state and the solvers' own state, without health bookkeeping (``HEALTH_KEYS``)."""
    from dataclasses import replace

    from alloy.precision.repeat import sim_parts

    parts = sim_parts(engine)
    parts = replace(parts, extra={k: v for k, v in parts.extra.items() if not _is_health(k)},
                    world_maps={k: v for k, v in parts.world_maps.items() if not _is_health(k)})
    return np.asarray(parts.digests(engine.state, True), dtype=np.uint64)


def robot_randomize_series(worlds: int, periods: int) -> list[np.ndarray]:
    """An arm in free air (no contact) with per-world mass and servo-gain draws, ramping to per-world targets."""
    import alloy

    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 1.0))
    scene.randomize("arm", "mass_scale", 0.8, 1.2)
    scene.randomize("arm", "kp_scale", 0.7, 1.3)
    sim = alloy.build(scene, seed=0)
    sim.set_targets({"shoulder": np.linspace(-0.5, 0.5, worlds), "elbow": np.linspace(0.8, -0.8, worlds)})
    out = [_engine_digests(sim.engine)]
    for _ in range(periods):
        sim.step(1)
        out.append(_engine_digests(sim.engine))
    return out


def vectorenv_reset_series(worlds: int, periods: int) -> list[np.ndarray]:
    """The guide's gantry task with episodes of 3 steps (autoreset) and a masked reset of every other world. No cube:
    a 5 cm cube resting on the ground beside the gantry made every world differ between two L4 runs from the first
    step (box-plane contact on MuJoCo Warp), while the gantry alone repeats with both resets (measured on an L4,
    2026-11-10)."""
    import alloy
    from alloy.envs import Task, VectorEnv
    from alloy.envs.robots import gantry_actions, gantry_gripper

    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    scene.add_robot("gripper", mjcf=gantry_gripper(), pos=(0.0, 0.0, 0.4))
    task = Task(obs=["gripper"], actions=gantry_actions("gripper", yaw=False), max_steps=3)
    env = VectorEnv(scene, task, worlds=worlds, backend="numpy", seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    shape = (worlds, len(env.action_names()))
    out = [_engine_digests(env.sim.engine)]
    for k in range(periods):
        env.step(rng.uniform(-0.3, 0.3, shape).astype(np.float32))
        if k == periods // 2:
            env.reset_worlds(np.arange(worlds) % 2 == 0)
        out.append(_engine_digests(env.sim.engine))
    env.close()
    return out


def _image_digests(images: dict[str, Any], worlds: int) -> np.ndarray:
    out = np.zeros(worlds, dtype=np.uint64)
    for w in range(worlds):
        h = hashlib.sha256()
        for name in sorted(images):
            h.update(name.encode())
            h.update(np.ascontiguousarray(np.asarray(images[name])[w]).tobytes())
        out[w] = np.frombuffer(h.digest()[:8], dtype="<u8")[0]
    return out


def camera_series(worlds: int, periods: int) -> list[np.ndarray]:
    """rgb, depth and segmentation of every world from the fast camera while a sphere lands beside a box."""
    import alloy

    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05), static=True)
    xs = np.stack([np.linspace(0.12, 0.2, worlds), np.zeros(worlds), np.full(worlds, 0.15)], axis=1)
    scene.add_sphere("ball", 0.03, pos=xs, material="steel_carbon")
    sim = alloy.build(scene, seed=0)
    cam = alloy.Camera("front", look_at=((0.6, -0.6, 0.4), (0.1, 0, 0.05)), width=64, height=48)
    outputs = ("rgb", "depth", "segmentation")
    out = [_image_digests(sim.render(cam, output=outputs), worlds)]
    for _ in range(periods):
        sim.step(1)
        out.append(_image_digests(sim.render(cam, output=outputs), worlds))
    return out
