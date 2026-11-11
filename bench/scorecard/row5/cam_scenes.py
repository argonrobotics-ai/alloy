# SPDX-License-Identifier: Apache-2.0
"""Scene constants shared by the Alloy and MuJoCo Warp row 5 harnesses."""
from __future__ import annotations

import numpy as np

CONTROL_DT = 0.01  # s per loop (one control step) in both engines

# cartpole: a position-servoed cart on a rail and a free-swinging pole starting 0.15 rad off vertical
CARTPOLE_MJCF = """<mujoco model="cartpole"><compiler angle="radian"/><option timestep="0.005"/>
<worldbody>{floor}
 <geom name="rail" type="capsule" fromto="-2 0 0.4 2 0 0.4" size="0.01" rgba="0.5 0.5 0.5 1"/>
 <body name="cart" pos="0 0 0.5">
  <joint name="slider" type="slide" axis="1 0 0" range="-1.8 1.8" limited="true" damping="1"/>
  <geom name="cartbox" type="box" size="0.12 0.08 0.05" mass="1" rgba="0.2 0.4 0.8 1"/>
  <body name="pole" pos="0 0 0" quat="0.99719 0 0.07473 0">
   <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01" range="-1.2 1.2" limited="true"/>
   <geom name="polebar" type="capsule" fromto="0 0 0 0 0 0.6" size="0.02" mass="0.1" rgba="0.9 0.3 0.2 1"/>
  </body></body></worldbody>
<actuator><position name="cart_servo" joint="slider" kp="200" kv="20"/></actuator></mujoco>"""
CARTPOLE_FLOOR = '<geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>'
CARTPOLE_JOINTS = ("slider",)
CARTPOLE_HOLD = np.array([0.0])
CARTPOLE_NOISE = 0.05  # m, uniform per world and step
CARTPOLE_CAM = ((0.0, -2.2, 0.8), (0.0, 0.0, 0.6))

# franka_cube: the shipped emb_e2_panda scene (own table, a free box at (0.5, -0.2), cylinder, mug), init keyframe
PANDA_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1")
PANDA_HOLD = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04])
# 0.25 rad makes the arm pose differ visibly from world to world.
PANDA_NOISE = 0.25  # rad (m for the finger), uniform per world and step
# Whole arm (about 0.9 m tall, base at the origin) and the 5 cm cube at (0.5, -0.2) in frame, ~1.2 m from the arm's middle.
PANDA_CAM = ((1.25, -0.95, 0.75), (0.35, -0.1, 0.42))
PANDA_NOISE_FINGER = 0.01  # m, uniform per world and step (finger_joint1 range 0-0.04 m)
CAM_FOV = 45.0  # degrees, vertical

SCENES = {"cartpole": (CARTPOLE_JOINTS, CARTPOLE_HOLD, CARTPOLE_NOISE, CARTPOLE_CAM),
          "franka_cube": (PANDA_JOINTS, PANDA_HOLD, PANDA_NOISE, PANDA_CAM)}


def noise_pool(scene: str, worlds: int, n: int = 16, seed: int = 0) -> np.ndarray:
    """(n, worlds, njoint) float32 hold-position targets with uniform noise, cycled by the loop."""
    _, hold, amp, _ = SCENES[scene]
    rng = np.random.default_rng(seed)
    amp = np.full(len(hold), amp)
    if scene == "franka_cube":
        amp[-1] = PANDA_NOISE_FINGER  # the finger is a slide joint (0-0.04 m): a joint-sized noise would pin it at a limit
    return (hold + amp * rng.uniform(-1, 1, (n, worlds, len(hold)))).astype(np.float32)


# The cell grid of PROTOCOL.md (the same in every engine).
SIZES = (64, 128, 256)
WORLDS = (256, 1024, 4096)
OUTPUTS = ("rgb", "rgb+depth", "rgb+depth+seg")
SCENE_NAMES = tuple(SCENES)
