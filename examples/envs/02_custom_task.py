# SPDX-License-Identifier: Apache-2.0
"""Your own environment: a scene, a robot from an MJCF string, and a task built from checkers.

A pusher on two stacked slide joints (x, then y: one joint per MJCF body) must push a cube onto a goal 15 cm away.
Reward: minus the cube-goal distance; success: the cube within 3 cm of the goal. A scripted policy (go behind the
cube, then push) shows the pieces work.
Run: python examples/envs/02_custom_task.py [num_envs]
"""
import json
import sys

import numpy as np

import alloy.api as alloy
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs import checks as C

PUSHER = """<mujoco><worldbody><body name="carriage" pos="0 0 0.03">
  <joint name="x" type="slide" axis="1 0 0" range="-0.3 0.3" damping="20"/>
  <geom type="sphere" size="0.005" mass="0.1" contype="0" conaffinity="0"/>
  <body name="tip"><joint name="y" type="slide" axis="0 1 0" range="-0.3 0.3" damping="20"/>
    <geom type="cylinder" size="0.015 0.02" mass="0.5" friction="0.5"/></body></body></worldbody>
  <actuator><position joint="x" kp="400" kv="40"/><position joint="y" kp="400" kv="40"/></actuator></mujoco>"""
GOAL = (0.15, 0.0, 0.025)

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.025), mass=0.2, friction=0.4)
scene.add_robot("pusher", mjcf=PUSHER, pos=(-0.08, 0.0, 0.0))     # starts 8 cm behind the cube

task = Task(
    name="PushToGoal",
    obs=["pusher", "cube", C.distance("cube", GOAL)],               # joints, cube pose/velocity, distance to goal
    actions=ActionSpec(["pusher/x", "pusher/y"], mode="delta", delta=0.005, low=[-0.3, -0.3], high=[0.3, 0.3]),
    reward={"distance": (-1.0, C.distance("cube", GOAL)), "effort": (-0.001, C.action_size())},
    success=C.near("cube", GOAL, 0.03),
    failure=~C.inside("cube", (-0.5, -0.5, -0.1), (0.5, 0.5, 0.5)),  # pushed off the table area
    max_steps=200)
env = VectorEnv(scene, task, worlds=num_envs, backend="numpy")      # NumPy here: the policy below is NumPy
obs, info = env.reset(seed=0)
print("obs columns:", env.obs_names())

finished = np.zeros(num_envs, bool)
for step in range(200):
    s = env.state
    tip = np.stack([s["pusher/x.q"], s["pusher/y.q"]], axis=1) + [-0.08, 0.0]   # tip position in the world
    behind = s["cube.pos"][:, :2] - [0.045, 0.0]                                 # where to push from
    push = np.where(np.abs(tip[:, 1:2] - behind[:, 1:2]) < 0.01, [1.0, 0.0], 0.0)
    action = np.clip((behind - tip) / 0.005, -1, 1) * (push == 0) + push        # line up, then push in +x
    obs, reward, terminated, truncated, info = env.step(action)
    if "_episode" in info:
        finished |= info["_episode"] & info["episode"]["success"]
    if finished.all():
        break
print(json.dumps({"steps": step + 1, "succeeded": int(finished.sum()), "of": num_envs,
                  "last_reward": [round(float(r), 3) for r in reward]}))
env.close()
