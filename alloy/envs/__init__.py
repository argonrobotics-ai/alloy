# SPDX-License-Identifier: Apache-2.0
"""Gym / PyTorch environments over Alloy's parallel worlds: one ``VectorEnv`` steps every world at once.

- Gymnasium's vector API (``reset``, ``step``, spaces, autoreset); a ``gymnasium.vector.VectorEnv`` when Gymnasium is
  installed.
- torch tensors on the simulation device (zero-copy views of the engine's arrays) when PyTorch is installed, else
  NumPy.
- State observations by name, camera images (``alloy.camera.Camera``), rewards and episode ends from task checkers
  (``alloy.envs.checks``), reset by world mask, and random starts per episode (``alloy.envs.randomize``).

Example::

    import alloy.api as alloy
    from alloy.envs import VectorEnv, Task, ActionSpec, checks as C

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=(0, 0, 0.03), mass=0.15)
    scene.add_robot("gripper", mjcf="gripper.xml", pos=(0, 0, 0.075))
    task = Task(obs=["gripper", "cube"], actions=ActionSpec(mode="delta"),
                reward={"lift": (10.0, C.rise("cube"))}, success=C.lifted("cube", by=0.08), max_steps=200)
    env = VectorEnv(scene, task, worlds=1024)                # GPU if there is one
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Ready-made: ``alloy.envs.make("GripperLift-v0", num_envs=256)``. Runnable examples: ``examples/envs/``.
RL libraries (rsl_rl, Stable-Baselines3, skrl): ``alloy.envs.wrappers``; one-command training:
``python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl``.
"""

from __future__ import annotations

from alloy.envs import checks, randomize
from alloy.envs.actions import ActionSpec
from alloy.envs.errors import EnvError
from alloy.envs.randomize import jitter_body, jitter_joint
from alloy.envs.state import BatchState
from alloy.envs.task import Task
from alloy.envs.tasks import REGISTRY, make, register_gymnasium
from alloy.envs.vector import VectorEnv

__all__ = ["REGISTRY", "ActionSpec", "BatchState", "EnvError", "Task", "VectorEnv", "checks", "jitter_body",
           "jitter_joint", "make", "randomize", "register_gymnasium"]
