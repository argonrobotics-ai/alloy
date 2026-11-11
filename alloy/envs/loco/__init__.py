# SPDX-License-Identifier: Apache-2.0
"""Legged locomotion on ``alloy.envs``: a reference Go2 velocity-tracking environment and its parts.

- ``go2_velocity(num_envs=...)``: the Unitree Go2 on flat ground following velocity commands, with pushes.
- ``LocoEnv``: the ``VectorEnv`` subclass behind it (commands, pushes, foot contacts, accelerometer); observations
  come in two groups, ``obs["policy"]`` and ``obs["critic"]`` (the policy's numbers plus privileged ones).
- ``python -m alloy.envs.loco.train``: PPO (rsl_rl) on the Go2 environment.
- ``terms``: observation and reward terms; ``VelocityCommands`` and ``Pushes``: their settings.

Example::

    from alloy.envs.loco import go2_velocity
    env = go2_velocity(num_envs=1024)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

``go2_rough(num_envs=...)`` is the same task on a field of rough terrain tiles with a difficulty curriculum
(``alloy.envs.loco.rough``, ``alloy.envs.loco.terrain``). Guide: ``docs/guide/locomotion.md``. No stairs or vertical
steps, no height scan of the ground, no actuator model yet.
"""

from __future__ import annotations

from alloy.envs.loco import terms
from alloy.envs.loco.commands import CommandSampler, VelocityCommands
from alloy.envs.loco.env import LocoConfig, LocoEnv, stand_action
from alloy.envs.loco.go2 import DEFAULT_REWARD, Go2VelocityConfig, go2_scene, go2_velocity, go2_velocity_task
from alloy.envs.loco.pushes import Pusher, Pushes
from alloy.envs.loco.robot import GO2, Foot, LeggedRobot
from alloy.envs.loco.rough import Go2RoughConfig, RoughEnv, go2_rough, go2_rough_scene
from alloy.envs.loco.state import LocoState
from alloy.envs.loco.terrain import Terrain, TerrainSpec, terrain_mjcf

__all__ = ["DEFAULT_REWARD", "GO2", "CommandSampler", "Foot", "Go2RoughConfig", "Go2VelocityConfig", "RoughEnv",
           "Terrain", "TerrainSpec", "go2_rough", "go2_rough_scene", "terrain_mjcf", "LeggedRobot", "LocoConfig",
           "LocoEnv", "LocoState", "Pusher", "Pushes", "VelocityCommands", "go2_scene", "go2_velocity",
           "go2_velocity_task", "stand_action", "terms"]
