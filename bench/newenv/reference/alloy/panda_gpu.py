# SPDX-License-Identifier: Apache-2.0
"""``PandaEnv`` for scenes with liquid or cloth: built on the GPU (``device=None`` picks it), optional settle at build and
a replaceable list of start jitters. Everything else is the pilot's ``panda_env.PandaEnv``, unchanged."""

from __future__ import annotations

from typing import Any

import numpy as np

import alloy
from alloy.envs import VectorEnv
from alloy.envs.randomize import Jitter

from panda_env import PandaConfig, PandaEnv


class PandaGpuEnv(PandaEnv):
    def __init__(self, num_envs: int, seed: int, cfg: PandaConfig, *, settle: float = 0.0,
                 jitters: list[Jitter] | None = None, device: str | None = None,
                 scene_kwargs: dict[str, Any] | None = None) -> None:
        self.cfg, self.num_envs, self.dt = cfg, int(num_envs), cfg.dt
        self.episode_steps = round(cfg.episode_s / cfg.dt)
        scene = alloy.Scene(worlds=self.num_envs, dt=cfg.dt, **(scene_kwargs or {}))
        scene.add_robot("panda", cfg.robot_file or self.robot_file())
        self.add_objects(scene)
        task = self._task()
        task.settle = settle
        if jitters is not None:
            task.randomize = list(jitters)
        self.venv = VectorEnv(scene, task, worlds=self.num_envs, seed=seed, device=device, backend="numpy")
        self.act_dim = 8 if cfg.gripper_action else 7
        self.hold = np.zeros(self.num_envs, np.int64)
        self.stage = np.zeros(self.num_envs, np.int64)
        self.clock = np.zeros(self.num_envs, np.int64)
        self.obs_dim = int(self.reset().shape[1])

    def robot_file(self) -> Any:
        from panda_env import panda_file  # noqa: PLC0415
        return panda_file()
