# SPDX-License-Identifier: Apache-2.0
"""Protocol task keys -> mjlab task ids, env cfg loaders, probes and matched PPO settings."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from metrics import LiftProbe, VelocityProbe


@dataclass(frozen=True)
class PpoSettings:
    gamma: float
    entropy_coef: float


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    protocol_ppo: PpoSettings | None
    probe: Callable[[], Any] | None
    substitution: str = ""
    unavailable: str = ""
    env_cfg_fn: Callable[[], Any] | None = None  # None = the registered task's own env cfg
    agent_task_id: str = ""  # registered task whose agent cfg / runner class to use (default: task_id)

    def env_cfg(self):
        from mjlab.tasks.registry import load_env_cfg

        import mjlab.tasks  # noqa: F401  (registers tasks)
        return self.env_cfg_fn() if self.env_cfg_fn else load_env_cfg(self.task_id)

    def agent_cfg(self):
        from mjlab.tasks.registry import load_rl_cfg

        import mjlab.tasks  # noqa: F401
        return load_rl_cfg(self.agent_task_id or self.task_id)

    def header_settings(self, env_cfg, agent_cfg) -> dict[str, Any]:
        a = agent_cfg
        return {
            "substitution": self.substitution,
            "ppo": {"steps_per_env": a.num_steps_per_env, "epochs": a.algorithm.num_learning_epochs,
                    "minibatches": a.algorithm.num_mini_batches, "lr": a.algorithm.learning_rate,
                    "schedule": a.algorithm.schedule, "desired_kl": a.algorithm.desired_kl, "gamma": a.algorithm.gamma,
                    "lam": a.algorithm.lam, "clip": a.algorithm.clip_param, "entropy": a.algorithm.entropy_coef,
                    "init_std": a.actor.distribution_cfg["init_std"], "hidden": list(a.actor.hidden_dims),
                    "actor_obs_norm": a.actor.obs_normalization, "critic_obs_norm": a.critic.obs_normalization},
            "env": {"dt_s": env_cfg.sim.mujoco.timestep * env_cfg.decimation, "decimation": env_cfg.decimation,
                    "episode_length_s": env_cfg.episode_length_s},
        }


def matched_agent_cfg(cfg, ppo: PpoSettings):
    """The protocol's matched PPO settings applied on top of the task's agent cfg (network: MLP [512,256,128] ELU)."""
    cfg = copy.deepcopy(cfg)
    a = cfg.algorithm
    a.num_learning_epochs, a.num_mini_batches = 5, 4
    a.learning_rate, a.schedule, a.desired_kl = 1e-3, "adaptive", 0.01
    a.lam, a.clip_param = 0.95, 0.2
    a.gamma, a.entropy_coef = ppo.gamma, ppo.entropy_coef
    cfg.num_steps_per_env = 24
    cfg.actor.hidden_dims = cfg.critic.hidden_dims = (512, 256, 128)
    cfg.actor.activation = cfg.critic.activation = "elu"
    cfg.actor.obs_normalization = True
    cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "init_std": 1.0}
    return cfg


def _go2_env_cfg():
    from go2 import go2_flat_env_cfg

    return go2_flat_env_cfg()


GO2_PORT = TaskSpec(
    task_id="local:Velocity-Flat-Unitree-Go2 (our port of Mjlab-Velocity-Flat-Unitree-Go1)",
    agent_task_id="Mjlab-Velocity-Flat-Unitree-Go1", protocol_ppo=PpoSettings(0.99, 0.01), probe=VelocityProbe,
    env_cfg_fn=_go2_env_cfg,
    substitution="OUR Go2 port of mjlab's Go1 flat task (Menagerie unitree_go2 MJCF, see go2.py); not shipped by mjlab",
)

TASKS: dict[str, TaskSpec] = {
    "go2_flat": TaskSpec(
        task_id="Mjlab-Velocity-Flat-Unitree-Go1", protocol_ppo=PpoSettings(0.99, 0.01), probe=VelocityProbe,
        substitution="Unitree Go1 (mjlab ships no Go2 task)",
    ),
    "franka_lift": TaskSpec(
        task_id="Mjlab-Lift-Cube-Yam", protocol_ppo=PpoSettings(0.995, 0.005), probe=LiftProbe,
        substitution="I2RT Yam arm (mjlab ships no Franka task)",
    ),
    "cube_reorient": TaskSpec(
        task_id="", protocol_ppo=None, probe=None, unavailable="mjlab 1.6.0 has no in-hand reorientation task",
    ),
}
