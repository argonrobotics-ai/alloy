# SPDX-License-Identifier: Apache-2.0
"""The protocol's matched PPO settings (PROTOCOL.md) as an rsl_rl 5.5.x runner config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STEPS_PER_ENV = 24


@dataclass(frozen=True)
class TaskPPO:
    """The two per-task PPO values of the protocol."""

    gamma: float
    entropy_coef: float


PER_TASK = {"go2_flat": TaskPPO(0.99, 0.01), "franka_lift": TaskPPO(0.995, 0.005),
            "cube_reorient": TaskPPO(0.998, 0.002)}


def matched_config(task: str, seed: int, max_iterations: int) -> dict[str, Any]:
    """rsl_rl config: MLP [512, 256, 128] ELU, actor obs normalisation, 24 steps, 5 epochs, 4 minibatches, lr 1e-3
    adaptive (KL 0.01), lam 0.95, clip 0.2, init std 1.0. Example: ``matched_config("franka_lift", 0, 3000)``."""
    ppo = PER_TASK[task]
    mlp = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True}
    return {
        "class_name": "OnPolicyRunner", "num_steps_per_env": STEPS_PER_ENV, "save_interval": 10**9,
        "experiment_name": f"alloy_{task}", "run_name": "", "seed": seed, "max_iterations": max_iterations,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]}, "logger": "tensorboard",
        "actor": {**mlp, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0,
                                              "std_type": "scalar"}},
        "critic": {**mlp, "obs_normalization": False},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-3,
                      "schedule": "adaptive", "gamma": ppo.gamma, "lam": 0.95, "entropy_coef": ppo.entropy_coef,
                      "desired_kl": 0.01, "max_grad_norm": 1.0, "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True, "clip_param": 0.2},
    }


def native_config(task: str, seed: int, max_iterations: int) -> dict[str, Any]:
    """go2_flat only: the PPO settings of FEAT-LOCO's documented command (``python -m alloy.envs.loco.train``),
    i.e. ``alloy.envs.loco.train.ppo_config``; differs from the matched config by a normalised critic input."""
    if task != "go2_flat":
        raise ValueError("native variant exists only for go2_flat (Alloy has no other native trainer)")
    from alloy.envs.loco.train import ppo_config
    cfg = ppo_config(max_iterations, seed)
    cfg["save_interval"] = 10**9
    return cfg
