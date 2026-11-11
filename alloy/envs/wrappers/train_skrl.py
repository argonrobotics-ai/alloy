# SPDX-License-Identifier: Apache-2.0
"""Train skrl PPO on an Alloy env in one call; prints one JSON line per chunk of iterations.

Needs ``pip install skrl`` (skrl's own TensorBoard and Weights & Biases writing is switched off, so neither package
is needed). One iteration is 32 steps of every world. Example::

    from alloy.envs.wrappers.train_skrl import train

    summary = train("GripperLift-v0", num_envs=512, iterations=200)
    print(summary["success_rate"], summary["checkpoint"])
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from typing import Any

from alloy.envs.wrappers._common import require
from alloy.envs.wrappers.for_skrl import BOOTSTRAP_OBS, BOOTSTRAP_STATES, SkrlEnv

log = logging.getLogger("alloy.envs")
STEPS_PER_ENV = 32
HIDDEN = 128
INITIAL_LOG_STD = -0.5
KL_TARGET = 0.01

ENTROPY = 0.001      # keeps the action spread from collapsing before the first grasp


@lru_cache(maxsize=1)
def model_classes() -> tuple[type, type]:
    """``(Policy, Value)``, the two skrl model classes (built on first use, so this module imports without skrl).

    The policy is a Gaussian on the observation (tanh 128x128, learned log standard deviation, start -0.5). The value
    model reads skrl's ``states`` input, which is the env's "critic" group, when the env has one, else the observation;
    skrl's PPO passes the state to the value model only, never to the policy. The policy does not clip its actions
    (the env does): with skrl's ``clip_actions=True`` the stored action and its log-probability are those of the
    clipped point, and in our GripperLift-v0 runs the share of actions beyond the limits then grew until learning
    fell back.

    Example: ``Policy, Value = model_classes()``."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from skrl.models.torch import DeterministicMixin, GaussianMixin, Model  # noqa: PLC0415

    def mlp(inputs: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                             nn.Linear(HIDDEN, outputs))

    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space: Any, action_space: Any, device: Any) -> None:
            Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
            GaussianMixin.__init__(self, clip_actions=False)       # the env clips; see model_classes
            self.net = mlp(self.num_observations, self.num_actions)
            self.log_std = nn.Parameter(torch.full((self.num_actions,), INITIAL_LOG_STD))

        def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[Any, dict[str, Any]]:
            return self.net(inputs["observations"]), {"log_std": self.log_std}

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space: Any, state_space: Any, action_space: Any, device: Any) -> None:
            Model.__init__(self, observation_space=observation_space, state_space=state_space,
                           action_space=action_space, device=device)
            DeterministicMixin.__init__(self)
            self.uses_state = state_space is not None
            self.net = mlp(self.num_states if self.uses_state else self.num_observations, 1)

        def compute(self, inputs: dict[str, Any], role: str = "") -> tuple[Any, dict[str, Any]]:
            return self.net(inputs["states" if self.uses_state else "observations"]), {}

    return Policy, Value


@lru_cache(maxsize=1)
def agent_class() -> type:
    """skrl's PPO with one change: at a time limit it bootstraps from the ended episode's last observation.

    The env restarts a finished world inside ``step``, so the observation skrl gets back is the next episode's
    first; stock PPO would bootstrap from that. ``SkrlEnv`` puts the last observations in ``infos``, and this class
    uses them as the next observations of the recorded transition (built on first use: no skrl import at module
    load).

    Example: ``agent = agent_class()(models=models, memory=memory, cfg=cfg, ...)``."""
    from skrl.agents.torch.ppo import PPO  # noqa: PLC0415

    class BootstrapPPO(PPO):
        def record_transition(self, **transition: Any) -> None:
            infos = transition.get("infos")
            if isinstance(infos, dict) and BOOTSTRAP_OBS in infos:
                transition["next_observations"] = infos[BOOTSTRAP_OBS]
                if transition.get("next_states") is not None:
                    transition["next_states"] = infos[BOOTSTRAP_STATES]
            super().record_transition(**transition)

    return BootstrapPPO


def build_agent(env: SkrlEnv) -> Any:
    """skrl PPO with the 128x128 tanh models, 32-step rollouts and no TensorBoard or Weights & Biases writing.

    ``value_clip`` is 0 (off): skrl's default of 0.2 limits how far each update can move the value estimate, and with
    returns of several hundred (scaled by the value preprocessor) the critic then lags the policy and learning stalls.

    The agent bootstraps a time limit from the episode's last observation (``agent_class``).

    Example: ``agent = build_agent(SkrlEnv(make("GripperLift-v0", num_envs=64, backend="torch")))``."""
    from skrl.agents.torch.ppo import PPO_CFG  # noqa: PLC0415
    from skrl.memories.torch import RandomMemory  # noqa: PLC0415
    from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: PLC0415
    from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: PLC0415

    Policy, Value = model_classes()
    dev, obs, state, act = env.device, env.observation_space, env.state_space, env.action_space
    models = {"policy": Policy(obs, act, dev), "value": Value(obs, state, act, dev)}
    scaler = {"size": obs, "device": dev}
    cfg = PPO_CFG(rollouts=STEPS_PER_ENV, learning_epochs=4, mini_batches=4, discount_factor=0.99, gae_lambda=0.95,
                  learning_rate=3e-4, ratio_clip=0.2, time_limit_bootstrap=True, value_clip=0.0,
                  entropy_loss_scale=ENTROPY,
                  grad_norm_clip=1.0,
                  learning_rate_scheduler=KLAdaptiveLR,
                  learning_rate_scheduler_kwargs={"kl_threshold": KL_TARGET},
                  observation_preprocessor=RunningStandardScaler,
                  observation_preprocessor_kwargs=scaler,
                  value_preprocessor=RunningStandardScaler, value_preprocessor_kwargs={"size": 1, "device": dev})
    if state is not None:
        cfg.state_preprocessor = RunningStandardScaler
        cfg.state_preprocessor_kwargs = {"size": state, "device": dev}
    cfg.experiment.write_interval = 0
    cfg.experiment.checkpoint_interval = 0
    memory = RandomMemory(memory_size=STEPS_PER_ENV, num_envs=env.num_envs, device=dev)
    return agent_class()(models=models, memory=memory, observation_space=obs, state_space=state, action_space=act,
               device=dev, cfg=cfg)


def train(env_id: str, *, num_envs: int = 512, iterations: int = 200, seed: int = 0, device: str | None = None,
          log_dir: str | None = None, report_every: int = 10, env_kwargs: dict[str, Any] | None = None,
          env: Any = None) -> dict[str, Any]:
    """Train PPO on ``env_id`` and return ``{success_rate, minutes, samples, checkpoint}``.

    One JSON line per ``report_every`` iterations goes to stdout. The final checkpoint is saved in ``log_dir``
    (default ``runs/skrl/<env_id>``). ``env`` is a ready torch-backend ``VectorEnv`` to train on instead of making
    ``env_id`` (``num_envs``, ``device`` and ``env_kwargs`` are then ignored); ``train`` closes it. An env with a
    "critic" observation group gives the value model that group (see ``model_classes``).
    Example: ``train("GripperLift-v0", num_envs=512, iterations=200)``."""
    require("skrl")
    from alloy.envs import make  # noqa: PLC0415

    t0 = time.perf_counter()
    log_dir = log_dir or os.path.join("runs", "skrl", env_id)
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    if env is None:
        env = make(env_id, num_envs=num_envs, device=device, seed=seed, backend="torch", **(env_kwargs or {}))
    wrapped = SkrlEnv(env)
    try:
        return _learn(wrapped, env_id, iterations, log_dir, max(1, report_every), t0)
    finally:
        wrapped.close()


def _learn(env: SkrlEnv, env_id: str, iterations: int, log_dir: str, chunk: int, t0: float) -> dict[str, Any]:
    from skrl.trainers.torch import SequentialTrainer  # noqa: PLC0415

    agent = build_agent(env)
    trainer = SequentialTrainer(env=env, agents=agent, cfg={"timesteps": STEPS_PER_ENV, "headless": True,
                                                            "disable_progressbar": True,
                                                            "close_environment_at_exit": False})
    done, last = 0, {}
    while done < iterations:
        step = min(chunk, iterations - done)
        trainer.cfg.timesteps = step * STEPS_PER_ENV
        trainer.train()
        done += step
        last = _report(env, env_id, done, t0)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint = os.path.join(log_dir, f"agent_{done}.pt")
    agent.save(checkpoint)
    log.info("skrl: %s trained %d iterations, checkpoint %s", env_id, done, checkpoint)
    return {"success_rate": last.get("success_rate"), "minutes": last.get("minutes"),
            "samples": done * STEPS_PER_ENV * env.num_envs, "checkpoint": checkpoint}


def _report(env: SkrlEnv, env_id: str, done: int, t0: float) -> dict[str, Any]:
    episodes, successes, mean_return = env.take_stats()
    line = {"lib": "skrl", "env": env_id, "num_envs": env.num_envs, "iter": done,
            "samples": done * STEPS_PER_ENV * env.num_envs, "minutes": round((time.perf_counter() - t0) / 60, 2),
            "episodes": episodes, "success_rate": round(successes / episodes, 3) if episodes else None,
            "mean_return": None if mean_return is None else round(mean_return, 3),
            "invalid_episodes": env.tally.take_invalid()}
    print(json.dumps(line), flush=True)
    return line


__all__ = ["build_agent", "model_classes", "train"]
