# SPDX-License-Identifier: Apache-2.0
"""``RslRlVecEnv``: an Alloy ``VectorEnv`` in the shape rsl_rl's ``OnPolicyRunner`` expects.

Needs ``pip install rsl-rl-lib`` and an environment on the torch backend. Observation groups
(``Task(obs={"policy": [...], "critic": [...]})``) arrive as keys of a TensorDict; a flat env is the group "policy".
Example::

    from rsl_rl.runners import OnPolicyRunner
    from alloy.envs import make
    from alloy.envs.wrappers.for_rsl_rl import RslRlVecEnv
    from alloy.envs.wrappers.train_rsl_rl import default_config

    venv = RslRlVecEnv(make("GripperLift-v0", num_envs=512, backend="torch"))
    cfg = default_config(iterations=200, seed=0)
    cfg["obs_groups"] = venv.default_obs_groups()
    OnPolicyRunner(venv, cfg, log_dir=None, device=str(venv.device)).learn(200)
"""

from __future__ import annotations

import logging
from typing import Any

try:  # optional libraries: the module imports without them, and the class says what to install when it is built
    import torch
    from rsl_rl.env import VecEnv as _Base
    from tensordict import TensorDict
except ImportError:
    torch = TensorDict = None  # type: ignore[assignment]
    _Base = object  # type: ignore[assignment, misc]

from alloy.envs.errors import EnvError
from alloy.envs.wrappers._common import (EpisodeTally, image_keys, require, require_same_step, require_torch_env,
                                         state_groups)

log = logging.getLogger("alloy.envs")


class RslRlVecEnv(_Base):  # type: ignore[misc, valid-type]
    """Wraps a torch-backend ``VectorEnv`` as an rsl_rl ``VecEnv``.

    ``clip_actions`` (default True) clips the policy's actions to the env's action space ([-1, 1] for most tasks,
    wider where the env says so); False passes them as they are.

    Example: ``venv = RslRlVecEnv(make("GripperLift-v0", num_envs=64, backend="torch"))``."""

    def __init__(self, env: Any, *, clip_actions: bool = True) -> None:
        require("rsl_rl")
        require_torch_env(env, "rsl_rl")
        require_same_step(env, "rsl_rl")
        if not isinstance(clip_actions, bool):
            raise EnvError(f"clip_actions is {clip_actions!r}", "pass True (clip to the env's action space) or False",
                           field="clip_actions")
        self.env, self.clip_actions = env, clip_actions
        self.num_envs, self.device = env.num_envs, env.xp.tdevice
        self.num_actions = int(env.single_action_space.shape[0])
        self.max_episode_length = int(env.task.max_steps)
        self.cfg = {"env": env.task.name, "num_envs": self.num_envs, "dt": float(env.sim.dt)}
        self.tally = EpisodeTally()
        self._return_sum = 0.0
        space = env.single_action_space
        self._low = torch.as_tensor(space.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(space.high, dtype=torch.float32, device=self.device)
        self._obs, _ = env.reset()                    # the env keeps the seed it was built with

    @property
    def unwrapped(self) -> Any:
        """The wrapped ``VectorEnv``. Example: ``venv.unwrapped.sim``."""
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Steps taken in each world's running episode, (W,). Example: ``venv.episode_length_buf[0]``."""
        return self.env.stats.steps

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # rsl_rl's init_at_random_ep_len writes this to stagger the episode ends
        self.env.stats.steps = value.to(device=self.env.stats.steps.device, dtype=self.env.stats.steps.dtype)

    def default_obs_groups(self) -> dict[str, list[str]]:
        """The runner's ``obs_groups``: the actor reads "policy", the critic "critic" when the env has that group.

        Example: ``cfg["obs_groups"] = venv.default_obs_groups()``."""
        groups = state_groups(self.env, self._obs)
        critic = "critic" if "critic" in groups else "policy"
        images = image_keys(self.env)
        if images:                                   # rsl_rl's MLP policies take flat vectors: say so, once
            log.warning("rsl_rl: the camera images %s are in the observations but not in these obs_groups, so the "
                        "policy does not see them; add them to obs_groups with an image encoder of your own, or "
                        "train on the state groups", sorted(images))
        return {"actor": ["policy"], "critic": [critic]}

    def get_observations(self) -> TensorDict:
        """The current observations, one TensorDict key per group (and per camera).

        Example: ``venv.get_observations()["policy"].shape``."""
        return self._tensordict(self._obs)

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        """Reset every world; returns ``(observations, {})``. Example: ``obs, _ = venv.reset()``."""
        self._obs, _ = self.env.reset()
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step every world: ``(observations, rewards, dones, extras)``; ``extras["time_outs"]`` marks time limits.

        Example: ``obs, rew, dones, extras = venv.step(torch.zeros(64, venv.num_actions))``."""
        if self.clip_actions:
            actions = torch.maximum(torch.minimum(actions, self._high), self._low)
        self._obs, reward, terminated, truncated, info = self.env.step(actions)
        # a frozen (INVALID) world has no future to bootstrap, whatever Task(on_invalid=) says
        extras = {"time_outs": truncated & ~info["invalid"], "log": self._log(info)}
        return self.get_observations(), reward, (terminated | truncated).long(), extras

    def take_stats(self) -> tuple[int, int, float | None]:
        """``(episodes, successes, mean_return)`` of the episodes ended since the last call, then start again.

        Example: ``episodes, successes, mean_return = venv.take_stats()``."""
        episodes, successes = self.tally.take()
        mean_return = self._return_sum / episodes if episodes else None
        self._return_sum = 0.0
        return episodes, successes, mean_return

    def seed(self, seed: int = 0) -> int:
        """Seed the action sampler and the randomisation and restart every world from that seed, so the episode the
        runner reads next (``get_observations``) is the seed's, as in ``make(..., seed=seed)``. Example:
        ``venv.seed(3)``."""
        self._obs, _ = self.env.reset(seed=seed)
        return seed

    def close(self) -> None:
        """Free the wrapped env. Example: ``venv.close()``."""
        self.env.close()

    def _tensordict(self, obs: Any) -> TensorDict:
        data = dict(state_groups(self.env, obs))
        data.update({k: obs[k] for k in image_keys(self.env)})
        return TensorDict(data, batch_size=[self.num_envs])

    def _log(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Values for rsl_rl's logger: means over the episodes that ended this step, and reward terms."""
        log_values = {f"Episode_Reward/{name}": value.float().mean() for name, value in info["terms"].items()}
        ended, _ = self.tally.add(info)
        if not ended:
            return log_values
        done, episode = info["_episode"], info["episode"]
        log_values["/success_rate"] = episode["success"][done].float().mean()
        log_values["/episode_return"] = episode["r"][done].float().mean()
        self._return_sum += float(episode["r"][done].float().sum())
        log_values["/episode_length"] = episode["l"][done].float().mean()
        return log_values


__all__ = ["RslRlVecEnv"]
