# SPDX-License-Identifier: Apache-2.0
"""``SkrlEnv``: an Alloy ``VectorEnv`` in the shape skrl's trainers expect, with every tensor staying on the GPU.

skrl's own wrapper for Gymnasium vector envs turns the action tensor into NumPy and the result back into tensors on
every step; this one passes the tensors straight through. Needs ``pip install skrl`` and a torch-backend env.
A second observation group named "critic" (``Task(obs={"policy": [...], "critic": [...]})``) becomes skrl's state.
Example::

    from alloy.envs import make
    from alloy.envs.wrappers.for_skrl import SkrlEnv

    env = SkrlEnv(make("GripperLift-v0", num_envs=64, backend="torch"))
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.zero_actions())   # reward: (64, 1)
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import torch
    from skrl.envs.wrappers.torch import Wrapper as _Base
except ImportError:                                 # the wrapper then raises a clear error when it is built
    torch = None  # type: ignore[assignment]
    _Base = object  # type: ignore[assignment, misc]

from alloy.envs import spaces
from alloy.envs.errors import EnvError
from alloy.envs.wrappers._common import EpisodeTally, image_keys, require, require_torch_env, state_groups

log = logging.getLogger("alloy.envs")
CRITIC = "critic"

BOOTSTRAP_OBS, BOOTSTRAP_STATES = "bootstrap_observations", "bootstrap_states"


class SkrlEnv(_Base):  # type: ignore[misc, valid-type]
    """Wraps a torch-backend ``VectorEnv`` as a skrl environment (observations, state, rewards as torch tensors).

    ``step`` returns rewards, terminated and truncated as ``(num_envs, 1)`` tensors, as skrl expects. Worlds restart
    inside ``step`` (the env's same-step autoreset), so ``reset`` is only needed once; later calls give back the
    current observation. ``tally`` counts finished episodes and successes; ``take_stats()`` reads and clears it.

    Example: ``env = SkrlEnv(make("GripperLift-v0", num_envs=64, backend="torch"))``."""

    def __init__(self, env: Any) -> None:
        require("skrl")
        require_torch_env(env, "skrl")
        if image_keys(env):
            raise EnvError("the skrl wrapper does not support camera observations yet",
                           "use an environment without cameras (a state-only task)", field="cameras")
        super().__init__(env)
        self.tally = EpisodeTally()
        self._tdevice = env.xp.tdevice
        self._obs: Any = None
        self._return_sum = 0.0
        self._started = False
        self._observation_space: Any = None
        self._state_space: Any = None

    # --- skrl's properties -----------------------------------------------------------------------------------------

    @property
    def unwrapped(self) -> Any:
        """The wrapped ``VectorEnv``. Example: ``env.unwrapped.sim``."""
        return self._env

    @property
    def observation_space(self) -> Any:
        """One world's policy observation as a ``Box``. Example: ``env.observation_space.shape`` -> (22,)."""
        if self._observation_space is None:
            self._observation_space = self._box_of(self._current()[0])
        return self._observation_space

    @property
    def state_space(self) -> Any:
        """One world's critic observation as a ``Box``, or None when the env has no "critic" group.

        Example: ``env.state_space``."""
        if self._state_space is None and self.has_critic:
            self._state_space = self._box_of(self._current()[1])
        return self._state_space

    @property
    def action_space(self) -> Any:
        """One world's action ``Box``. Example: ``env.action_space.shape`` -> (3,)."""
        return self._env.single_action_space

    @property
    def has_critic(self) -> bool:
        """True when the env has a separate "critic" observation group. Example: ``env.has_critic``."""
        return CRITIC in (getattr(self._env, "obs_groups", ()) or ())

    # --- skrl's methods --------------------------------------------------------------------------------------------

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset every world once; later calls return the current observation (worlds restart inside ``step``).

        Example: ``obs, info = env.reset()``."""
        if not self._started:
            self._obs, _ = self._env.reset()
            self._started = True
        return self._current()[0], {}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Step every world with a ``(num_envs, actions)`` tensor; nothing is copied through NumPy.

        Returns ``(observation, reward, terminated, truncated, info)``; the last three tensors are ``(num_envs, 1)``.
        Example: ``obs, rew, term, trunc, info = env.step(torch.zeros(64, 3))``."""
        if not self._started:
            self.reset()
        if actions.device != self._tdevice:
            actions = actions.to(self._tdevice)
        self._obs, reward, terminated, truncated, info = self._env.step(actions)
        frozen = info["invalid"]                       # an INVALID world ends for good: no time-limit bootstrap
        terminated, truncated = terminated | frozen, truncated & ~frozen
        self._count(info)
        n = self.num_envs
        # skrl adds its bootstrap to the reward in place, so it gets a copy, never the env's own buffer
        info = self._with_bootstrap(info)
        return self._current()[0], reward.reshape(n, 1).clone(), terminated.view(n, 1), truncated.view(n, 1), info

    def _with_bootstrap(self, info: dict[str, Any]) -> dict[str, Any]:
        """Add what a value bootstrap at a time limit needs: for worlds whose episode just ended, their last
        observation (the env has already restarted them, so the returned observation is the next episode's first).
        ``info["bootstrap_observations"]`` (and ``"bootstrap_states"`` with a critic group) are full (num_envs, D)
        tensors: the last observation where an episode ended, the returned one elsewhere."""
        if "final_obs" not in info:
            return info
        done = info["_final_obs"].view(self.num_envs, 1)
        final = state_groups(self._env, info["final_obs"])
        now_obs, now_state = self._current()
        out = dict(info)
        out[BOOTSTRAP_OBS] = torch.where(done, final["policy"], now_obs)
        if now_state is not None:
            out[BOOTSTRAP_STATES] = torch.where(done, final[CRITIC], now_state)
        return out

    def state(self) -> torch.Tensor | None:
        """The latest critic observation ``(num_envs, D)``, or None without a "critic" group.

        Example: ``env.state()``."""
        if not self.has_critic:
            return None
        return self._current()[1]

    def render(self, *args: Any, **kwargs: Any) -> Any:
        """An image of world 0 when the env was made with ``render_mode="rgb_array"``, else None.

        Example: ``env.render()``."""
        return self._env.render()

    def close(self) -> None:
        """Free the wrapped env. Example: ``env.close()``."""
        self._env.close()

    # --- extras ----------------------------------------------------------------------------------------------------

    def zero_actions(self) -> torch.Tensor:
        """A ``(num_envs, actions)`` tensor of zeros on the env's device. Example: ``env.step(env.zero_actions())``."""
        return torch.zeros((self.num_envs, int(self.action_space.shape[0])), device=self._tdevice)

    def take_stats(self) -> tuple[int, int, float | None]:
        """``(episodes, successes, mean_return)`` of the episodes ended since the last call, then start again.

        Example: ``episodes, successes, mean_return = env.take_stats()``."""
        episodes, successes = self.tally.take()
        mean_return = self._return_sum / episodes if episodes else None
        self._return_sum = 0.0
        return episodes, successes, mean_return

    # --- helpers ---------------------------------------------------------------------------------------------------

    def _current(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._obs is None:
            self._obs, _ = self._env.reset()
            self._started = True
        groups = state_groups(self._env, self._obs)
        return groups["policy"], groups.get(CRITIC)

    @staticmethod
    def _box_of(batch: torch.Tensor) -> Any:
        return spaces.box(-float("inf"), float("inf"), (int(batch.shape[1]),))

    def _count(self, info: dict[str, Any]) -> None:
        ended, _ = self.tally.add(info)
        if ended:
            done = info["_episode"]
            self._return_sum += float(info["episode"]["r"][done].float().sum())


__all__ = ["BOOTSTRAP_OBS", "BOOTSTRAP_STATES", "SkrlEnv"]
