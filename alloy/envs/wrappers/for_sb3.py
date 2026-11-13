# SPDX-License-Identifier: Apache-2.0
"""Stable-Baselines3 adapter: train ``stable_baselines3.PPO`` (or any SB3 algorithm) on an Alloy ``VectorEnv``.

``Sb3VecEnv`` implements SB3's vector-environment interface on top of the batched env, so no glue is needed. All data
crossing to SB3 is NumPy (copied, because on the CPU the torch and NumPy views alias the engine's memory). For each
world that ends in a step, ``infos[i]`` carries ``terminal_observation`` (the last observation before the reset),
``TimeLimit.truncated`` (True for a time-limit end, so SB3 bootstraps the value), ``is_success`` and
``episode`` = ``{"r", "l"}`` (so SB3's own logger shows ``ep_rew_mean`` and ``success_rate``).

Observations: a flat env gives its Box; an env with cameras gives a Dict {"state", "<camera>"} (use
``MultiInputPolicy``); an env with observation groups gives only its "policy" group, because SB3 has one observation.
Only the "same_step" autoreset mode is supported.

Example::

    from stable_baselines3 import PPO
    from alloy.envs import make
    from alloy.envs.wrappers.for_sb3 import Sb3VecEnv

    venv = Sb3VecEnv(make("GripperLift-v0", num_envs=64))
    PPO("MlpPolicy", venv, n_steps=32, batch_size=512, n_epochs=4).learn(64 * 32 * 10)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from alloy.envs.wrappers._common import (POLICY, EpisodeTally, image_keys, require, require_same_step,
                                         require_vector_env, state_groups)

log = logging.getLogger("alloy.envs")

try:  # Stable-Baselines3's VecEnv when it is installed; Sb3VecEnv() then raises with the pip command
    from stable_baselines3.common.vec_env import VecEnv as _VecEnv
except ImportError:  # pragma: no cover - exercised where Stable-Baselines3 is missing
    _VecEnv = object


class Sb3VecEnv(_VecEnv):  # type: ignore[misc, valid-type]
    """A ``VectorEnv`` as a Stable-Baselines3 ``VecEnv``. ``tally`` counts finished episodes and successes.

    Example: ``venv = Sb3VecEnv(make("GripperLift-v0", num_envs=64)); venv.reset().shape`` -> (64, 22)."""

    def __init__(self, env: Any) -> None:
        require("sb3")
        require_vector_env(env, "sb3")
        require_same_step(env, "Stable-Baselines3")
        self.env, self.tally = env, EpisodeTally()
        self._images = image_keys(env)
        self._actions: Any = None
        self._returns = np.zeros(env.num_envs, np.float64)
        self._lengths = np.zeros(env.num_envs, np.int64)
        obs_space = _obs_space(env, self._images)
        super().__init__(env.num_envs, obs_space, _gym_space(env.single_action_space))

    # --- observations ----------------------------------------------------------------------------------------------

    def _np(self, x: Any) -> np.ndarray:
        return np.array(self.env.xp.to_numpy(x), copy=True)

    def _obs(self, obs: Any) -> Any:
        state = self._np(state_groups(self.env, obs)[POLICY])
        if not self._images:
            return state
        cams = {k: self._np(obs[k]) for k in self._images}
        return {"state": state, **cams} if state.shape[-1] else cams      # a camera-only task has no state entry

    def _row(self, obs: Any, i: int) -> Any:
        if isinstance(obs, dict):
            return {k: v[i] for k, v in obs.items()}
        return obs[i]

    # --- VecEnv API ------------------------------------------------------------------------------------------------

    def reset(self) -> Any:
        """Reset every world and return the observation. Example: ``venv.reset()``."""
        seed = next((s for s in self._seeds if s is not None), None)
        obs, _ = self.env.reset(seed=seed)
        self._seeds = [None] * self.num_envs
        self._returns[:], self._lengths[:] = 0.0, 0
        return self._obs(obs)

    def step_async(self, actions: np.ndarray) -> None:
        """Store the actions for ``step_wait``. Example: ``venv.step_async(actions)``."""
        self._actions = actions

    def step_wait(self) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step every world: ``(obs, rewards, dones, infos)``. Example: ``venv.step_wait()``."""
        obs, rew, term, trunc, info = self.env.step(self._actions)
        obs, rew = self._obs(obs), self._np(rew)
        term, trunc = self._np(term), self._np(trunc)
        frozen = self._np(info["invalid"])            # an INVALID world ends for good: no time-limit bootstrap
        term, trunc = term | frozen, trunc & ~frozen
        self._returns += rew
        self._lengths += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        if "_final_obs" in info:
            self._fill_ended(infos, info, term, trunc)
        return obs, rew, term | trunc, infos

    def _fill_ended(self, infos: list[dict[str, Any]], info: dict[str, Any], term: np.ndarray,
                    trunc: np.ndarray) -> None:
        final = self._obs(info["final_obs"])
        success = self._np(info["episode"]["success"])
        self.tally.add({"_episode": self._np(info["_episode"]),
                        "episode": {"success": success, "invalid": self._np(info["episode"]["invalid"])}})
        for i in np.flatnonzero(self._np(info["_final_obs"])):
            infos[i] = {"terminal_observation": self._row(final, i),
                        "TimeLimit.truncated": bool(trunc[i] and not term[i]),
                        "is_success": bool(success[i]),
                        "episode": {"r": float(self._returns[i]), "l": int(self._lengths[i])}}
            self._returns[i], self._lengths[i] = 0.0, 0

    def close(self) -> None:
        """Free the env. Example: ``venv.close()``."""
        self.env.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """The env's attribute, once per selected world. Example: ``venv.get_attr("render_mode")``."""
        return [getattr(self.env, attr_name)] * len(self._indices(indices))

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Set an attribute on the env. Example: ``venv.set_attr("render_mode", None)``."""
        setattr(self.env, attr_name, value)

    def env_method(self, method_name: str, *args: Any, indices: Any = None, **kwargs: Any) -> list[Any]:
        """Call a method of the env once. Example: ``venv.env_method("obs_names")``."""
        return [getattr(self.env, method_name)(*args, **kwargs)] * len(self._indices(indices))

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> list[bool]:
        """Always False: the env has no Gymnasium wrappers. Example: ``venv.env_is_wrapped(object)``."""
        return [False] * len(self._indices(indices))

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        """Seed the next ``reset``. Example: ``venv.seed(0)``."""
        return super().seed(seed)

    def get_images(self) -> Sequence[np.ndarray | None]:
        """One picture (world 0) when the env renders, else None per world. Example: ``venv.get_images()``."""
        frame = self.env.render()
        return [frame] + [None] * (self.num_envs - 1) if frame is not None else [None] * self.num_envs

    def _indices(self, indices: Any) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        return [int(indices)] if isinstance(indices, int) else [int(i) for i in indices]


def _gym_space(space: Any) -> Any:
    """The env's space as a real Gymnasium space (the env falls back to stand-ins without Gymnasium)."""
    import gymnasium  # noqa: PLC0415

    if isinstance(space, gymnasium.spaces.Space):
        return space
    return gymnasium.spaces.Box(space.low, space.high, space.shape, space.dtype)


def _obs_space(env: Any, images: list[str]) -> Any:
    """Box for a flat env or one with groups (the "policy" width), Dict with the cameras otherwise."""
    import gymnasium  # noqa: PLC0415

    if not images and not getattr(env, "obs_groups", None):
        return _gym_space(env.single_observation_space)
    shape = tuple(int(d) for d in state_groups(env, env._observe(env.state))[POLICY].shape[1:])  # (D,) or (H, D)
    state = gymnasium.spaces.Box(-np.inf, np.inf, shape, np.float32)
    if not images:
        return state
    cams = env.single_observation_space.spaces
    images_only = {k: _gym_space(cams[k]) for k in images}
    return gymnasium.spaces.Dict({"state": state, **images_only} if shape[-1] else images_only)


__all__ = ["Sb3VecEnv"]
