# SPDX-License-Identifier: Apache-2.0
"""Playground env stack with protocol metrics, and the rsl_rl VecEnv on top of it.

Stack (same order as ``mujoco_playground.wrapper.wrap_for_brax_training``, one layer added):
  raw env -> Vmap / DomainRandomizationVmap -> MetricsWrapper -> EpisodeWrapper -> AutoResetWrapper.
MetricsWrapper sits before the auto-reset so it sees the last state of an episode. Nothing in the installed
package is edited.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jp
from brax.envs.wrappers import training as brax_training
from mujoco_playground import registry
from mujoco_playground._src import wrapper as pg_wrapper

from tasks import TaskSpec

PREFIX = "pg_"


class MetricsWrapper(pg_wrapper.Wrapper):
    """Adds the task's protocol metrics to ``state.metrics`` (batched, pre-reset)."""

    def __init__(self, env, spec: TaskSpec, episode_length: int = 0):
        super().__init__(env)
        self._spec = spec
        self._episode_length = episode_length
        self._names = spec.step_names + spec.end_names
        self._extra = ("fell",) + tuple("endg_" + n for n in spec.end_names)

    def reset(self, rng):
        state = self.env.reset(rng)
        n = rng.shape[0]
        metrics = dict(state.metrics)
        for name in self._names:
            metrics[PREFIX + name] = jp.zeros(n)
        for name in self._extra:
            metrics[PREFIX + name] = jp.zeros(n)
        return state.replace(metrics=metrics)

    def step(self, state, action):
        new = self.env.step(state, action)
        vals = self._spec.metric_fn(self.env.unwrapped, state, new)
        metrics = dict(new.metrics)
        for name in self._names:
            metrics[PREFIX + name] = vals[name].astype(jp.float32)
        # Brax eval sums state.metrics over an episode: these are non-zero only on the episode's last step
        fell = new.done.astype(jp.float32)
        last = jp.maximum(fell, ((state.info["steps"] + 1) >= self._episode_length).astype(jp.float32))
        metrics[PREFIX + "fell"] = fell
        for name in self._spec.end_names:
            metrics[PREFIX + "endg_" + name] = vals[name].astype(jp.float32) * last
        return new.replace(metrics=metrics)


def build_stack(spec: TaskSpec, num_envs: int, impl: str):
    """Load the raw env (jax | warp) and wrap it. Returns (raw_env, stack, episode_length)."""
    cfg = registry.get_default_config(spec.env_name)
    cfg.impl = impl
    if impl == "warp":
        scale = max(1.0, num_envs / spec.ref_envs)  # shipped Warp contact buffers are sized for ref_envs
        for key in ("naconmax", "naccdmax"):
            if key in cfg:
                cfg[key] = int(cfg[key] * scale)
    raw = registry.load(spec.env_name, config=cfg)
    rand = registry.get_domain_randomizer(spec.env_name)
    return raw, cfg, rand


def wrap_stack(env, spec, randomizer, episode_length, num_envs=None, rng=None):
    """Vmap (or DR-vmap) -> MetricsWrapper -> Episode -> AutoReset."""
    if randomizer is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = pg_wrapper.BraxDomainRandomizationVmapWrapper(env, functools.partial(randomizer, rng=rng))
    env = MetricsWrapper(env, spec, episode_length)
    env = brax_training.EpisodeWrapper(env, episode_length, 1)
    return pg_wrapper.BraxAutoResetWrapper(env, full_reset=False)


def wrap(raw, spec: TaskSpec, randomizer, seed: int, num_envs: int, episode_length: int):
    key = jax.random.PRNGKey(seed)
    key_reset, key_rand = jax.random.split(key)
    rng = jax.random.split(key_rand, num_envs)
    return wrap_stack(raw, spec, randomizer, episode_length, rng=rng), jax.random.split(key_reset, num_envs)
