# SPDX-License-Identifier: Apache-2.0
"""rsl_rl 5.x VecEnv over the JAX Playground stack (torch <-> JAX through dlpack, zero copy on the GPU).

The installed ``mujoco_playground.wrapper_torch.RSLRLBraxWrapper`` targets rsl_rl 3.x (``get_observations``
resets the env, obs dict keys, ``extras['observations']``); rsl_rl 5.x calls ``get_observations`` as a plain
read and builds the actor/critic from ``obs_groups``, so this is the same wrapper idea rewritten for 5.x.
Per-iteration protocol sums are accumulated on the GPU inside one jitted step and read once per iteration.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import torch
import torch.utils.dlpack as tpack
from jax.dlpack import from_dlpack as jax_from_dlpack
from tensordict import TensorDict

from env import PREFIX
from tasks import TaskSpec

try:
    from rsl_rl.env import VecEnv
except ImportError:  # pragma: no cover
    VecEnv = object


def _to_torch(x):
    return tpack.from_dlpack(x)


def _strong(tree):
    """Drop JAX weak types (python scalars in env info) so reset and step outputs share one jit signature."""
    return jax.tree.map(lambda x: jax.lax.convert_element_type(x, jp.asarray(x).dtype), tree)


class PlaygroundVecEnv(VecEnv):
    def __init__(self, stack, key_reset, spec: TaskSpec, num_envs: int, episode_length: int, raw_env, device="cuda:0"):
        self.num_envs = num_envs
        self.num_actions = raw_env.action_size
        self.max_episode_length = episode_length
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.device = device
        self.cfg = {}
        self._spec = spec
        self._stack = stack
        self._key_reset = key_reset
        self.ctrl_dt = float(raw_env.dt)
        self.asymmetric = isinstance(raw_env.observation_size, dict)
        self._reset = jax.jit(lambda k: _strong(stack.reset(k)))
        self._step = jax.jit(self._step_fn)
        self._state = None
        self._acc = None
        self.reset_all()

    # ---- jitted step + accumulation -------------------------------------------------------------------
    def _zero_acc(self):
        names = self._spec.step_names + self._spec.end_names
        z = jp.zeros((), jp.float32)
        return {"n_steps": z, "n_end": z, "n_fall": z, "rew": z, "ret": z,
                **{"s_" + n: z for n in names}}

    def _step_fn(self, state, action, acc):
        new = _strong(self._stack.step(state, action))
        ended = (new.done > 0.5).astype(jp.float32)
        timeout = new.info["truncation"]
        acc = dict(acc)
        acc["n_steps"] += new.done.shape[0]
        acc["n_end"] += ended.sum()
        acc["n_fall"] += (ended * (1.0 - timeout)).sum()
        acc["rew"] += new.reward.sum()
        acc["ret"] += (ended * new.info["episode_metrics"]["sum_reward"]).sum()
        for n in self._spec.step_names:
            acc["s_" + n] += new.metrics[PREFIX + n].sum()
        for n in self._spec.end_names:
            acc["s_" + n] += (ended * new.metrics[PREFIX + n]).sum()
        return new, {k: jp.asarray(v, jp.float32) for k, v in acc.items()}  # fixed dtype: one compile

    # ---- rsl_rl VecEnv --------------------------------------------------------------------------------
    def reset_all(self):
        self._state = jax.device_put(self._reset(self._key_reset), jax.devices('gpu')[0])  # committed leaves: one jit signature
        self._acc = self._zero_acc()

    def warmup(self, steps: int = 60):
        """Compile reset + step the way training calls them (torch-made action arrays), then restore the start."""
        for _ in range(steps):
            a = jax_from_dlpack(torch.randn(self.num_envs, self.num_actions, device=self.device).clip(-1, 1))
            self._state, self._acc = self._step(self._state, a, self._acc)
        jax.block_until_ready(self._acc)
        self.pop_metrics()
        self.reset_all()

    def _obs(self):
        o = self._state.obs
        if self.asymmetric:
            d = {"state": _to_torch(o["state"]), "privileged_state": _to_torch(o["privileged_state"])}
        else:
            d = {"state": _to_torch(o)}
        return TensorDict(d, batch_size=[self.num_envs])

    def get_observations(self):
        return self._obs()

    def step(self, actions):
        a = jax_from_dlpack(torch.clip(actions, -1.0, 1.0).contiguous())
        self._state, self._acc = self._step(self._state, a, self._acc)
        s = self._state
        done = _to_torch(s.done).long()
        extras = {"time_outs": _to_torch(s.info["truncation"]), "log": {}}
        return self._obs(), _to_torch(s.reward), done, extras

    # ---- protocol metrics -----------------------------------------------------------------------------
    def pop_metrics(self, last: dict[str, float] | None = None) -> dict[str, float]:
        """Protocol metrics of the steps since the last call; ``last`` carries episode-end values forward."""
        acc = {k: float(v) for k, v in jax.device_get(self._acc).items()}
        self._acc = self._zero_acc()
        n, ne = max(acc["n_steps"], 1.0), acc["n_end"]
        last = dict(last or {})
        out = {n_: acc["s_" + n_] / n for n_ in self._spec.step_names}
        for name in self._spec.end_names:
            out[name] = acc["s_" + name] / ne if ne > 0 else last.get(name, float("nan"))
        out["reward"] = acc["ret"] / ne if ne > 0 else last.get("reward", float("nan"))
        out["reward_step"] = acc["rew"] / n
        out["episodes_ended"] = ne
        if self._spec.key == "go2_flat":
            out["fall_rate"] = acc["n_fall"] / ne if ne > 0 else 0.0
        if self._spec.key == "cube_reorient":
            out["drop_rate"] = acc["n_fall"] / ne if ne > 0 else 0.0
            out["goals_per_min"] = out.pop("goal_hit") * 60.0 / self.ctrl_dt
        return out
