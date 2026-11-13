# SPDX-License-Identifier: Apache-2.0
"""``Recorder``: record episodes from a ``VectorEnv`` while you step it, one episode per world at a time.

It wraps the env from outside (``recorder.step`` calls ``env.step``); the env and the simulation are not changed, so
a run steps the same with or without it. Example::

    import alloy.envs
    from alloy.il import Recorder, HDF5Writer

    env = alloy.envs.make("GripperLift-v0", num_envs=8)
    rec = Recorder(env, [HDF5Writer("lift.hdf5")], task="lift the cube", keep="success")
    obs, info = rec.reset(seed=0)
    while rec.saved < 20:
        obs, reward, terminated, truncated, info = rec.step(policy(obs))
    rec.close()

Every finished episode goes through ``keep`` ("success": only successful ones, "all", or a function of the
``Episode``) and then to every writer. Works with all three autoreset modes of the env.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.il.episode import DatasetSpec, Episode, Writer
from alloy.il.snapshot import WorldSnapshot
from alloy.il.support import labels, require_vector_env, unrestorable

log = logging.getLogger("alloy.il")

KEEP = ("success", "all")


class Recorder:
    """Records the episodes of ``env`` into ``writers``. Example: see the module docstring."""

    def __init__(self, env: Any, writers: Sequence[Writer], *, task: str | None = None,
                 keep: str | Callable[[Episode], bool] = "success", states: bool = True,
                 max_episodes: int | None = None, env_kwargs: dict[str, Any] | None = None,
                 manual: bool = False) -> None:
        require_vector_env(env, "Recorder")
        if not callable(keep) and keep not in KEEP:
            raise EnvError(f"keep is {keep!r}", f"use one of {KEEP} or a function of the Episode", field="keep")
        self.env, self.writers, self.keep, self.max_episodes = env, list(writers), keep, max_episodes
        self.manual = manual                               # the operator decides: see ``finish`` (teleop sets it)
        self.world_quota: int | None = None                # save at most this many episodes per world (evaluate)
        self.num_envs, self.xp = int(env.num_envs), env.xp
        self._ended = np.zeros(self.num_envs, np.int64)    # episodes each world ended since the last full reset
        self.snapshot = WorldSnapshot(env) if states else None
        why = unrestorable(env, self.snapshot.has_particles) if self.snapshot is not None else None
        if why is not None:
            log.warning("Recorder: %s; Episode.states and Episode.start lack that part and replay refuses this "
                        "env (the actions, observations and rewards are recorded in full)", why)
        self.spec = self._spec(task, env_kwargs or {})
        for w in self.writers:
            w.open(self.spec)
        self.saved = self.dropped = 0
        self.invalid = 0                                   # ended episodes whose world was INVALID at some step
        self.last_invalid = np.zeros(self.num_envs, bool)  # of the episodes that ended in the last step
        self._bad = np.zeros(self.num_envs, bool)          # worlds INVALID at some step of the running episode
        self._frames: list[dict[str, Any]] = []            # one entry per recorded step, arrays over all worlds
        self._base = 0                                     # step number of self._frames[0]
        self._step = 0                                     # steps recorded so far
        self._begin = np.zeros(self.num_envs, np.int64)    # step each world's running episode began at
        self._idle = np.zeros(self.num_envs, bool)         # worlds between an episode's end and their reset
        self._obs: dict[str, np.ndarray] | None = None
        self._state: dict[str, np.ndarray] | None = None
        self._start: dict[str, np.ndarray] = {}
        self.closed = False

    # --- description -----------------------------------------------------------------------------------------------

    def _spec(self, task: str | None, env_kwargs: dict[str, Any]) -> DatasetSpec:
        import alloy  # noqa: PLC0415

        env = self.env
        single = env.single_observation_space
        parts = dict(single.spaces) if hasattr(single, "spaces") else {"state": single}
        groups = tuple(getattr(env, "obs_groups", ()))
        named = {g: labels(env, g) for g in groups if g in parts} or {"state": labels(env, None)}
        named = {k: v for k, v in named.items() if k in parts and len(parts[k].shape) == 1
                 and len(v) == int(parts[k].shape[0])}
        return DatasetSpec(
            env=str(env.task.name), task=task if task is not None else str(env.task.name),
            fps=1.0 / float(env.sim.dt),
            obs_shapes={k: tuple(int(n) for n in s.shape) for k, s in parts.items()},
            obs_dtypes={k: str(np.dtype(s.dtype)) for k, s in parts.items()},
            obs_names=list(next(iter(named.values()), [])), obs_key_names=named,
            action_names=list(env.action_names()),
            action_mode=str(env.writer.spec.mode),
            state_names=self.snapshot.flat_names() if self.snapshot is not None else [],
            device=str(env.device), alloy_version=str(alloy.__version__), env_kwargs=dict(env_kwargs))

    @property
    def full(self) -> bool:
        """True once ``max_episodes`` episodes are saved. Example: ``while not rec.full: rec.step(a)``."""
        return self.max_episodes is not None and self.saved >= self.max_episodes

    # --- the env's API ---------------------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """``env.reset``; the running episodes of the worlds it resets are dropped. Example: ``rec.reset(seed=0)``."""
        obs, info = self.env.reset(seed=seed, options=options)
        mask = (options or {}).get("reset_mask")
        mask = np.ones(self.num_envs, bool) if mask is None else np.asarray(self.xp.to_numpy(mask), bool)
        if mask.all():
            self._ended[:] = 0
        self._restart(mask, self._host(obs))
        return obs, info

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """``env.step(actions)``, recorded. Returns what the env returns. Example: ``rec.step(policy(obs))``."""
        self._require_ready()
        out = self.env.step(actions)
        obs, reward, terminated, truncated, info = out
        to_np = self.xp.to_numpy
        obs_np = self._host(obs)
        term, trunc = np.asarray(to_np(terminated), bool), np.asarray(to_np(truncated), bool)
        done = term | trunc
        self._frames.append({"obs": self._obs, "next_obs": self._next_obs(obs_np, info, done),
                             "action": self._action(actions), "reward": np.asarray(to_np(reward), np.float32).copy(),
                             "terminated": term, "truncated": trunc, "state": self._flat_state()})
        self._step += 1
        woke = self._idle & ~done if self._auto("next_step") else np.zeros(self.num_envs, bool)
        ended = done & ~self._idle
        self._note_invalid(info, ended)
        if done.any():
            self._finish(ended, np.asarray(to_np(info["success"]), bool))
            self._bad[done] = False
        self._after_step(done, woke, obs_np)
        return out

    def finish(self, save: bool = True) -> Any:
        """End every world's running episode now (teleoperation: the operator decides), write them when ``save``
        (their last step is marked truncated), reset the env and return the new observation. ``save=True`` is the
        operator's approval: it saves an episode that did not pass the success check under ``keep="success"``. A
        ``keep`` function still decides, and an INVALID episode is saved only with ``keep="all"``.

        Example: ``obs = rec.finish(save=True)``."""
        self._require_ready()
        live = ~self._idle & (self._step > self._begin)
        if save and live.any():
            self._frames[-1]["truncated"] = self._frames[-1]["truncated"] | (live & ~self._frames[-1]["terminated"])
            success = np.asarray(self.xp.to_numpy(self._success_now()), bool)
            self._finish(live, success, force=True)
        elif live.any():
            self.dropped += int(live.sum())
        self._begin[:] = self._step                        # accounted for: the reset below must not count them
        obs, _ = self.reset()
        return obs

    def close(self) -> None:
        """Close every writer (running episodes are dropped). Example: ``rec.close()``."""
        if self.closed:
            return
        for w in self.writers:
            w.close()
        self.closed = True
        log.info("Recorder %s: %d episodes saved, %d dropped", self.spec.env, self.saved, self.dropped)

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- bookkeeping -----------------------------------------------------------------------------------------------

    def _auto(self, mode: str) -> bool:
        return self.env.autoreset == mode

    def _after_step(self, done: np.ndarray, woke: np.ndarray, obs_np: dict[str, np.ndarray]) -> None:
        """Where each world's next episode begins, by autoreset mode; then remember what the policy sees next."""
        if self._auto("same_step"):
            begins = done                                  # reset inside this step: obs is the new episode's first
        else:
            self._idle = (self._idle & ~woke) | done       # next_step: the reset happens in the following step,
            begins = woke                                  # whose action is void; disabled: until reset(mask)
        self._obs = obs_np
        self._state = self.snapshot.capture(randomized=bool(begins.any())) if self.snapshot is not None else None
        self._mark_begin(begins)
        self._trim()

    def _note_invalid(self, info: dict[str, Any], ended: np.ndarray) -> None:
        """Read ``info["invalid"]`` at every step: the env's reset clears the engine's health record, so a later
        ``sim.health()`` no longer shows a world that froze."""
        if "invalid" in info:
            self._bad |= np.asarray(self.xp.to_numpy(info["invalid"]), bool) & ~self._idle
        self.last_invalid = ended & self._bad

    def _restart(self, mask: np.ndarray, obs_np: dict[str, np.ndarray]) -> None:
        self.dropped += int((mask & ~self._idle & (self._step > self._begin)).sum())
        self._idle = self._idle & ~mask
        self._bad[mask] = False
        self._obs = obs_np
        self._state = self.snapshot.capture(randomized=True) if self.snapshot is not None else None
        self._mark_begin(mask)
        self._trim()

    def _mark_begin(self, mask: np.ndarray) -> None:
        if not mask.any():
            return
        self._begin[mask] = self._step
        for key, value in (self._state or {}).items():
            if key not in self._start:
                self._start[key] = value.copy()
            self._start[key][mask] = value[mask]

    def _trim(self) -> None:
        """Forget steps no running episode needs."""
        oldest = int(self._begin[~self._idle].min()) if (~self._idle).any() else self._step
        drop = oldest - self._base
        if drop > 0:
            del self._frames[:drop]
            self._base = oldest

    def _finish(self, mask: np.ndarray, success: np.ndarray, force: bool = False) -> None:
        self.invalid += int((mask & self._bad).sum())
        for w in np.flatnonzero(mask):
            episode = self._episode(int(w), bool(success[w]) and not self._bad[w])   # frozen: no success
            over = self.world_quota is not None and self._ended[w] >= self.world_quota
            self._ended[w] += 1
            if over or not self._keeps(episode, force) or self.full:
                self.dropped += 1
                continue
            episode.index = self.saved
            for writer in self.writers:
                writer.write(episode)
            self.saved += 1

    def _keeps(self, episode: Episode, force: bool) -> bool:
        """``force``: the operator approved it (``finish(save=True)``). Otherwise, in ``manual`` mode an episode
        the env ended is kept only when it succeeded: a timed-out attempt nobody approved is not a demonstration."""
        if self.manual and not force and not episode.success:
            return False
        if callable(self.keep):
            return bool(self.keep(episode))
        if self.keep == "all":
            return True
        return (episode.success or force) and not episode.invalid

    def _episode(self, w: int, success: bool) -> Episode:
        frames = self._frames[int(self._begin[w]) - self._base:]
        stack = lambda key: np.stack([f[key][w] for f in frames])                      # noqa: E731
        # one (T + 1, ...) array per key: obs is its first T rows and next_obs its last T (views, not two copies)
        both = {k: np.stack([frames[0]["obs"][k][w], *(f["next_obs"][k][w] for f in frames)])
                for k in frames[0]["obs"]}
        obs, nxt = {k: v[:-1] for k, v in both.items()}, {k: v[1:] for k, v in both.items()}
        states = stack("state") if frames[0]["state"] is not None else None
        return Episode(obs=obs, next_obs=nxt, actions=stack("action"), rewards=stack("reward"),
                       terminated=stack("terminated"), truncated=stack("truncated"), success=success, states=states,
                       start={k: v[w].copy() for k, v in self._start.items()}, world=w,
                       invalid=bool(self._bad[w]))

    # --- array helpers ---------------------------------------------------------------------------------------------

    def _host(self, obs: Any) -> dict[str, np.ndarray]:
        """The env's observation as host NumPy copies, always a dict ("state" plus camera keys)."""
        parts = obs if isinstance(obs, dict) else {"state": obs}
        return {k: np.array(self.xp.to_numpy(v)) for k, v in parts.items()}

    def _next_obs(self, obs_np: dict[str, np.ndarray], info: dict[str, Any], done: np.ndarray) -> dict[str, np.ndarray]:
        """What each world saw after this step: for a world reset inside the step, the observation it ended on."""
        if "final_obs" not in info or not done.any():
            return obs_np
        final = self._host(info["final_obs"])
        return {k: np.where(done.reshape(-1, *([1] * (v.ndim - 1))), final[k], v) for k, v in obs_np.items()}

    def _action(self, actions: Any) -> np.ndarray:
        a = np.array(self.xp.to_numpy(self.xp.asarray(actions, "float32")), np.float32)
        if self.spec.action_mode != "absolute":
            return np.clip(a, -1.0, 1.0)
        writer = self.env.writer                           # absolute: the env clips to the ActionSpec's low / high
        low, high = getattr(writer, "low_np", None), getattr(writer, "high_np", None)
        return a if low is None or high is None else np.clip(a, low, high).astype(np.float32, copy=False)

    def _flat_state(self) -> np.ndarray | None:
        return self.snapshot.flat(self._state) if self.snapshot is not None and self._state is not None else None

    def _success_now(self) -> Any:
        task, s = self.env.task, self.env.state
        return task.success(s) if task.success is not None else self.xp.zeros(self.num_envs, "bool")

    def _require_ready(self) -> None:
        if self.closed:
            raise EnvError("the recorder is closed", "make a new Recorder")
        if self._obs is None:
            raise EnvError("step before reset", "call recorder.reset() first: the recorder needs the observation "
                           "each episode starts from", field="reset")


__all__ = ["KEEP", "Recorder"]
