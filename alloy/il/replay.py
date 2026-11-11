# SPDX-License-Identifier: Apache-2.0
"""Replay recorded episodes in a ``VectorEnv``: restore each episode's start, apply its actions, compare.

One episode runs per world, many at once. Each world is put into the episode's recorded start state (joint
coordinates and velocities, control targets, goal points), then steps the recorded actions; the result says whether
the success came out the same and how far the simulator state drifted from the recorded one. Example::

    from alloy.il import load_hdf5, replay

    spec, episodes = load_hdf5("lift.hdf5")
    result = replay(env, episodes)                      # env built like the one that recorded
    print(result.success.mean(), result.matches.mean(), result.state_error.max())

A replay repeats the recording exactly only on the CPU, in an env with the same number of worlds and the same build
seed, and only for scenes of rigid bodies and robots. On a GPU it is not exact once contact is chaotic (contacts
there do not repeat bit for bit, ``docs/known_gaps/determinism-gpu-rigid.md``): in one audit of 64 PickPlace-v0
demonstrations replayed on the same GPU with the same world count and seed, 1-2 flipped from success to failure and
the state error had a median of 4.6 and a maximum of 30; two fresh envs given the same actions ended with the object
up to 0.20 m apart in 19 of 64 worlds. On a GPU compare with a tolerance (``state_error``) and judge by success
(``matches``), not by an exact repeat.
Scenes with cloth, liquid or other particles have no restorable start here: ``replay`` refuses them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.il.episode import Episode
from alloy.il.snapshot import WorldSnapshot


@dataclass
class ReplayResult:
    """Per-episode outcome of a replay, in the order the episodes were given.

    Example: ``replay(env, episodes).summary()``."""

    success: np.ndarray            # (N,) bool: the task's success check on the replay's last step
    recorded_success: np.ndarray   # (N,) bool: what the recording says
    state_error: np.ndarray        # (N,) float: largest |replayed - recorded| over all state columns and steps
    obs_error: np.ndarray          # (N,) float: the same for the "state" observation after each step
    ended_early: np.ndarray        # (N,) bool: the env ended the episode before its last recorded action

    @property
    def matches(self) -> np.ndarray:
        """(N,) bool: success came out as recorded and the episode did not end early. Example: ``r.matches.all()``."""
        return (self.success == self.recorded_success) & ~self.ended_early

    def summary(self) -> dict[str, Any]:
        """A JSON-ready summary. Example: ``json.dumps(result.summary())``."""
        n = int(self.success.shape[0])
        return {"episodes": n, "success": int(self.success.sum()),
                "recorded_success": int(self.recorded_success.sum()), "matches": int(self.matches.sum()),
                "ended_early": int(self.ended_early.sum()),
                "max_state_error": float(self.state_error.max()) if n else 0.0,
                "max_obs_error": float(self.obs_error.max()) if n else 0.0}


def replay(env: Any, episodes: Sequence[Episode]) -> ReplayResult:
    """Replay ``episodes`` in ``env`` (a ``VectorEnv`` built like the recording one, default autoreset), one per
    world, ``env.num_envs`` at a time. Example: see the module docstring."""
    if env.autoreset != "same_step":
        raise EnvError(f"replay needs autoreset='same_step', this env has {env.autoreset!r}",
                       "build the env with the default autoreset", field="autoreset")
    snapshot = WorldSnapshot(env)
    if snapshot.has_particles:
        raise EnvError("this scene has cloth, liquid or other particles, whose start state a replay cannot restore",
                       "replay rigid-body and robot scenes; for this one compare policies with alloy.il.evaluate",
                       field="episodes")
    for k, ep in enumerate(episodes):
        _check(env, ep, k)
    worlds = int(env.num_envs)
    parts = [_replay_batch(env, snapshot, list(episodes[i:i + worlds])) for i in range(0, len(episodes), worlds)]
    fields = ("success", "recorded_success", "state_error", "obs_error", "ended_early")
    if not parts:
        return ReplayResult(*(np.zeros(0, bool if f not in ("state_error", "obs_error") else float) for f in fields))
    return ReplayResult(*(np.concatenate([getattr(p, f) for p in parts]) for f in fields))


def _check(env: Any, ep: Episode, k: int) -> None:
    if not ep.start:
        raise EnvError(f"episode {k} has no start state", "record with Recorder(..., states=True) (the default) and "
                       "store it with HDF5Writer (the LeRobot layout does not keep it)", field="episodes")
    if ep.actions.shape[1] != env.writer.size:
        raise EnvError(f"episode {k} has {ep.actions.shape[1]} action columns, the env has {env.writer.size}",
                       "replay in an env built like the one that recorded", field="episodes")
    if ep.length > int(env.task.max_steps):
        raise EnvError(f"episode {k} has {ep.length} steps, the task ends episodes at {env.task.max_steps}",
                       "replay with the task's recorded max_steps", field="max_steps")


def _replay_batch(env: Any, snapshot: WorldSnapshot, batch: list[Episode]) -> ReplayResult:
    """Replay up to ``env.num_envs`` episodes at once; worlds without an episode idle on zero actions."""
    worlds, n, to_np = int(env.num_envs), len(batch), env.xp.to_numpy
    env.reset()
    snapshot.restore(_mask(worlds, n), _stack_starts(snapshot.capture(), batch))
    lengths = np.array([ep.length for ep in batch])
    actions = np.zeros((int(lengths.max()), worlds, batch[0].actions.shape[1]), np.float32)
    for w, ep in enumerate(batch):
        actions[:ep.length, w] = ep.actions
        actions[ep.length:, w] = ep.actions[-1] if _absolute(env) else 0.0
    result = ReplayResult(np.zeros(n, bool), np.array([ep.success for ep in batch], bool), np.zeros(n), np.zeros(n),
                          np.zeros(n, bool))
    live = np.ones(n, bool)
    for t in range(actions.shape[0]):
        _compare_states(result, snapshot, batch, live & (t < lengths), t)
        obs, _, terminated, truncated, info = env.step(env.xp.asarray(actions[t]))
        done = (np.asarray(to_np(terminated), bool) | np.asarray(to_np(truncated), bool))[:n]
        last = live & (t == lengths - 1)
        result.success[last] = np.asarray(to_np(info["success"]), bool)[:n][last]
        _compare_obs(result, env, batch, live & (t < lengths), t, obs, info)
        result.ended_early |= live & done & (t < lengths - 1)
        live &= ~done & ~last
    return result


def _absolute(env: Any) -> bool:
    return env.writer.spec.mode == "absolute"


def _mask(worlds: int, n: int) -> np.ndarray:
    mask = np.zeros(worlds, bool)
    mask[:n] = True
    return mask


def _stack_starts(current: dict[str, np.ndarray], batch: list[Episode]) -> dict[str, np.ndarray]:
    """(W, ...) start arrays: world w gets episode w's start; worlds past the batch keep their current state."""
    out = {k: v.copy() for k, v in current.items()}
    for w, ep in enumerate(batch):
        for key, value in ep.start.items():
            if key in out and out[key][w].shape == np.shape(value):
                out[key][w] = value
            else:
                raise EnvError(f"the recorded start has {key} of shape {np.shape(value)}, which this env does not "
                               "have", "replay in an env built like the one that recorded (same scene and task)",
                               field=key)
    return out


def _compare_states(result: ReplayResult, snapshot: WorldSnapshot, batch: list[Episode], active: np.ndarray,
                    t: int) -> None:
    """Largest state difference before action ``t``, for the episodes that recorded states."""
    now = snapshot.flat(snapshot.capture())
    for w in np.flatnonzero(active):
        states = batch[w].states
        if states is not None and states.shape[1] == now.shape[1]:
            result.state_error[w] = max(result.state_error[w], float(np.abs(now[w] - states[t]).max(initial=0.0)))


def _compare_obs(result: ReplayResult, env: Any, batch: list[Episode], active: np.ndarray, t: int, obs: Any,
                 info: dict[str, Any]) -> None:
    """Largest difference of the "state" observation after action ``t`` (a reset world's is its ``final_obs``)."""
    vec = env.xp.to_numpy(obs["state"] if isinstance(obs, dict) else obs)
    if "final_obs" in info:
        final, mask = info["final_obs"], np.asarray(env.xp.to_numpy(info["_final_obs"]), bool)
        final = env.xp.to_numpy(final["state"] if isinstance(final, dict) else final)
        vec = np.where(mask[:, None], final, vec)
    for w in np.flatnonzero(active):
        recorded = batch[w].next_obs.get("state")
        if recorded is not None:
            result.obs_error[w] = max(result.obs_error[w], float(np.abs(vec[w] - recorded[t]).max(initial=0.0)))


__all__ = ["ReplayResult", "replay"]
