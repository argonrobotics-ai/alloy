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
from alloy.il.snapshot import RANDOM, WorldSnapshot
from alloy.il.support import require_vector_env, unrestorable


@dataclass
class ReplayResult:
    """Per-episode outcome of a replay, in the order the episodes were given.

    Example: ``replay(env, episodes).summary()``."""

    success: np.ndarray            # (N,) bool: the task's success check on the replay's last step
    recorded_success: np.ndarray   # (N,) bool: what the recording says
    state_error: np.ndarray        # (N,) float: largest |replayed - recorded| over all state columns and steps
    obs_error: np.ndarray          # (N,) float: the same for the "state" observation after each step
    ended_early: np.ndarray        # (N,) bool: the env ended the episode before its last recorded action
    invalid: np.ndarray            # (N,) bool: the engine froze the world (INVALID) at some step of the replay
    recorded_invalid: np.ndarray   # (N,) bool: what the recording says (False for files that do not store it)

    @property
    def matches(self) -> np.ndarray:
        """(N,) bool: success and INVALID came out as recorded and the episode did not end early.

        Example: ``r.matches.all()``."""
        same = (self.success == self.recorded_success) & (self.invalid == self.recorded_invalid)
        return same & ~self.ended_early

    def summary(self) -> dict[str, Any]:
        """A JSON-ready summary. Example: ``json.dumps(result.summary())``."""
        n = int(self.success.shape[0])
        return {"episodes": n, "success": int(self.success.sum()),
                "recorded_success": int(self.recorded_success.sum()), "matches": int(self.matches.sum()),
                "ended_early": int(self.ended_early.sum()), "invalid": int(self.invalid.sum()),
                "recorded_invalid": int(self.recorded_invalid.sum()),
                "max_state_error": float(self.state_error.max()) if n else 0.0,
                "max_obs_error": float(self.obs_error.max()) if n else 0.0}


def replay(env: Any, episodes: Sequence[Episode]) -> ReplayResult:
    """Replay ``episodes`` in ``env`` (a ``VectorEnv`` built like the recording one, default autoreset), one per
    world, ``env.num_envs`` at a time. Example: see the module docstring."""
    require_vector_env(env, "replay")
    if env.autoreset != "same_step":
        raise EnvError(f"replay needs autoreset='same_step', this env has {env.autoreset!r}",
                       "build the env with the default autoreset", field="autoreset")
    snapshot = WorldSnapshot(env)
    why = unrestorable(env, snapshot.has_particles)
    if why is not None:
        raise EnvError(why, "replay rigid-body and robot scenes whose actions are joint targets; for this one "
                       "compare policies with alloy.il.evaluate", field="episodes")
    for k, ep in enumerate(episodes):
        _check(env, ep, k)
    fields = ("success", "recorded_success", "state_error", "obs_error", "ended_early", "invalid",
              "recorded_invalid")
    kinds = {f: float if f in ("state_error", "obs_error") else bool for f in fields}
    out = ReplayResult(*(np.zeros(len(episodes), kinds[f]) for f in fields))
    for batch in _batches(episodes, int(env.num_envs)):
        numbers, slots = np.array([k for k, _ in batch]), np.array([w for _, w in batch])
        part = _replay_batch(env, snapshot, [episodes[k] for k in numbers], slots)
        for f in fields:
            getattr(out, f)[numbers] = getattr(part, f)
    return out


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


def _batches(episodes: Sequence[Episode], worlds: int) -> list[list[tuple[int, int]]]:
    """Episodes sorted into batches of ``(episode number, world)``: each replays in the world that recorded it
    (per-world build randomization makes worlds differ), or in any free world when the env has fewer worlds."""
    batches: list[dict[int, int]] = []
    for k, ep in enumerate(episodes):
        home = int(ep.world) if 0 <= int(ep.world) < worlds else None
        for batch in batches:
            slot = home if home is not None else next((w for w in range(worlds) if w not in batch), None)
            if slot is not None and slot not in batch:
                batch[slot] = k
                break
        else:
            batches.append({home if home is not None else 0: k})
    return [[(k, w) for w, k in sorted(b.items())] for b in batches]


def _replay_batch(env: Any, snapshot: WorldSnapshot, batch: list[Episode], slots: np.ndarray) -> ReplayResult:
    """Replay ``batch[i]`` in world ``slots[i]``, all at once; the other worlds idle."""
    worlds, n, to_np = int(env.num_envs), len(batch), env.xp.to_numpy
    env.reset()
    mask = np.zeros(worlds, bool)
    mask[slots] = True
    _check_randomized(snapshot, batch, slots)
    snapshot.restore(mask, _stack_starts(snapshot.capture(randomized=True), batch, slots))
    _restart_layers(env)
    lengths = np.array([ep.length for ep in batch])
    result = ReplayResult(np.zeros(n, bool), np.array([ep.success for ep in batch], bool), np.zeros(n), np.zeros(n),
                          np.zeros(n, bool), np.zeros(n, bool), np.array([ep.invalid for ep in batch], bool))
    live = np.ones(n, bool)
    for t in range(int(lengths.max())):
        running = live & (t < lengths)
        _compare_states(result, snapshot, batch, slots, running, t)
        actions = _idle_actions(env)
        for i in np.flatnonzero(running):
            actions[slots[i]] = batch[i].actions[t]
        obs, _, terminated, truncated, info = env.step(env.xp.asarray(actions))
        done = (np.asarray(to_np(terminated), bool) | np.asarray(to_np(truncated), bool))[slots]
        last = live & (t == lengths - 1)
        result.success[last] = np.asarray(to_np(info["success"]), bool)[slots][last]
        if "invalid" in info:
            result.invalid |= running & np.asarray(to_np(info["invalid"]), bool)[slots]
        _compare_obs(result, env, batch, slots, running, t, obs, info)
        result.ended_early |= live & done & (t < lengths - 1)
        live &= ~done & ~last
    return result


def _idle_actions(env: Any) -> np.ndarray:
    """(W, A) actions that leave a world as it is: zeros, or in "absolute" mode the targets it holds now (a world
    whose episode is over must not be driven by that episode's last action)."""
    writer = env.writer
    zeros = np.zeros((int(env.num_envs), int(writer.size)), np.float32)
    if not _absolute(env) or not hasattr(writer, "target") or not hasattr(writer, "rows"):
        return zeros
    held = np.asarray(env.xp.to_numpy(env.xp.view(writer.target)[writer.rows]), np.float32)
    return held.reshape(zeros.shape).copy() if held.size == zeros.size else zeros


def _check_randomized(snapshot: WorldSnapshot, batch: list[Episode], slots: np.ndarray) -> None:
    """Raise when a world's randomized model values (``Scene.randomize`` on robots and gravity) are not the ones
    the episode was recorded with: the replay would run other dynamics."""
    now = snapshot.randomized()
    for ep, w in zip(batch, slots):
        for key, value in now.items():
            recorded = ep.start.get(key)
            if recorded is not None and not np.allclose(np.asarray(recorded).reshape(-1), value[w].reshape(-1),
                                                        rtol=1e-6, atol=1e-9):
                raise EnvError(
                    f"world {int(w)} has {key[len(RANDOM):]} = {value[w].reshape(-1)[:3].tolist()}, the episode was "
                    f"recorded with {np.asarray(recorded).reshape(-1)[:3].tolist()} (Scene.randomize)",
                    "replay in an env built with the recording's world count and seed; values drawn at every reset "
                    "(when='reset') cannot be put back, so compare such policies with alloy.il.evaluate",
                    field="episodes")


def _restart_layers(env: Any) -> None:
    """Observation layers (history, delay) start again from the restored state, as they did from the recording's
    reset; without layers nothing is read."""
    if getattr(env, "layers", None):
        env._observe(env.state, only=env.xp.zeros(int(env.num_envs), "bool") == 0)


def _absolute(env: Any) -> bool:
    return env.writer.spec.mode == "absolute"


def _stack_starts(current: dict[str, np.ndarray], batch: list[Episode], slots: np.ndarray
                  ) -> dict[str, np.ndarray]:
    """(W, ...) start arrays: world ``slots[i]`` gets episode i's start; the other worlds keep their state."""
    out = {k: v.copy() for k, v in current.items()}
    for w, ep in zip(slots, batch):
        for key, value in ep.start.items():
            if key in out and out[key][w].shape == np.shape(value):
                out[key][w] = value
            else:
                raise EnvError(f"the recorded start has {key} of shape {np.shape(value)}, which this env does not "
                               "have", "replay in an env built like the one that recorded (same scene and task)",
                               field=key)
    return out


def _compare_states(result: ReplayResult, snapshot: WorldSnapshot, batch: list[Episode], slots: np.ndarray,
                    active: np.ndarray, t: int) -> None:
    """Largest state difference before action ``t``, for the episodes that recorded states."""
    now = snapshot.flat(snapshot.capture())
    for i in np.flatnonzero(active):
        states = batch[i].states
        if states is not None and states.shape[1] == now.shape[1]:
            error = float(np.abs(now[slots[i]] - states[t]).max(initial=0.0))
            result.state_error[i] = max(result.state_error[i], error)


def _compare_obs(result: ReplayResult, env: Any, batch: list[Episode], slots: np.ndarray, active: np.ndarray,
                 t: int, obs: Any, info: dict[str, Any]) -> None:
    """Largest difference of the vector observations after action ``t`` (a reset world's is its ``final_obs``):
    ``"state"``, or every group of ``Task(obs={...})``. Camera images are not compared."""
    now = _vectors(env, obs)
    if "final_obs" in info:
        mask = np.asarray(env.xp.to_numpy(info["_final_obs"]), bool)
        final = _vectors(env, info["final_obs"])
        now = {k: np.where(mask.reshape(-1, *([1] * (v.ndim - 1))), final[k], v) for k, v in now.items()}
    for i in np.flatnonzero(active):
        for key, vec in now.items():
            recorded = batch[i].next_obs.get(key)
            if recorded is not None and recorded[t].shape == vec[slots[i]].shape:
                error = float(np.abs(vec[slots[i]] - recorded[t]).max(initial=0.0))
                result.obs_error[i] = max(result.obs_error[i], error)


def _vectors(env: Any, obs: Any) -> dict[str, np.ndarray]:
    """The float observations of ``obs`` as host arrays by key (a bare array is ``"state"``)."""
    parts = obs if isinstance(obs, dict) else {"state": obs}
    host = {k: np.asarray(env.xp.to_numpy(v)) for k, v in parts.items()}
    return {k: v for k, v in host.items() if v.dtype != np.uint8}


__all__ = ["ReplayResult", "replay"]
