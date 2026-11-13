# SPDX-License-Identifier: Apache-2.0
"""One recorded episode of one world, the description of a dataset, and the writer interface.

An ``Episode`` holds NumPy arrays with time first: ``T`` steps of one world. ``obs[k][t]`` is what the policy saw
before ``actions[t]``; ``next_obs[k][t]`` what it saw after (for the last step, the observation the episode ended
on, before the world was reset). Example::

    from alloy.il import Episode
    ep = recorder.episodes[0]                       # with a MemoryWriter
    print(ep.length, ep.success, ep.obs["state"].shape, ep.actions.shape)      # T, bool, (T, D), (T, A)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class DatasetSpec:
    """What every episode of one recording shares; writers get it once, in ``open``.

    Example: ``DatasetSpec(env="GripperLift-v0", task="lift the cube", fps=100.0, obs_shapes={"state": (22,)},
    obs_dtypes={"state": "float32"}, obs_names=[...], action_names=[...])``."""

    env: str                                   # the task's name (``env.task.name``)
    task: str                                  # the instruction text stored with every frame ("lift the cube")
    fps: float                                 # control steps per simulated second (1 / env.sim.dt)
    obs_shapes: dict[str, tuple[int, ...]]     # per observation key, the shape of one step ("state": (D,))
    obs_dtypes: dict[str, str]                 # per observation key, "float32" or "uint8" (camera images)
    obs_names: list[str]                       # one label per column of the main vector (``state_key``)
    action_names: list[str]                    # one label per action column
    action_mode: str = "delta"                 # the env's ActionSpec.mode: what an action value means
    state_names: list[str] = field(default_factory=list)   # one label per column of Episode.states ([] = none)
    device: str = "cpu"
    alloy_version: str = ""
    env_kwargs: dict[str, Any] = field(default_factory=dict)   # what the caller says rebuilds the env (free form)
    obs_key_names: dict[str, list[str]] = field(default_factory=dict)   # labels of every vector key the env names

    @property
    def state_key(self) -> str:
        """The main vector observation: ``"state"``, or the first group of ``Task(obs={...})`` (``"policy"``).

        Example: ``episode.obs[spec.state_key]``."""
        if "state" in self.obs_shapes or not self.obs_key_names:
            return "state"
        return next(iter(self.obs_key_names))

    @property
    def image_keys(self) -> list[str]:
        """Observation keys that are camera images (uint8, (H, W, 3)). Example: ``spec.image_keys``."""
        return [k for k, d in self.obs_dtypes.items() if d == "uint8" and len(self.obs_shapes[k]) == 3]

    @property
    def action_dim(self) -> int:
        """Action columns. Example: ``spec.action_dim``."""
        return len(self.action_names)

    def to_json(self) -> dict[str, Any]:
        """A JSON-ready dict of every field. Example: ``json.dumps(spec.to_json())``."""
        return {"env": self.env, "task": self.task, "fps": self.fps,
                "obs_shapes": {k: list(v) for k, v in self.obs_shapes.items()}, "obs_dtypes": dict(self.obs_dtypes),
                "obs_names": list(self.obs_names), "action_names": list(self.action_names),
                "action_mode": self.action_mode, "state_names": list(self.state_names), "device": self.device,
                "alloy_version": self.alloy_version, "env_kwargs": dict(self.env_kwargs),
                "obs_key_names": {k: list(v) for k, v in self.obs_key_names.items()}}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DatasetSpec:
        """The inverse of ``to_json``. Example: ``DatasetSpec.from_json(json.loads(text))``."""
        data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}     # a newer file's extra fields
        data["obs_shapes"] = {k: tuple(v) for k, v in data["obs_shapes"].items()}
        return cls(**data)


@dataclass
class Episode:
    """``T`` steps of one world, as NumPy arrays (time first). Example: see the module docstring."""

    obs: dict[str, np.ndarray]                 # key -> (T, ...): seen before each action
    next_obs: dict[str, np.ndarray]            # key -> (T, ...): seen after each action
    actions: np.ndarray                        # (T, A) float32, as the policy gave them (clipped as the env clips)
    rewards: np.ndarray                        # (T,) float32
    terminated: np.ndarray                     # (T,) bool: True on the last step when the task ended the episode
    truncated: np.ndarray                      # (T,) bool: True on the last step when max_steps (or a crash) did
    success: bool                              # the task's success check on the last step; False when ``invalid``
    states: np.ndarray | None = None           # (T, S) float32 simulator state before each action (see snapshot.py)
    start: dict[str, np.ndarray] = field(default_factory=dict)   # this world's start: what ``replay`` restores
    world: int = 0                             # the world it ran in
    index: int = -1                            # its number in the dataset (set when it is saved)
    invalid: bool = False                      # the engine marked the world INVALID (froze it) at some step

    @property
    def length(self) -> int:
        """Number of steps T. Example: ``ep.length``."""
        return int(self.actions.shape[0])

    @property
    def dones(self) -> np.ndarray:
        """(T,) bool, True on the last step only (terminated or truncated). Example: ``ep.dones[-1]``."""
        return self.terminated | self.truncated

    @property
    def episode_return(self) -> float:
        """Sum of the rewards. Example: ``ep.episode_return``."""
        return float(self.rewards.sum())


class Writer(Protocol):
    """What a ``Recorder`` writes to: ``open`` once, ``write`` per kept episode, ``close`` at the end."""

    def open(self, spec: DatasetSpec) -> None:
        """Start the dataset described by ``spec``."""

    def write(self, episode: Episode) -> None:
        """Add one finished episode."""

    def close(self) -> None:
        """Flush and close the files."""


class MemoryWriter:
    """Keeps the episodes in a list (``writer.episodes``): for tests, replay and small runs.

    Example: ``rec = Recorder(env, [MemoryWriter()])``, then ``rec.writers[0].episodes``."""

    def __init__(self) -> None:
        self.spec: DatasetSpec | None = None
        self.episodes: list[Episode] = []
        self.closed = False

    def open(self, spec: DatasetSpec) -> None:
        """Remember the dataset description. Example: ``w.open(spec)``."""
        self.spec = spec

    def write(self, episode: Episode) -> None:
        """Append the episode. Example: ``w.write(ep)``."""
        self.episodes.append(episode)

    def close(self) -> None:
        """Nothing to flush. Example: ``w.close()``."""
        self.closed = True


__all__ = ["DatasetSpec", "Episode", "MemoryWriter", "Writer"]
