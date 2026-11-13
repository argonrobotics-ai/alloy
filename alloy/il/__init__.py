# SPDX-License-Identifier: Apache-2.0
"""Imitation learning on Alloy: record demonstrations from a ``VectorEnv``, write them as datasets, replay them
and evaluate policies.

- ``Recorder``: wraps a ``VectorEnv`` and cuts what you step into episodes, one per world at a time.
- Writers: ``HDF5Writer`` (the robomimic file layout; needs ``h5py``), ``LeRobotWriter`` (LeRobotDataset v3.0
  through the ``lerobot`` package), ``MemoryWriter`` (a list).
- ``replay``: restore each recorded start, apply the recorded actions, compare. ``evaluate``: a policy's success
  rate with a 95 % interval.
- ``generate``: run a scripted policy (``PhaseScript``, ``demonstrator(env)``) in every world until enough
  episodes are saved.
- Teleoperation: ``KeyboardTeleop`` (reads the terminal, works over ssh), ``SpaceMouseTeleop`` (needs ``hidapi``),
  ``teleop_loop``.

Example::

    import alloy.envs
    from alloy.il import Recorder, HDF5Writer, load_hdf5, replay

    env = alloy.envs.make("GripperLift-v0", num_envs=8)
    with Recorder(env, [HDF5Writer("lift.hdf5")], task="lift the cube") as rec:
        obs, _ = rec.reset(seed=0)
        while rec.saved < 16:
            obs, *_ = rec.step(policy(obs))
    spec, episodes = load_hdf5("lift.hdf5")
    print(replay(env, episodes).summary())

Guide: ``docs/guide/imitation.md``. Runnable examples: ``examples/il/``.
"""

from __future__ import annotations

from alloy.il.episode import DatasetSpec, Episode, MemoryWriter, Writer
from alloy.il.evaluate import EvalResult, evaluate, wilson_interval
from alloy.il.generate import DEMONSTRATORS, GenerateResult, demonstrator, generate
from alloy.il.hdf5 import HDF5Writer, copy_hdf5, load_hdf5
from alloy.il.lerobot import LeRobotWriter
from alloy.il.recorder import Recorder
from alloy.il.replay import ReplayResult, replay
from alloy.il.script import Phase, PhaseScript
from alloy.il.snapshot import WorldSnapshot
from alloy.il.spacemouse import SpaceMouseTeleop
from alloy.il.teleop import KeyboardTeleop, ScriptKeys, TeleopCommand, TerminalKeys, teleop_loop

__all__ = ["DEMONSTRATORS", "DatasetSpec", "Episode", "EvalResult", "GenerateResult", "HDF5Writer", "KeyboardTeleop",
           "LeRobotWriter", "MemoryWriter", "Phase", "PhaseScript", "Recorder", "ReplayResult", "ScriptKeys",
           "SpaceMouseTeleop", "TeleopCommand", "TerminalKeys", "WorldSnapshot", "Writer", "demonstrator", "evaluate",
           "copy_hdf5", "generate", "load_hdf5", "replay", "teleop_loop", "wilson_interval"]
