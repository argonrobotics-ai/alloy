# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T1-019: the task ports take their seed, and an autoreset does not replay the episode before.

The five Concrete*-v0 ports built their task without the seed (every seed ran seed 0), and ``PortEnv`` restarted a
finished run with the same seed, so ``autoreset="same_step"`` replayed the first episode for ever. Pure checks: the
port's run function is called with its module's ``run`` replaced, and the restart is run on stand-in slots."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs.ports import PORTS
from alloy.envs.ports.env import PortEnv

CONCRETE = {"ConcretePadEdge-v0": "pad_edge", "ConcretePatchFlush-v0": "concrete_patch",
            "ConcreteToolJoints-v0": "concrete_joints", "ConcreteToolSealant-v0": "concrete_sealant",
            "ConcretePatchDrywall-v0": "concrete_drywall"}


@pytest.mark.parametrize(("name", "module"), sorted(CONCRETE.items()))
def test_concrete_ports_pass_their_seed_to_the_task(name, module, monkeypatch):
    mod = importlib.import_module(f"alloy.tasks.{module}")
    seen = []
    monkeypatch.setattr(mod, "run", lambda task, worlds, device: seen.append(task))
    PORTS[name].run(worlds=2, device="cpu", seed=42)
    PORTS[name].run(worlds=2, device="cpu", seed=0)
    assert [t.seed for t in seen] == [42, 0]                                  # was: the default seed both times
    assert seen[1] == type(seen[1])()                                         # seed 0: the port's default task


def test_an_autoreset_moves_each_finished_runs_seed_on():
    slots = [SimpleNamespace(worlds=1, seed=7 + k, starts=0) for k in range(3)]
    for s in slots:
        s.start = lambda s=s: setattr(s, "starts", s.starts + 1)
    env = SimpleNamespace(_slots=slots, num_envs=3)
    PortEnv._restart(env, np.array([True, False, True]))
    assert [s.seed for s in slots] == [10, 8, 12] and [s.starts for s in slots] == [1, 0, 1]   # was: 7, 8, 9
    PortEnv._restart(env, np.array([True, True, True]))
    assert [s.seed for s in slots] == [13, 11, 15]                            # no seed comes back
