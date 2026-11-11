# SPDX-License-Identifier: Apache-2.0
"""MPM rheology's smoothing grid keeps its own descriptor across CUDA graph replays (sph/grid_descriptor.py), GPU.

SolverRheologyMPM rebuilds a float32 warp HashGrid every step (the strain-rate smoothing). Warp's device build copies
the grid descriptor from one static host struct shared by every float32 grid, and a CUDA graph replays that copy
from the static's address. Without the guard, two paste sims stepped in turn swap grids: each replay installs the
descriptor of whichever grid the host built last (OSS-4's find for SPH, 2026-11-08). A yogurt paste (Herschel-Bulkley,
paste_mpm on SolverRheologyMPM with smoothing), 2 worlds:
- after interleaved steps each sim's grid holds its own descriptor (the one its first eager build wrote) and not the
  other sim's;
- two sims stepped in turn end bitwise equal (the same scene and seed).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA")]


def _paste() -> alloy.Sim:
    sc = alloy.Scene(dt=0.01, worlds=2)
    sc.add_ground()
    sc.add_open_box("tray", inner=(0.06, 0.06, 0.03), wall=0.004)
    sc.add_liquid("paste", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.016), material="yogurt_plain")
    sim = alloy.build(sc, device="cuda:0", seed=0)
    assert type(sim.engine.solvers[0]).__name__ == "SolverRheologyMPM"
    return sim


def _desc(sim) -> np.ndarray:
    return sim.engine.solvers[0]._grid_desc.view.numpy().copy()


def test_each_paste_sims_grid_keeps_its_own_descriptor_under_replay():
    a, b = _paste(), _paste()
    a.step(1)
    b.step(1)
    own = (_desc(a), _desc(b))                      # the first (eager) builds: each grid's own descriptor
    assert not np.array_equal(*own)                 # two grids, two sets of buffers
    for _ in range(3):
        a.step(4)
        b.step(4)
    assert np.array_equal(_desc(a), own[0]) and np.array_equal(_desc(b), own[1])


def test_two_paste_sims_stepped_in_turn_stay_bitwise_equal():
    a, b = _paste(), _paste()
    for _ in range(3):
        a.step(4)
        b.step(4)
    qa, qb = a.engine.state.particle_q.numpy(), b.engine.state.particle_q.numpy()
    assert np.array_equal(qa, qb), float(np.abs(qa - qb).max())
