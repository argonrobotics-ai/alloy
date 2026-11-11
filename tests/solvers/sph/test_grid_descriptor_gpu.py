# SPDX-License-Identifier: Apache-2.0
"""SPH neighbour grids keep their own descriptor across CUDA graph replays (alloy.solvers.sph.grid_descriptor), GPU.

Warp's device HashGrid build copies its descriptor from one static host struct shared by every float32 grid, and a
graph replays that copy from the static's address. Each replay therefore installs the last-built grid's descriptor
(OSS-4's find, 2026-11-08). Before the guard, on an L4:
- two identical pour sims stepped in turn differed by up to 5.7e-5 m in 696 of 896 particles;
- a float32 grid built between steps (a camera) sent the build kernel out of bounds.
- Two identical pour sims stepped in turn, 4 periods each, three times: bitwise equal.
- A sim stepped with a foreign float32 HashGrid rebuilt between its periods equals a twin stepped without it, bitwise.
- The guard's layout check passes on this warp (the descriptor offsets are warp 1.18's).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA")]


def _pour() -> alloy.Sim:
    sc = alloy.Scene(dt=0.01, worlds=2)
    sc.add_ground()
    sc.add_open_cylinder("cup", inner_radius=0.03, depth=0.06, wall=0.004)
    sc.add_pour("water", 20e-6, pos=(0.0, 0.0, 0.12), rate=1e-4)
    return alloy.build(sc, device="cuda:0", seed=0)


def _q(sim) -> np.ndarray:
    return sim.engine.state.particle_q.numpy().copy()


def test_two_pour_sims_stepped_in_turn_stay_bitwise_equal():
    a, b = _pour(), _pour()
    for _ in range(3):
        a.step(4)
        b.step(4)
    qa, qb = _q(a), _q(b)
    assert np.array_equal(qa, qb), (int((qa != qb).any(axis=1).sum()), float(np.abs(qa - qb).max()))
    from alloy.api._liquid import liquid_solvers

    assert all(s._nbrs.desc.ready for s in liquid_solvers(a.engine))


def test_a_foreign_float32_grid_between_periods_changes_nothing():
    twin, sim = _pour(), _pour()
    other = wp.HashGrid(32, 32, 32, device="cuda:0")
    pts = wp.array(np.random.default_rng(0).uniform(-1.0, 1.0, (500, 3)).astype(np.float32), dtype=wp.vec3,
                   device="cuda:0")
    twin.step(12)                                   # alone first: only the foreign grid can differ below
    for _ in range(12):
        sim.step(1)
        other.build(pts, 0.05)
    assert np.array_equal(_q(sim), _q(twin)), float(np.abs(_q(sim) - _q(twin)).max())
