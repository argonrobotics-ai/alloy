# SPDX-License-Identifier: Apache-2.0
"""GAPS-NOW #5a: Newton's Gauss-Seidel rheology sweep skips the worlds Alloy's per-world stop has frozen
(alloy/_vendor/patches/0006-mpm-skip-converged.patch, ``alloy.integration.mpm_worlds.SKIP_CONVERGED``). A frozen world is
restored to its converged state after the loop, so skipping its rows must leave every result bit for bit the same.

Scene: WP12's perf scene (a dough ball under a kinematic press moving down at 5 cm/s), 2 worlds on the CPU, world 1's
press 1 cm higher, so the worlds converge after different numbers of sweeps."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy._vendor.newton._src.solvers.implicit_mpm import solve_rheology as SR
from alloy.integration import mpm_worlds


def _run(periods: int = 3) -> dict[str, np.ndarray]:
    s = alloy.Scene(worlds=2, dt=0.01)
    s.add_ground()
    s.add_object("dough", "dough_ball", pos=(0, 0, 0.040))
    s.add_box("press", size=(0.12, 0.12, 0.01), pos=np.array([[0.0, 0.0, 0.079], [0.0, 0.0, 0.089]]), kinematic=True,
              material="steel_carbon")
    sim = alloy.build(s, device="cpu", seed=0)
    try:
        for _ in range(periods):
            sim.drive("press", velocity=(0, 0, -0.05), duration=sim.dt)
            sim.step(1)
        assert sim.check()["ok"]
        return {k: np.asarray(v).copy() for k, v in sim.state().items()}
    finally:
        sim.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_skipping_converged_worlds_leaves_every_result_bit_for_bit(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    calls: list[bool] = []
    skip = SR._GaussSeidelSolver.skip_converged

    def recorded(self, done):
        out = skip(self, done)
        calls.append(out if done is not None else False)
        return out
    monkeypatch.setattr(SR._GaussSeidelSolver, "skip_converged", recorded)
    skipped = _run()
    assert calls and all(calls[::2])                         # every per-world solve handed its done flags over
    monkeypatch.setattr(mpm_worlds, "SKIP_CONVERGED", False)
    swept = _run()
    assert skipped.keys() == swept.keys()
    assert not np.array_equal(skipped["dough.points"][0], skipped["dough.points"][1])   # the worlds differ
    for k in skipped:
        assert np.array_equal(skipped[k], swept[k]), k


def test_the_reordered_and_single_world_sweeps_do_not_skip():
    assert SR._GaussSeidelSolver.skips_converged and not SR._ReorderedGaussSeidelSolver.skips_converged
    assert not hasattr(SR._BatchedGaussSeidelSolver, "skip_converged")
