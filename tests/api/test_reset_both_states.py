# SPDX-License-Identifier: Apache-2.0
"""``sim.reset()`` leaves both of the engine's states at the start (physics' ruling, the API-16 class).

The engine swaps ``state`` and ``_next`` every exchange, and the reset restored ``state`` alone. In an MPM-only scene no
solver writes a kinematic body, so the tool's row from before the reset came back every second exchange: the dough
under a tool that had been driven and stopped moved up to 6.6 um apart from a fresh build's within 10 ms. Law (CPU):
after drive -> stop -> reset the next 20 periods equal a fresh build's first 20 bit for bit (also with the tool's start
randomized per world), and both states agree after the reset. VectorEnv's per-world reset (``engine.reset(mask)``)
refuses an MPM scene (its solver's arrays have no per-world axis: ``SnapshotError``), so it has no such path here.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy


def _sim(randomized: bool) -> alloy.Sim:
    scene = alloy.Scene(dt=0.01, worlds=2)
    scene.add_ground()
    scene.add_box("dough", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.01), material="dough_wheat")
    scene.add_sphere("tool", 0.008, pos=(0, 0, 0.03), kinematic=True)
    if randomized:
        scene.randomize("tool", "x", -0.005, 0.005)
    return alloy.build(scene, device="cpu", seed=0)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
@pytest.mark.parametrize("randomized", [False, True], ids=["fixed_start", "randomized_start"])
def test_after_a_drive_and_a_reset_the_run_repeats_a_fresh_build(randomized):
    fresh = _sim(randomized)
    ref = fresh.run(0.2, record=["dough", "tool"])
    fresh.close()
    sim = _sim(randomized)
    sim.drive("tool", velocity=(0, 0, -0.05))
    sim.step(10)                                             # 5 mm down: 3 mm into the dough
    sim.stop_drives()
    sim.reset()
    eng = sim.engine
    for key in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
        assert np.array_equal(getattr(eng.state, key).numpy(), getattr(eng._next, key).numpy()), key
    got = sim.run(0.2, record=["dough", "tool"])
    for key in ("dough.points", "tool.pos"):
        assert np.array_equal(got[key], ref[key]), key       # was up to 6.6 um apart on the dough
    sim.close()
