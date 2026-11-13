# SPDX-License-Identifier: Apache-2.0
"""A drive moves a kinematic body for exactly its time (API-17, AUDIT-STACK; 0.1.1).

The drive kernel writes only the state a physics step ends in. Before: a new drive's first step still read the old
velocity (a lag of one step), and at the end its float32 clock fell just short of ``t1``, so the last step wrote the
velocity once more and the next step read it (an overrun of one step): on WP7's press-and-hold the hold force read
3.2 % higher than after ``stop_drives`` (2.867 N against 2.776 N). Now the start velocity is written into both states
when the drive is made (``Drive.prime``) and the end is tested in float64, the step ending nearest ``t1`` ending at
rest. Laws (CPU): both states move at the drive's velocity before the first step; a drive whose duration is whole
periods ends at rest in the state its last period ends in; a press that ends by its duration then holds the material
bit for bit as one stopped with ``stop_drives`` at the same time.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy


def _press_scene() -> alloy.Sim:
    scene = alloy.Scene(dt=0.01, worlds=2)
    scene.add_ground()
    scene.add_box("dough", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.01), material="dough_wheat")
    scene.add_sphere("tool", 0.008, pos=(0, 0, 0.0285), kinematic=True)          # 0.5 mm over the dough
    return alloy.build(scene, device="cpu")


def _vz(sim: alloy.Sim, st) -> np.ndarray:
    return st.body_qd.numpy()[sim.index.per_world(sim.index.body("tool"), "body")][:, 2]


@pytest.mark.compile_heavy
def test_a_drive_moves_from_its_first_step_and_ends_at_rest_on_its_last():
    sim = _press_scene()
    sim.drive("tool", velocity=(0, 0, -0.02), duration=0.05)
    for st in (sim.engine.state, sim.engine._next):
        assert np.allclose(_vz(sim, st), -0.02)              # read by the first step: it was 0 (one step late)
    sim.step(5)                                              # the whole duration
    assert not _vz(sim, sim.engine.state).any()              # was -0.02: the last step wrote the velocity again
    z = float(sim.state(["tool"], world=0)["tool.pos"][2])
    assert z == pytest.approx(0.0285 - 0.001, abs=1e-6)      # 2 cm/s for 50 ms
    sim.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_a_press_ended_by_its_duration_holds_like_one_stopped_at_that_time():
    held = []
    for stop in (True, False):
        sim = _press_scene()
        sim.drive("tool", velocity=(0, 0, -0.02), duration=float("inf") if stop else 0.1)
        sim.step(10)                                         # 2 mm: the 0.5 mm gap, then 1.5 mm into the dough
        if stop:
            sim.stop_drives()
        held.append(sim.run(0.1, record=["dough", "tool"]))
        sim.close()
    for key in ("dough.points", "tool.pos"):
        assert np.array_equal(held[0][key], held[1][key]), key   # differed by the duration's extra step before
