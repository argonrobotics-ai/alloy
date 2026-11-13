# SPDX-License-Identifier: Apache-2.0
"""A free-base robot starts the same whichever way ``pos`` is given (GAP-E1's finding, physics' ruling; 0.1.1).

A ``pos`` tuple moves the whole robot file, so the root body ends at ``pos`` plus where the file places it. A per-world
``pos`` array, and a tuple whose ``x`` / ``y`` / ``z`` ``scene.randomize`` draws, put the root body itself at the row:
a root body 0.3 m up in its file started in the ground (START_PEN, PEN_FAIL). Now every form moves the whole file.
Laws (CPU): a per-world array equal to the broadcast tuple gives the same start, bit for bit; each row moves the root
by that row; a randomized coordinate keeps the file's height; ``sim.reset()`` returns to those starts.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy

BOX = """<mujoco><worldbody><body name="box" pos="0 0 0.3"><freejoint/>
  <geom type="box" size="0.05 0.04 0.03" mass="0.1"/></body></worldbody></mujoco>"""


def _start(pos, randomize: bool = False) -> np.ndarray:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_robot("c", mjcf=BOX, pos=pos)
    if randomize:
        scene.randomize("c", "x", 0.2, 0.3)
    sim = alloy.build(scene, device="cpu", seed=0)
    body = sim.index.per_world(sim.index.body("c/box"), "body")
    q = sim.engine.state.body_q.numpy()[body].copy()
    sim.step(2)
    sim.reset()
    assert np.array_equal(sim.engine.state.body_q.numpy()[body], q)        # reset returns to the start
    sim.close()
    return q


@pytest.mark.compile_heavy
def test_a_per_world_array_starts_like_the_tuple_it_repeats():
    tup = _start((0.0, 0.0, 0.1))
    arr = _start(np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]))
    assert np.allclose(tup[:, 2], 0.4, atol=1e-6)                         # 0.1 plus the file's 0.3
    assert np.array_equal(arr, tup)                                       # was z = 0.1: the root put at the row


@pytest.mark.compile_heavy
def test_each_row_moves_the_whole_file_and_a_drawn_coordinate_keeps_its_height():
    arr = _start(np.array([[0.0, 0.0, 0.1], [0.5, 0.0, 0.2]]))
    assert np.allclose(arr[:, :3], [[0.0, 0.0, 0.4], [0.5, 0.0, 0.5]], atol=1e-6)
    drawn = _start((0.0, 0.0, 0.0), randomize=True)
    assert np.allclose(drawn[:, 2], 0.3, atol=1e-6)                       # was 0: in the ground
    assert np.all((drawn[:, 0] >= 0.2) & (drawn[:, 0] <= 0.3))
