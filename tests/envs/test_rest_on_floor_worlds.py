# SPDX-License-Identifier: Apache-2.0
"""``rest_on_floor`` rests the object on the floor in every world, each by its own shift (audit T2-013).

It measured world 0 only and moved every world by that amount, so an object started at another height in another
world (``pos=(W, 3)``, or ``scene.randomize(name, "z", ...)``) began buried in the floor or floating above it.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs.placement import lowest_z, rest_on_floor

HEIGHTS = (0.03, 0.01, 0.11)             # cube centres: resting (0.03), 2 cm in the floor, 8 cm above it


@pytest.mark.timeout(300)
def test_each_world_rests_on_the_floor():
    scene = alloy.Scene(worlds=3, dt=0.02)
    scene.add_ground()
    scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=np.array([[0.0, 0.0, z] for z in HEIGHTS]), mass=0.2)
    sim = alloy.build(scene, device="cpu", seed=0)
    shift = rest_on_floor(sim, ["cube"])["cube"]
    low = np.array([lowest_z(sim, "cube", world=w) for w in range(3)])
    assert np.allclose(low, 0.0005, atol=1e-6)                           # gap 0.5 mm above the floor in every world
    assert np.allclose(shift, 0.0005 - (np.array(HEIGHTS) - 0.03), atol=1e-6)
    sim.reset()
    assert np.allclose([lowest_z(sim, "cube", world=w) for w in range(3)], 0.0005, atol=1e-6)   # the new start


@pytest.mark.timeout(300)
def test_identical_starts_get_one_shift_as_before():
    scene = alloy.Scene(worlds=3, dt=0.02)
    scene.add_ground()
    scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.08), mass=0.2)
    sim = alloy.build(scene, device="cpu", seed=0)
    before = 0.0005 - lowest_z(sim, "cube")                               # what the one shared shift was
    shift = rest_on_floor(sim, ["cube"])["cube"]
    assert shift.shape == (3,) and np.allclose(shift, before, atol=1e-9)
