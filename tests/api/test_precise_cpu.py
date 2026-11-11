# SPDX-License-Identifier: Apache-2.0
"""A peg dropped into add_block_with_hole lands well under 1 mm deep (T3 of the fresh-agent tasks), on CPU."""

from __future__ import annotations

import pytest

import alloy.api as alloy


@pytest.mark.compile_heavy
def test_peg_lands_in_a_precise_hole_under_half_a_millimetre():
    scene = alloy.Scene(dt=0.001)
    scene.add_ground()
    scene.add_block_with_hole("block", size=(0.10, 0.10, 0.05), hole=(0.0102, 0.0102), depth=0.03)
    scene.add_box("peg", size=(0.01, 0.01, 0.04), pos=(0, 0, 0.075), mass=0.05)
    sim = alloy.build(scene, device="cpu")
    assert sim.dt == pytest.approx(0.001)                      # the user's period; the engine steps 0.25 ms inside
    z = sim.run(0.15, record=["peg"]).world(0)["peg.pos"][:, 2]   # lands on the hole floor at ~0.085 s, ~0.83 m/s
    assert sim.physics_steps == 600
    floor_depth = 0.02 + 0.02 - z.min()                         # hole floor at z 0.02; peg half-length 0.02
    assert 0.0 < floor_depth < 0.0005 and sim.check()["ok"]
    sim.close()
