# SPDX-License-Identifier: Apache-2.0
"""A cloth sheet placed metres from the origin builds: the mass readback's expected value uses the builder's rest
areas, not areas from float32 positions (a 0.24 x 0.12 m sheet of 13 x 7 cells at x = 3 m failed the 1e-6 readback
with 0.02999997 against 0.03 kg; 2026-11-14)."""

from __future__ import annotations

import numpy as np

import alloy


def test_a_sheet_far_from_the_origin_builds_and_keeps_its_mass() -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    for name, x in (("near", 0.0), ("three", 3.0), ("ten", 10.0)):
        scene.add_cloth(name, size=(0.24, 0.12), pos=(x, 0.0, 0.01), resolution=(13, 7), mass=0.03)
    sim = alloy.build(scene, device="cpu")                  # raised BuildError (ReadbackError) before the fix
    model = sim.engine.model
    for name in ("near", "three", "ten"):
        assert sim.state(world=0)[f"{name}.points"].shape == (14 * 8, 3)
    total = float(np.asarray(model.particle_mass.numpy(), np.float64).sum())
    assert abs(total - 0.09) <= 1e-6 * 0.09, total
