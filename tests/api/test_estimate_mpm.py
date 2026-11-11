# SPDX-License-Identifier: Apache-2.0
"""``alloy.estimate``'s memory for MPM particles uses WP12's measured model (``alloy.integration.mpm_memory``), the
numbers the build refuses on: 400 B per particle was right for SPH and VBD and 15-80x low for MPM (soil_dig at 64
worlds needs about 30 GB and crashed an L4). CPU, no build (``estimate`` assembles a one-world template)."""

from __future__ import annotations

import pytest

import alloy.api as alloy
from alloy.api import _estimate
from alloy.integration.mpm_memory import BYTES_MOVING, BYTES_STATIC, predict

RIGID = {"bodies": 1, "dofs": 6, "particles": 0, "shapes": 2}


def test_memory_adds_the_mpm_model_and_keeps_400_bytes_for_other_particles():
    base = _estimate._memory(RIGID, 64)
    assert _estimate._memory({**RIGID, "particles": 1000}, 64) == base + 64 * 1000 * _estimate.BYTES_PER_PARTICLE
    mpm = {**RIGID, "particles": 1500, "mpm_particles": 1000}
    assert _estimate._memory(mpm, 64) == pytest.approx(base + 64 * 500 * 400 + predict(64 * 1000, moving=True), abs=2)
    still = {**mpm, "bodies": 0}
    assert _estimate._memory(still, 64) - _estimate._memory({**still, "mpm_particles": 0, "particles": 500}, 64) \
        == pytest.approx(predict(64 * 1000, moving=False), abs=2)


def test_estimate_of_a_sand_bed_counts_its_mpm_particles():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_object("soil", "soil_bed_dry", pos=(0.0, 0.0, 0.0))
    still = alloy.estimate(scene, worlds=4, device="cpu")
    n = still["counts"]["mpm_particles"]
    assert n > 10_000 and still["memory_bytes"] >= 4 * n * BYTES_STATIC                 # 15,360 grains a world
    scene.add_box("scoop", size=(0.03, 0.03, 0.03), pos=(0.0, 0.0, 0.2), kinematic=True)
    moving = alloy.estimate(scene, worlds=4, device="cpu")
    assert moving["memory_bytes"] >= 4 * n * BYTES_MOVING > still["memory_bytes"]
