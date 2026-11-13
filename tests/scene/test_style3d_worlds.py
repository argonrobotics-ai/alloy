# SPDX-License-Identifier: Apache-2.0
"""The Style3D route in more than one world (SCR-WP0-S3D-WORLDS, vendored patches 0002 + 0003): its collision and its
PCG keep each world to itself, so 2+ worlds build and stay healthy. Before the patches every world went INVALID at the
first step ([internal note] notes/wp2-fabric/style3d_worlds) and the build refused the route with worlds > 1."""

from __future__ import annotations

import alloy


def _sheet(worlds: int) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_cloth("napkin", size=(0.1, 0.1), pos=(0.0, 0.0, 0.05), resolution=5, material="cotton_woven_aniso")
    scene.items["napkin"]["card_extra"]["route"] = {"route": "cloth_style3d", "reason": "test: the Style3D solver"}
    return scene


def test_one_world_builds_on_style3d():
    sim = alloy.build(_sheet(1), device="cpu")
    assert sim.routes()["napkin"]["route"] == "cloth_style3d"
    sim.close()


def test_two_worlds_build_and_stay_healthy():
    sim = alloy.build(_sheet(2), device="cpu")
    try:
        assert sim.routes()["napkin"]["route"] == "cloth_style3d"
        sim.step(3)
        assert sim.health()["invalid_worlds"] == [] and sim.check()["ok"], sim.health()
    finally:
        sim.close()
