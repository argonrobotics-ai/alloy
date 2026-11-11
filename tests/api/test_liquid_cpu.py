# SPDX-License-Identifier: Apache-2.0
"""Liquid at rest on CPU: a cylinder of liquid in a round cup stays put; argument checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api.errors import SceneError

R, H, WALL = 0.02, 0.02, 0.004


def test_cylinder_liquid_arguments():
    scene = alloy.Scene()
    scene.add_liquid("water", size=(2 * R, 2 * R, H), pos=(0, 0, WALL + H / 2), shape="cylinder")
    item = scene.items["water"]
    assert item["shape"]["primitive"] == {"kind": "cylinder", "size": [R, H / 2]}
    assert item["volume"] == pytest.approx(math.pi * R * R * H)
    with pytest.raises(SceneError, match="x == y"):
        scene.add_liquid("oval", size=(0.04, 0.03, H), shape="cylinder")
    with pytest.raises(SceneError, match="shape"):
        scene.add_liquid("blob", size=(0.04, 0.04, H), shape="sphere")
    again = alloy.Scene.from_dict(scene.to_dict())
    assert again.items["water"]["shape"] == item["shape"]


@pytest.mark.compile_heavy
def test_cylinder_of_liquid_rests_in_a_round_cup():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_cylinder("cup", inner_radius=R + 0.001, depth=0.04, wall=WALL)
    scene.add_liquid("water", size=(2 * R, 2 * R, H), pos=(0, 0, WALL + H / 2 + 0.0005), shape="cylinder")
    sim = alloy.build(scene, device="cpu")
    assert sim.routes()["water"]["route"] == "liquid_sph"
    assert sim.liquid_volume("water") == pytest.approx(math.pi * R * R * H, rel=0.15)
    sim.run(0.4)
    p = sim.state(["water"], world=0)["water.points"]
    assert np.hypot(p[:, 0], p[:, 1]).max() < R + 0.001 + 0.003 and p[:, 2].max() < WALL + 0.04
    assert sim.check()["ok"] and sim.health()["retired_particles"] == [0]
    sim.close()
