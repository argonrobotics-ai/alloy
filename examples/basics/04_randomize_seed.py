# SPDX-License-Identifier: Apache-2.0
"""Per-world object properties: a free object's density, friction and start position, drawn per world at build
from a seed; same seed, same worlds.

Shows: Scene.randomize, alloy.build(seed=), bitwise-identical repeat runs, sim.reset().
Run: python examples/basics/04_randomize_seed.py
"""
import json

import numpy as np

import alloy


def make_scene() -> alloy.Scene:
    scene = alloy.Scene(worlds=8, dt=0.01)
    scene.add_ground()
    scene.add_box("crate", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.3), mass=1.0)
    scene.randomize("crate", "x", -0.05, 0.05)             # start x, metres
    scene.randomize("crate", "density", 500, 1500)        # kg/m^3 (overrides the mass split per world)
    scene.randomize("crate", "friction", 0.5, 0.9)
    return scene


def final_positions(seed: int) -> np.ndarray:
    sim = alloy.build(make_scene(), seed=seed)
    sim.run(0.5, record=[])
    return sim.state(["crate"])["crate.pos"]               # (worlds, 3)


a, b, c = final_positions(1), final_positions(1), final_positions(2)
print(json.dumps({"same_seed_identical": bool(np.array_equal(a, b)),
                  "other_seed_differs": bool(not np.allclose(a, c)),
                  "x_per_world_seed1": np.round(a[:, 0], 4).tolist()}))
