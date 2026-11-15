# SPDX-License-Identifier: Apache-2.0
"""``scene.randomize(name, "x" | "y" | "z", lo, hi)`` on objects whose particles move (Gemini audit S1-025).

Before, a soft body, an MPM box and the library's sponge, towel, water and rice cards failed the build with
"pose randomization needs a free root joint", though ``pos=(worlds, 3)`` worked on every one of them. Now the draw
shifts the object's particles in each world, keeping the documented meaning: the drawn value replaces that start
coordinate of the object's placement.

Laws (CPU):
- soft box and MPM box (placed by their centre): each world's particle centroid sits at its draw (x), the other two
  coordinates where they were placed;
- the four library cards build, and world w's particles are world 0's shifted by draw[w] - draw[0];
- no compounding: the build draws at the assembly seed and again at the build seed, and the centroid is at the build
  seed's draw; a world redrawn at another seed and then reset is back at its draw, the other world bit for bit as it
  was;
- a rigid free body still takes the draw as its free joint's coordinate, exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy.api as alloy

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]

W, LO, HI, SEED = 2, -0.15, 0.15, 7
FIELD = "pose.0"


def _scene(add) -> alloy.Scene:
    scene = alloy.Scene(worlds=W, dt=0.01, cloth_accuracy="fast")
    scene.add_ground()
    add(scene)
    scene.randomize("obj", "x", LO, HI)
    return scene


def _points(sim: alloy.Sim) -> np.ndarray:
    st = sim.state()
    return np.asarray(st["obj.points"] if "obj.points" in st else st["obj.particles"], np.float64)


def _draws(sim: alloy.Sim) -> np.ndarray:
    return np.asarray(sim.engine.built.randomized[("obj", FIELD)], np.float64)


PRIMITIVES = {
    "soft_box": lambda s: s.add_box("obj", size=(0.05, 0.05, 0.03), pos=(0.0, 0.02, 0.08), material="pu_foam_sponge",
                                    rigid=False),
    "mpm_box": lambda s: s.add_box("obj", size=(0.04, 0.04, 0.03), pos=(0.0, 0.02, 0.08), material="sand_silica_dry"),
}
CARDS = ["sponge_kitchen", "towel_cotton_hand", "water_box_200ml", "rice_cup_fill"]


@pytest.mark.parametrize("kind", list(PRIMITIVES))
def test_each_worlds_centroid_lands_at_its_draw(kind):
    sim = alloy.build(_scene(PRIMITIVES[kind]), device="cpu", seed=SEED)
    draws, centroid = _draws(sim), _points(sim).mean(axis=1)
    assert np.all((draws >= LO) & (draws < HI)) and abs(draws[0] - draws[1]) > 1e-3
    np.testing.assert_allclose(centroid[:, 0], draws, atol=2e-5)
    np.testing.assert_allclose(centroid[:, 1:], [[0.02, 0.08]] * W, atol=2e-5)
    sim.close()


@pytest.mark.parametrize("card", CARDS)
def test_library_cards_build_and_shift_by_their_draws(card):
    sim = alloy.build(_scene(lambda s: s.add_object("obj", card, pos=(0.0, 0.0, 0.08))), device="cpu", seed=SEED)
    draws, pts = _draws(sim), _points(sim)
    np.testing.assert_allclose(pts[1] - pts[0], np.broadcast_to([draws[1] - draws[0], 0.0, 0.0], pts[0].shape),
                               atol=2e-5)
    sim.close()


def test_a_redraw_does_not_compound_and_a_reset_brings_the_draw_back():
    from alloy.scene.randomize import apply_randomization

    sim = alloy.build(_scene(PRIMITIVES["soft_box"]), device="cpu", seed=SEED)   # draws at 0, then again at 7
    draws = _draws(sim).copy()
    np.testing.assert_allclose(_points(sim).mean(axis=1)[:, 0], draws, atol=2e-5)
    sim.step(2)
    apply_randomization(sim.engine.built, 11, wp.array([1], dtype=wp.int32))       # world 1 drawn at another seed
    before = _points(sim)
    m = np.zeros(W, np.int32)
    m[1] = 1
    sim.engine.reset(wp.array(m, dtype=wp.int32, device="cpu"))
    after = _points(sim)
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_allclose(after[1].mean(axis=0)[0], draws[1], atol=2e-5)
    np.testing.assert_array_equal(_draws(sim), draws)
    sim.close()


def test_a_rigid_body_takes_the_draw_on_its_free_joint_as_before():
    sim = alloy.build(_scene(lambda s: s.add_box("obj", size=(0.05,) * 3, pos=(0.0, 0.02, 0.08), mass=0.2)),
                      device="cpu", seed=SEED)
    x = sim.state()["obj.pos"][:, 0]
    np.testing.assert_array_equal(x, _draws(sim).astype(x.dtype))
    sim.close()
