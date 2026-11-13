# SPDX-License-Identifier: Apache-2.0
"""alloy vary keeps fills with their containers and particle items on their supports
(``alloy.scene.variants.base.particle_links``), AUDIT S2-030 (Gemini 0.1.2): vary moved a free mug away from the
water in it (composed fills: 0.186 m apart, health and check --quick passing; add_liquid and add_pour stayed where
they were), and pushed an 8 cm cloth off a 16 cm pedestal (up to 0.21 m) without a redraw. The laws: a liquid or a
pour in or over a free container moves with it (one group, the same move); a fill of a static container stays; a
cloth resting on a still support keeps its centre over it."""

from __future__ import annotations

import numpy as np

import alloy
from alloy.scene.variants import vary


def _xy(item) -> np.ndarray:
    return np.asarray(item["pose"][:2], np.float64)


def _cup_scene(static_cup: bool = False) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_box("bin", inner=(0.08, 0.08, 0.05), wall=0.005, pos=(0.3, 0.0, 0.0))          # static: its fill stays
    s.add_liquid("still", size=(0.06, 0.06, 0.02), pos=(0.3, 0.0, 0.015), material="water")
    s.add_object("mug", "mug_stoneware", pos=(0.0, 0.0, 0.0005), static=static_cup)
    s.add_liquid("water", size=(0.05, 0.05, 0.03), pos=(0.0, 0.0, 0.03), material="water", shape="cylinder")
    s.add_pour("stream", 20e-6, pos=(0.0, 0.0, 0.16))
    return s


def test_a_fill_and_a_pour_move_with_their_free_mug_and_a_static_fill_stays():
    """Two variants (CPU): water and the pour's nozzle move exactly as the mug; the bin's water stays. ~30 s."""
    base = _cup_scene()
    for v in vary(base, 2, pos=0.12, yaw_deg=0.0, textures="keep", seed=1):
        move = _xy(v.items["mug"]) - _xy(base.items["mug"])
        assert np.linalg.norm(move) > 1e-3
        assert np.allclose(_xy(v.items["water"]) - _xy(base.items["water"]), move, atol=1e-9)
        assert np.allclose(np.subtract(v.items["stream"]["pour"]["pos"][:2], base.items["stream"]["pour"]["pos"][:2]),
                           move, atol=1e-9)
        assert np.array_equal(_xy(v.items["still"]), _xy(base.items["still"]))


def test_a_fill_of_a_static_cup_stays():
    base = _cup_scene(static_cup=True)
    (v,) = vary(base, 1, pos=0.12, textures="keep", seed=1)
    for name in ("water", "still"):
        assert np.array_equal(_xy(v.items[name]), _xy(base.items[name]))
    assert v.items["stream"]["pour"]["pos"] == base.items["stream"]["pour"]["pos"]


def test_a_cloth_keeps_its_centre_over_a_still_pedestal():
    """The audit's pedestal (16 cm, top at 0.30 m) and 8 cm cloth, vary pos 0.25: every variant keeps the cloth's
    centre over the pedestal (before: 3 of 4 at 0.195-0.21 m, in the air). ~10 s."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("pedestal", size=(0.16, 0.16, 0.30), pos=(0.0, 0.0, 0.15), static=True)
    s.add_cloth("cloth", size=(0.08, 0.08), pos=(0.0, 0.0, 0.303), resolution=8, material="cotton_woven")
    moved = [np.abs(_xy(v.items["cloth"])).max() for v in vary(s, 4, pos=0.25, textures="keep", seed=0)]
    assert max(moved) <= 0.08 and max(moved) > 0.0, moved


def test_a_composed_fill_card_moves_with_its_mug(tmp_path):
    """A fill from a card (compose's "a mug with 80 ml of water": the water is a card on a liquid route, not an
    add_liquid item) groups with its mug and takes its move. ~20 s."""
    import alloy.compose
    from alloy.compose.catalog import Catalog

    alloy.compose.compose("a mug with 80 ml of water on a table", check="load", catalog=Catalog(imports=False),
                          device="cpu").save(tmp_path)
    base = alloy.Scene.load(str(tmp_path / "scenes" / "scene.yaml"))
    (v,) = vary(base, 1, pos=0.12, yaw_deg=0.0, textures="keep", seed=1)
    mug = [n for n in v.items if n.startswith("mug")][0]
    water = [n for n in v.items if "water" in n][0]
    pose = lambda s, n: np.asarray(s.items[n]["raw_placement"]["pose"][:2], np.float64)      # noqa: E731
    move = pose(v, mug) - pose(base, mug)
    assert np.linalg.norm(move) > 1e-3 and np.allclose(pose(v, water) - pose(base, water), move, atol=1e-9)
