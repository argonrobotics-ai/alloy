# SPDX-License-Identifier: Apache-2.0
"""Scene variants keep large surfaces apart (alloy.scene.variants.contrast, CPU): a table and the floor under it differ
in texture family or in mean albedo by 0.15; max_albedo caps them. The pool's albedos are replaced per test, spread
so that many same-family draws are too close; no test reads a texture image.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

import alloy
from alloy.scene.variants import contrast as C
from alloy.scene.variants import looks as L
from alloy.scene.variants import LOOK_TRIES, vary


def _scene() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()                                                   # plywood_marine: wood
    s.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")
    s.add_box("a", size=(0.04, 0.04, 0.04), pos=(-0.05, 0.0, 0.42), material="wood_beech")
    s.add_box("b", size=(0.04, 0.04, 0.04), pos=(0.05, 0.0, 0.42), material="abs")
    return s


POOL: dict = {}


def _spread(albedo):
    """Patch the texture pool: every texture's albedo from ``albedo(k, n)`` (k-th of n in its family)."""
    real = L.texture_pool

    def pool(lib):
        out = {f: [dataclasses.replace(x, albedo=albedo(k, len(ts))) for k, x in enumerate(ts)]
               for f, ts in real(lib).items()}
        POOL.clear()
        POOL.update(out)
        return out
    return pool


@pytest.fixture
def albedos(monkeypatch):
    """Wood albedos from 0.10 to 0.80 (steps under 0.15); images are never read."""
    monkeypatch.setattr(L, "texture_pool", _spread(lambda k, n: 0.1 + 0.7 * k / max(1, n - 1)))
    monkeypatch.setattr(C, "texture_albedo", lambda ref, res: pytest.fail(f"read the image of {ref}"))


def _surface(v, name):
    look = v.items[name]["look"]
    return look["materials"][look["default"]]


def _ok(v, gap=0.15, top="table", floor="ground") -> bool:
    """The table and the floor under it: on a floor only the albedo counts."""
    return abs(C.albedo(_surface(v, top), POOL) - C.albedo(_surface(v, floor), POOL)) >= gap


def _family(v, name) -> str:
    return L.family_of(_surface(v, name), POOL)


def test_the_table_and_the_floor_under_it_are_the_contrast_pair(albedos, tmp_path):
    vary(_scene(), 1, out=tmp_path)
    surf = json.loads((tmp_path / "variants.json").read_text())["surfaces"]
    assert surf == {"large": ["table", "ground"], "pairs": [["table", "ground"]], "floors": ["ground"]}


def test_every_variant_keeps_the_table_apart_from_the_floor(albedos, tmp_path):
    kept = vary(_scene(), 8, seed=0, out=tmp_path)
    assert all(_ok(v) for v in kept)
    rows = json.loads((tmp_path / "variants.json").read_text())["variants"]
    assert all(r["look_rule"] is None for r in rows) and max(r["look_draws"] for r in rows) > 1
    unruled = vary(_scene(), 8, seed=0, contrast=0.0)
    assert not all(_ok(v) for v in unruled)              # without the rule some tables melt into the floor
    assert [v.to_dict() for v in vary(_scene(), 8, seed=0)] == [v.to_dict() for v in kept]


def test_the_ground_takes_floor_textures_and_the_table_keeps_wood(albedos):
    for v in vary(_scene(), 6, seed=3):
        assert _family(v, "ground") == "floor" and _family(v, "table") == "wood" and _ok(v)


def test_a_large_static_floor_under_the_table_is_a_floor_too(albedos, tmp_path):
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("deck", size=(2.0, 2.0, 0.02), pos=(0, 0, 0.01), static=True, material="wood_pine")
    s.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.22), static=True, material="wood_beech")
    s.add_box("a", size=(0.04, 0.04, 0.04), pos=(0.0, 0.0, 0.44), material="abs")
    kept = vary(s, 4, seed=0, out=tmp_path)
    surf = json.loads((tmp_path / "variants.json").read_text())["surfaces"]
    assert surf["floors"] == ["ground", "deck"] and surf["pairs"] == [["table", "deck"]]
    for v in kept:
        assert _family(v, "deck") == "floor" and _family(v, "table") == "wood" and _ok(v, floor="deck")


def test_max_albedo_caps_the_large_surfaces(albedos):
    for v in vary(_scene(), 6, seed=1, max_albedo=0.4):
        assert C.albedo(_surface(v, "table"), POOL) <= 0.4 and C.albedo(_surface(v, "ground"), POOL) <= 0.4
        assert _ok(v)


def test_the_real_pool_needs_no_images(monkeypatch):
    monkeypatch.setattr(C, "texture_albedo", lambda ref, res: pytest.fail(f"read the image of {ref}"))
    for v in vary(_scene(), 3, seed=2, max_albedo=0.5):
        assert C.albedo(_surface(v, "table"), L.texture_pool(_lib())) <= 0.5


def test_pool_albedo_metadata_needs_no_images(monkeypatch):
    monkeypatch.setattr(C, "texture_albedo", lambda ref, res: pytest.fail("read an image"))
    pool = {"wood": [L.Texture("x:dark", 0.5, "wood", "t", "1k", None, 0.1),
                     L.Texture("x:pale", 0.5, "wood", "t", "1k", None, 0.6)]}
    spec = {"preset": "wood_oak", "texture": "x:pale", "color_gain": 0.5}
    assert C.albedo(spec, pool) == pytest.approx(0.3)


def test_unknown_albedo_keeps_the_last_draw_with_a_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(L, "texture_pool", _spread(lambda k, n: None))
    monkeypatch.setattr(C, "texture_albedo", lambda ref, res: None)
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        vary(_scene(), 1, out=tmp_path)
    (row,) = json.loads((tmp_path / "variants.json").read_text())["variants"]
    assert row["look_draws"] == LOOK_TRIES and "unknown" in row["look_rule"]
    assert any("surface rule" in r.message for r in caplog.records)


def test_untextured_base_colour_albedo_is_linear_luminance():
    assert C.albedo({"preset": "plastic_matte", "base_color": [1.0, 1.0, 1.0]}) == pytest.approx(1.0)
    assert C.albedo({"preset": "plastic_matte", "base_color": [0.5, 0.5, 0.5]}) == pytest.approx(0.214, abs=1e-3)


def _lib():
    from alloy.cards.library import CardLibrary  # noqa: PLC0415

    return CardLibrary()
