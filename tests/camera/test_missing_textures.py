# SPDX-License-Identifier: Apache-2.0
"""Laws: a texture that is not on disk never stops a render or turns a surface white, and one fetched later is drawn.

- The full-material tracer draws a library card's look (``card_looks.fallback_rules``) with its texture when it is
  cached, else in its family's flat tone, with the one warning naming ``python -m alloy.looks --fetch-textures``
  (Gemini audit S2-028: every snapshot of a liquid scene, which uses that tracer, raised CameraError on a fresh
  install when a card such as ``table_top_plywood`` named a texture; the error's fix named internal calls).
- The variants contact sheet draws a missing texture in the same flat tone as ``sim.render()`` (S2-026: it kept the
  white placeholder colour of textured presets, 212 of 255 against 171 on the render).
- The item-look layer remembers which textures it found missing and draws them once they are on disk, in the same
  process (S2-026: it kept the flat tones until its memo was cleared).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

import alloy
import alloy.camera.item_looks as IL
from alloy.camera.item_looks import FAMILY_TONE
from tests.camera.helpers import fake_texture

pytestmark = pytest.mark.timeout(600)


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    """A fresh install: no render texture cache, nothing remembered from one."""
    monkeypatch.setenv("ALLOY_RENDER_ASSETS", str(tmp_path))
    monkeypatch.setattr(IL, "_MEMO", {})
    return tmp_path


def _tones(materials: dict) -> dict:
    return {k: tuple(np.round(m["base_color"], 3)) for k, m in materials.items() if m.get("texture") is None
            and "base_color" in m}


def test_the_full_tracer_draws_a_cards_missing_texture_in_its_familys_tone(empty_cache, caplog):
    from alloy.camera.model_pack import _materials, _template_shapes
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_object("table_top_plywood", "table_top_plywood", pos=(0, 0, 0))   # named as its card, as compose names it
    s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.9), mass=0.2)
    model = alloy.build(s, device="cpu").engine.model
    caplog.set_level(logging.INFO, logger="alloy.camera.item_looks")
    from alloy.camera.card_looks import fallback_rules
    labels = list(model.shape_label)
    card = fallback_rules(labels)["materials"]
    assert any(IL._texture(m, k)[0] for k, m in card.items()), card       # its preset names a texture
    names, mats, tex = _materials(model, _template_shapes(model), {})     # raised CameraError before
    assert not tex and "python -m alloy.looks --fetch-textures" in caplog.text, tex
    from alloy.camera.item_looks import textures_on_disk
    tones = set(_tones(textures_on_disk(fallback_rules(labels))["materials"]).values())
    assert tuple(np.round(FAMILY_TONE["wood"], 3)) in tones, tones        # plywood: wood, not the white placeholder


def test_the_texture_error_names_the_fetch_command(empty_cache):
    from alloy.camera.errors import CameraError
    from alloy.camera.looks.textures import local_texture_files
    with pytest.raises(CameraError) as err:
        local_texture_files("polyhaven:plywood", "2k")
    assert "python -m alloy.looks --fetch-textures" in err.value.fix and "quality='fast'" in err.value.fix


def _varied():
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")
    s.add_box("a", size=(0.04, 0.04, 0.04), pos=(-0.06, 0, 0.42), material="wood_beech")
    return alloy.vary(s, 1, pos=0.0, seed=3)[0]


def test_the_contact_sheet_draws_a_missing_texture_as_sim_render_does(empty_cache):
    from alloy.scene.variants.looks import scene_look
    from alloy.scene.variants.sheet import drawable
    scene = _varied()
    raw = scene_look(scene)
    textured = [k for k, m in raw["materials"].items() if m.get("texture")]
    assert textured, raw["materials"]
    sheet, dropped = drawable(raw)
    render = IL.items_look(scene)
    for key in textured:
        assert sheet["materials"][key]["base_color"] == render["materials"][key]["base_color"], key
        assert max(sheet["materials"][key]["base_color"]) < 0.9, sheet["materials"][key]   # not the white placeholder
    assert any("a flat colour of its family" in d for d in dropped), dropped


def test_a_texture_fetched_later_is_drawn_in_the_same_process(empty_cache):
    from alloy.scene.variants.looks import scene_look
    scene = _varied()
    before = IL.items_look(scene)
    flat = {k for k, m in before["materials"].items() if m.get("texture") is None}
    refs = {(IL._texture(m, k)) for k, m in scene_look(scene)["materials"].items() if m.get("texture")}
    assert flat and refs, (flat, refs)
    assert IL.items_look(scene) is before                         # remembered while nothing changes on disk
    for ref, res in refs:
        fake_texture(empty_cache, ref, res)
    after = IL.items_look(scene)                                  # no memo clear
    assert all(after["materials"][k].get("texture") for k in flat), after["materials"]
