# SPDX-License-Identifier: Apache-2.0
"""Scene item looks in the camera (CPU): a scene variant's per-item look (alloy.scene.variants.vary) reaches the
materials the camera renders, through sim.render / alloy.camera.render and Camera.render."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import alloy
from alloy.camera import Camera
from alloy.camera.item_looks import items_look
from alloy.camera.looks.material import rgba
from alloy.camera.model_pack import pack_from_model
from alloy.scene.variants import vary


def _scene() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.1), pos=(0, 0, 0.05), static=True, material="wood_pine")
    s.add_box("a", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.13), material="steel_carbon")
    return s


def _variant() -> alloy.Scene:
    (v,) = vary(_scene(), 1, pos=0.0, textures="keep", tint=0.5, seed=2)   # colour only: no texture cache needed
    return v


def test_item_looks_become_the_camera_materials_of_their_shapes():
    base, var = alloy.build(_scene(), device="cpu"), alloy.build(_variant(), device="cpu")
    assert items_look(base) is None                                        # no item carries a look
    layer = items_look(var)
    assert any(k.startswith("a:") for k in layer["materials"]) and "lights" not in layer
    pk, frame, _ = pack_from_model(var.engine.model, items=layer)
    used = {m.meta["materials"][0] for m in frame.meshes}
    assert {k for k in layer["materials"] if k.startswith("a:")} & used   # the box renders with its variant look
    tinted = layer["materials"][next(k for k in layer["materials"] if k.startswith("a:"))]
    plain = pack_from_model(base.engine.model)[1].meshes
    assert not any(m.meta["materials"][0].startswith("a:") for m in plain) and tinted.get("color_gain", 1.0) != 1.0


def test_sim_render_and_camera_render_show_the_variant_look():
    base, var = alloy.build(_scene(), device="cpu"), alloy.build(_variant(), device="cpu")
    view = {"look_at": ((0.25, -0.25, 0.3), (0.0, 0.0, 0.13)), "width": 24, "height": 18, "spp": 4}
    a = base.render(quality="full", **view).astype(float)
    b = var.render(**view).astype(float)                                   # item looks: full materials by itself
    plain = var.render(quality="full", items=None, **view).astype(float)   # the same poses without item looks
    assert np.abs(plain - a).mean() < 1.0                                  # pos=0: the variant moved nothing
    assert (np.abs(b - plain).mean(axis=2) > 8.0).mean() > 0.2            # its tinted looks show
    cam = Camera("c", look_at=view["look_at"], width=24, height=18, spp=4, fov=50.0)
    assert np.abs(cam.render(var).numpy()[0].astype(float) - b).mean() < 4.0


def test_a_swapped_texture_missing_from_the_cache_falls_back_to_the_items_own(monkeypatch):
    import alloy.camera.item_looks as IL
    import alloy.camera.looks.textures as T
    (v,) = vary(_scene(), 1, pos=0.0, textures="same-family", seed=3)
    swapped = v.items["table"]["look"]["materials"]["wood_pine"]["texture"]
    own = "polyhaven:coated_pine"                                          # wood_pine's library texture
    assert swapped != own

    def cache(*have):
        def files(ref, res, root=None):
            if ref not in have:
                raise FileNotFoundError(ref)
            return {"color": Path(f"/cache/{ref}.png")}
        return files

    for have, want in (((own,), own), ((), None)):                         # library texture cached / nothing cached
        IL._MEMO.clear()
        monkeypatch.setattr(T, "local_texture_files", cache(*have))
        assert IL.items_look(v)["materials"]["table:wood_pine"].get("texture") == want
    IL._MEMO.clear()


def test_a_texture_missing_from_the_cache_draws_its_familys_tone_with_one_warning(monkeypatch, caplog):
    import alloy.camera.item_looks as IL
    import alloy.camera.looks.textures as T

    def nothing(ref, res, root=None):
        raise FileNotFoundError(ref)
    monkeypatch.setattr(T, "local_texture_files", nothing)
    monkeypatch.setattr(IL, "_MEMO", {})
    monkeypatch.setattr(IL, "_WARNED", [])
    (v,) = vary(_scene(), 1, pos=0.0, textures="same-family", seed=3)
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.items["ground"]["look"] = {"materials": {"floor": {"preset": "wood_oak", "texture": "polyhaven:wood_floor"}},
                                 "assign": [], "default": "floor"}            # a floor-family texture: not wood
    with caplog.at_level("INFO", logger="alloy.camera.item_looks"):
        table = IL.items_look(v)["materials"]["table:wood_pine"]
        floor = IL.items_look(s)["materials"]["ground:floor"]
    assert table["texture"] is None and table["base_color"] == list(IL.FAMILY_TONE["wood"])
    assert floor["texture"] is None and floor["base_color"] == list(IL.FAMILY_TONE["floor"])
    warned = [r for r in caplog.records if "--fetch-textures" in r.getMessage()]
    assert [r.levelname for r in warned] == ["WARNING", "INFO"]                 # one warning per process
    for rgb in IL.FAMILY_TONE.values():                                           # mid-tones, never white
        assert 0.15 <= float(np.dot(rgba(rgb)[:3], [0.2126, 0.7152, 0.0722])) <= 0.37
