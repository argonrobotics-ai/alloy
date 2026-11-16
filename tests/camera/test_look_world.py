# SPDX-License-Identifier: Apache-2.0
"""Law: the full-material camera draws every key of a look's ``world`` section the Blender renderer reads, and raises
for one it cannot draw (sample render item 11).

``sim.render(quality="full", look=...)`` read only ``hdri`` and ``strength``: ``rotation_deg``, ``show_hdri``,
``background_color`` and ``background_from_hdri`` were dropped without a word, so the camera always saw the HDRI
unturned. Now the HDRI turns as the Blender build turns it, camera rays see the flat background unless
``show_hdri``, missing keys take the look schema's defaults, a key the camera cannot draw (``fade``, a typo) raises
CameraError naming the keys it reads, and a written pack keeps the turn and the background for Cycles.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.camera.errors import CameraError
from alloy.camera.look_world import WORLD_DEFAULTS, WORLD_KEYS, look_world, z_turn
from alloy.camera.pack_write import write_hdr

VIEW = ((0.5, -0.5, 0.3), (0.0, 0.0, 0.05))
DIFFERENT = 3.0    # sRGB levels, mean |a - b|: well above one render's own noise at a fixed seed (0 here)


@pytest.fixture(scope="module")
def hdri(tmp_path_factory) -> str:
    """A lopsided sky: a bright warm patch on one side of the horizon, dim blue elsewhere."""
    img = np.full((32, 64, 3), (0.05, 0.06, 0.12), np.float32)
    img[10:20, 0:12] = (6.0, 3.0, 1.0)
    path = tmp_path_factory.mktemp("sky") / "lopsided.hdr"
    write_hdr(path, img)
    return str(path)


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05), mass=1.0)
    s = alloy.build(scene, device="cpu")
    yield s
    s.close()


def _render(sim, world: dict) -> np.ndarray:
    look = {"world": world, "lights": []}
    return sim.render(width=48, height=32, spp=4, quality="full", look=look, look_at=VIEW).astype(np.float64)


@pytest.mark.parametrize("show", [True, False], ids=["sky_seen", "sky_lights_only"])
def test_an_hdri_turned_150_deg_renders_differently_from_0_deg(sim, hdri, show):
    straight = _render(sim, {"hdri": hdri, "rotation_deg": 0.0, "show_hdri": show})
    turned = _render(sim, {"hdri": hdri, "rotation_deg": 150.0, "show_hdri": show})
    again = _render(sim, {"hdri": hdri, "rotation_deg": 0.0, "show_hdri": show})
    assert np.abs(straight - again).mean() == 0.0      # same look, same image: the difference is the turn
    assert np.abs(straight - turned).mean() > DIFFERENT


def test_camera_rays_see_the_background_colour_unless_show_hdri(sim, hdri):
    red = _render(sim, {"hdri": hdri, "background_color": [0.8, 0.0, 0.0]})
    corner = red[:4, :4].reshape(-1, 3).mean(0)        # sky behind the cube: the flat colour, not the HDRI
    assert corner[0] > 40 and corner[1] < 0.5 * corner[0] and corner[2] < 0.5 * corner[0]
    shown = _render(sim, {"hdri": hdri, "background_color": [0.8, 0.0, 0.0], "show_hdri": True})
    assert np.abs(shown[:4, :4] - red[:4, :4]).mean() > DIFFERENT


def test_the_world_defaults_are_the_look_schemas(hdri, monkeypatch):
    look_py = Path(alloy.__file__).parent / "render" / "blender" / "look.py"
    spec = importlib.util.spec_from_file_location("_blender_look", look_py)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(look_py.parent))   # look.py imports the Blender side's presets module
    monkeypatch.setitem(sys.modules, "_blender_look", mod)
    had_presets = "presets" in sys.modules
    try:
        spec.loader.exec_module(mod)
    finally:
        if not had_presets:
            sys.modules.pop("presets", None)
    assert WORLD_DEFAULTS == mod.WORLD_DEFAULTS
    world = look_world({"world": {"hdri": hdri}})
    from alloy.render.blender.motion import srgb_to_linear
    assert np.allclose(world["camera_color"], srgb_to_linear(WORLD_DEFAULTS["background_color"]))
    assert np.allclose(world["rotation"], np.eye(3))


def test_background_from_hdri_is_its_mean_radiance_times_strength(hdri):
    a = look_world({"world": {"hdri": hdri, "strength": 1.0, "background_from_hdri": True}})["camera_color"]
    b = look_world({"world": {"hdri": hdri, "strength": 2.5, "background_from_hdri": True}})["camera_color"]
    assert a[0] > a[2] and np.allclose(b, 2.5 * np.asarray(a))      # the warm patch shows in the mean


@pytest.mark.parametrize("world", [{"fade": {"material": "floor", "start_m": 100, "depth_m": 500}},
                                   {"rotation": 150}, {"show_hdri": "yes"}, {"rotation_deg": "half"}],
                         ids=["fade", "unknown_key", "show_hdri_not_bool", "rotation_not_number"])
def test_a_world_key_the_camera_cannot_draw_raises(sim, world):
    with pytest.raises(CameraError) as err:
        _render(sim, world)
    if set(world) - set(WORLD_KEYS):
        assert all(k in err.value.fix for k in WORLD_KEYS), err.value.fix


def test_a_written_pack_keeps_the_turn_and_the_camera_background(sim, hdri, tmp_path):
    from alloy.camera.pack_write import write_sim_pack
    from alloy.camera.pbr.world_nodes import camera_background, lighting_background, mapping_rotation
    look = {"world": {"hdri": hdri, "rotation_deg": 150.0, "background_color": [0.5, 0.2, 0.1]}, "lights": []}
    root = write_sim_pack(sim, tmp_path / "pack", VIEW, look=look)
    tree = json.loads((root / "pack.json").read_text())["world"]["tree"]     # what load_pack and Cycles read
    assert np.allclose(mapping_rotation(tree), z_turn(150.0), atol=1e-9)
    assert np.allclose(camera_background(tree), look_world(look)["camera_color"])
    assert lighting_background(tree)["name"] == "Background"                  # every other ray sees the HDRI
