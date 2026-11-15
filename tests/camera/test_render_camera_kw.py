# SPDX-License-Identifier: Apache-2.0
"""Law: ``sim.render(camera, ...)`` uses or refuses every keyword, and depth of a varied scene raises, not crashes.

Gemini audit S2-039: with a Camera, ``alloy.camera.render`` passed only ``output`` on: ``path=`` wrote no file,
``exposure=`` / ``seed=`` changed nothing, ``items=None`` was ignored and a typo (``outputs=``) returned colour, all
without a word; and ``output="depth"`` on a scene from ``alloy.vary`` (drawn by the full-material renderer) ran the
fast tracer's depth kernel on that renderer's scene and killed the process (SIGFPE). Now ``exposure``, ``seed`` and
``items`` reach ``Camera.render``, anything else raises CameraError, ``items=None`` draws a varied scene on the fast
camera (depth works), and depth on the full-material renderer raises the documented CameraError.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera.errors import CameraError

LOOK = ((0.35, -0.35, 0.28), (0.0, 0.0, 0.06))

pytestmark = pytest.mark.timeout(600)


def _scene():
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("blk", size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.06), material="wood_beech")
    return s


@pytest.fixture(scope="module")
def sim():
    s = alloy.build(_scene(), device="cpu")
    yield s
    s.close()


def _cam(name="front"):
    return alloy.Camera(name, look_at=LOOK, width=32, height=32, spp=2)


@pytest.mark.parametrize("kw", [{"path": "never.png"}, {"outputs": "depth"}, {"width": 64}, {"world": 0},
                                {"liquids": False}])
def test_a_keyword_a_camera_render_does_not_take_raises(sim, kw, tmp_path):
    with pytest.raises(CameraError, match="does not take"):
        sim.render(_cam(), **kw)
    assert not (tmp_path / "never.png").exists()


def test_exposure_and_seed_reach_the_camera(sim):
    cam = _cam()
    dark = sim.render(cam, exposure=-4.0, seed=1)
    bright = sim.render(cam, exposure=4.0, seed=1)
    assert np.abs(bright.astype(float) - dark.astype(float)).mean() > 20.0
    assert np.array_equal(dark, cam.render(sim, exposure=-4.0, seed=1).numpy())
    assert not np.array_equal(sim.render(cam, seed=1), sim.render(cam, seed=2))


def test_depth_of_a_varied_scene_with_items_none_and_without(tmp_path):
    varied = alloy.vary(_scene(), 1, seed=0)[0]
    s = alloy.build(varied, device="cpu")
    try:
        depth = s.render(_cam("v"), output="depth", items=None)            # the fast camera: no item looks
        assert depth.shape == (1, 32, 32) and np.isfinite(depth).any()
        with pytest.raises(CameraError, match="full-material camera"):    # killed the process before
            s.render(_cam("v2"), output="depth")
    finally:
        s.close()
