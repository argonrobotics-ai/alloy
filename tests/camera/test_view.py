# SPDX-License-Identifier: Apache-2.0
"""Display transforms: AgX keeps highlights below white, the scene's view settings pick the transform, and a
blown-out first image logs a warning (CPU)."""
from __future__ import annotations

import logging

import numpy as np
import warp as wp

from alloy.camera.agx import agx_u8
from alloy.camera.view import STANDARD, View, clipped_fraction, standard_u8, to_display, view_from_settings


def test_agx_keeps_mid_grey_and_compresses_highlights():
    img = agx_u8(np.array([[0.18] * 3, [1.0] * 3, [16.0] * 3, [0.0] * 3], np.float32))
    assert 110 <= img[0, 0] <= 140                    # middle grey stays near the middle
    assert img[1, 0] < 230 and img[2, 0] < 255        # white and 4 stops over are not clipped
    assert img[3, 0] < 5
    assert standard_u8(np.array([[2.0] * 3], np.float32))[0, 0] == 255   # Standard clips


def test_device_agx_matches_the_host_version():
    rgb = np.random.default_rng(0).exponential(0.5, (1, 4, 8, 3)).astype(np.float32)
    dev = to_display(wp.array(rgb, dtype=wp.vec3, device="cpu"), view=View("agx", 0.0, 1.2)).numpy()
    host = agx_u8(rgb, contrast=1.2)
    assert np.abs(dev.astype(int) - host.astype(int)).max() <= 1


def test_view_settings_pick_the_transform_and_exposure():
    assert view_from_settings(None) == STANDARD
    v = view_from_settings({"view_transform": "AgX", "look": "AgX - Medium High Contrast", "exposure": -1.0})
    assert v.transform == "agx" and v.contrast > 1.0 and v.exposure == -1.0
    assert view_from_settings({"view_transform": "Standard", "look": "None"}).transform == "standard"
    lo = to_display(wp.array(np.full((1, 2, 2, 3), 0.18, np.float32), dtype=wp.vec3, device="cpu"),
                    exposure=-1.0, view=v).numpy()
    hi = to_display(wp.array(np.full((1, 2, 2, 3), 0.18, np.float32), dtype=wp.vec3, device="cpu"),
                    view=v).numpy()
    assert lo.mean() < hi.mean()                       # exposure adds to the view's own


def test_model_scenes_render_through_the_look_view():
    from alloy.camera.model_pack import _settings
    assert _settings({})["view_settings"]["view_transform"] == "AgX"
    vs = _settings({"render": {"view": "Standard", "exposure": 0.5}})["view_settings"]
    assert vs["view_transform"] == "Standard" and vs["exposure"] == 0.5


def test_a_blown_out_first_image_logs_a_warning(caplog):
    from alloy.camera.camera import _warn_if_clipped
    white = wp.array(np.full((2, 4, 4, 3), 255, np.uint8), device="cpu")
    with caplog.at_level(logging.WARNING, logger="alloy.camera.camera"):
        _warn_if_clipped(white, "front", 0.0, STANDARD)
    assert "100% of pixels clip" in caplog.text and "exposure" in caplog.text
    assert clipped_fraction(np.zeros((4, 4, 3), np.uint8)) == 0.0
