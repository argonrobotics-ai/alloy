# SPDX-License-Identifier: Apache-2.0
"""Images from the engine's gel fields (OSS-21 side of the WP13 GelSensor interface, plans section 6)."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.camera.errors import CameraError
from alloy.camera.tactile import Augment
from alloy.camera.tactile.engine import EngineTactile

W, S, STRIDE = 2, 2, 4
H_IMG, W_IMG, MM = 480, 640, 0.0295  # GelSight calibration grid


def cap_mm(x, y, cx, cy, radius=2.0, depth=0.8):
    return np.clip(np.sqrt(np.clip(radius ** 2 - (x - cx) ** 2 - (y - cy) ** 2, 0, None)) - (radius - depth), 0, None)


def section6_depth(cx: float, cy: float) -> np.ndarray:
    """[W, S, Hg, Wg] depth in m of a ball press at (cx, cy) mm, cells laid out as written in section 6."""
    hg, wg = H_IMG // STRIDE, W_IMG // STRIDE
    i, j = np.mgrid[0:hg, 0:wg].astype(np.float64)
    x = (j + 0.5) * STRIDE * MM - W_IMG * MM / 2
    y = H_IMG * MM / 2 - (i + 0.5) * STRIDE * MM
    d = cap_mm(x, y, cx, cy) * 1e-3
    return np.broadcast_to(d, (W, S, hg, wg)).astype(np.float32).copy()


def test_depth_grid_lands_where_the_sensor_frame_says():
    view = EngineTactile("gelsight", num_worlds=W, sensors=S, width=320, device="cpu", augment=Augment(noise_std=0))
    out = view.render({"depth": section6_depth(2.0, 3.0)}, output="all")
    press = out.press.numpy()
    assert out.rgb.shape == (W * S, 240, 320, 3)
    mm = view.renderer.mm_per_px
    yy, xx = np.mgrid[0:240, 0:320].astype(np.float64)
    want = cap_mm((xx + 0.5 - 160) * mm, (yy + 0.5 - 120) * mm, 2.0, 3.0)  # image rows run along +y (down)
    assert np.abs(press[0] - want).max() < 0.03 and np.allclose(press[0], press[3])
    r, c = np.unravel_index(np.argmax(press[0]), press[0].shape)
    assert r > 120 and c > 160  # +y is down the image, +x right


def test_engine_markers_follow_marker_disp_in_any_order():
    pitch = 1.5
    gx, gy = np.meshgrid(pitch * (np.arange(12) - 5.5), pitch * (np.arange(9) - 4))
    rest_m = np.stack([gx.ravel(), gy.ravel()], -1)[::-1] * 1e-3  # engine order differs from the grid's
    view = EngineTactile("gelsight", num_worlds=W, sensors=S, width=160, marker_rest_m=rest_m, device="cpu")
    disp = np.zeros((W, S, len(rest_m), 3), np.float32)
    disp[1, 0, :, 0] = 0.1e-3
    disp[1, 0, 0, 1] = 0.2e-3  # engine marker 0 only
    view.render({"depth": section6_depth(0.0, 0.0), "marker_disp": disp})
    r = view.renderer
    pos = r.marker_pos.numpy()
    moved = pos[2] - r.grid.rest_mm  # image 2 = world 1, sensor 0
    assert np.allclose(moved[:, 0], 0.1, atol=1e-5) and np.allclose(pos[0], r.grid.rest_mm)
    j0 = int(np.flatnonzero(r.marker_order == 0)[0])
    assert moved[j0, 1] == pytest.approx(0.2, abs=1e-5) and np.allclose(np.delete(moved[:, 1], j0), 0.0)


def test_engine_shapes_are_checked():
    view = EngineTactile("digit", num_worlds=W, sensors=S, width=80, device="cpu")
    with pytest.raises(CameraError, match="depth has shape"):
        view.render({"depth": np.zeros((W, 1, 30, 40), np.float32)})
    with pytest.raises(CameraError, match="no 'depth'"):
        view.render({})
    with pytest.raises(CameraError, match="regular grid"):
        EngineTactile("gelsight", width=80, marker_rest_m=np.random.default_rng(0).uniform(-5e-3, 5e-3, (20, 2)),
                      device="cpu")


def test_physics_gel_sensor_end_to_end():
    """The physics side's GelSensor on its own press scene (5 mm ball on a gel pad), rendered by EngineTactile:
    its depth field matches this package's own ray cast on the same state, and markers come from its layout."""
    from alloy.camera.tactile import TactileSensor
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.tactile import GelSensor
    from alloy.plugins.tactile.gel_laws import press_scene
    scene = press_scene()
    gel = GelSensor(GelSensor.Config(markers=(7, 9)))
    sim = build_simulation(scene, [gel], "cpu", worlds=1)
    for _ in range(scene.periods):
        sim.step_period()
    view = EngineTactile.from_gel_sensor(gel, width=320, augment=Augment(noise_std=0.0))
    assert view.renderer.preset.name == "gelsight" and (view.renderer.grid.rows, view.renderer.grid.cols) == (7, 9)
    eng = view.render(gel.outputs(), output="all")
    own = TactileSensor("gel", attach_to="pad", width=320, markers=False, augment=Augment(noise_std=0.0))
    mine = own.render(sim.model, sim.state, output="all").press.numpy()[0]
    theirs = eng.press.numpy()[0]
    touched = (mine > 0) | (theirs > 0)
    assert mine.max() == pytest.approx(theirs.max(), abs=0.005) and 0.3 < mine.max() < 1.0  # mm
    assert np.abs(mine - theirs)[touched].mean() < 0.005 and np.abs(mine - theirs).max() < 0.03
    bg = view.renderer._bg.numpy()
    assert np.abs(eng.rgb.numpy()[0].astype(float) - bg)[touched].max() > 20
