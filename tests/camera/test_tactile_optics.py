# SPDX-License-Identifier: Apache-2.0
"""Tactile optics (OSS-21): presets, the Taxim-style colour table on the device vs numpy, and real ball presses.

The real presses are crops of Taxim's published calibration images (MIT, see test_tactile_data/README.md): a 4 mm
ball pressed into a GelSight and a DIGIT gel. A sphere press rendered with the shipped calibration must be much
closer to the real image than the unpressed gel is.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy.camera.errors import CameraError
from alloy.camera.tactile import Augment, TactileRenderer, list_presets, load_preset
from alloy.camera.tactile.optics import GelOptics, gaussian_weights
from alloy.camera.tactile.presets import output_size

DATA = Path(__file__).with_name("test_tactile_data")
BALL_MM = 2.0  # Taxim's calibration ball radius


def sphere_press(h: int, w: int, mm_px: float, cx: float, cy: float, contact_mm: float) -> np.ndarray:
    """Press map [mm] of a 2 mm-radius ball whose contact circle has radius contact_mm, centred at (cx, cy) px."""
    a = min(contact_mm, BALL_MM * 0.999)
    depth = BALL_MM - math.sqrt(BALL_MM ** 2 - a * a)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - cx) ** 2 + (yy - cy) ** 2) * mm_px ** 2
    return np.clip(np.sqrt(np.clip(BALL_MM ** 2 - r2, 0, None)) - (BALL_MM - depth), 0, None).astype(np.float32)


def test_presets_load_with_their_licence():
    assert sorted(list_presets()) == ["digit", "gelsight"]
    assert (Path(load_preset("gelsight").source) is not None)
    assert "MIT License" in (Path(__file__).parents[2] / "alloy/camera/tactile/data/LICENSE-Taxim.txt").read_text()
    for name, mm in (("gelsight", 0.0295), ("digit", 0.0266)):
        p = load_preset(name)
        assert p.table.shape == (3, p.bins, p.bins, 6) and p.background.shape == (480, 640, 3)
        assert p.mm_per_px == pytest.approx(mm) and p.field_mm[0] == pytest.approx(640 * mm)
        x0, y0, x1, y1 = p.fit_box
        assert 0 <= x0 < x1 <= 639 and 0 <= y0 < y1 <= 479


def test_output_size_keeps_the_gel_aspect():
    p = load_preset("gelsight")
    assert output_size(p, None, None) == (240, 320)
    assert output_size(p, 160, None) == (120, 160) and output_size(p, None, 60) == (60, 80)
    with pytest.raises(CameraError, match="aspect"):
        output_size(p, 160, 160)
    with pytest.raises(CameraError, match="no tactile preset"):
        load_preset("gelsight_mini_x")


def _numpy_delta(p, press: np.ndarray, mm_px: float) -> np.ndarray:
    """Reference colour change: central-difference slopes -> (magnitude, direction) bins -> polynomial (u, v)."""
    h, w = press.shape
    gr = np.zeros_like(press, np.float64)
    gc = np.zeros_like(press, np.float64)
    gr[1:-1] = (press[2:] - press[:-2]) / (2 * mm_px)
    gc[:, 1:-1] = (press[:, 2:] - press[:, :-2]) / (2 * mm_px)
    gr[0], gr[-1] = (press[1] - press[0]) / mm_px, (press[-1] - press[-2]) / mm_px
    gc[:, 0], gc[:, -1] = (press[:, 1] - press[:, 0]) / mm_px, (press[:, -1] - press[:, -2]) / mm_px
    t = np.hypot(gr, gc)
    mag, ang, b = np.arctan(t), np.arctan2(gr, gc), p.bins
    ix = np.clip((mag / (0.5 * np.pi / (b - 1))).astype(int), 0, b - 1)
    iy = np.clip(((ang + np.pi) / (2 * np.pi / (b - 1))).astype(int), 0, b - 1)
    ch, cw = p.calib_size
    s = (cw - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x0, y0, x1, y1 = p.fit_box
    u = np.clip(((xx + 0.5) * cw / w - 0.5 - (cw - 1) / 2) / s, (x0 - (cw - 1) / 2) / s, (x1 - (cw - 1) / 2) / s)
    v = np.clip(((yy + 0.5) * ch / h - 0.5 - (ch - 1) / 2) / s, (y0 - (ch - 1) / 2) / s, (y1 - (ch - 1) / 2) / s)
    terms = np.stack([u * u, v * v, u * v, u, v, np.ones_like(u)], -1)
    coef = p.table[:, ix, iy]  # (3, H, W, 6)
    fade = np.minimum(mag / math.radians(p.flat_fade_deg), 1.0)
    return (coef * terms[None]).sum(-1).transpose(1, 2, 0) * fade[..., None]


@pytest.mark.parametrize("preset", ["gelsight", "digit"])
def test_colour_table_matches_numpy(preset):
    p = load_preset(preset)
    opt = GelOptics(p, 60, 80, 2, device="cpu")
    press = np.stack([sphere_press(60, 80, opt.mm_per_px, 30.0, 25.0, 1.5), np.zeros((60, 80), np.float32)])
    got = opt.shade(wp.array(press, device="cpu")).numpy()
    want = _numpy_delta(p, press[0], opt.mm_per_px)
    assert np.abs(got[0] - want).max() < 0.05
    assert np.abs(got[1]).max() == 0.0  # no press: exactly the unpressed gel


def test_spreading_keeps_the_core_and_softens_the_rim():
    p = load_preset("gelsight")
    opt = GelOptics(p, 120, 160, 1, device="cpu")
    press = sphere_press(120, 160, opt.mm_per_px, 80.0, 60.0, 1.6)
    gel = opt.spread(wp.array(press[None], device="cpu")).numpy()[0]
    core = press > p.contact_scale * press.max()
    assert np.allclose(gel[core], press[core], atol=0.03)  # only the final light blur touches the core
    ring = (press == 0) & (gel > 1e-3)
    assert ring.any() and gel[ring].max() < 0.5 * press.max()  # the gel around the ball dips smoothly
    assert np.isclose(gaussian_weights(4.0).sum(), 1.0) and len(gaussian_weights(0.1)) == 1


@pytest.mark.parametrize("preset,ratio", [("gelsight", 0.45), ("digit", 0.75)])
def test_ball_press_looks_like_the_real_sensor(preset, ratio):
    """Sim vs Taxim's real calibration presses, half resolution, inside 1.6 x the contact radius."""
    data = np.load(DATA / "presses.npz", allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    keys = [k for k in meta if meta[k]["preset"] == preset]
    tr = TactileRenderer(preset, width=320, num_worlds=len(keys), device="cpu", markers=False)
    mm = tr.mm_per_px
    press = np.stack([sphere_press(240, 320, mm, meta[k]["center_full_px"][0] / 2 - 0.25,
                                   meta[k]["center_full_px"][1] / 2 - 0.25, meta[k]["radius_full_px"] / 2 * mm)
                      for k in keys])
    img = tr.render(press, seed=0).numpy().astype(np.float64)
    bg = tr._bg.numpy()
    sims, blanks = [], []
    for i, k in enumerate(keys):
        x0, y0, x1, y1 = meta[k]["crop_half_px"]
        real = data[k].astype(np.float64)
        cx, cy, r = (meta[k]["center_full_px"][0] / 2 - x0, meta[k]["center_full_px"][1] / 2 - y0,
                     meta[k]["radius_full_px"] / 2)
        yy, xx = np.mgrid[0:real.shape[0], 0:real.shape[1]]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 < (1.6 * r) ** 2
        sims.append(np.abs(img[i, y0:y1, x0:x1] - real)[disk].mean())
        blanks.append(np.abs(bg[y0:y1, x0:x1] - real)[disk].mean())
    assert np.mean(sims) < ratio * np.mean(blanks), (sims, blanks)


def test_validate_tool_scores_a_pack(tmp_path):
    """A pack whose 'real' images are our own renders scores L1 0 and SSIM 1; the blank scores worse."""
    from alloy.camera.tactile.validate import evaluate, ssim_map
    img = np.random.default_rng(0).integers(0, 255, (24, 32, 3)).astype(np.uint8)
    assert np.allclose(ssim_map(img, img), 1.0)
    tr = TactileRenderer("gelsight", width=640, num_worlds=2, device="cpu", markers=False,
                         augment=Augment(noise_std=0.0))
    centers, radii = np.array([[300.0, 200.0], [420.0, 260.0]]), np.array([60, 50])
    press = np.stack([sphere_press(480, 640, tr.mm_per_px, cx, cy, r * tr.mm_per_px)
                      for (cx, cy), r in zip(centers, radii)])
    imgs = tr.render(press, seed=0).numpy()[..., ::-1]  # packs are BGR
    np.savez(tmp_path / "pack.npz", imgs=imgs, touch_center=centers, touch_radius=radii,
             f0=np.zeros((480, 640, 3), np.uint8))
    res = evaluate(tmp_path / "pack.npz", "gelsight", batch=2)
    assert res["sim"]["l1"] < 1e-6 and res["sim"]["ssim"] > 0.9999 and res["blank"]["l1_disk"] > 5.0
