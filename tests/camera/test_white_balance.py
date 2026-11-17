# SPDX-License-Identifier: Apache-2.0
"""Law: the full-material camera applies a look's white balance as Blender 4.5 does (``render.white_balance``).

"auto" (Blender's default, also when the key is left out) adapts the light reaching the subject to white, measured
as Blender measures it (a probe render of a grey ball at the subject), so a grey ball under a warm 2800 K light reads
grey, and the probe of two shipped settings gives Blender 4.5.15's white points (Cycles on an L4); "off" leaves the
light's colour, today's image; a temperature adapts that white (3000 K and 6500 K give different images). The
matrix is the render gate's method (Robertson temperature and tint, Bradford adaptation), so camera images and the
gate's display of Cycles references agree. CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera import white_balance as wb
from alloy.camera.errors import CameraError
from alloy.camera.view import view_from_settings

VIEW = ((0.35, -0.35, 0.25), (0.0, 0.0, 0.05))
BALL = (slice(30, 42), slice(42, 54))  # the ball's lit centre at 96 x 72
WARM = {"materials": {"grey": {"base_color": [0.18, 0.18, 0.18], "roughness": 1.0}},
        "assign": [{"match": ".*", "material": "grey"}],
        "lights": [{"type": "area", "watts": 60, "kelvin": 2800, "eye": [0.3, -0.3, 0.5], "target": [0, 0, 0.05],
                    "size": 0.3}],
        "world": {"strength": 0.05}}


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.05, pos=(0.0, 0.0, 0.05), mass=0.2)
    s = alloy.build(scene, device="cpu")
    yield s
    s.close()


def render(sim, white=...) -> np.ndarray:
    look = {**WARM, "render": {} if white is ... else {"white_balance": white}}
    return sim.render(width=96, height=72, spp=16, quality="full", look=look, look_at=VIEW).astype(np.float64)


def red_over_blue(img: np.ndarray) -> float:
    c = img[BALL].reshape(-1, 3).mean(0)
    return float(c[0] / c[2])


def test_a_grey_ball_reads_grey_under_auto(sim, caplog):
    with caplog.at_level("WARNING", logger=wb.__name__):
        auto = render(sim, "auto")
    assert not [r for r in caplog.records if "probe failed" in r.getMessage()]   # measured, not estimated
    assert 0.97 < red_over_blue(auto) < 1.03
    assert np.array_equal(render(sim), render(sim, "auto"))   # Blender's default when the key is left out
    assert red_over_blue(render(sim, "off")) > 1.8             # the warm light itself


def test_off_is_todays_image(sim, monkeypatch):
    off = render(sim, "off")
    monkeypatch.setattr(wb, "look_view_white", lambda *args: {})   # the camera before white balance
    assert np.array_equal(off, render(sim, "off"))


def test_3000k_and_6500k_differ(sim):
    k3000, k6500 = render(sim, {"kelvin": 3000}), render(sim, {"kelvin": 6500})
    assert np.abs(k3000 - k6500).mean() > 10.0
    assert red_over_blue(k3000) < red_over_blue(k6500)        # adapting to a warmer white takes more red out


def test_a_bad_setting_says_what_to_use():
    with pytest.raises(CameraError, match="kelvin"):
        wb.view_white({"render": {"white_balance": {"kelvin": 500}}}, (0, 0, 0))
    with pytest.raises(CameraError, match="'auto', 'off'"):
        wb.view_white({"render": {"white_balance": "daylight"}}, (0, 0, 0))


@pytest.mark.parametrize("temperature, tint", [(2800.0, 10.0), (5000.0, -4.0), (6500.0, 10.0), (9000.0, 0.0)])
def test_the_matrix_is_the_render_gate_method(temperature, tint):
    from bench.render_gates import whitebalance as gate
    assert np.allclose(wb.white_point_matrix(temperature, tint), gate.white_point_matrix(temperature, tint))
    white = gate.XYZ_TO_REC709 @ gate.whitepoint_xyz(temperature, tint)
    assert np.allclose(wb.temp_tint(white), gate.temp_tint_from_whitepoint_rgb(white))
    vs = wb.view_white({"render": {"white_balance": {"kelvin": temperature, "tint": tint}}}, (0, 0, 0))
    out = np.asarray(view_from_settings(vs).white).reshape(3, 3) @ np.asarray(vs["white_balance_whitepoint"])
    assert np.allclose(out / out[1], 1.0, atol=1e-6)               # the light's white shows white


def test_a_temperature_alone_takes_blenders_tint():
    vs = wb.view_white({"render": {"white_balance": {"kelvin": 4000}}}, (0, 0, 0))
    assert vs["white_balance_temperature"] == 4000.0 and vs["white_balance_tint"] == wb.DEFAULT_TINT == 10.0


def test_a_full_render_without_a_look_has_no_white_balance(sim, monkeypatch):
    def boom(*args):
        raise AssertionError("white balance asked for without a look")
    monkeypatch.setattr(wb, "look_view_white", boom)
    sim.render(width=32, height=24, spp=1, quality="full", look_at=VIEW)  # studio lighting, as before


# Blender 4.5.15's auto white balance (its probe, Cycles on one L4) for the shipped kitchen and workshop settings,
# subject (0, 0, 0.32) seen from (0.45, -0.5, 0.55): temperature K, tint
BLENDER_AUTO = {"kitchen": (2584.6, -39.07), "workshop": (6280.6, 33.84)}


@pytest.mark.parametrize("setting", sorted(BLENDER_AUTO))
def test_the_probe_gives_blenders_white_point(sim, setting):
    """Through the scene build: the look's camera key is the subject, the probe measures the light there."""
    from alloy.camera import camera_scene
    from alloy.clips.kit.look import SETTINGS
    from alloy.render.blender.assets import default_cache
    hdri = SETTINGS[setting]["hdri"].split(":", 1)[1]
    if not (default_cache() / "polyhaven" / hdri / f"{hdri}_4k.hdr").exists():
        pytest.skip(f"the {hdri} HDRI is not in the local texture cache")
    key = {"t": 0.0, "eye": [0.45, -0.5, 0.55], "target": [0.0, 0.0, 0.32]}
    look = {**WARM, "lights": [], "world": {**SETTINGS[setting], "show_hdri": True},
            "cameras": {"main": {"lens_mm": 50.0, "keys": [key]}}}
    vs = camera_scene(sim, look=look).pack.static["settings"]["view_settings"]
    temperature, tint = vs["white_balance_temperature"], vs["white_balance_tint"]
    want_t, want_tint = BLENDER_AUTO[setting]
    assert abs(temperature - want_t) < 25.0 and abs(tint - want_tint) < 0.5, (temperature, tint)


def test_no_look_and_off_carry_no_white_balance():
    assert view_from_settings({"view_transform": "AgX"}).white is None
    assert view_from_settings({"view_transform": "AgX", "use_white_balance": False}).white is None
    assert wb.view_white({"render": {"white_balance": "off"}}, (0, 0, 0)) == {"use_white_balance": False}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
