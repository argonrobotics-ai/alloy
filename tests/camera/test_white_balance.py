# SPDX-License-Identifier: Apache-2.0
"""Law: the full-material camera applies a look's white balance as Blender 4.5 does (``render.white_balance``).

"auto" (Blender's default, also when the key is left out) adapts the light reaching the subject to white, measured
as Blender measures it (a probe render of a grey ball at the subject, the look's camera key), so a grey ball under a
warm 2800 K light reads grey in both views (AgX and Standard), and the probe of two shipped settings gives Blender
4.5.15's white points (Cycles on an L4); the probe runs once per look (a redraw's new sim reuses it); a light too
coloured for a temperature keeps Blender's 6500 K, tint 10, with a warning, never a negative kelvin. "off" leaves
the light's colour, today's image; a temperature adapts that white (3000 K and 6500 K give different images; held
to Blender's ranges). The matrix is the render gate's method (Robertson temperature and tint, Bradford adaptation),
so camera images and the gate's display of Cycles references agree. CPU.
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


def ball_sim(x: float = 0.0, y: float = 0.0):
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.05, pos=(x, y, 0.05), mass=0.2)
    return alloy.build(scene, device="cpu")


@pytest.fixture(scope="module")
def sim():
    s = ball_sim()
    yield s
    s.close()


def render(sim, white=..., view: str = "AgX") -> np.ndarray:
    look = {**WARM, "render": {"view": view, **({} if white is ... else {"white_balance": white})}}
    return sim.render(width=96, height=72, spp=16, quality="full", look=look, look_at=VIEW).astype(np.float64)


def red_over_blue(img: np.ndarray) -> float:
    c = img[BALL].reshape(-1, 3).mean(0)
    return float(c[0] / c[2])


@pytest.mark.parametrize("view", ["AgX", "Standard"])
def test_a_grey_ball_reads_grey_under_auto(sim, caplog, view):
    with caplog.at_level("WARNING", logger=wb.__name__):
        auto = render(sim, "auto", view)
    assert not [r for r in caplog.records if "probe failed" in r.getMessage()]   # measured, not estimated
    assert 0.97 < red_over_blue(auto) < 1.03
    assert np.array_equal(render(sim, view=view), auto)        # Blender's default when the key is left out
    assert red_over_blue(render(sim, "off", view)) > 1.8       # the warm light itself


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


@pytest.mark.parametrize("kelvin, tint, want", [(1200, 10, (1800.0, 10.0)), (2e5, 900, (1e5, 500.0)),
                                                (3000, -800, (3000.0, -500.0))])
def test_a_set_white_is_held_to_blenders_ranges(kelvin, tint, want):
    vs = wb.view_white({"render": {"white_balance": {"kelvin": kelvin, "tint": tint}}}, (0, 0, 0))
    assert (vs["white_balance_temperature"], vs["white_balance_tint"]) == want


@pytest.mark.parametrize("white", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.2, 1.0, 0.2), (1.0, 0.2, 1.0)])
def test_a_light_too_coloured_for_a_temperature_keeps_blenders_default(white, caplog):
    """Red past the locus's 600 mired end, blue past its 0, green and magenta more than 150 tint off it: Blender
    leaves its 6500 K, tint 10 (Robertson's lines would give -101 K for pure red)."""
    assert wb.temp_tint(white) is None
    with caplog.at_level("WARNING", logger=wb.__name__):
        vs = wb.view_white({"render": {}}, (0, 0, 0), light=white)
    assert (vs["white_balance_temperature"], vs["white_balance_tint"]) == (6500.0, 10.0)
    assert np.allclose(vs["white_balance_whitepoint"], wb._white_of(6500.0, 10.0))
    assert "too coloured" in caplog.text
    assert np.allclose(view_from_settings(vs).white, wb.white_point_matrix(6500.0, 10.0).ravel())


def view_settings(sim, look: dict) -> dict:
    from alloy.camera import camera_scene
    return camera_scene(sim, look=look).pack.static["settings"]["view_settings"]


def test_the_probe_runs_once_per_look(sim, monkeypatch):
    """A redraw builds a new sim per frame: the same look (its ball moved 1 mm) measures once; a warmer light again."""
    calls, probe = [], wb.probe_light
    monkeypatch.setattr(wb, "_PROBED", {})
    monkeypatch.setattr(wb, "probe_light", lambda *args: calls.append(args) or probe(*args))
    look = {**WARM, "render": {"exposure": 0.0}}
    first = view_settings(sim, look)
    moved = ball_sim(0.001)
    try:
        assert view_settings(moved, look) == first and len(calls) == 1
    finally:
        moved.close()
    warmer = {**look, "lights": [{**WARM["lights"][0], "kelvin": 2000}]}
    assert view_settings(sim, warmer)["white_balance_temperature"] < first["white_balance_temperature"]
    assert len(calls) == 2


def probed_at(monkeypatch) -> list:
    """(target, eye) of every probe from now on (the probe itself skipped: a white light)."""
    seen = []
    monkeypatch.setattr(wb, "_PROBED", {})
    monkeypatch.setattr(wb, "probe_light", lambda look, target, eye, device: seen.append((target, eye)) or np.ones(3))
    return seen


def test_the_probe_measures_at_the_looks_camera_key(sim, monkeypatch):
    seen = probed_at(monkeypatch)
    key = {"t": 0.0, "eye": [1.0, -1.0, 0.8], "target": [0.3, 0.2, 0.1]}
    view_settings(sim, {**WARM, "render": {"exposure": 0.0}, "cameras": {"main": {"keys": [key]}}})
    assert len(seen) == 1 and np.allclose(seen[0][0], key["target"]) and np.allclose(seen[0][1], key["eye"])


def test_without_a_camera_key_the_probe_measures_where_the_parts_are(monkeypatch):
    """A free ball placed at (0.5, 0.3): its pack mesh is in its body's frame, the subject is where it is."""
    seen = probed_at(monkeypatch)
    far = ball_sim(0.5, 0.3)
    try:
        view_settings(far, WARM)
    finally:
        far.close()
    assert len(seen) == 1 and np.allclose(seen[0][0], (0.5, 0.3, 0.05)), seen


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
    from alloy.clips.kit.look import SETTINGS
    from alloy.render.blender.assets import default_cache
    hdri = SETTINGS[setting]["hdri"].split(":", 1)[1]
    if not (default_cache() / "polyhaven" / hdri / f"{hdri}_4k.hdr").exists():
        pytest.skip(f"the {hdri} HDRI is not in the local texture cache")
    key = {"t": 0.0, "eye": [0.45, -0.5, 0.55], "target": [0.0, 0.0, 0.32]}
    look = {**WARM, "lights": [], "world": {**SETTINGS[setting], "show_hdri": True},
            "cameras": {"main": {"lens_mm": 50.0, "keys": [key]}}}
    vs = view_settings(sim, look)
    temperature, tint = vs["white_balance_temperature"], vs["white_balance_tint"]
    want_t, want_tint = BLENDER_AUTO[setting]
    assert abs(temperature - want_t) < 25.0 and abs(tint - want_tint) < 0.5, (temperature, tint)


def test_no_look_and_off_carry_no_white_balance():
    assert view_from_settings({"view_transform": "AgX"}).white is None
    assert view_from_settings({"view_transform": "AgX", "use_white_balance": False}).white is None
    assert wb.view_white({"render": {"white_balance": "off"}}, (0, 0, 0)) == {"use_white_balance": False}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
