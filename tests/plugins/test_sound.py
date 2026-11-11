# SPDX-License-Identifier: Apache-2.0
"""ModalSound plug-in (engine item 10, sound) on CPU: registry, config, the harness's quick check (laws S-1..S-6, null,
rest), no double counting of an impact, dt vs dt/2 audio, and the audio ring buffer."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import fresh
from alloy.plugins.harness.run import run_scene
from alloy.plugins.sound import ModalSound
from alloy.plugins.sound import laws as SL
from alloy.plugins.sound.analysis import magnitude_spectrum

HARD = {"contact_radius": SL.FLAT, "partner": SL.STEEL_BALL}


def _strike(dt: float = 2e-3, periods: int = 10, x: float = 0.14, worlds: int = 3, solver: dict | None = None, **cfg):
    snd = fresh(ModalSound(), {**HARD, **cfg})
    scene = SL.strike_scene(fixed=True, balls=(("ball", x, SL.GAP),), dt=dt, periods=periods, worlds=worlds)
    run = run_scene(dataclasses.replace(scene, solver_kwargs=solver) if solver else scene, [snd], "cpu")
    assert run.error is None, run.error
    return run, snd


def test_registered_as_a_builtin():
    assert "modal_sound" in [p.name for p in ap.available()]


def test_config_is_validated():
    with pytest.raises(PluginError, match="one entry, or one per body"):
        ModalSound(bodies=("a", "b", "c"), presets=("free_beam", "plate"))
    with pytest.raises(PluginError, match="restitution"):
        ModalSound(restitution=(1.5,))
    with pytest.raises(ValueError, match="settle"):
        ModalSound.Config(settle=0)


def test_an_impact_sounds_the_same_on_mujoco_c_in_one_world():
    """Newton 1.8 exports MuJoCo C's contacts with forces (``use_mujoco_cpu``), so the plug-in runs there too: one
    strike in one world kicks the modes by the dt-free prediction, as on MuJoCo Warp, and nothing goes through the
    sustained path. Newton's C backend steps only the template world of a multi-world model, so there worlds 1 and
    up stay silent: the multi-world laws (S-3 linearity, S-5 contact duration) do not apply."""
    run, snd = _strike(worlds=1, solver={"use_mujoco_cpu": True})
    r = snd.read()
    assert r["kicks"][0] == pytest.approx(SL.predicted_kicks(run, 0, 1.0, 0.14), rel=1e-5, abs=1e-12)
    assert np.all(r["sustained"] == 0.0)
    assert np.all(r["pairs"] == 0)


@pytest.mark.law
def test_quick_check_passes_every_law():
    report = ap.check("modal_sound", quick=True)
    assert report.passed, report.summary()


def test_an_impact_is_counted_once():
    """WP13: strike the bar once. The kicks equal the dt-free prediction, and MuJoCo's multi-step soft impact
    (including its v_n = 0 turning point) adds nothing through the sustained path."""
    run, snd = _strike()
    r = snd.read()
    want = SL.predicted_kicks(run, 0, 1.0, 0.14)
    assert r["kicks"][0] == pytest.approx(want, rel=1e-5, abs=1e-12)
    assert np.all(r["sustained"] == 0.0)
    assert np.all(r["pairs"] == 0)                       # the ball has left: no pair in contact


def test_the_same_strike_sounds_the_same_at_dt_and_dt_over_2():
    """S-6 on the audio: the mode peaks of one strike agree within 5 % at 1.25 and 0.625 ms steps."""
    mags = []
    for dt in (2e-3, 1e-3):
        run, snd = _strike(dt=dt)
        x = snd.audio_of()
        freqs, mag = magnitude_spectrum(x[np.argmax(np.abs(x) > 0):], 44100, 1 << 18)
        mags.append([mag[np.argmin(np.abs(freqs - f))] for f in snd.sets[0].freq_hz[:6]])
    ratio = np.array(mags[1]) / np.array(mags[0])
    assert ratio == pytest.approx(np.ones(6), rel=0.05)


def test_ring_buffer_keeps_the_last_duration():
    _, long = _strike(periods=10)
    _, short = _strike(periods=10, duration=0.05)
    a, b = long.audio_of(), short.audio_of()
    assert len(b) == int(round(0.05 * 44100)) and len(a) == int(long.samples.numpy()[0])
    assert np.array_equal(b, a[-len(b):])
    assert np.any(b != 0.0)
