# SPDX-License-Identifier: Apache-2.0
"""ModalSound host math (no simulation): mode tables, presets, Hertz pulse, the exact oscillator and audio analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from alloy.plugins.sound import analysis as A
from alloy.plugins.sound.hertz import contact_time, effective, effective_modulus, pulse_weight
from alloy.plugins.sound.modes import ModeSet, beam_modes, beam_roots, plate_modes
from alloy.plugins.sound.presets import DC6_TAB, material, preset
from alloy.plugins.sound.sensor import oscillator

FREE = (4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966)     # Blevins table 8-1
CLAMPED = (1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839)


def test_beam_roots_match_the_tables():
    assert beam_roots(5, False) == pytest.approx(FREE, abs=1e-7)
    assert beam_roots(5, True) == pytest.approx(CLAMPED, abs=1e-7)
    big = beam_roots(30, False)                                   # (2k+3)π/2 asymptote
    assert big[-1] == pytest.approx(61 * math.pi / 2, abs=1e-6)


def test_beam_frequencies_and_unit_mean_square_shapes():
    steel = material("steel")
    ms = beam_modes(0.3, 0.02, 0.005, steel, 32, samples=2001)
    c = math.sqrt(steel.youngs * 0.005 ** 2 / (12 * steel.density))
    want = np.array(FREE) ** 2 / (2 * math.pi * 0.09) * c
    assert ms.freq_hz[:5] == pytest.approx(want, rel=1e-8)
    ms2 = np.mean(ms.gains[:, :, 2] ** 2, axis=1)                 # stable up to βL ≈ 100
    assert ms2 == pytest.approx(np.ones(32), rel=2e-3) and np.all(np.isfinite(ms.gains))
    assert np.all(ms.gains[:, :, :2] == 0.0) and ms.mass[0] == pytest.approx(steel.density * 0.3 * 0.02 * 0.005)
    assert abs(ms.gains[0, 0, 2]) == pytest.approx(2.0, rel=1e-3)   # free end: |φ| = 2 at unit mean square
    tab = preset("dc6_tab", (), material("pc"), 3)
    assert np.abs(tab.gains[:, 0, 2]).max() < 1e-9                 # clamped root does not move
    assert tab.freq_hz[0] == pytest.approx(CLAMPED[0] ** 2 / (2 * math.pi * DC6_TAB[0] ** 2) * math.sqrt(
        material("pc").youngs * DC6_TAB[2] ** 2 / (12 * 1200.0)), rel=1e-8)


def test_plate_and_two_plane_beam():
    al = material("aluminium")
    ms = plate_modes(0.2, 0.15, 0.002, al, 6)
    d = al.youngs * 0.002 ** 3 / (12 * (1 - al.poisson ** 2))
    f11 = 0.5 * math.pi * (1 / 0.2 ** 2 + 1 / 0.15 ** 2) * math.sqrt(d / (al.density * 0.002))
    assert ms.freq_hz[0] == pytest.approx(f11, rel=1e-12) and np.all(np.diff(ms.freq_hz) >= 0)
    assert ms.gains[0, len(ms.points) // 2, 2] == pytest.approx(2.0)          # (1, 1) at the centre: unit mean square
    two = preset("free_beam_2", (0.3, 0.02, 0.005), material("steel"), 4)
    assert two.count == 8 and np.all(two.gains[4:, :, 2] == 0.0) and np.any(two.gains[4:, :, 1] != 0.0)
    assert two.freq_hz[4] / two.freq_hz[0] == pytest.approx(0.02 / 0.005)    # f ∝ thickness


def test_modeset_below_save_load(tmp_path):
    ms = preset("free_beam", (0.3, 0.02, 0.005), material("steel"), 12)
    cut = ms.below(5000.0, limit=4)
    assert cut.count == 4 and cut.freq_hz.max() < 5000.0
    ms.save(tmp_path / "m.npz")
    back = ModeSet.load(tmp_path / "m.npz")
    assert np.array_equal(back.gains, ms.gains) and back.area == ms.area and back.name == ms.name
    with pytest.raises(ValueError, match="gains"):
        ModeSet(ms.freq_hz, ms.zeta, ms.mass, ms.radiation, ms.points[:3], ms.gains, 1.0)


def test_material_loss_table_and_override():
    pc = material("pc")
    assert pc.zeta(201.5) == pytest.approx(0.0035) and pc.zeta(10.0) == pytest.approx(0.0035)   # held at the ends
    assert pc.zeta(1831.0) == pytest.approx(0.0175)
    assert material("pc", loss=0.02).zeta(5000.0) == pytest.approx(0.01)
    with pytest.raises(ValueError, match="known"):
        material("unobtainium")


def test_hertz_time_scales_and_pulse_weight():
    e = effective_modulus(200e9, 0.29, 200e9, 0.29)
    assert e == pytest.approx(200e9 / (2 * (1 - 0.29 ** 2)))
    assert effective(0.005, math.inf) == 0.005 and effective(1.0, 1.0) == 0.5
    t1, t32 = contact_time(0.004, 0.005, e, 1.0), contact_time(0.004, 0.005, e, 32.0)
    assert t1 / t32 == pytest.approx(2.0, rel=1e-12)                # t_c ∝ v^(-1/5)
    assert 10e-6 < t1 < 100e-6                                       # a 4 g steel ball at 1 m/s: tens of µs
    assert pulse_weight(0.0, 1.0) == pytest.approx(1.0)
    assert pulse_weight(math.pi, 1.0) == pytest.approx(math.pi / 4)  # the removable singularity
    assert pulse_weight(3 * math.pi, 1.0) == pytest.approx(0.0, abs=1e-12)   # first spectral zero
    tau = 1e-3                                                       # against a numerical Fourier transform
    t = np.linspace(0, tau, 20001)
    f = np.sin(np.pi * t / tau) * np.pi / (2 * tau)                  # unit impulse
    for om in (1000.0, 5000.0, 20000.0):
        num = abs(np.trapezoid(f * np.exp(-1j * om * t), t))
        assert pulse_weight(om, tau) == pytest.approx(num, rel=1e-6)


def test_oscillator_is_the_exact_one_sample_update():
    f, z, h = np.array([300.0, 9000.0]), np.array([0.005, 0.2]), 1.0 / 44100
    phi, gam, damp = oscillator(f, z, h)
    for k in range(2):
        w = 2 * math.pi * f[k]
        a = np.array([[0.0, 1.0, 0.0], [-w * w, -2 * z[k] * w, 1.0], [0.0, 0.0, 0.0]])   # [q, v, u]
        e = expm(a * h)
        assert phi[k] == pytest.approx(e[:2, :2].ravel(), rel=1e-9, abs=1e-12)
        assert gam[k] == pytest.approx(e[:2, 2], rel=1e-9, abs=1e-15)
        assert damp[k] == pytest.approx([2 * z[k] * w, w * w])


def test_analysis_on_synthetic_signals(tmp_path):
    fs = 44100
    t = np.arange(int(0.5 * fs)) / fs
    x = np.exp(-20.0 * t) * np.sin(2 * math.pi * 1234.5 * t) + 0.3 * np.exp(-60.0 * t) * np.sin(2 * math.pi * 3000 * t)
    freqs, mag = A.magnitude_spectrum(x, fs, 1 << 18)
    assert A.peak_near(freqs, mag, 1230.0) == pytest.approx(1234.5, rel=2e-4)
    assert A.mode_decay_rate(x, fs, 1234.5) == pytest.approx(20.0, rel=0.01)
    assert A.mode_decay_rate(x, fs, 3000.0) == pytest.approx(60.0, rel=0.01)
    assert 1234.5 < A.centroid(x, fs) < 3000.0
    assert A.top_peaks(freqs, mag, k=2) == pytest.approx([1234.5, 3000.0], abs=2.0)
    scale = A.write_wav(str(tmp_path / "x.wav"), x, fs)
    back, fs2 = A.read_wav(str(tmp_path / "x.wav"))
    assert fs2 == fs and np.max(np.abs(back - x * scale)) < 2.0 / 32767
