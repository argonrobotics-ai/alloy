# SPDX-License-Identifier: Apache-2.0
"""The tactile noise layer on synthetic inputs (CPU): each term against its closed form, and repeatability."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.tactile import noise as N

W, P, T = 4, 2, 16          # worlds, pads, taxels (a 4 x 4 grid)
SLOTS = W * P * T


def _run(raw: np.ndarray, steps: int = 1, dt: float = 0.01, *, noise=0.0, drift=0.0, tau=(0.0, 0.0), dead=None,
         crosstalk=0.0, saturation=0.0, lsb=0.0, seed=0) -> np.ndarray:
    dev = "cpu"
    raw_a = wp.array(raw.astype(np.float32).reshape(-1), dtype=wp.float32, device=dev)
    nb = wp.array(N.grid_neighbours(4, 4), dtype=wp.int32, device=dev)
    tick = wp.zeros(W, dtype=wp.int32, device=dev)
    dead_a = wp.array(np.zeros(SLOTS, np.int32) if dead is None else dead, dtype=wp.int32, device=dev)
    hyst, bias, out = (wp.zeros(SLOTS * 3, dtype=wp.float32, device=dev) for _ in range(3))
    for _ in range(steps):
        wp.launch(N.apply_noise, dim=SLOTS, device=dev,
                  inputs=[raw_a, nb, T, P, tick, dead_a, wp.vec4(noise, drift, *tau), wp.vec4(crosstalk, saturation,
                          lsb, 0.0), seed, dt, hyst, bias], outputs=[out])
        wp.launch(N.advance_tick, dim=W, device=dev, inputs=[tick])
    return out.numpy().reshape(W, P, T, 3)


def test_all_off_passes_the_signal_through():
    raw = np.random.default_rng(0).normal(size=(SLOTS, 3))
    np.testing.assert_array_equal(_run(raw).reshape(SLOTS, 3), raw.astype(np.float32))


def test_drift_is_a_random_walk():
    out = _run(np.zeros((SLOTS, 3)), steps=400, dt=0.01, drift=1.0)
    assert abs(out.std() / math.sqrt(400 * 0.01) - 1.0) < 0.1


def test_white_noise_std():
    out = _run(np.zeros((SLOTS, 3)), noise=0.5)
    assert abs(out.std() / 0.5 - 1.0) < 0.1


def test_hysteresis_lags_by_its_time_constants():
    up = _run(np.ones((SLOTS, 3)), steps=1, dt=0.01, tau=(0.1, 0.4))
    np.testing.assert_allclose(up, 1.0 - math.exp(-0.1), rtol=1e-5)


def test_dead_taxels_read_zero_and_count_exactly():
    mask = N.dead_mask(W, P * T, 0.25, seed=3)
    assert (mask.reshape(W, -1).sum(1) == 8).all()
    out = _run(np.ones((SLOTS, 3)), dead=mask).reshape(SLOTS, 3)
    assert (out[mask == 1] == 0).all() and (out[mask == 0] == 1).all()


def test_crosstalk_shares_to_the_four_neighbours():
    raw = np.zeros((W, P, T, 3))
    raw[:, :, 5, 0] = 1.0                       # taxel (1, 1) of the 4 x 4 grid
    out = _run(raw.reshape(SLOTS, 3), crosstalk=0.1)
    assert out[0, 0, 5, 0] == pytest.approx(1.0)
    for j in (4, 6, 1, 9):
        assert out[0, 0, j, 0] == pytest.approx(0.1)
    assert out[0, 0, 0, 0] == 0.0


def test_saturation_and_adc_step():
    raw = np.full((SLOTS, 3), 7.26)
    out = _run(raw, saturation=5.0, lsb=0.5)
    np.testing.assert_allclose(out, 5.0)
    out = _run(np.full((SLOTS, 3), 1.26), lsb=0.5)
    np.testing.assert_allclose(out, 1.5)


def test_draws_repeat_per_seed_and_differ_between_worlds():
    a = _run(np.zeros((SLOTS, 3)), steps=3, noise=1.0, seed=7)
    b = _run(np.zeros((SLOTS, 3)), steps=3, noise=1.0, seed=7)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[0], a[1])
    assert not np.array_equal(a, _run(np.zeros((SLOTS, 3)), steps=3, noise=1.0, seed=8))


def test_settings_are_validated():
    with pytest.raises(ValueError):
        N.NoiseParams(noise=-1.0)
    with pytest.raises(ValueError):
        N.NoiseParams(dead=1.5)
