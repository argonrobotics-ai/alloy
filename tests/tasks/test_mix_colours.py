# SPDX-License-Identifier: Apache-2.0
"""Mix two colours (alloy/tasks/mix_colours.py): the mixing measures on synthetic particles, and a short stirred run
on the CPU (exact conservation, no spill, the dyes start mixing)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.mix_colours import MixSpec, particle_colours, run
from alloy.tasks.mix_stir import Cavity, particle_cv, portion_cv, spill_fraction

CAV = Cavity((0.0, 0.0, 0.0), (0.048, 0.048, 0.06))


def _block(n: int = 12, layers: int = 6, sp: float = 0.004) -> np.ndarray:
    g = (np.arange(n) - (n - 1) / 2.0) * sp
    x, y, z = np.meshgrid(g, g, (np.arange(layers) + 0.5) * sp, indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _split(x: np.ndarray) -> np.ndarray:
    """Red (species 0) left of x = 0, blue right, one unit per particle."""
    left = x[:, 0] < 0.0
    return np.stack([left, ~left], axis=1).astype(float)


def test_streaks_finer_than_a_portion_read_as_mixed():
    """Dyes in 4 mm streaks: 24 mm portions (2 x 2 x 2) see them as mixed, the particle CV sees every particle pure;
    two unmixed halves read as unmixed on both."""
    x = _block()
    vol, on = np.ones(len(x)), np.ones(len(x), int)
    streak = (np.round(x[:, 0] / 0.004 + 0.5).astype(int) % 2 == 0)
    amt = np.stack([streak, ~streak], axis=1).astype(float)
    assert np.allclose(portion_cv(x, amt, vol, on, CAV, (2, 2, 2)), 0.0)
    assert np.allclose(particle_cv(x, amt, vol, on, CAV), 1.0)
    halves = _split(x)
    assert np.allclose(portion_cv(x, halves, vol, on, CAV, (4, 4, 2)), 1.0)
    assert np.allclose(particle_cv(x, halves, vol, on, CAV), 1.0)


def test_uniform_dyes_have_zero_cv():
    x = _block()
    amt, vol, on = np.full((len(x), 2), 0.5), np.ones(len(x)), np.ones(len(x), int)
    assert np.allclose(portion_cv(x, amt, vol, on, CAV, (4, 4, 2)), 0.0)
    assert np.allclose(particle_cv(x, amt, vol, on, CAV), 0.0)


def test_spill_counts_liquid_outside_the_footprint_only():
    x = np.array([[0.0, 0.0, 0.01], [0.03, 0.0, 0.01], [0.0, 0.0, 0.2], [0.0, 0.0, -0.01], [0.03, 0.0, 0.01]])
    on = np.array([1, 1, 1, 1, 0])
    assert spill_fraction(x, on, CAV) == pytest.approx(0.5)   # outside in x, below the floor; above the rim is not


def test_particle_colours_blend_the_dyes():
    rgb = particle_colours(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]]))
    assert np.allclose(rgb[0], (0.85, 0.10, 0.10)) and np.allclose(rgb[1], (0.10, 0.20, 0.85))
    assert np.allclose(rgb[2], (0.475, 0.15, 0.475)) and np.allclose(rgb[3], 0.5)


@pytest.mark.compile_heavy
def test_short_stir_conserves_and_starts_mixing():
    spec = MixSpec(cavity=0.024, depth=0.012, paddle=0.008, seconds=0.5, sample=0.25, bins=(2, 2, 2))
    out = run(spec, device="cpu")
    assert out["conservation_error"] == 0
    assert out["spill"].max() == 0.0 and out["health_ok"]
    assert np.allclose(out["particle_cv"][0], 1.0)
    assert (out["particle_cv"][-1] < 0.9).all()
