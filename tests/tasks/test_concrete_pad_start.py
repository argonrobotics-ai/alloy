# SPDX-License-Identifier: Apache-2.0
"""task concrete_pad_start: Gym-Anything's pad-darby / pad-trowel start state (layout.py:224-343), CPU, fast.

The port was also checked cell for cell against their ``pad_region`` (b, tb, ts, E, a, D, w, q, S identical for seeds
0, 2, 3, 5 at 21.1 and 15 C); these tests pin the spec's numbers and the maths without needing their package."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_pad_start as PS

FT = PS.FORM_H


@pytest.fixture(scope="module")
def darby1():
    return PS.pad_fields("pad-darby", 1)       # odd seed: the low corner at (PAD_SIZE[0] - 0.2, 0.18) (layout.py:246)


@pytest.fixture(scope="module")
def trowel0():
    return PS.pad_fields("pad-trowel", 0)


def _smooth_x(a, cells=30):
    """Moving mean over cells along x (60 mm = 30 cells kills the ridges)."""
    k = np.ones(cells) / cells
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, a)


def test_grid_and_interior_as_theirs(darby1):
    assert darby1["h"].shape == (650, 497) and darby1["x"].shape == (650,) and darby1["y"].shape == (497,)
    assert darby1["cell"] == 0.002 and np.allclose(np.diff(darby1["x"]), 0.002)
    m = darby1["paste_mask"]
    xs, ys = darby1["x"][m.any(1)], darby1["y"][m.any(0)]
    assert abs((xs[-1] - xs[0]) - 1.219) < 0.003 and abs((ys[-1] - ys[0]) - 0.914) < 0.003
    assert abs(xs.mean()) < 0.002 and abs(ys.mean()) < 0.002          # origin at the interior's centre
    # origin one form width + 2 mm outside the interior: first cell centre 1 mm further in
    assert darby1["x"][0] == pytest.approx(-0.6095 - 0.040 + 0.001) and darby1["y"][0] == pytest.approx(-0.457 - 0.039)
    assert np.array_equal(darby1["form_mask"], ~m)
    assert np.all(darby1["b"][m] == 0) and np.all(darby1["b"][~m] == 0.089) and darby1["form_top"] == 0.089


def test_darby_struck_surface(darby1):
    f, m = darby1, darby1["paste_mask"]
    rel = (f["h"] - FT)[m]
    assert -0.0085 < rel.min() < -0.004 and 0.003 < rel.max() < 0.0075          # ridges + features + 0.5 mm noise
    # ridges: 60 mm wavelength, +-2 mm, on a row away from the low corner and the high band
    row = f["h"][:, np.argmin(np.abs(f["y"] - 0.40))][m[:, np.argmin(np.abs(f["y"] - 0.40))]]
    hp = row - _smooth_x(row[:, None], 90)[:, 0]
    hp = hp[100:-100]
    spec = np.abs(np.fft.rfft(hp - hp.mean()))
    assert 1 / np.fft.rfftfreq(hp.size, 0.002)[spec.argmax()] == pytest.approx(0.060, rel=0.1)
    assert 0.0012 < hp.std() * np.sqrt(2) < 0.0026
    # low corner at (1.019, 0.18) from the south-west corner: about -5 mm; high band: the smoothed maximum, about +4 mm
    sm = _smooth_x(f["h"] - FT)
    lo = (np.argmin(np.abs(f["x"] + 0.6095 - 1.019)), np.argmin(np.abs(f["y"] + 0.457 - 0.18)))
    assert -0.0058 < sm[lo] < -0.0035
    assert sm[lo] < sm[m].min() + 0.0006
    assert 0.0025 < sm[m].max() < 0.0045
    assert f["x"][np.unravel_index(np.where(m, sm, -1).argmax(), sm.shape)[0]] + 0.6095 < 0.9      # not the low corner


def test_darby_layers_and_age(darby1):
    f, m = darby1, darby1["paste_mask"]
    assert f["E"][m].mean() == pytest.approx(0.30, abs=0.01) and 0.05 < f["E"][m].std() < 0.2
    assert np.all(f["a"][m] == np.float32(8.0e-4)) and np.all(f["ts"][m] == 0.0008) and np.all(f["D"] == 0)
    assert np.allclose(f["tb"][m] + f["ts"][m] + FT * 0, f["h"][m]) and np.all(f["tb"][m] > 0.08)
    assert f["age"] == pytest.approx(300 * CM.masco_factor(21.1)) and abs(f["age"] - 300) < 1
    assert PS.pad_fields("pad-darby", 1, temp_c=27.0)["age"] == pytest.approx(300 * CM.masco_factor(27.0))
    assert PS.pad_fields("pad-darby", 1, temp_c=27.0)["age"] < 300          # warmer: same stage, shorter clock
    assert np.all(f["bl"] == 0) and np.all(f["q"] == 0) and np.all(f["S"] == 0)


def test_darby_smear_on_two_form_tops(darby1):
    f = darby1
    sm, X, Y = f["smear"], f["x"][:, None] + 0.6095, f["y"][None, :] + 0.457
    south = sm[(Y < 0) & (Y > -0.038) & (X > 0.2) & (X < 0.75)]
    west = sm[(X < 0) & (X > -0.038) & (Y > 0.3) & (Y < 0.6)]
    assert 0.001 < south.max() <= 0.003 and 0.001 < west.max() <= 0.003
    assert (south > 0).mean() > 0.6 and (west > 0).mean() > 0.6
    assert np.count_nonzero(sm) == np.count_nonzero(south) + np.count_nonzero(west)      # north and east tops clean
    forms = f["form_mask"]
    assert np.allclose((f["h"] - FT)[forms], sm[forms]) and np.all(sm[f["paste_mask"]] == 0)


def test_trowel_floated_flat_with_edged_arris(trowel0):
    f, m, d = trowel0, trowel0["paste_mask"], trowel0["d_edge"]
    field = m & (d > PS.EDGER_TRACK_W + 0.002)
    rel = (f["h"] - FT)[field]
    assert np.abs(rel).max() < 0.0016 and 0.0002 < rel.std() < 0.0004          # flat within about +-1.3 mm
    track = m & (d > 0.015) & (d < 0.07)
    assert abs((f["h"] - FT)[track].mean()) < 0.0002                          # the 1.2 mm track floated out
    r = PS.EDGER_RADIUS
    for dd in (0.001, 0.003, 0.006):                                          # the arris: r - sqrt(r^2 - (r - d)^2)
        ring = m & (np.abs(d - dd) < 0.0011)
        assert (f["h"] - FT)[ring].mean() == pytest.approx(-(r - np.sqrt(r * r - (r - dd) ** 2)), abs=0.0008)
    assert f["edger_radius"] == 0.0095 and np.all(f["edge_drop"][m & (d < 0.0095)] > 0)
    assert f["track"][m & (d < 0.07)].all()
    assert f["compact"][m & (d < 0.0025)].min() == 1 and f["compact"][field].max() == 0


def test_trowel_layers_and_age(trowel0):
    f, m = trowel0, trowel0["paste_mask"]
    assert np.all(f["a"][m] == np.float32(2.5e-4)) and np.all(f["ts"][m] == 0.003)
    assert np.all(f["D"][m] == np.float32(0.10)) and np.all(f["E"][m] == np.float32(0.01))
    assert f["age"] == pytest.approx(165 * 60 * CM.masco_factor(21.1)) and abs(f["age"] - 9900) < 10
    assert np.all(f["tb"][m] > 0.07) and np.allclose(f["b"] + f["tb"] + f["ts"] + f["bl"], f["h"])


def _integrate_bleed(age, temp_c, air, mean):
    """Independent float64 integration of the bleed film: 10 s steps, mid-step age, evaporation, clipped at 0."""
    mat, w, clock = CM.Concrete(bleed_mean=mean), 0.0, -age
    air_c, rh, wind = CM.AIR[air]
    evap = CM.evaporation_kg_m2_h(temp_c, air_c + temp_c - 21.1, rh, wind)
    while clock < -1e-9:
        step = min(10.0, -clock)
        rate = float(mat.bleed_rate(clock + step / 2 + age, temp_c))
        w = max(w + step * rate / 1000 - step * evap / 3600 / 1000, 0.0)
        clock += step
    return w


def test_bleed_history_equals_independent_integration(trowel0):
    # at 21.1 C in still air the film has evaporated by 165 min (as theirs: w = 0, q = 0 at the start)
    assert trowel0["w"].max() == 0 and trowel0["q"].max() == 0
    assert _integrate_bleed(trowel0["age"], 21.1, "still", CM.BLEED_MEAN) == 0.0
    cases = [(3000.0, 21.1, "still", 0.15), (9900.0, 15.0, "still", 0.18), (5000.0, 21.1, "still", 0.5),
             (9900.0, 15.0, "door", 0.18)]
    for age, t, air, mean in cases:
        got = PS.bleed_history(age, t, air, mean / CM.BLEED_MEAN)
        want = _integrate_bleed(age, t, air, mean)
        assert want > 1e-6 or (age, t) == (9900.0, 15.0)
        assert got == pytest.approx(want, rel=2e-3, abs=1e-9)
    assert _integrate_bleed(5000.0, 21.1, "still", 0.5) > 1e-4
    # the field carries the same film on the paste only
    f = PS.pad_fields("pad-trowel", 0, temp_c=15.0, air="still", bleed_factor=1.2)
    assert f["w"][f["paste_mask"]].min() == f["w"][f["paste_mask"]].max() > 0 and np.all(f["w"][f["form_mask"]] == 0)
    assert f["w"].max() == pytest.approx(_integrate_bleed(f["age"], 15.0, "still", 0.18), rel=2e-3)


def test_seeds_differ_same_seed_identical():
    a, b, a2 = PS.pad_fields("pad-darby", 4), PS.pad_fields("pad-darby", 5), PS.pad_fields("pad-darby", 4)
    assert np.array_equal(a["h"], a2["h"]) and np.array_equal(a["E"], a2["E"])
    assert np.abs(a["h"] - b["h"]).max() > 0.002 and np.abs(a["E"] - b["E"]).max() > 0.1
    t0, t1 = PS.pad_fields("pad-trowel", 0), PS.pad_fields("pad-trowel", 1)
    assert np.abs(t0["h"] - t1["h"]).max() > 1e-4
    with pytest.raises(ValueError):
        PS.pad_fields("pad-sweep", 0)


def test_block_conserves_mean_and_masks(darby1):
    f = darby1
    for factor in (1, 2, 5):
        coarse = PS.blocked(f, factor)
        n0, n1 = (650 // factor) * factor, (497 // factor) * factor
        assert coarse["h"].shape == (650 // factor, 497 // factor) and coarse["cell"] == pytest.approx(0.002 * factor)
        assert coarse["h"].mean() == pytest.approx(f["h"][:n0, :n1].mean(), abs=1e-12)
        assert coarse["tb"].mean() == pytest.approx(f["tb"][:n0, :n1].mean(), abs=1e-12)
        assert coarse["paste_mask"].dtype == bool and coarse["x"].shape == (650 // factor,)
        assert coarse["paste_mask"].sum() * factor**2 == pytest.approx(f["paste_mask"].sum(), rel=0.02)
    five = PS.block(f["paste_mask"], 5)
    assert five[0, 0] == (f["paste_mask"][:5, :5].mean() >= 0.5)
    assert PS.block(np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1]], bool), 2).tolist() == [[True, False]]
    with pytest.raises(ValueError):
        PS.block(f["h"], 0)
