# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything laundry gap (coordinator 2026-11-06 18:40Z): the towel fold task's gates and the fold rig's geometry
on synthetic sheets (CPU, no simulation), then the bath towel half and thirds folds through the cloth route
(``alloy.tasks.laundry_fold``, slow)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import laundry_fold as T
from alloy.tasks.laundry_crossings import FoldRig, FoldState, _corners, _Flap


def _grid(nx: int = 15, ny: int = 10, lx: float = 0.70, ly: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.linspace(-lx / 2, lx / 2, nx + 1), np.linspace(-ly / 2, ly / 2, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    q = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    t = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1
            t += [[a, b, c], [a, c, d]]
    return q, np.asarray(t)


def _folded_half(q0: np.ndarray, h: float = 6e-4) -> np.ndarray:
    """The ideal half fold: y < 0 mirrored onto y > 0, one ply up."""
    q = q0.copy()
    low = q0[:, 1] < 0
    q[low, 1] = -q0[low, 1]
    q[low, 2] = h
    return q


def _folded_thirds(q0: np.ndarray, h: float = 6e-4) -> np.ndarray:
    ly = np.ptp(q0[:, 1])
    w = ly / 3.0
    y0 = q0[:, 1].min()
    q = q0.copy()
    lo, hi = q0[:, 1] < y0 + w - 1e-9, q0[:, 1] > y0 + 2 * w + 1e-9
    q[lo, 1] = 2 * (y0 + w) - q0[lo, 1]
    q[lo, 2] = h
    q[hi, 1] = 2 * (y0 + 2 * w) - q0[hi, 1]
    q[hi, 2] = 2 * h
    return q


def _state(q0, q, tri, h=6e-4, log=((1.0, 0),)) -> FoldState:
    return FoldState(q0, q, tri, h, list(log), 0.0, 1e-3, {})


def test_an_ideal_half_fold_passes_the_half_gates():
    q0, tri = _grid()
    gates, m = T.evaluate(_state(q0, _folded_half(q0), tri), "half")
    assert all(gates.values()), (gates, m)
    assert m["hem_rms"] < 1e-9 and m["area_ratio"] == pytest.approx(0.5, abs=1e-6)


def test_a_flat_towel_fails_the_half_gates():
    q0, tri = _grid()
    gates, m = T.evaluate(_state(q0, q0.copy(), tri), "half")
    assert not gates["hem_rms"] and not gates["area"]
    assert m["hem_rms"] == pytest.approx(0.45, rel=1e-6)


def test_an_ideal_thirds_fold_passes_the_thirds_gates():
    q0, tri = _grid(14, 30)
    gates, m = T.evaluate(_state(q0, _folded_thirds(q0), tri), "thirds")
    assert all(gates.values()), (gates, m)
    assert m["three_ply"] >= 0.8 and m["area_ratio"] == pytest.approx(1 / 3, abs=1e-6)


def test_a_half_fold_is_not_a_thirds_fold():
    q0, tri = _grid(14, 30)
    gates, _ = T.evaluate(_state(q0, _folded_half(q0), tri), "thirds")
    assert not gates["area"] and not gates["three_ply"]


def test_a_crossing_at_the_end_fails_and_stretch_is_measured():
    q0, tri = _grid()
    q = _folded_half(q0)
    gates, m = T.evaluate(_state(q0, q, tri, log=[(0.5, 3), (1.0, 2)]), "half")
    assert not gates["no_crossing_end"] and m["crossings_max"] == 3
    q[:, 0] *= 1.5
    assert T.stretch(q0, q, tri) == pytest.approx(0.5, rel=1e-6)


def test_ply_counts_see_stacked_sheets():
    q0, tri = _grid(4, 4, 0.1, 0.1)
    h = 1e-3
    layers = [q0 + np.array([0.0, 0.0, k * h]) for k in range(3)]
    q = np.concatenate(layers)
    t = np.concatenate([tri + k * len(q0) for k in range(3)])
    pts = np.array([[0.0, 0.0], [0.012, -0.03], [0.2, 0.2]])
    assert T.ply_counts(q, t, pts, h).tolist() == [3, 3, 0]


def test_the_swing_starts_at_the_grasp_and_lands_mirrored_over_the_fold_line():
    start = np.array([[0.1, -0.2, 3e-4], [0.2, -0.2, 3e-4]])
    flap = _Flap(np.array([0, 1]), start, y_fold=0.0, z_land=9e-4, apex=0.5)
    assert np.allclose(flap.at(0.0), start)
    assert np.allclose(flap.at(1.0), [[0.1, 0.2, 9e-4], [0.2, 0.2, 9e-4]])
    assert flap.at(0.5)[0, 2] == pytest.approx(3e-4 + 0.5 * 0.2 + 0.5 * 6e-4)
    assert math.isclose(flap.at(0.5)[0, 1], 0.0, abs_tol=1e-12)


def test_corner_grasps_are_two_patches_at_the_row_ends():
    q0, _ = _grid(28, 4)
    row = np.flatnonzero(np.isclose(q0[:, 1], q0[:, 1].min()))
    ids = _corners(q0, row, FoldRig(corner_inset=0.03, patch=0.015))
    x = np.sort(q0[ids, 0])
    assert len(ids) >= 2 and x[0] < -0.30 and x[-1] > 0.30
    assert np.all(np.minimum(np.abs(x + 0.32), np.abs(x - 0.32)) <= 0.015 + 1e-9)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(5400)
def test_the_bath_towel_folds_in_half():
    """Row (CPU, 0.70 x 0.45 m bath towel, two corner grasps): hem RMS 1.8 mm, max 2.8 mm, area 50.0 %, 0 crossings
    in every frame, stretch 0.8 % (~4 min)."""
    v = T.run("half")
    assert v.success, v
    assert v.metrics["crossings_max"] == 0, v.metrics


@pytest.fixture(scope="module")
def thirds_fold():
    """One thirds fold through the cloth route (CPU, ~2 h at the 40-iteration default), shared by the two laws below."""
    return T.run("thirds")


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(10800)
def test_the_bath_towel_folds_in_thirds(thirds_fold):
    """Every Gym-Anything gate but the normal extent (the next law). Row (CPU, SCR-WP11-F32 3057f60a7): area 34.2 %,
    hems inside, short hem 0.152 m, length 0.700 m, three-ply 91.1 %, 0 crossings in every frame, stretch 0.85 %
    (main 7f7e9a937 at 40 iterations: 34.3 %, 0.153 m, 0.700 m, 89.8 %, 0.91 %)."""
    v = thirds_fold
    assert {k for k, ok in v.gates.items() if not ok} <= {"normal_extent"}, v
    assert v.metrics["crossings_max"] == 0, v.metrics


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(10800)
@pytest.mark.xfail(strict=False, reason="known gap cloth-fold-unsettled-default: the folded towel is still moving at "
                   "the measuring point, so its height is a draw: with the default's self-contact hold and "
                   "SCR-WP11-F32, 2 of 8 nudged starts are over 25 mm (26.0 and 30.5 mm; without F32 0 of 8). Kept "
                   "by the coordinator's ruling of 2026-11-10 02:42Z; the 35 mm guard below is hard")
def test_the_thirds_fold_lies_within_25_mm(thirds_fold):
    """Gym-Anything's normal extent (ptp z <= 25 mm). Rows (CPU, the start nudged 1 um per seed; WP2's 8-seed
    spreads, 2026-11-09): with the default's self-contact hold, main before SCR-WP11-F32 0 of 8 over 25 mm, with
    SCR-WP11-F32 2 of 8 (26.0 and 30.5 mm). Without the hold: main 3 of 10 over (17.9-39.0 mm), F32 5 of 10
    (12.7-35.7 mm), the towel moving at 0.06-0.13 m/s ([internal note] notes/wp11-f32/spread; fold_spread.py)."""
    assert thirds_fold.gates["normal_extent"], thirds_fold.metrics


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(10800)
def test_the_thirds_fold_lies_within_35_mm(thirds_fold):
    """The hard guard beside the 25 mm draw (coordinator 2026-11-10 02:42Z): the worst of 8 nudged starts with the
    default's hold and SCR-WP11-F32 is 30.5 mm. A fold standing higher than 35 mm is a regression, not the draw."""
    assert thirds_fold.metrics["normal_extent"] <= 0.035, thirds_fold.metrics
