# SPDX-License-Identifier: Apache-2.0
"""Cost model (SPEC §3.9, parallel §4.1, §4.3, §4.7): fit, knee, pack size, memory and dense-H prediction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.core.errors import CapacityError
from alloy.parallel import costmodel as cm
from alloy.parallel.types import PackPolicy
from alloy.testing.parallel import GB, profile

@pytest.mark.parametrize("t_floor,c_w", [(3.27e-3, 27.9e-6), (2.5e-4, 1.0e-6), (1.0e-2, 4.0e-4)])
def test_fit_recovers_exact_synthetic(t_floor, c_w):
    n = [1, 64, 1024]
    t = [t_floor + k * c_w for k in n]
    tf, cw = cm.fit_cost_model(n, t)
    assert tf == pytest.approx(t_floor, rel=1e-9) and cw == pytest.approx(c_w, rel=1e-9)


def test_fit_recovers_noisy_synthetic_within_1pct():
    # T12 numbers from parallel §3 (3.27 ms floor, 27.9 µs per world); ±0.3 % deterministic jitter on 7 rungs
    t_floor, c_w = 3.27e-3, 27.9e-6
    n = np.array([1, 4, 16, 64, 256, 1024, 4096])
    jitter = 1.0 + 0.003 * np.array([1, -1, 0.5, -0.5, 1, -1, 0.3])
    tf, cw = cm.fit_cost_model(n, (t_floor + n * c_w) * jitter)
    assert abs(tf / t_floor - 1.0) < 0.01 and abs(cw / c_w - 1.0) < 0.01


def test_fit_clamps_negative_parameters():
    tf, cw = cm.fit_cost_model([1, 2, 4], [1e-3, 1e-3, 1e-3])  # flat: no per-world cost visible
    assert cw == 0.0 and tf == pytest.approx(1e-3)
    tf, cw = cm.fit_cost_model([10, 20], [1e-3, 2e-3])  # line through the origin
    assert tf == pytest.approx(0.0, abs=1e-12) and cw == pytest.approx(1e-4)


@pytest.mark.parametrize("n,t", [([1, 1], [1.0, 2.0]), ([1, 2], [0.0, 1.0]), ([1, 2], [1.0, float("nan")]),
                                 ([0, 2], [1.0, 2.0]), ([1, 2, 3], [1.0, 2.0])])
def test_fit_rejects_bad_input(n, t):
    with pytest.raises(ValueError):
        cm.fit_cost_model(n, t)


def test_knee_throughput_occupancy():
    p = profile()
    knee = 3.27e-3 / 27.9e-6  # 117.2 (parallel §3: T12 knee ≈ 117)
    assert cm.n_knee(p) == pytest.approx(knee)
    assert cm.throughput(p, 64) == pytest.approx(64 * 0.02 / (3.27e-3 + 64 * 27.9e-6))
    assert cm.occupancy(p, round(knee)) == pytest.approx(0.5, abs=0.01)
    assert cm.n_knee(profile(c_w_s=0.0)) == math.inf


def test_plateau_is_nine_knees_at_90pct():
    p = profile()
    n = cm.plateau_n(p, 0.9)
    assert n == math.ceil(9 * 3.27e-3 / 27.9e-6)  # 1055 (parallel §3: 90 % of plateau at ~1,050)
    theta_max = p.dt / p.c_w_s
    assert cm.throughput(p, n) >= 0.9 * theta_max > cm.throughput(p, n - 1)


def test_mjw_padding_matches_mujoco_warp():
    # io.py:1410-1417: nv <= 32 rounds to 4; above, nv + 1 rounds to the 16-wide tile
    assert cm.mjw_nv_pad(9) == 12 and cm.mjw_nv_pad(32) == 32 and cm.mjw_nv_pad(33) == 48
    assert cm.mjw_nv_pad(1200) == 1216


def test_200_box_dense_h_and_oom_prediction():
    # parallel §3: 200 boxes, nv_pad 1216 -> h + hfactor = 2 * 4 * 1216² = 11.8 MB per world; 8,192 worlds need
    # 97 GB > 80 GB (measured OOM on an 80 GB H100)
    h = cm.dense_h_bytes(1200)
    assert h == 2 * 4 * 1216**2 and h / 1e6 == pytest.approx(11.83, abs=0.01)
    per_world = cm.estimate_bytes_per_world(cm.MemoryTerms(nv=1200, rows_max=4800, contacts_max=1600,
                                                           pairs_max=20_000, n_bodies=200))
    assert per_world > h
    p = profile(nv=1200, island_nv_max=1200, bytes_per_world=per_world, bytes_fixed=0, t_floor_s=5e-3,
                c_w_s=4e-4)
    assert 8192 * per_world > 80 * GB
    assert cm.n_mem(p, 80 * GB, PackPolicy()) < 8192
    assert cm.pack_size(p, 8192, 80 * GB, PackPolicy()) < 8192


def test_pack_size_takes_the_smallest_cap():
    pol = PackPolicy()
    p = profile()
    assert cm.pack_size(p, 100, 24 * GB, pol) == 100  # count
    assert cm.pack_size(p, 100_000, 24 * GB, pol) == cm.plateau_n(p, 0.9)  # knee
    tight = cm.pack_size(profile(bytes_per_world=10_000_000), 100_000, 2 * GB, pol)
    assert tight == math.floor((0.9 * 2 * GB - 50_000_000) / 10_000_000)  # memory
    dense = profile(nv=600, bytes_per_world=1, c_w_s=1e-9)  # dense H 2*4*608² B, 25 % of 1 GB
    assert cm.pack_size(dense, 100_000, GB, pol) == math.floor(0.25 * GB / cm.dense_h_bytes(600))


def test_pack_size_raises_when_one_world_does_not_fit():
    with pytest.raises(CapacityError):
        cm.pack_size(profile(bytes_fixed=GB), 10, GB, PackPolicy())


def test_capacities_headroom():
    caps = cm.capacities(profile(contacts_p99=10, rows_p99=41, pairs_peak=8), 64, PackPolicy())
    assert caps == {"rigid_contact_max": 15 * 64, "njmax": 62, "nconmax": 15, "shape_pairs_max": 10 * 64}
    assert cm.capacities(profile(contacts_p99=0, rows_p99=0, pairs_peak=0), 4, PackPolicy())["nconmax"] == 1


def test_capacities_from_counts_matches_profile_rule():
    p = profile(contacts_p99=96, rows_p99=402, pairs_peak=10)
    assert cm.capacities_from_counts(96, 402, 10, 8, PackPolicy()) == cm.capacities(p, 8, PackPolicy())
    assert cm.capacities_from_counts(96, 402, 0, 8, PackPolicy())["shape_pairs_max"] == 8  # at least 1 per world
