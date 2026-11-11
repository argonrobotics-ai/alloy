# SPDX-License-Identifier: Apache-2.0
"""LIQ-2 / LIQ-3 and solver iterations on a resting column (SPEC §5.2, §7 WP9 CPU acceptance).

Column 6 x 6 x 50 (H/(2r) = 50, 1800 particles, r = 1 mm) in a box tank, dt = 1.25 ms (D4 grid at 0.02 s),
default tolerances. ``prime`` gives it its hydrostatic pressure; 1 s is simulated. With ``ALLOY_WP9_RESULTS`` set,
the metrics are also written to ``$ALLOY_WP9_RESULTS/wp9_rest_column_cpu.json`` (cited by docs/notes/wp9-results.md).
The iteration bound (SPEC §7 WP9: the tolerance is not over-tight) is checked at the base tolerance
``max_density_error`` without the resting-jitter rule; the rule's iterations on the same column are reported beside it
(SCR-SC-C2: its cost shows, it does not block).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.testing.sph import column_height, default_config, water_column, write_result

DT = 1.25e-3


def _run_column(nz: int, steps: int, nx: int = 6, device: str | None = None, **cfg_kw) -> dict:
    sc = water_column(nx, nx, nz, cfg=default_config(**cfg_kw), device=device)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    prime_iters = s.prime(a, DT)
    h0 = float(column_height(s, a, sc.base_area)[0])
    den, den_max, it_den, it_div = [], [], [], []
    for _ in range(steps):
        s.step(a, b, None, None, DT)
        a, b = b, a
        r = s.report()
        den.append(float(r.density_err[0]))
        den_max.append(float(s.stats.diagnostics()["density_err_max"][0]))
        it_den.append(int(r.iters_density[0]))
        it_div.append(int(r.iters_divergence[0]))
    h1 = float(column_height(s, a, sc.base_area)[0])
    q = a.particle_q.numpy()
    lo, hi = np.array(sc.inner_lo), np.array(sc.inner_hi)
    outside = int(np.sum(np.any(q[:, :2] < lo[:2], axis=1) | np.any(q[:, :2] > hi[:2], axis=1) | (q[:, 2] < lo[2])))
    return {
        "nx": nx, "nz": nz, "H_over_2r": nz, "particles": nx * nx * nz, "steps": steps, "dt": DT, "prime_iterations": prime_iters,
        "liq2_drift_pct": 100.0 * (h1 - h0) / h0, "liq3_mean_density_err": float(np.mean(den)),
        "liq3_max_density_err": float(np.max(den_max)), "mean_iters_density": float(np.mean(it_den)),
        "mean_iters_divergence": float(np.mean(it_div)), "max_speed_end": float(s.report().max_speed[0]),
        "escaped": int(s.report().escaped[0]), "outside_tank": outside,
    }


@pytest.fixture(scope="module")
def column_1s():
    out = _run_column(50, int(round(1.0 / DT)))
    base = _run_column(50, int(round(1.0 / DT)), rest_jitter_speed=None)
    out["base_tolerance"] = {k: base[k] for k in ("mean_iters_density", "mean_iters_divergence",
                                                    "liq3_mean_density_err", "liq2_drift_pct")}
    write_result("wp9_rest_column_cpu", out)
    return out


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_liq2_column_height_drift_1s(column_1s):
    assert abs(column_1s["liq2_drift_pct"]) <= 0.5
    assert column_1s["escaped"] == 0 and column_1s["outside_tank"] == 0


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_liq3_density_error(column_1s):
    assert column_1s["liq3_mean_density_err"] <= 1.0e-3
    assert column_1s["liq3_max_density_err"] <= 1.0e-2


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_mean_iterations_below_10(column_1s):
    """At the base tolerance (resting-jitter rule off); the rule's own iterations are in the result file."""
    assert column_1s["base_tolerance"]["mean_iters_density"] < 10.0
    assert column_1s["base_tolerance"]["mean_iters_divergence"] < 10.0


def test_rest_state_density_exact_at_walls():
    """The initial lattice in the tank is the SPH rest state: interior and wall layers at rho0, edges and corners
    at most rho0 (boundary_fit + inclusion-exclusion)."""
    sc = water_column(6, 6, 8)
    s = sc.make_solver()
    ratio, active = s.density_ratio(sc.model.state())
    q = sc.model.particle_q.numpy()
    ix = np.round(q / sc.config.spacing - 0.5).astype(int)
    below_top = ix[:, 2] < 7
    on_x_wall = (ix[:, 0] == 0) | (ix[:, 0] == 5)
    on_y_wall = (ix[:, 1] == 0) | (ix[:, 1] == 5)
    on_floor = ix[:, 2] == 0
    walls = (on_x_wall.astype(int) + on_y_wall.astype(int) + on_floor.astype(int))
    assert np.all(np.abs(ratio[below_top & (walls <= 1)] - 1.0) < 2e-4)
    assert np.all(ratio[below_top & (walls >= 2)] <= 1.0 + 2e-4)
    assert np.all(ratio[active] > 0.8)


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_liq2_liq3_column_10s_gpu():
    """GPU acceptance: LIQ-2 over 10 s (SPEC §5.2) on a 16 x 16 x 60 column (H/(2r) = 60, 15 360 particles)."""
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    out = _run_column(60, int(round(10.0 / DT)), nx=16, device="cuda:0")
    write_result("wp9_rest_column_gpu", out)
    assert abs(out["liq2_drift_pct"]) <= 0.5 and out["escaped"] == 0 and out["outside_tank"] == 0
    assert out["liq3_mean_density_err"] <= 1.0e-3 and out["liq3_max_density_err"] <= 1.0e-2
