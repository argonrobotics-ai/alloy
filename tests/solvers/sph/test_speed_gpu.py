# SPDX-License-Identifier: Apache-2.0
"""LIQ-9 (SPEC §5.2): SolverDFSPH speed on one L4, reported (targets: 1 world x 30k particles >= 0.5x real time;
256 worlds x 4k >= 50 world-s per wall-s). Timing counts only from an exclusive run (PRF-08); the numbers go to
``$ALLOY_WP9_RESULTS/wp9_speed_gpu.json`` and docs/notes/wp9-results.md."""

from __future__ import annotations

import pytest
import warp as wp

from alloy.solvers.sph.config import snap_down, stable_dt
from alloy.testing.sph import dam_break, default_config, time_steps, water_column, write_result

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(3600)]
DEVICE = "cuda:0"


@pytest.fixture(autouse=True)
def _cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")


def test_liq9_speed_report():
    cfg = default_config(r=2.0e-3)  # router default radius
    out = {}
    rest = water_column(32, 32, 30, cfg=cfg, device=DEVICE)  # 30 720 particles at rest, dt_max
    t = time_steps(rest.make_solver(), rest.model, cfg.dt_max, 400)
    out["one_world_30k_rest"] = {**t, "dt": cfg.dt_max, "rt_factor": cfg.dt_max / t["wall_per_step_s"]}
    dam = dam_break(32, 30, 32, 96, cfg=default_config(r=2.0e-3), device=DEVICE)  # 30 720, surge front
    dt = snap_down(stable_dt(cfg, 2.0 * (9.81 * 30 * cfg.spacing) ** 0.5), 0.02)  # CFL at the Ritter speed
    t = time_steps(dam.make_solver(), dam.model, dt, 400)
    out["one_world_30k_dam_break"] = {**t, "dt": dt, "rt_factor": dt / t["wall_per_step_s"]}
    many = water_column(16, 16, 16, worlds=256, cfg=cfg, device=DEVICE)  # 256 x 4096
    t = time_steps(many.make_solver(), many.model, cfg.dt_max, 200)
    out["256_worlds_4k_rest"] = {**t, "dt": cfg.dt_max, "world_s_per_wall_s": 256 * cfg.dt_max / t["wall_per_step_s"]}
    out["device"] = wp.get_device(DEVICE).name
    write_result("wp9_speed_gpu", out)
    assert all(v["wall_per_step_s"] > 0.0 for k, v in out.items() if isinstance(v, dict))
