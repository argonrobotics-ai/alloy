# SPDX-License-Identifier: Apache-2.0
"""PRC-1 (SPEC §5.2): resting drift without tangential load, max body, rest-16 and 3-stack, dt 1.25 ms and 0.156 ms
on the grid, 10 sim-min, precision preset checks, sleeping off, SolverMuJoCo (the default rigid entry). Drift is the
largest body displacement from its pose after a 1 s settle. Target <= 0.01 mm / 10 min. The peg-in-0.2 mm-hole
variant needs the precision contact pipeline inside a full Simulation (WP3 + WP2/WP12) and is not_run here."""

from __future__ import annotations

import os

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.testing.precision import DetSim, resting_boxes, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.nightly, pytest.mark.slow, pytest.mark.timeout(3600)]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_rest_drift_gpu.py -m nightly' --vm <gpu-vm>"
SIM_S = float(os.environ.get("ALLOY_PRC1_SECONDS", "600"))
WORLDS = 16


@pytest.mark.parametrize("j", [4, 7], ids=["dt1.25ms", "dt0.156ms"])
@pytest.mark.parametrize("layout", ["rest-16", "3-stack"])
def test_prc1_rest_drift(layout: str, j: int) -> None:
    model = resting_boxes(WORLDS, layout)
    # MuJoCo's default njmax (64 rows per world) overflows on 16 resting boxes and they sink (NaN within 0.1 s)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, njmax=1024, nconmax=256)
    sim = DetSim(model, solver, dt=0.02 / 2**j, use_graph=True)
    periods_settle, periods = 50, int(round(SIM_S / 0.02))
    for _ in range(periods_settle):
        sim.step_period()
    p0 = sim.state.body_q.numpy()[:, :3].astype(np.float64)
    for _ in range(periods):
        sim.step_period()
    p1 = sim.state.body_q.numpy()[:, :3].astype(np.float64)
    overflow = int(np.bitwise_or.reduce(solver.mjw_data.overflow.numpy()))
    drift = float(np.linalg.norm(p1 - p0, axis=1).max())
    per_10min = drift * 600.0 / SIM_S
    ok = per_10min <= 1e-5 and np.isfinite(drift) and overflow == 0
    write_row("PRC-1", f"{layout} dt {0.02 / 2**j * 1e3:.4g} ms: max drift {drift * 1e3:.4g} mm over {SIM_S:.0f} s "
              f"({per_10min * 1e3:.4g} mm / 10 min), {WORLDS} worlds, MuJoCo, sleeping off, MJW overflow bits {overflow}",
              "pass" if ok else "fail",
              CMD, layout=layout, dt=0.02 / 2**j, seconds=SIM_S)
    assert np.isfinite(drift) and overflow == 0
