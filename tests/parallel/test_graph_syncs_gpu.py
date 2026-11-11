# SPDX-License-Identifier: Apache-2.0
"""PAR-14: host syncs inside a captured control period = 0 (SPEC §5.2), with a FakeStepper period graph.

The period is 16 exchanges of a ``FakeStepper`` around SolverMuJoCo (MJWarp's iterative solve runs inside the graph
through ``wp.capture_while``), with an even number of ping-pong swaps. The negative controls put a host read and a
device sync inside the period: both must be caught at capture, counted in ``host_syncs`` and raised.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import GraphCaptureError
from alloy.parallel.graphs import PeriodGraph
from alloy.testing.fakes import FakeStepper
from alloy.testing.parallel import record_metric
from alloy.testing.scenes import box_on_plane

pytestmark = pytest.mark.gpu
PERIODS = 100
SUBSTEPS = 16
DT = 1.25e-3


class Period:
    """One control period of FakeStepper exchanges on ``n`` box-on-plane worlds."""

    def __init__(self, device: str, n: int = 256) -> None:
        self.model = box_on_plane(n, device=device).model
        self.stepper = FakeStepper(newton.solvers.SolverMuJoCo(self.model, njmax=32, nconmax=8), dt_x=DT)
        self.a, self.b = self.model.state(), self.model.state()
        self.control = self.model.control()

    def run(self) -> None:
        for _ in range(SUBSTEPS):
            self.stepper.step(self.a, self.b, self.control, None, DT)
            self.a, self.b = self.b, self.a

    def buffers(self):
        return (self.a.body_q, self.b.body_q)


@pytest.mark.smoke
def test_par14_zero_host_syncs_in_captured_periods(device):
    if not device.startswith("cuda"):
        pytest.skip("PAR-14 is a CUDA graph metric")
    graph_p, eager_p = Period(device), Period(device)
    g = PeriodGraph(graph_p.run, device, buffers=graph_p.buffers)
    for _ in range(PERIODS):
        g.launch()
        eager_p.run()
    wp.synchronize_device(device)
    assert g.mode == "graph" and g.captures == 1 and g.launches == PERIODS
    assert g.host_syncs == 0
    assert graph_p.stepper.exchanges == SUBSTEPS  # the Python body ran once, at capture; replays are pure graph
    z_graph, z_eager = graph_p.a.body_q.numpy()[:, 2], eager_p.a.body_q.numpy()[:, 2]
    np.testing.assert_allclose(z_graph, z_eager, atol=1e-5)  # SolverMuJoCo is not bitwise (D9); same physics
    record_metric("PAR-14", g.host_syncs, "pass" if g.host_syncs == 0 else "fail",
                  "pytest tests/parallel/test_graph_syncs_gpu.py", periods=PERIODS, worlds=256,
                  max_dz_graph_vs_eager_m=float(np.max(np.abs(z_graph - z_eager))))


@pytest.mark.parametrize("leak", ["numpy", "synchronize"])
def test_host_sync_inside_period_is_caught(device, leak):
    if not device.startswith("cuda"):
        pytest.skip("CUDA capture semantics")
    p = Period(device, n=4)

    def bad() -> None:
        p.run()
        if leak == "numpy":
            p.a.body_q.numpy()
        else:
            wp.synchronize_device(device)

    g = PeriodGraph(bad, device, buffers=p.buffers)
    with pytest.raises(GraphCaptureError):
        g.launch()
    assert g.host_syncs == 1 and g.mode == "eager"
    assert not wp.get_device(device).is_capturing  # the broken capture was closed
    g2 = PeriodGraph(p.run, device, buffers=p.buffers)  # the device is usable again
    g2.launch()
    assert g2.mode == "graph" and g2.host_syncs == 0
