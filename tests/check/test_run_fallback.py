# SPDX-License-Identifier: Apache-2.0
"""OSS-7: a check run whose CUDA graph capture fails (a part of the scene reads the host inside a step: CUDA error
906, task_dc_show_service in the 11-08 and 11-09 nightlies) runs again without graphs instead of stopping. CPU, with
the capture failure faked on the first simulation."""

from __future__ import annotations

import alloy.check
from alloy.cards import types as ct
from alloy.parallel.graphs import GraphCaptureError

TABLE = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))
CUBE = ct.Placement("cube_beech_30", (0.0, 0.0, 0.015, 0.0, 0.0, 0.0, 1.0))


def test_capture_failure_runs_again_without_graphs(monkeypatch):
    from alloy.integration import wiring

    real, calls = wiring.build_simulation, []

    def build(built, prof="standard", **kw):
        calls.append(kw.get("graph"))
        sim = real(built, prof, **kw)
        if kw.get("graph") != "off":
            def capture_fails():
                raise GraphCaptureError("host sync inside a captured control period on cuda:0: Warp CUDA error 906")
            sim.step_period = capture_fails
        return sim

    monkeypatch.setattr(wiring, "build_simulation", build)
    report = alloy.check(ct.SceneCard(robots=(), objects=(TABLE, CUBE)), worlds=1, seconds=0.1, device="cpu")
    assert calls == [None, "off"]
    assert report.ok, report.summary()
    assert report.checks["rest_drift"] == "pass" and report.metrics["run"]["periods"] == 5
    assert report.metrics["run"]["graph_error"].startswith("GraphCaptureError: host sync inside a captured")
    (f,) = [f for f in report.findings if f.check == "health"]
    assert f.level == "info" and "ran again without graphs" in f.message and "traceback" in f.fix
