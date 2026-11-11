# SPDX-License-Identifier: Apache-2.0
"""GPU: WP3 laws through ``run_law`` in quick mode (A7.hydroelastic, A8.clearance, A8.push), each with the twin
factory the bench registers (``law_device.CudaOnly`` around the twin); one row per law in
bench/results/wp3/laws.jsonl with the verdicts and the twin's measured numbers (monitors).

Run: scripts/gpu_run.sh WP3 "tests/contact/test_laws_gpu.py -q -p no:cacheprovider" --vm <gpu-vm>
"""

from __future__ import annotations

import os

import pytest

from alloy.bench import law as L
from alloy.bench.runner import run_law
from alloy.bench.types import LawResult, LawSpec
from alloy.contact.laws import A7_HYDROELASTIC, A8_CLEARANCE, A8_PUSH, register

from peg_harness import record

pytestmark = pytest.mark.gpu
DEVICE = os.environ.get("ALLOY_DEVICE", "cuda:0")


def _run(spec: LawSpec) -> tuple[LawResult, dict]:
    register()
    factory = {s.id: f for s, f in L.registered_laws()}[spec.id]
    res = run_law(spec, factory(), None, device=DEVICE, solver=spec.solvers[0], quick=True)
    row = dict(metric=spec.id, verdict=res.verdict, device=DEVICE, wall_s=res.wall_s, config=dict(res.config),
               metrics={v.name: dict(error=v.error, bound=v.bound, verdict=v.verdict, sim_mean=v.sim.mean)
                        for v in res.verdicts},
               monitors=dict(res.monitors))
    record("laws", row)
    return res, row


def _assert_passed(res: LawResult, row: dict) -> None:
    assert "error" not in res.monitors, res.monitors.get("error")
    assert sum(res.monitors.get("overflow", {}).values()) == 0, res.monitors.get("overflow")
    assert [v.verdict for v in res.verdicts] == ["pass"] * len(res.verdicts), row["metrics"]


@pytest.mark.timeout(480)
def test_a7_hydroelastic_pressure_law() -> None:
    res, row = _run(A7_HYDROELASTIC)
    _assert_passed(res, row)


@pytest.mark.timeout(480)
def test_a8_clearance_gap_and_tilt() -> None:
    res, row = _run(A8_CLEARANCE)
    _assert_passed(res, row)


@pytest.mark.timeout(480)
def test_a8_push_penetration() -> None:
    res, row = _run(A8_PUSH)
    _assert_passed(res, row)
    assert res.monitors["tunnels"] == 0 and res.monitors["samples"] > 0, row["monitors"]
