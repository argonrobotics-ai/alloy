# SPDX-License-Identifier: Apache-2.0
"""GPU: the analytic and determinism suites on CUDA (quick), reports copied back by scripts/gpu_run.sh (SPEC §6.4).

Run: scripts/gpu_run.sh WP6 "tests/bench/test_gpu_suite.py -m gpu". Both suites are compared to the cuda:0 keys of
bench/goldens/analytic_baseline.json (audit laws-4: they ran without `--baseline`, so no GPU row could fail, and the
CUDA-only laws A7.hydroelastic, A8.clearance and A8.push were compared nowhere): a verdict that got worse, an error
past its key's plus the bound, a keyless law that does not pass, or a keyed law with no result fails the test. Known
GPU failures are in the baseline under their keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alloy.bench import cli

pytestmark = [pytest.mark.gpu, pytest.mark.law]
OUT = Path(__file__).resolve().parents[2] / "bench" / "results"
BASELINE = Path(__file__).resolve().parents[2] / "bench" / "goldens" / "analytic_baseline.json"


@pytest.mark.timeout(3600)   # a cold Warp cache compiles every law (the analytic suite: 13 laws on 10-31, 52 now)
@pytest.mark.parametrize("suite", ["analytic", "determinism"])
def test_suite_on_cuda(device: str, suite: str) -> None:
    assert cli.main(["run", "--suite", suite, "--device", device, "--quick", "--out", str(OUT),
                     "--baseline", str(BASELINE)]) == 0, "regressions against the cuda keys (REGRESSION lines above)"
    reports = [json.loads(p.read_text()) for p in OUT.rglob("*.json") if p.parent.name == suite]
    mine = [r for r in reports if r["device"] == device and r["law"][0] in "AD"]
    assert mine, "no reports written"
    if suite == "analytic":
        gate = [r for r in mine if r["law"] == "A16.selftest"]
        assert gate and all(m["verdict"] == "pass" for m in gate[-1]["metrics"])
