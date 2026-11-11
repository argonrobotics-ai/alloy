# SPDX-License-Identifier: Apache-2.0
"""GPU determinism acceptance (SPEC §7 WP5): PRC-5 (1,024 identical worlds, all digests equal; XPBD and Steel in
deterministic mode), PRC-4 on the D9 population (FakeSolver: R1, R2, R3 graph vs eager, R4) and PRC-6 (batch
invariance at N = 1, 64, 1024, 4096). Rows go to bench/results/wp5_results.jsonl."""

from __future__ import annotations

import functools

import numpy as np
import pytest

from alloy.precision.repeat import repeat_test, run_digests
from alloy.testing.precision import det_sim, write_row

pytestmark = [pytest.mark.gpu, pytest.mark.det]
CMD = "scripts/gpu_run.sh WP5 'tests/precision/test_identical_worlds_gpu.py -k {k}' --vm <gpu-vm>"
PERIODS = 30  # 480 steps at 1.25 ms (0.6 s): the 27 boxes land and pile


def fake_sim(n: int, use_graph: bool = False):
    return det_sim("fake", n, use_graph=use_graph)


def steel_sim(n: int, use_graph: bool = False):
    return det_sim("steel", n, use_graph=use_graph)


@pytest.mark.parametrize("kind", ["xpbd", "steel"])
def test_prc5_identical_worlds(kind: str) -> None:
    digests = run_digests(det_sim(kind, 1024), PERIODS)[-1]
    equal = int((digests == digests[0]).all(axis=1).sum())
    control = run_digests(det_sim(kind, 1024, det=False), PERIODS)[-1]
    distinct_ng = len({tuple(d) for d in control})
    verdict = "pass" if equal == 1024 else "fail"
    write_row("PRC-5", f"{equal}/1024 worlds equal ({kind} det); NONE control: {distinct_ng} distinct digests",
              verdict, CMD.format(k=f"prc5 and {kind}"), solver=kind, worlds=1024, periods=PERIODS)
    assert equal == 1024


@pytest.mark.parametrize("case", [pytest.param("R1", marks=pytest.mark.smoke), "R2", "R3"])
def test_prc4_d9_population(case: str) -> None:
    res = repeat_test(fake_sim, case, steps=8, world_counts=(1, 1024))
    ok = bool(res.equal) and all(res.equal.values())
    write_row("PRC-4", f"{case} FakeSolver equal {dict(res.equal)} native_share {res.native_share}",
              "pass" if ok else "fail", CMD.format(k=f"prc4 and {case}"), case=case, notes=dict(res.notes),
              det_mode=res.det_mode.name)
    assert ok and res.native_share == 1.0


@pytest.mark.parametrize("make", [fake_sim, steel_sim], ids=["fake", "steel"])
def test_prc6_batch_invariance(make) -> None:
    res = repeat_test(make, "R4", steps=8, world_counts=(1, 64, 1024, 4096))
    ok = all(res.equal.values())
    write_row("PRC-6", f"R4 {make.__name__}: equal {dict(res.equal)}; first divergent {dict(res.first_divergent_step)}",
              "pass" if ok else "fail", CMD.format(k="prc6"), notes=dict(res.notes))
    assert ok


def test_steel_graph_r3() -> None:
    res = repeat_test(steel_sim, "R3", steps=8, world_counts=(1024,))
    write_row("PRC-4", f"R3 Steel det graph vs eager equal {dict(res.equal)}", "pass" if all(res.equal.values())
              else "fail", CMD.format(k="steel_graph"), notes=dict(res.notes))
    assert all(res.equal.values())
