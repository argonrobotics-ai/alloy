# SPDX-License-Identifier: Apache-2.0
"""BEN-1: the A16 harness self-test passes 100 % (tests/bench/test_selftest.py named in SPEC §5.2)."""

from __future__ import annotations

from alloy.bench.analytic import selftest
from alloy.bench.runner import run_law


def test_every_comparator_case_matches() -> None:
    rate, detail = selftest.run_cases()
    assert rate == 1.0, {k: v for k, v in detail.items() if v != "ok"}
    assert len(detail) >= 20
    kinds = {name.split("_")[0] for name in detail}
    assert {"scalar", "curve", "dist", "rate", "errors"} <= kinds


def test_a16_law_through_run_law() -> None:
    r = run_law(selftest.LAW, selftest.SelfTestTwin(), None, device="cpu", solver="none")
    assert r.verdicts[0].verdict == "pass" and r.verdicts[0].error == 0.0
    assert r.monitors["n_cases"] == len(selftest.cases())


def test_a16_detects_a_broken_comparator(monkeypatch) -> None:
    from alloy.bench import compare as C

    monkeypatch.setattr(C, "compare_rate", lambda *a, **k: C.placeholder_verdict("r", "1", "fail", "rate"))
    rate, detail = selftest.run_cases()
    assert rate < 1.0 and any(v != "ok" for k, v in detail.items() if k.startswith("rate"))
