# SPDX-License-Identifier: Apache-2.0
"""report/1 round trip and the D23 baseline gate (seeded regression fails)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from alloy.bench import baseline as BL
from alloy.bench import report as R
from alloy.bench.compare import placeholder_verdict
from alloy.bench.types import LawResult, MetricVerdict, Quantity


def verdict(name: str = "m", error: float = 0.1, bound: float = 1.0, v: str = "pass") -> MetricVerdict:
    return MetricVerdict(name, "m", Quantity(name, "m", 1.0, 0.1, 10, 0.05), Quantity(name, "m", 1.1, 0.0, 16, 0.0),
                         error, bound, 3.5, "mean", v)  # type: ignore[arg-type]


def result(*vs: MetricVerdict, law: str = "A1.freefall", solver: str = "xpbd", quick: bool = True) -> LawResult:
    prov = {"suite": "analytic", "quick": str(quick), "alloy": "abc1234", "date": "2026-10-31T22:00:00-0700"}
    return LawResult(law, solver, "cpu", tuple(vs) or (verdict(),), {"steps": 3, "nan": math.nan}, prov, 1.5, False,
                     command="python -m alloy.bench run", config={"dt": 1e-3}, data={})


def test_report_round_trip(tmp_path) -> None:
    r = result(verdict(), placeholder_verdict("x", "1", "not_run"))
    path = R.write_report(r, tmp_path)
    assert path.name == "A1.freefall.xpbd.cpu.json" and path.parent.name == "analytic"
    d = json.loads(path.read_text())  # strict JSON: NaN written as null
    assert d["schema"] == "alloy.bench.report/1" and d["monitors"]["nan"] is None and not R.validate(d)
    back = R.read_report(path)
    assert back.verdicts[0] == r.verdicts[0] and back.command == r.command and back.config == r.config
    assert back.verdicts[1].verdict == "not_run" and math.isnan(back.verdicts[1].error)


def test_report_schema_validation() -> None:
    d = R.to_dict(result())
    d["metrics"][0]["verdict"] = "great"
    del d["wall_s"]
    problems = R.validate(d)
    assert any("verdict" in p for p in problems) and any("wall_s" in p for p in problems)
    with pytest.raises(ValueError):
        R.from_dict(d)


def test_baseline_seeded_regression(tmp_path) -> None:
    path = tmp_path / "base.json"
    base = result(verdict(error=0.1, bound=1.0, v="pass"), verdict("n", error=2.0, bound=1.0, v="fail"))
    assert BL.update_baseline([base], path, reason="initial") == ["A1.freefall|xpbd|cpu|quick"]
    assert BL.compare_to_baseline([base], path) == []
    worse = result(verdict(error=1.2, bound=1.0, v="fail"), verdict("n", error=2.0, bound=1.0, v="fail"))
    regs = BL.compare_to_baseline([worse], path)
    assert any("pass -> fail" in r for r in regs)
    drift = result(verdict(), verdict("n", error=3.5, bound=1.0, v="fail"))  # 3.5 > 2.0 + 1.0
    assert any("error 3.5" in r for r in BL.compare_to_baseline([drift], path))
    gone = result(verdict(), dataclasses.replace(verdict("n"), verdict="not_run"))
    assert any("fail -> not_run" in r for r in BL.compare_to_baseline([gone], path))


def test_baseline_new_keys_and_updates(tmp_path) -> None:
    path = tmp_path / "base.json"
    BL.update_baseline([result()], path, reason="initial")
    other = result(solver="mujoco")
    assert BL.compare_to_baseline([other], path) == []  # a new key that passes is not a regression
    assert BL.update_baseline([result(verdict(v="fail"))], path) == []  # never overwrites without a reason
    with pytest.raises(ValueError):
        BL.update_baseline([result()], path, overwrite=True)
    keys = BL.update_baseline([result(verdict(v="fail"))], path, reason="golden-update: test", overwrite=True)
    data = BL.load_baseline(path)
    assert keys and data["entries"][keys[0]]["verdict"] == "fail" and data["history"][-1]["reason"].startswith("golden")


@pytest.mark.parametrize("own", ["fail", "inconclusive", "not_run", "error"])
def test_a_keyless_result_is_held_to_its_own_verdict(tmp_path, own) -> None:
    """Audit AB-laws-1: with no key to compare against, a law that does not pass is a regression (it used to be
    skipped as a new key, so a broken law without a key passed the gate)."""
    path = tmp_path / "base.json"
    BL.update_baseline([result()], path, reason="initial")
    planted = result(verdict(), verdict("n", error=1.0, bound=0.01, v=own), law="A11.torsion", solver="mujoco")
    regs = BL.compare_to_baseline([planted], path)
    assert len(regs) == 1 and regs[0].startswith(f"A11.torsion|mujoco|cpu|quick: no baseline key and its own verdict "
                                                 f"is {own} (n error 1, bound 0.01)"), regs
    assert BL.compare_to_baseline([planted], tmp_path / "no_such_file.json") == regs   # no baseline file: the same
    BL.update_baseline([planted], path, reason="known failure")       # recorded under a key: compared as before
    assert BL.compare_to_baseline([planted], path) == []
    assert BL.compare_to_baseline([dataclasses.replace(planted, verdicts=())], tmp_path / "x.json")[0].endswith(
        "failure under a key (--add-new-keys, reviewed with the baseline file)")       # no metrics at all: error


def test_a_keyed_law_with_no_result_is_a_regression(tmp_path) -> None:
    """The other way to pass unseen: stop running. A whole-suite run that has no result for a key of its device and
    mode is a regression; other devices, modes and suites are out of scope, and a retired key leaves with a reason."""
    path = tmp_path / "base.json"
    kept, lost, full = result(), result(law="A11.torsion", solver="mujoco"), result(law="A2.x", quick=False)
    gpu, det = dataclasses.replace(result(law="A3.x"), device="cuda:0"), result(law="D1.repeat")
    BL.update_baseline([kept, lost, full, gpu, det], path, reason="initial")
    scope = dict(device="cpu", mode="quick", in_scope=lambda law: law.startswith("A"))
    assert BL.missing_from_run([kept, lost], path, **scope) == []
    regs = BL.missing_from_run([kept], path, **scope)
    assert len(regs) == 1 and regs[0].startswith("A11.torsion|mujoco|cpu|quick: in the baseline (pass) but this run "
                                                 "has no result for it"), regs
    with pytest.raises(ValueError):
        BL.drop_keys(["A11.torsion|mujoco|cpu|quick"], path, reason="")
    assert BL.drop_keys(["A11.torsion|mujoco|cpu|quick", "no.such|key"], path, reason="law retired (test)") == [
        "A11.torsion|mujoco|cpu|quick"]
    data = BL.load_baseline(path)
    assert data["history"][-1]["dropped"] and data["history"][-1]["reason"] == "law retired (test)"
    assert BL.missing_from_run([kept], path, **scope) == [] and len(data["entries"]) == 4


def test_a_failing_zero_bound_metric_is_not_compared_by_size(tmp_path) -> None:
    """WP5's flake, 2026-11-10: D3.spread|mujoco|cuda:0 read 5.13e-9 against a stored 4.31e-9 with bound 0. A GPU
    MuJoCo run does not repeat, so the stored size is one sample; such a row is held to its verdict only. A passing
    zero-bound metric that turns non-zero still regresses, and so does a failing metric with a real bound."""
    path = tmp_path / "base.json"
    base = result(verdict("spread", error=4.31e-9, bound=0.0, v="fail"), verdict("exact", error=0.0, bound=0.0),
                  verdict("n", error=2.0, bound=1.0, v="fail"), law="D3.spread", solver="mujoco")
    BL.update_baseline([base], path, reason="initial")
    more = dataclasses.replace(base, verdicts=(verdict("spread", error=5.13e-9, bound=0.0, v="fail"),) + base.verdicts[1:])
    assert BL.compare_to_baseline([more], path) == []
    broke = dataclasses.replace(base, verdicts=(base.verdicts[0], verdict("exact", error=1e-9, bound=0.0, v="fail"),
                                                verdict("n", error=3.5, bound=1.0, v="fail")))
    regs = BL.compare_to_baseline([broke], path)
    assert any("exact verdict pass -> fail" in r for r in regs) and any("n error 3.5" in r for r in regs), regs


def test_mode_is_part_of_the_key() -> None:
    assert BL.key_of(result(quick=True)).endswith("|quick") and BL.key_of(result(quick=False)).endswith("|full")
    left, right = (dataclasses.replace(result(quick=True), data={"dataset_id": f"yam1-{arm}-20261103"})
                   for arm in ("left", "right"))
    assert BL.key_of(left) != BL.key_of(right) and BL.key_of(left).endswith("|quick")  # one result per arm
    assert R.report_path(left, Path("r")) != R.report_path(right, Path("r"))
    assert BL.law_verdict(result(verdict(), verdict(v="error"))) == "error"
