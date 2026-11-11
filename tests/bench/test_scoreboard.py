# SPDX-License-Identifier: Apache-2.0
"""Scoreboard: X01 baseline row, not_run never passes, deferred rows excluded (SPEC §5, WP6 acceptance)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from alloy.bench import report as R
from alloy.bench import scoreboard as SB
from alloy.bench.types import LawResult, MetricVerdict, Quantity

DATA = Path(__file__).parent / "data" / "run1"


def _result(law: str, name: str, err: float, v: str, solver: str = "xpbd") -> LawResult:
    mv = MetricVerdict(name, "1", None, Quantity(name, "1", err, 0.0, 1, 0.0), err, 1.0, None, "mean", v)  # type: ignore[arg-type]
    prov = {"suite": "analytic", "quick": "True", "alloy": "abc", "date": "2026-10-31"}
    return LawResult(law, solver, "cpu", (mv,), {}, prov, 0.1, False, command="cmd")


def test_x01_baseline_row_reproduced() -> None:
    rows = SB.scoreboard([DATA])
    x01 = next(r for r in rows if r.metric_id == "X01_free_fall")
    # SCENARIOS.md §4: 4.089 mm = 1/2 g h t at h = 1/600 s, t = 0.5 s, all four rigid solvers pass
    assert x01.value == pytest.approx(4.0888, abs=1e-4) and x01.unit == "mm" and x01.verdict == "pass"
    assert x01.target.startswith("<= 8.175 mm") and x01.source_rows and "bench.run" in x01.command


def test_not_run_never_counts_as_pass(tmp_path) -> None:
    for r in (_result("A1.freefall", "pos_err_ratio", 0.5, "pass", "mujoco"),
              _result("A9.thread", "advance", 0.0, "not_run")):
        R.write_report(r, tmp_path)
    rows = SB.scoreboard([tmp_path])
    sb12 = next(r for r in rows if r.metric_id == "SB-12")
    # SB-12 counts the laws that ran (1/1 pass); the not_run law keeps it from passing and is listed
    assert sb12.value == 1.0 and sb12.verdict == "inconclusive" and "A9.thread" in sb12.reason
    assert all(r.verdict != "pass" for r in rows if r.source_rows == ())  # empty rows are not_run
    assert SB.pass_rate(rows) < 1.0


def test_deferred_rows_excluded_from_release(tmp_path) -> None:
    rows = SB.scoreboard([tmp_path])
    deferred = {r.metric_id for r in rows if r.status == "deferred"}
    assert {"SB-1", "PAR-2", "PRC-8", "TOP-2", "SB-2"} <= deferred  # SB-2 deferred while no real rig data
    assert not deferred & {r.metric_id for r in SB.release_rows(rows)}


def test_ben1_and_ben3_fed_from_laws(tmp_path) -> None:
    R.write_report(_result("A16.selftest", "selftest_pass_rate", 0.0, "pass", "none"), tmp_path)
    R.write_report(_result("A5.onset", "creep", 3e-6, "pass", "mujoco"), tmp_path)
    R.write_report(_result("A5.onset", "creep", 0.03, "fail", "xpbd"), tmp_path)
    rows = {r.metric_id: r for r in SB.scoreboard([tmp_path])}
    assert rows["BEN-1"].verdict == "pass" and rows["BEN-1"].value == 1.0  # pass rate, not the self-test error
    assert rows["BEN-3"].value == pytest.approx(0.03) and rows["BEN-3"].verdict == "fail"  # worst over solvers
    assert rows["BEN-3:mujoco"].verdict == "pass" and rows["BEN-3:mujoco"].value == pytest.approx(3e-6)
    assert rows["law:A5.onset:mujoco:cpu"].verdict == "pass"


def test_write_and_render(tmp_path) -> None:
    shutil.copytree(DATA, tmp_path / "run1")
    rows = SB.scoreboard([tmp_path])
    out = SB.write_scoreboard(rows, tmp_path / "SCOREBOARD.json")
    assert out.exists() and "| X01_free_fall |" in SB.render_markdown(rows)


def test_the_board_says_when_and_from_which_tree_it_was_built(tmp_path) -> None:
    """Audit AB-inv-reports-12: the board names its build date and tree, each row the date and commit of its own
    run (a note row: the commit only, its date here would be the collection's), and SB-7's population table."""
    import dataclasses
    import json

    from alloy.bench.provenance import commit_short

    shutil.copytree(DATA, tmp_path / "run1")
    rows = SB.scoreboard([tmp_path])
    law = dataclasses.replace(rows[0], metric_id="law:x", date="2026-11-09T10:00:00+0000", commit="abc1234",
                              source_rows=("bench/results/x.json",))
    noted = dataclasses.replace(law, metric_id="SB-x", source_rows=("docs/notes/wp7-results.md:83",))
    md = SB.render_markdown([law, noted]).splitlines()
    assert md[0].startswith("Built 20") and f"from tree {commit_short()}" in md[0]
    assert md[2].split(" | ")[6] == "run" and "| 2026-11-09 @ abc1234 |" in md[4] and "| deferred | abc1234 |" in md[5]
    assert "## SB-7 counted population" in md
    data = json.loads(SB.write_scoreboard(rows, tmp_path / "SB.json").read_text())
    assert data["tree"] == commit_short() and len(data["sb7_population"]) >= 10


def test_ben2_held_out_gap(tmp_path) -> None:
    mv = MetricVerdict("e@c0", "1", Quantity("e", "1", 0.7, 0.0, 10, 0.005), Quantity("e", "1", 0.71, 0.0, 8, 0.0),
                       0.012, 0.03, 5.0, "mean", "pass")
    prov = {"suite": "rig", "quick": "True", "alloy": "abc", "date": "d"}
    r = LawResult("R4.drop_bounce", "xpbd", "cpu", (mv,), {"calib_error": {"e@c0": 0.002}}, prov, 0.1, False)
    assert SB.held_out_gaps(r) == pytest.approx([(0.012 - 0.002) / 0.03])
    R.write_report(r, tmp_path)
    row = next(x for x in SB.scoreboard([tmp_path]) if x.metric_id == "BEN-2")
    assert row.verdict == "pass" and row.value == pytest.approx(1 / 3)


def test_every_not_run_row_has_a_reason(tmp_path) -> None:
    from alloy.bench.law import not_run_result

    R.write_report(not_run_result("A9", "any", "cpu", "producer module alloy.contact.laws not merged"), tmp_path)
    rows = SB.scoreboard([tmp_path])
    assert all(r.reason for r in rows if r.verdict in ("not_run", "error"))
    by = {r.metric_id: r for r in rows}
    assert by["CT-1"].reason == "no result fed CT-1 (producer: WP3)"
    assert by["SB-1"].reason.startswith("deferred (SPEC §5)") and "synthetic" in by["SB-2"].reason
    assert by["law:A9:any:cpu"].reason == "producer module alloy.contact.laws not merged"


def test_reason_from_multi_verdict_report(tmp_path) -> None:
    # WP12's collector: one report per package, verdicts "<id>:<WPn>:<i>", reasons in monitors["reasons"]
    q = Quantity("x", "1", float("nan"), float("nan"), 0, float("nan"))
    vs = (MetricVerdict("CT-1:WP3:0", "1", None, q, float("nan"), float("nan"), None, "mean", "not_run"),
          MetricVerdict("CT-7:WP3:1", "1", None, Quantity("x", "1", 1e-7, 0.0, 1, 0.0), 1e-7, 1e-6, None, "mean",
                        "pass"))
    prov = {"suite": "analytic", "quick": "False", "alloy": "abc", "date": "d"}
    r = LawResult("WP3.results", "-", "cpu", vs, {"reasons": {"CT-1:WP3:0": "needs the GPU peg-hole run"}}, prov,
                  0.0, False)
    assert SB.verdict_reason(r, vs[0]) == "needs the GPU peg-hole run" and SB.verdict_reason(r, vs[1]) == ""
    R.write_report(r, tmp_path)
    by = {x.metric_id: x for x in SB.scoreboard([tmp_path])}
    assert by["CT-1"].verdict == "not_run" and by["CT-1"].reason == "needs the GPU peg-hole run"
    assert by["CT-7"].verdict == "pass" and by["CT-7"].reason == ""
    assert "| reason |" in SB.render_markdown(list(by.values()))
