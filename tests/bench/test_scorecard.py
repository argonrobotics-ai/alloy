# SPDX-License-Identifier: Apache-2.0
"""The scorecard command: record rules, the restricted split, the row 3 conversion, the page."""

from __future__ import annotations

import dataclasses
import json

import pytest

from alloy.bench import law as L
from alloy.bench.scorecard import cli, page, row3, rows_io
from alloy.bench.scorecard.registry import ROWS
from alloy.bench.scorecard.stats import wilson
from alloy.bench.scorecard.types import Headline, Row
from alloy.bench.types import LawResult, MetricVerdict, Quantity

RESTRICTED_ENGINE = "isaac_lab_physx"          # an engine whose measured numbers may not be published
RESTRICTED_LABEL = "PhysX"


def _row(**kw) -> Row:
    base = {"row": "1", "engine": "alloy", "task": "franka_reach", "metric": "steps_per_s", "value": 1.0}
    return Row(**{**base, **kw})


def _result(law: str, verdict: str, error: float = 0.001, bound: float = 0.01) -> LawResult:
    sim = Quantity("x", "m", 1.0, 0.0, 16, 0.0)
    mv = MetricVerdict("x_error", "m", None, sim, error, bound, None, "mean", verdict)
    prov = {"alloy": "abc1234", "date": "2026-11-10T00:00:00+0000", "quick": "False"}
    return LawResult(law, "mujoco", "cpu", (mv,), {}, prov, 1.5, False, command="cmd")


def test_wilson_bounds():
    lo, hi = wilson(0, 256)
    assert lo == 0.0 and hi == pytest.approx(0.01478, abs=1e-4)
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson(8, 8)[1] == 1.0
    with pytest.raises(ValueError, match="wilson needs"):
        wilson(3, 2)


def test_measured_physx_number_must_be_internal(tmp_path):
    with pytest.raises(ValueError, match="must be internal=True"):
        rows_io.write_rows([_row(engine=RESTRICTED_ENGINE)], tmp_path)
    capability = _row(engine=RESTRICTED_ENGINE, value=None, metric="status", verdict="yes")
    assert rows_io.problems(capability) == []          # a yes / no fact from public docs is not a measured number
    tally = _row(engine=RESTRICTED_ENGINE, metric="cells_yes", k=3, n=26, value=3 / 26, documented=True)
    assert rows_io.problems(tally) == []               # nor is a count of such facts


def test_row6_headline_is_three_counts_and_no_pass_figure(tmp_path):
    from alloy.bench.scorecard import row6
    from alloy.bench.scorecard.types import RunOptions

    cells = [{"capability": "cloth", "engine": "alloy", "status": "partial", "evidence": ["x"]},
             {"capability": "liquid", "engine": "alloy", "status": "yes"},
             {"capability": "cloth", "engine": "isaaclab_physx", "status": "?"}]
    tasks = [{"task_id": f"t{i}", "verdict_class": "yes" if i < 3 else "partial", "alloy_env": "Port-v0" if i < 2 else None,
              "evidence_tier": row6.FULL_PORT_TIER if i == 0 else "E_class_judgment_only"} for i in range(4)]
    (tmp_path / "capabilities.jsonl").write_text("\n".join(json.dumps(c) for c in cells))
    (tmp_path / "ga_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    rows, head = row6.run(RunOptions(out=tmp_path, data=tmp_path, command="cmd"))
    assert head.number.startswith("Gym-Anything's 4 tasks: run and scored by Gym-Anything's own verifier 0; "
                                  "ports with our own checks 2 (k of n gates each; 1 judge all n); "
                                  "physics class supported 3 of 4 (by reading, not run).")
    assert "pass" not in head.number and "strict" not in "".join(r.metric for r in rows)
    census = [r for r in rows if r.task == "gym-anything"]
    assert census and all(r.lo is None and r.hi is None for r in census) and "not a sample" in head.bound
    assert "1 yes, 1 partial, 0 no of 2" in head.number and "not run through GA's verifier" in head.caveat
    unknown = [r for r in rows if r.engine == "isaaclab_physx" and r.task == "capability/cloth"]
    assert unknown[0].verdict == "unknown"
    rows_io.write_rows(rows, tmp_path / "out")          # no measured rival number: nothing internal
    assert not (tmp_path / "out" / rows_io.INTERNAL_NAME).exists()
    with pytest.raises(FileNotFoundError, match="--data"):
        row6.run(RunOptions(out=tmp_path, data=tmp_path / "none"))


def test_internal_rows_go_to_their_own_file_with_the_header(tmp_path):
    rows = [_row(), _row(engine="isaac_lab_physx", internal=True)]
    paths = rows_io.write_rows(rows, tmp_path)
    assert [p.name for p in paths] == [rows_io.PUBLIC_NAME, rows_io.INTERNAL_NAME]
    assert "physx" not in paths[0].read_text()
    first = json.loads(paths[1].read_text().splitlines()[0])
    assert first == {"_header": rows_io.INTERNAL_HEADER}
    assert rows_io.read_rows(paths[1]) == [rows[1]]
    assert rows_io.read_rows(paths[0]) == [rows[0]]


@pytest.mark.parametrize("bad, text", [
    ({"value": float("nan")}, "finite or null"), ({"k": 3}, "k and n come together"),
    ({"lo": 2.0, "hi": 1.0, "bound_kind": "min_max"}, "lo > hi"), ({"hi": 1.0}, "needs bound_kind"),
    ({"verdict": "great"}, "unknown verdict"), ({"task": ""}, "all required"),
])
def test_bad_records_are_refused(bad, text):
    assert any(text in p for p in rows_io.problems(_row(**bad)))


def test_row3_counts_only_laws_that_ran():
    results = [_result("A1.freefall", "pass"), _result("A5.coulomb", "fail", error=0.05),
               L.not_run_result("A9", "any", "cpu", "producer not merged")]
    rows = [r for res in results for r in row3.law_rows(res)]
    assert [r.verdict for r in rows] == ["pass", "fail", "not_run"]
    assert rows[0].hi == 0.01 and rows[0].bound_kind == "tolerance" and rows[0].commit == "abc1234"
    assert "producer not merged" in rows[2].caveat
    head = row3.headline(rows, "cmd")
    assert head.number.startswith("1 of 2 laws") and "1 laws did not run" in head.caveat
    assert head.state == "measured" and "no interval" in head.bound      # the laws are a fixed list, not a sample


def test_row3_headline_leaves_out_comparison_solvers():
    stock = dataclasses.replace(_result("A1.freefall", "fail"), solver="xpbd")
    rows = row3.law_rows(_result("A1.freefall", "pass")) + row3.law_rows(stock)
    assert [r.engine for r in rows] == ["alloy/mujoco", "newton/xpbd"]
    assert row3.headline(rows, "cmd").number.startswith("1 of 1 laws")


def test_row3_quick_run_is_a_draft():
    res = _result("A1.freefall", "pass")
    res = dataclasses.replace(res, provenance={**res.provenance, "quick": "True"})
    head = row3.headline(row3.law_rows(res), "cmd")
    assert head.state == "draft" and "not the launch number" in head.caveat


def test_page_shows_unmeasured_rows_and_hides_internal_ones(tmp_path):
    rows_io.write_headline(Headline(row="3", title="Laws", number="40 of 44", bound="b", command="c",
                                    date="2026-11-10T01:00", commit="abc", state="measured"),
                           page.row_dir(tmp_path, "3"))
    rows_io.write_headline(Headline(row="1", title="Speed", number=f"0.19x {RESTRICTED_LABEL}", internal=True, state="measured"),
                           page.row_dir(tmp_path, "1"))
    text = page.render(page.load_headlines(tmp_path))
    assert "| 3 | Laws | 40 of 44 | b | `c` | 2026-11-10 | abc |" in text
    assert RESTRICTED_LABEL not in text and text.count("not measured") == len(ROWS) - 1
    internal = page.render(page.load_headlines(tmp_path, internal=True), internal=True)
    assert internal.startswith(rows_io.INTERNAL_HEADER) and f"0.19x {RESTRICTED_LABEL}" in internal


def test_cli_names_the_keeper_of_an_unwired_row(capsys):
    assert cli.main(["4"]) == cli.EXIT_NOT_WIRED
    assert "keeper" in capsys.readouterr().err
    assert cli.main(["list"]) == 0


def test_row3_law_resumes_from_its_partial_file(tmp_path):
    """A stopped parallel run keeps each finished law: the worker reads its records back and never re-runs it."""
    import dataclasses

    from alloy.bench.scorecard import row3
    from alloy.bench.scorecard.types import Row, RunOptions

    kept = Row(row="3", engine="alloy/mujoco", task="A1.fall", metric="height", value=1e-4, verdict="pass")
    partial = tmp_path / "partial" / "A1.fall.jsonl"
    partial.parent.mkdir()
    partial.write_text(json.dumps(dataclasses.asdict(kept)) + "\n")
    assert row3._run_one_law("A1.fall", RunOptions(out=tmp_path, jobs=2)) == [kept]


def test_public_rows_1_and_2_are_whole_and_public(tmp_path):
    """The shipped row 1 / row 2 files read back, hold Alloy's own numbers only (the other engine as a ratio), and
    row 1's mean is its tasks' mean."""
    import math
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "bench" / "scorecard"
    r1 = rows_io.read_rows(root / "row1" / "rows.jsonl")
    r2 = rows_io.read_rows(root / "row2" / "rows.jsonl")
    assert all(not rows_io.problems(r) and not r.internal for r in r1 + r2)
    tasks = [r for r in r1 if not r.task.startswith("geometric_mean")]
    (mean,) = [r for r in r1 if r.task.startswith("geometric_mean")]
    assert mean.task == f"geometric_mean_{len(tasks)}_tasks"
    assert math.exp(sum(math.log(r.value) for r in tasks) / len(tasks)) == pytest.approx(mean.value, abs=1e-3)
    assert {r.engine for r in r1 + r2} == {"alloy"} and all(r.k is not None and r.command for r in r2)
    assert all(not any(key.startswith("mujoco_warp_best") for key in r.extra) for r in r1)
