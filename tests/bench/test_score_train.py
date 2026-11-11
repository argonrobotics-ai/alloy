# SPDX-License-Identifier: Apache-2.0
"""Row 4 scorer: threshold-with-hold, censored seeds, torn files, the public filter."""

from __future__ import annotations

import json

from bench.scorecard.train import score
from bench.scorecard.train.schema import Point, Run, RunHeader, RunWriter, read_runs

TH = score.Threshold(metric="err", value=0.2, higher_is_better=False, hold=3)


def _run(errs, seed=0, framework="alloy", internal=False):
    header = RunHeader(framework, "go2_flat", "X-v0", seed, 4096, "matched", "L4", internal_only=internal)
    points = [Point(i, (i + 1) * 1000, float(i + 1), float(i + 11), {"err": e}) for i, e in enumerate(errs)]
    return Run(header, points)


def test_reach_needs_the_hold():
    # A single good reading at index 1 does not count; the streak starting at index 3 does.
    r = score.time_to_threshold(_run([0.5, 0.1, 0.5, 0.2, 0.1, 0.15, 0.1]), TH)
    assert r.reached and r.iteration == 3 and r.wall_s == 4.0 and r.env_steps == 4000


def test_never_reached_and_nan():
    assert not score.time_to_threshold(_run([0.5, 0.1, 0.1, 0.5]), TH).reached
    assert not score.time_to_threshold(_run([float("nan")] * 5), TH).reached


def test_cell_median_counts_censored_seeds():
    good = [_run([0.5] * k + [0.1] * 3, seed=k) for k in (1, 2, 3)]  # reach at wall 2, 3, 4 s
    bad = [_run([0.5] * 6, seed=s) for s in (4, 5)]
    cell = score.score_cell(good + bad, TH)
    assert (cell.seeds, cell.seeds_reached) == (5, 3)
    assert cell.wall_s_median == 4.0  # sorted [2, 3, 4, inf, inf]
    assert cell.wall_s_max is None  # not every seed reached: no finite worst case
    minority = score.score_cell(good[:2] + bad + [_run([0.5] * 6, seed=6)], TH)
    assert minority.wall_s_median is None and minority.seeds_reached == 2


def test_writer_round_trip_and_torn_tail(tmp_path):
    path = tmp_path / "r.jsonl"
    w = RunWriter(path, _run([]).header)
    w.start_training()
    w.point(0, 1000, {"err": 0.3})
    w.point(1, 2000, {"err": 0.1})
    with path.open("a") as f:
        f.write('{"kind": "point", "iterat')  # preempted mid-line
    (run,) = read_runs(path)
    assert [p.env_steps for p in run.points] == [1000, 2000]
    assert run.points[1].wall_s >= run.points[0].wall_s >= 0.0


def test_cli_public_filter(tmp_path, capsys):
    for name, internal in (("alloy", False), ("restricted_fw", True)):
        run = _run([0.1] * 3, framework=name, internal=internal)
        lines = [json.dumps(vars(run.header))] + [json.dumps(vars(p)) for p in run.points]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    th = tmp_path / "th.json"
    th.write_text(json.dumps({"go2_flat": {"metric": "err", "value": 0.2, "higher_is_better": False}}))
    assert score.main([str(tmp_path), "--thresholds", str(th), "--public"]) == 0
    cells = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [c["framework"] for c in cells] == ["alloy"]


def test_report_public_hides_internal_cells():
    from bench.scorecard.train import report

    cells = [
        score.score_cell([_run([0.1] * 3, seed=s) for s in range(3)], TH),
        score.score_cell([_run([0.1] * 3, seed=s, framework="restricted_fw", internal=True) for s in range(3)], TH),
    ]
    public = report.render(cells, public=True, header="NOTICE")
    assert "restricted_fw" not in public and "NOTICE" in public
    assert "| alloy |" in public
    internal = report.render(cells, public=False, header="NOTICE")
    assert internal.startswith("NOTICE") and "restricted_fw" in internal
