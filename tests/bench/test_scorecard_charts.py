# SPDX-License-Identifier: Apache-2.0
"""Scorecard charts: each builds from small made-up rows, internal rows stay out, failures carry a text label."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from alloy.bench.scorecard import charts, rows_io  # noqa: E402
from alloy.bench.scorecard.types import Row  # noqa: E402

RESTRICTED_ENGINE = "isaac_lab_physx"          # an engine whose measured numbers may not be published


def _law(task, engine="alloy/mujoco", value=0.001, hi=0.01, verdict="pass", **kw) -> Row:
    return Row(row="3", engine=engine, task=task, metric="err", value=value, lo=0.0, hi=hi, bound_kind="tolerance",
               verdict=verdict, device="cpu", **kw)


def _cap(cap, engine, status) -> Row:
    return Row(row="6", engine=engine, task=f"capability/{cap}", metric="status", verdict=status)


def _task(i, tier) -> Row:
    return Row(row="6", engine="alloy", task=f"gym-anything/t{i}", metric="verdict_class", verdict="yes",
               extra={"evidence_tier": tier})


def _speed(task, engine, value, internal=False, **kw) -> Row:
    return Row(row="1", engine=engine, task=task, metric="steps_per_s", value=value, internal=internal, **kw)


def _texts(fig) -> str:
    out = [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text"))]
    return "\n".join(out)


def test_law_chart_labels_failures_and_unrun_laws():
    rows = [_law("A1.ok"), _law("A2.bad", value=0.021, hi=0.01, verdict="fail"),
            _law("A7.skip", value=None, hi=None, verdict="not_run"), _law("A16.selftest"),
            _law("A9.other", engine="newton/vbd", verdict="pass")]
    fig = charts.law_error_chart(rows)
    text = _texts(fig)
    assert "fail 2.1x" in text and "pass 0.1x" in text and "not run" in text
    assert "A16.selftest" not in text and "A9.other" not in text
    assert fig.axes[0].get_xscale() == "log"


def test_law_chart_without_alloy_rows_refuses():
    with pytest.raises(ValueError, match="no rows"):
        charts.law_error_chart([_law("A1.x", engine="newton/vbd")])


def test_law_chart_keeps_a_failure_with_no_number_visible():
    fig = charts.law_error_chart([_law("A2.nan", value=None, verdict="fail")])
    assert "fail, no ratio" in _texts(fig)


def test_capability_grid_puts_alloy_first_and_marks_every_cell():
    rows = [_cap("cloth", "genesis", "yes"), _cap("cloth", "alloy", "partial"), _cap("rods", "alloy", "no"),
            _cap("rods", "genesis", "unknown")]
    caps, engines, status = charts.capability_grid(rows)
    assert engines[0] == "alloy" and caps == ["cloth", "rods"] and status["rods", "genesis"] == "unknown"
    text = _texts(charts.capability_chart(rows))
    for mark in charts.STATUS_MARK.values():
        assert mark in text


def test_ga_chart_counts_tiers():
    rows = [_task(i, "A_port_judges_every_GA_gate") for i in range(2)] + [_task(9, "E_class_judgment_only")] * 5
    assert charts.tier_counts(rows) == {"A": 2, "B": 0, "C": 0, "D": 0, "E": 5}
    text = _texts(charts.ga_tier_chart(rows))
    assert "5" in text and "A: port judges every GA gate (2)" in text


def test_ratio_chart_never_draws_internal_rows_by_default():
    rows = [_speed("reach", "mujoco_warp", 100.0), _speed("reach", "alloy", 80.0, lo=70.0, hi=90.0,
                                                          bound_kind="min_max", caveat="quick mode", device="cpu"),
            _speed("reach", RESTRICTED_ENGINE, 400.0, internal=True)]
    fig = charts.ratio_chart(rows, "steps_per_s", "mujoco_warp")
    text = _texts(fig)
    assert RESTRICTED_ENGINE not in text and "RESTRICTED" not in text
    assert "0.8x" in text and "quick mode" in text           # a loss is drawn and labelled like a win
    assert fig.axes[0].get_xscale() == "log"


def test_ratio_chart_internal_flag_draws_them_under_a_banner():
    rows = [_speed("reach", "mujoco_warp", 100.0), _speed("reach", RESTRICTED_ENGINE, 400.0, internal=True)]
    text = _texts(charts.ratio_chart(rows, "steps_per_s", "mujoco_warp", internal=True))
    assert "RESTRICTED" in text and RESTRICTED_ENGINE in text and "4x" in text


def test_ratio_chart_needs_a_baseline():
    with pytest.raises(ValueError, match="baseline"):
        charts.ratio_chart([_speed("reach", "alloy", 1.0)], "steps_per_s", "mujoco_warp")


def test_write_charts_skips_rows_with_no_file(tmp_path):
    (tmp_path / "row6").mkdir()
    rows_io.write_rows([_cap("cloth", "alloy", "yes"), _task(1, "E_class_judgment_only")], tmp_path / "row6")
    paths = charts.write_charts(tmp_path, tmp_path / "out")
    assert sorted(p.name for p in paths) == ["row6_capabilities.svg", "row6_gym_anything.svg"]
    assert all(p.read_text().lstrip().startswith(("<?xml", "<svg")) for p in paths)
    assert charts.write_charts(tmp_path / "none", tmp_path / "out2") == []


def test_charts_module_imports_without_matplotlib():
    import subprocess
    import sys

    code = ("import sys; sys.modules['matplotlib'] = None; import alloy.bench.scorecard.charts as c; "
            "from alloy import FeatureUnavailableError\n"
            "try:\n    c.law_error_chart([c.Row(row='3', engine='alloy/x', task='A1', metric='m', value=1.0, hi=2.0)])\nexcept FeatureUnavailableError as e: print('ok', e.fix)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "pip install matplotlib" in out.stdout
