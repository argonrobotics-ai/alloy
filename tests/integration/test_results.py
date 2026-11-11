# SPDX-License-Identifier: Apache-2.0
"""WP12 scoreboard collector: results tables / JSON lines -> report/1 files the WP6 scoreboard values (CPU, fast)."""

from __future__ import annotations

import json
import math

from alloy.bench.report import read_report, validate
from alloy.bench.scoreboard import scoreboard
from alloy.integration.results import collect, jsonl_rows, markdown_rows

MD = """# WPX results

| metric id | value | verdict | command | commit |
|---|---|---|---|---|
| CT-1 peg p99 | 3.1e-5 m (n = 4096) | pass | `pytest tests/contact/x.py` | abc1234 |
| CT-2 | not_run: needs the bully thin-wall sweep (WP7) | not_run | - | abc1234 |
| WPX-internal | 1.0 | pass | x | abc1234 |
| PAR-14 | 1 host sync | **fail** | y | abc1234 |
| CPL-1 | 0.4 % | info | z | abc1234 |
| PAR-1 overhead | max \\|Δq\\| 1e-6 m, 3 % | pass | w | abc1234 |

text between tables

| id | value | status | command | commit |
|---|---|---|---|---|
| LIQ-2 resting column | 0.8 % | pass (<= 1 %) | `pytest a` | def5678 |
"""


def test_markdown_rows(tmp_path):
    p = tmp_path / "wpx-results.md"
    p.write_text(MD)
    rows = markdown_rows(p, "WPX")
    assert [(r.metric_id, r.verdict) for r in rows] == [("CT-1", "pass"), ("CT-2", "not_run"), ("PAR-14", "fail"),
                                                       ("PAR-1", "pass"), ("LIQ-2", "pass")]
    assert rows[3].value_text == "max |Δq| 1e-6 m, 3 %" and rows[3].command == "w"
    assert rows[0].value == 3.1e-5 and math.isnan(rows[1].value) and rows[4].commit == "def5678"
    assert rows[0].command == "pytest tests/contact/x.py"


def test_jsonl_rows(tmp_path):
    p = tmp_path / "results.jsonl"
    lines = [{"metric": "PAR-14", "value": 0.0, "status": "pass", "command": "c", "commit": "x1", "scene": "peg"},
             {"metric": "WP12-T09", "value": 1.0, "status": "pass"},
             {"metric": "SB-3:peg", "value": None, "status": "not_run", "reason": "no sdf"}]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = jsonl_rows(p, "WP12")
    assert [(r.metric_id, r.verdict) for r in rows] == [("PAR-14", "pass"), ("SB-3", "not_run")]


def test_collect_feeds_the_scoreboard(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "wpx-results.md").write_text(MD)
    paths = collect(notes, [], tmp_path / "out", "2026-11-01")
    assert len(paths) == 1
    data = json.loads(paths[0].read_text())
    assert not validate(data)
    result = read_report(paths[0])
    assert any(n.startswith("CT-2:") for n in result.monitors["reasons"])
    rows = {r.metric_id: r for r in scoreboard([tmp_path / "out"])}
    assert rows["CT-1"].verdict == "pass" and rows["CT-1"].value == 3.1e-5
    assert rows["CT-2"].verdict == "not_run" and rows["PAR-14"].verdict == "fail"
    assert rows["LIQ-2"].verdict == "pass" and rows["CPL-1"].verdict == "not_run"


def test_scoreboard_build_writes_json_and_md(tmp_path):
    from alloy.integration.scoreboard import build, summary

    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "wpx-results.md").write_text(MD)
    out = tmp_path / "SCOREBOARD.json"
    rows = build(notes, [], [], out, tmp_path / "collected")
    data = json.loads(out.read_text())
    assert data["schema"] == "alloy.bench.scoreboard/1" and out.with_suffix(".md").exists()
    active = [r for r in rows if r.status == "active" and ":" not in r.metric_id]
    assert all(r.verdict != "not_run" or r.reason for r in active)
    assert "not_run without a reason: none" in summary(rows)
