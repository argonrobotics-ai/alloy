# SPDX-License-Identifier: Apache-2.0
"""OSS-7: alloy.check runs a scene of only static and kinematic parts (posed parts, nothing free) for the seconds asked,
as alloy.build steps it (with no solver: ``alloy.api._static``), and reports the findings of that run. Before, the
check built its simulation without that stepper, so every such scene stopped at t=0 with "scene has no dynamic entry
to step" and no stage after the load checks ran.

Laws (CPU): a table with a posed tray, and a table alone, run 0.3 s, every stage checked, ok; a posed part 20 m out
under the precision profile (INVALID past 16 m) fails the run's health check by name in every world."""

from __future__ import annotations

from pathlib import Path

import pytest

import alloy
import alloy.check

SECONDS = 0.3
WORLDS = 2


def _posed(folder: Path, *, tray: bool = True, far: bool = False) -> str:
    scene = alloy.Scene(worlds=WORLDS, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    if tray:
        scene.add_box("tray", size=(0.2, 0.2, 0.02), pos=(0, 0, 0.5), kinematic=True)
    if far:
        scene.add_box("marker", size=(0.05, 0.05, 0.05), pos=(20.0, 0, 0.5), kinematic=True)
    return str(scene.save(folder / "posed"))


def _ran_it_all(report) -> None:
    run = report.metrics["run"]
    assert run["seconds_run"] == pytest.approx(SECONDS) and run["periods"] == round(SECONDS / 0.01)
    assert not run["truncated"]
    assert not [f for f in report.findings if "stopped" in f.message]
    assert all(not str(v).startswith("skipped") for v in report.checks.values()), report.checks


@pytest.mark.parametrize("tray", [True, False], ids=["kinematic_tray", "static_only"])
def test_a_posed_scene_runs_its_seconds(tmp_path: Path, tray: bool):
    report = alloy.check(_posed(tmp_path, tray=tray), worlds=WORLDS, seconds=SECONDS, device="cpu")
    _ran_it_all(report)
    assert report.ok and report.checks["health"] == "pass" and report.checks["rest_drift"] == "pass"


def test_a_posed_scene_reports_the_findings_of_its_run(tmp_path: Path):
    report = alloy.check(_posed(tmp_path, far=True), worlds=WORLDS, seconds=SECONDS, device="cpu",
                         profile="precision")
    _ran_it_all(report)
    (bad,) = [f for f in report.findings if f.check == "health" and f.level == "error"]
    assert bad.objects == ("marker",) and bad.worlds == tuple(range(WORLDS)) and not report.ok
