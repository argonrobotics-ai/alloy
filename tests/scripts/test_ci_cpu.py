# SPDX-License-Identifier: Apache-2.0
"""scripts/ci_cpu.sh's caller knobs (SCR-GATE-2: WP1's fast lane and post-merge runner call it).
- ALLOY_CI_PATHS: unset means `tests`; set but empty means the caller's args are the whole selection.
- ALLOY_CI_DIST picks the xdist mode (default loadfile); an unknown mode fails before anything runs.
- ALLOY_CI_SKIP_ANALYTIC=1 skips the analytic suite and prints no "analytic suite exit" line.
- The "ci_cpu: pytest exit N" line the merge gate reads is unchanged.
- Both suites run on the warp CPU route, or on ALLOY_TEST_CPU_ROUTE's (release tiers; tests/conftest.py).
A fake PYTHON records each call, so nothing real runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci_cpu.sh"
HAVE_ANALYTIC = (ROOT / "alloy/bench/__main__.py").exists() and (ROOT / "bench/goldens/analytic_baseline.json").exists()
_FAKE = """#!/usr/bin/env bash
echo "$*" >> "$CI_FAKE_LOG"
echo "${ALLOY_CPU_ROUTE:-unset}" >> "$CI_FAKE_LOG.route"
[[ "$2" == "pytest" ]] && exit "${CI_FAKE_RC:-0}"
exit 0
"""


def _run(tmp_path: Path, *args: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    fake = tmp_path / "python"
    fake.write_text(_FAKE)
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    base = {k: v for k, v in os.environ.items() if not k.startswith("ALLOY_CI_") and k != "ALLOY_TEST_CPU_ROUTE"}
    base.update(PYTHON=str(fake), CI_FAKE_LOG=str(log), ALLOY_CI_SKIP_WARM="1", ALLOY_CI_OUT=str(tmp_path / "out"))
    (tmp_path / "out").mkdir(exist_ok=True)
    proc = subprocess.run(["bash", str(SCRIPT), *args], env={**base, **env}, capture_output=True, text=True,
                          timeout=60)
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def _pytest_call(calls: list[str]) -> str:
    (call,) = [c for c in calls if c.startswith("-m pytest")]
    return call


def test_default_runs_tests_serially_with_the_analytic_suite(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert _pytest_call(calls) == "-m pytest tests -q -m not gpu and not slow"
    assert any(c.startswith("-m alloy.bench run --suite analytic") for c in calls) == HAVE_ANALYTIC
    assert "ci_cpu: pytest exit 0, wall" in proc.stdout


def test_empty_paths_take_the_callers_selection(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, "tests/core/test_types.py", "--deselect", "tests/a.py::t", ALLOY_CI_PATHS="")
    assert proc.returncode == 0, proc.stderr
    assert _pytest_call(calls) == "-m pytest -q -m not gpu and not slow tests/core/test_types.py --deselect tests/a.py::t"


def test_paths_split_on_spaces(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, ALLOY_CI_PATHS="tests/core tests/scene")
    assert proc.returncode == 0, proc.stderr
    assert _pytest_call(calls).startswith("-m pytest tests/core tests/scene -q ")


def test_empty_paths_without_args_fail(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, ALLOY_CI_PATHS="")
    assert proc.returncode == 2 and "no test args" in proc.stderr and calls == []


@pytest.mark.parametrize("dist", ["loadfile", "worksteal"])
def test_workers_use_the_dist_mode(tmp_path: Path, dist: str) -> None:
    env = {"ALLOY_CI_WORKERS": "4"} | ({"ALLOY_CI_DIST": dist} if dist != "loadfile" else {})
    proc, calls = _run(tmp_path, **env)
    assert proc.returncode == 0, proc.stderr
    assert _pytest_call(calls).endswith(f"-n 4 --dist {dist}")
    assert f"with -n 4 --dist {dist}" in proc.stdout


def test_unknown_dist_fails_first(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, ALLOY_CI_WORKERS="4", ALLOY_CI_DIST="bogus")
    assert proc.returncode == 2 and "not an xdist mode" in proc.stderr and calls == []


def test_skip_analytic_keeps_the_pytest_exit_line(tmp_path: Path) -> None:
    proc, calls = _run(tmp_path, ALLOY_CI_SKIP_ANALYTIC="1", CI_FAKE_RC="1")
    assert proc.returncode == 1
    assert not any("alloy.bench" in c for c in calls)
    assert "ci_cpu: pytest exit 1, wall" in proc.stdout and "analytic suite exit" not in proc.stdout


def test_ci_runs_on_the_warp_cpu_route(tmp_path: Path) -> None:
    """SCR-WP8-8: pytest and the analytic suite run on the Warp CPU route even when the caller exports mjc."""
    proc, calls = _run(tmp_path, ALLOY_CPU_ROUTE="mjc")
    assert proc.returncode == 0, proc.stderr
    routes = (tmp_path / "calls.log.route").read_text().split()
    assert len(routes) == len(calls) >= 1 and set(routes) == {"warp"}


def test_the_release_opt_in_moves_both_suites(tmp_path: Path) -> None:
    """ALLOY_TEST_CPU_ROUTE=mjc (a release tier on the route pip users get) moves pytest and the analytic suite."""
    proc, calls = _run(tmp_path, ALLOY_TEST_CPU_ROUTE="mjc", ALLOY_CPU_ROUTE="warp")
    assert proc.returncode == 0, proc.stderr
    routes = (tmp_path / "calls.log.route").read_text().split()
    assert len(routes) == len(calls) >= 1 and set(routes) == {"mjc"}
    assert "ci_cpu: CPU route mjc" in proc.stdout
