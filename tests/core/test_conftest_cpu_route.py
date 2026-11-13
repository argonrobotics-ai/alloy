# SPDX-License-Identifier: Apache-2.0
"""The suite's CPU route switch in tests/conftest.py (physics, 0.1.3 release tiers): unset, the suite runs on the warp
CPU route (coordinator ruling 2026-11-06, an exported ALLOY_CPU_ROUTE=mjc ignored); ALLOY_TEST_CPU_ROUTE=mjc or warp
picks the route, which is what alloy then resolves; any other value stops the run with the fix. Each case runs pytest
in a subprocess on a probe beside a copy of this repo's conftest."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE = '''
import os
from alloy.route.mjc_core_config import requested_route

def test_route():
    assert os.environ["ALLOY_CPU_ROUTE"] == os.environ["EXPECT_ROUTE"]
    assert requested_route({}) == os.environ["EXPECT_ROUTE"]   # what an entry without alloy:cpu_route gets
'''


def _run(tmp_path: Path, expect: str, **env: str) -> subprocess.CompletedProcess:
    (tmp_path / "conftest.py").write_text((REPO / "tests" / "conftest.py").read_text())
    (tmp_path / "test_probe.py").write_text(PROBE)
    base = {k: v for k, v in os.environ.items()
            if k not in ("ALLOY_TEST_CPU_ROUTE", "ALLOY_CPU_ROUTE", "PYTEST_ADDOPTS", "PYTEST_XDIST_WORKER")}
    full = {**base, "PYTHONPATH": str(REPO), "EXPECT_ROUTE": expect, **env}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", str(tmp_path)]
    return subprocess.run(cmd, cwd=tmp_path, env=full, capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize(("env", "route"), [
    ({}, "warp"),
    ({"ALLOY_CPU_ROUTE": "mjc"}, "warp"),             # a developer's exported user setting does not move the suite
    ({"ALLOY_TEST_CPU_ROUTE": "mjc"}, "mjc"),
    ({"ALLOY_TEST_CPU_ROUTE": " MJC "}, "mjc"),
    ({"ALLOY_TEST_CPU_ROUTE": "warp", "ALLOY_CPU_ROUTE": "mjc"}, "warp"),
    ({"ALLOY_TEST_CPU_ROUTE": ""}, "warp"),
], ids=["unset", "user-env-ignored", "mjc", "mjc-spaced", "warp", "empty"])
def test_the_switch_picks_the_suites_cpu_route(tmp_path: Path, env: dict[str, str], route: str) -> None:
    res = _run(tmp_path, route, **env)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert f"alloy CPU route: {route} (" in res.stdout   # the log says which route ran


def test_an_unknown_route_stops_the_run_with_the_fix(tmp_path: Path) -> None:
    res = _run(tmp_path, "warp", ALLOY_TEST_CPU_ROUTE="gpu")
    assert res.returncode != 0
    assert "ALLOY_TEST_CPU_ROUTE='gpu': use 'mjc' or 'warp'" in res.stdout + res.stderr
