# SPDX-License-Identifier: Apache-2.0
"""The suite's CPU route switch in tests/conftest.py (physics, 0.1.3 release tiers): unset, the suite runs on the warp
CPU route (coordinator ruling 2026-11-06, an exported ALLOY_CPU_ROUTE=mjc ignored); ALLOY_TEST_CPU_ROUTE=mjc or warp
picks the route, which is what alloy then resolves; any other value stops the run with the fix. A test marked
warp_route("<why>") stays on warp under mjc, its module-scoped fixtures too, and the run lists it by reason; a
module fixture that sets the route itself keeps it for every test that uses it; a mark
without a reason stops the run at collection. Each case runs pytest in a subprocess on probes beside a copy of this
repo's conftest."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.timeout(300)   # each case starts pytest + Warp in a subprocess: past 60 s on a loaded host
PROBE = '''
import os
from alloy.route.mjc_core_config import requested_route

def test_route():
    assert os.environ["ALLOY_CPU_ROUTE"] == os.environ["EXPECT_ROUTE"]
    assert requested_route({}) == os.environ["EXPECT_ROUTE"]   # what an entry without alloy:cpu_route gets
'''


def _run(tmp_path: Path, expect: str, files: dict[str, str] | None = None, **env: str) -> subprocess.CompletedProcess:
    (tmp_path / "conftest.py").write_text((REPO / "tests" / "conftest.py").read_text())
    for name, text in (files or {"test_probe.py": PROBE}).items():
        (tmp_path / name).write_text(text)
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


MARKED = {
    "test_marked.py": '''
import os
import pytest
from alloy.route.mjc_core_config import requested_route

@pytest.fixture(scope="module")
def built():
    return requested_route({})       # a fixture built for the first test that uses it

@pytest.mark.warp_route("probe: reads MuJoCo Warp's data")
def test_marked(built):
    assert (requested_route({}), built) == ("warp", "warp")

def test_unmarked_beside_it():
    assert requested_route({}) == os.environ["EXPECT_ROUTE"]
''',
    "test_module_marked.py": '''
import pytest
from alloy.route.mjc_core_config import requested_route

pytestmark = pytest.mark.warp_route(reason="probe: the world clock")

@pytest.fixture(scope="module")
def built():
    return requested_route({})

def test_a(built):
    assert (requested_route({}), built) == ("warp", "warp")

def test_b(built):
    assert requested_route({}) == "warp"
''',
    "test_own_route.py": '''
import pytest
from alloy.route.mjc_core_config import requested_route

@pytest.fixture(scope="module", params=["mjc", "warp"])
def route(request):
    with pytest.MonkeyPatch.context() as patch:        # a module fixture that picks the route itself
        patch.setenv("ALLOY_CPU_ROUTE", request.param)
        yield request.param

def test_first(route):
    assert requested_route({}) == route

def test_second(route):
    assert requested_route({}) == route                 # stands for every test that uses it
''',
    "test_z_after.py": '''
import os
from alloy.route.mjc_core_config import requested_route

def test_the_next_module_is_back_on_the_suites_route():
    assert requested_route({}) == os.environ["EXPECT_ROUTE"]
''',
}


@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_a_marked_test_stays_on_warp_and_is_listed_by_reason(tmp_path: Path, route: str) -> None:
    env = {"ALLOY_TEST_CPU_ROUTE": route} if route == "mjc" else {}
    res = _run(tmp_path, route, MARKED, **env)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "9 passed" in res.stdout
    kept = [line.strip() for line in res.stdout.splitlines() if "kept on warp" in line]
    if route == "mjc":
        assert kept == ["2 kept on warp: probe: the world clock", "1 kept on warp: probe: reads MuJoCo Warp's data"]
    else:
        assert kept == []       # the suite is on warp already: nothing is kept, nothing is listed


@pytest.mark.parametrize("mark", ['pytest.mark.warp_route()', 'pytest.mark.warp_route("  ")'], ids=["none", "blank"])
def test_a_mark_without_its_reason_stops_the_run_at_collection(tmp_path: Path, mark: str) -> None:
    probe = f"import pytest\n\n@{mark}\ndef test_x():\n    pass\n"
    res = _run(tmp_path, "warp", {"test_probe.py": probe})
    assert res.returncode not in (0, 1)
    assert "needs its reason" in res.stdout + res.stderr
