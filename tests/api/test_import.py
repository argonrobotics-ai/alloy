# SPDX-License-Identifier: Apache-2.0
"""``import alloy`` stays light. The checks that need the lazy alloy/__init__.py
skip until it lands; ``test_import_under_one_second`` is a wall-clock test (listed in scripts/timing_tests.txt)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

import alloy

ROOT = Path(alloy.__file__).resolve().parents[1]
LAZY = "after_import" in (ROOT / "alloy" / "__init__.py").read_text()
needs_lazy_init = pytest.mark.skipif(not LAZY, reason="alloy/__init__.py is not lazy yet")


def _fresh(code: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}      # keep HOME / WARP_CACHE_PATH: the xdist worker's own cache
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == alloy.__version__


@needs_lazy_init
def test_import_leaves_engine_unloaded():
    got = _fresh("import sys, alloy; print(sorted(m for m in ('newton', 'warp', 'numpy') if m in sys.modules))")
    assert got == "[]"


@needs_lazy_init
def test_top_level_names():
    got = _fresh("import alloy; print(alloy.Scene.__module__, callable(alloy.build), 'build' in dir(alloy))")
    assert got == "alloy.api.scene True True"


@needs_lazy_init
@pytest.mark.parametrize("first", ["from alloy._vendor import newton", "import alloy.scene.coordinate_carry"])
def test_carry_installed_in_either_import_order(first: str):
    code = (f"import alloy; {first}; from alloy._vendor import newton; "
            "print(getattr(newton.solvers.SolverMuJoCo, '_alloy_carry_default', False))")
    assert _fresh(code) == "True"


@needs_lazy_init
def test_import_under_one_second():
    best = min(_time_import() for _ in range(3))
    assert best < 1.0, f"import alloy took {best:.2f} s"


def _time_import() -> float:
    t0 = time.perf_counter()
    _fresh("import alloy")
    return time.perf_counter() - t0
