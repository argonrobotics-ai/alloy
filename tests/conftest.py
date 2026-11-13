# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration (SPEC §6.1, WP0): markers, device fixtures, timeouts, GPU skipping."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import warp as wp


_WORKER_CACHE: tuple[Path, Path] | None = None  # (base, this xdist worker's cache) once set up


def _xdist_worker_warp_cache() -> None:
    """Under pytest-xdist each worker gets its own Warp kernel cache (SCR-GATE-1): ``<base>_xdist/<worker>``, where
    base is ``$WARP_CACHE_PATH`` (a gate slot's cache) or ``~/.cache/warp``. It is seeded, and topped up on later runs,
    with hard links to the base (as the merge gate seeds a slot; Warp only adds files by os.replace). A shared cache
    raced: one process's clear deleted another's in-flight build dir. Runs before Warp initializes; the worker hands
    its new kernels back to the base at session finish (``pytest_sessionfinish``)."""
    global _WORKER_CACHE
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return
    base = Path(os.environ.get("WARP_CACHE_PATH") or Path.home() / ".cache" / "warp")
    mine = base.parent / f"{base.name}_xdist" / worker
    mine.mkdir(parents=True, exist_ok=True)
    if base.is_dir():
        subprocess.run(["cp", "-aln", f"{base}/.", f"{mine}/"], check=False, capture_output=True)  # GNU cp; else cold
    os.environ["WARP_CACHE_PATH"] = str(mine)
    _WORKER_CACHE = (base, mine)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """An xdist worker links the kernels it compiled back into the base cache (existing files kept), so the next run
    does not compile them again in every worker."""
    if _WORKER_CACHE is not None and _WORKER_CACHE[0].is_dir():
        base, mine = _WORKER_CACHE
        subprocess.run(["cp", "-aln", f"{mine}/.", f"{base}/"], check=False, capture_output=True)


_xdist_worker_warp_cache()
wp.config.quiet = True

# SCR-WP8-8: users get the MuJoCo C route on CPU by default (alloy.route.mjc_core_config.DEFAULT_CPU_ROUTE); the suite
# stays on MuJoCo Warp on CPU, which stands in for the GPU kernels (coordinator ruling 2026-11-06 14:18Z). Forced, not
# setdefault: an ALLOY_CPU_ROUTE=mjc exported in a developer's shell must not move the suite off the Warp kernels. The
# route's own tests ask for "mjc" per entry (alloy:cpu_route) or per test (monkeypatch). Opt-in for release tiers
# (physics, 0.1.3): ALLOY_TEST_CPU_ROUTE=mjc runs the suite on the route pip users get; the merge gate drops it, so
# gates stay on warp (scripts/merge_gate.py test_env).
TEST_CPU_ROUTE_ENV = "ALLOY_TEST_CPU_ROUTE"


def _test_cpu_route() -> str:
    """The suite's CPU route: ``$ALLOY_TEST_CPU_ROUTE`` ("mjc" or "warp") when set, else "warp"."""
    route = os.environ.get(TEST_CPU_ROUTE_ENV, "").strip().lower()
    if route not in ("", "mjc", "warp"):
        raise pytest.UsageError(f"{TEST_CPU_ROUTE_ENV}={route!r}: use 'mjc' or 'warp' (unset runs the suite on warp)")
    return route or "warp"


os.environ["ALLOY_CPU_ROUTE"] = _test_cpu_route()


def pytest_terminal_summary(terminalreporter) -> None:  # noqa: ANN001 - pytest's TerminalReporter
    """Which CPU route this run tested, in every log, also under -q (a release tier runs the suite on both)."""
    how = f"{TEST_CPU_ROUTE_ENV}={os.environ[TEST_CPU_ROUTE_ENV]}" if os.environ.get(TEST_CPU_ROUTE_ENV) else "default"
    terminalreporter.write_line(f"alloy CPU route: {os.environ['ALLOY_CPU_ROUTE']} ({how})")

REPO = Path(__file__).resolve().parents[1]
COMPILE_HEAVY_TIMEOUT_S = 180
GPU_TIMEOUT_S = 600     # first CUDA run compiles kernels (~73 s on the L4) before any test logic


def _cuda_available() -> bool:
    try:
        return bool(wp.is_cuda_available())
    except Exception:  # noqa: BLE001 - a broken driver means "no CUDA" for test selection
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``gpu`` tests without CUDA; give ``compile_heavy`` 180 s and ``gpu`` 600 s unless a test sets its own."""
    has_cuda = _cuda_available()
    skip_gpu = pytest.mark.skip(reason="no CUDA device (gpu marker)")
    for item in items:
        explicit = item.get_closest_marker("timeout") is not None
        if "gpu" in item.keywords:
            if not has_cuda:
                item.add_marker(skip_gpu)
            elif not explicit:
                item.add_marker(pytest.mark.timeout(GPU_TIMEOUT_S))
        elif "compile_heavy" in item.keywords and not explicit:
            item.add_marker(pytest.mark.timeout(COMPILE_HEAVY_TIMEOUT_S))


@pytest.fixture(scope="session")
def device() -> str:
    """"cpu", or the device in ``ALLOY_DEVICE`` (e.g. "cuda:0"); skips if that CUDA device is missing."""
    dev = os.environ.get("ALLOY_DEVICE", "cpu")
    if dev.startswith("cuda") and not _cuda_available():
        pytest.skip(f"ALLOY_DEVICE={dev} but no CUDA device")
    return dev


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy generator (seed 0)."""
    return np.random.default_rng(0)


@pytest.fixture
def tmp_library(tmp_path: Path) -> Path:
    """A writable copy of ``library/`` (empty tree if the library has not landed yet)."""
    dst = tmp_path / "library"
    src = REPO / "library"
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        for sub in ("materials", "pairs", "objects", "scenes"):
            (dst / sub).mkdir(parents=True)
    return dst


@pytest.fixture(scope="session")
def newton_pin() -> str:
    """The installed Newton commit; fails loudly if it is not the pin (SPEC header)."""
    from alloy.core.newton_compat import check_newton_pin

    return check_newton_pin(strict=True)
