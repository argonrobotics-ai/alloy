# SPDX-License-Identifier: Apache-2.0
"""Regression tests for concurrent worker pipe draining and shared timeouts in launch_mode_a."""

from __future__ import annotations

import time

import pytest

from alloy.core.errors import LaunchError
from alloy.parallel import multigpu as mg


def test_launch_mode_a_drains_rank_pipes_concurrently(tmp_path):
    """Rank 1 writing >256 KB before signalling rank 0 must not stall while waiting on rank 0."""
    flag = tmp_path / "rank1_done"
    worker = tmp_path / "sync_worker.py"
    worker.write_text(
        "import os, pathlib, sys, time\n"
        f"flag = pathlib.Path({str(flag)!r})\n"
        "if os.environ['ALLOY_RANK'] == '1':\n"
        "    sys.stdout.write('x' * 400000); sys.stdout.flush()\n"
        "    flag.write_text('ok')\n"
        "else:\n"
        "    t0 = time.monotonic()\n"
        "    while not flag.exists() and time.monotonic() - t0 < 10.0:\n"
        "        time.sleep(0.01)\n"
        "    assert flag.exists(), 'rank 1 blocked on stdout pipe'\n"
    )
    t0 = time.monotonic()
    mg.launch_mode_a(str(worker), ["cpu", "cpu"], 4, timeout_s=3.0)
    assert time.monotonic() - t0 < 2.5


def test_launch_mode_a_enforces_shared_deadline(tmp_path):
    """timeout_s must apply as one shared wall deadline across all ranks rather than scaling by rank count."""
    worker = tmp_path / "hang.py"
    worker.write_text("import time; time.sleep(10)\n")
    t0 = time.monotonic()
    timeout = 0.8
    with pytest.raises(LaunchError, match="rank 0: timeout") as exc:
        mg.launch_mode_a(str(worker), ["cpu", "cpu", "cpu"], 6, timeout_s=timeout)
    elapsed = time.monotonic() - t0
    assert "rank 2: timeout" in str(exc.value)
    assert elapsed < 2 * timeout, f"timeout scaled additively across ranks: {elapsed:.2f}s"
