# SPDX-License-Identifier: Apache-2.0
"""An unmodified Isaac Lab task steps on Alloy (the shipped example, in a fresh process: Isaac Lab keeps one simulation
per process). Needs Isaac Lab 3 (skipped without it); the CPU run is slow (kernel compile), the GPU run needs CUDA."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("isaaclab_tasks", reason="Isaac Lab is not installed")

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "isaaclab" / "01_switch_physics.py"


def _run(*args: str, timeout: float) -> str:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    out = subprocess.run([sys.executable, str(EXAMPLE), *args], capture_output=True, text=True, timeout=timeout,
                         env=env, cwd=EXAMPLE.parent)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def _check(stdout: str, envs: int) -> None:
    assert "-> AlloySolverCfg" in stdout
    assert "[alloy] physics: Alloy" in stdout and f"on {envs} worlds" in stdout
    assert f"0 of {envs} worlds INVALID now" in stdout


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cartpole_steps_on_alloy_cpu():
    _check(_run("--num_envs", "2", "--steps", "5", "--device", "cpu", timeout=1700), 2)


@pytest.mark.gpu
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("task", ["Isaac-Cartpole-Direct", "Isaac-Velocity-Flat-AnymalD"])
def test_task_steps_on_alloy_gpu(task):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    out = _run("--task", task, "--num_envs", "64", "--steps", "50", "--device", "cuda:0", timeout=1700)
    _check(out, 64)
    if "robot base height" in out:   # standing, not fallen through the ground
        assert float(out.split("robot base height ")[1].split(" m")[0]) > 0.3
