# SPDX-License-Identifier: Apache-2.0
"""bench/newenv/drive.py end to end on the numpy cart-pole test double (no engine)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bench.newenv.checkers import common

ROOT = Path(__file__).resolve().parents[2] / "bench" / "newenv"
FAKE = ROOT / "reference" / "fake"


def _task(tmp_path: Path, **request_extra) -> Path:
    task = tmp_path / "task"
    task.mkdir()
    shutil.copy(ROOT / "drive.py", task)
    for name in ("env.py", "script.py"):
        shutil.copy(FAKE / name, task)
    request = {**json.loads((FAKE / "request.json").read_text()), **request_extra}
    (task / "request.json").write_text(json.dumps(request))
    return task


def _drive(task: Path, *args: str, fault: str = "") -> tuple[int, dict, Path]:
    env = {**os.environ, "FAKE_ENV_FAULT": fault}
    proc = subprocess.run([sys.executable, "drive.py", *args], cwd=task, env=env, capture_output=True, text=True,
                          timeout=120, check=False)
    out = task / ("out_quick" if "--quick" in args else "out")
    return proc.returncode, json.loads((out / "result.json").read_text()), out


def test_full_run_reaches_level_2(tmp_path):
    task = _task(tmp_path)
    code, res, out = _drive(task)
    assert code == 0 and res["completed"] and res["errors"] == []
    assert res["phase_order"] == ["rest", "act", "random"]
    assert (res["num_envs"], res["obs_dim"], res["act_dim"], res["episode_steps"]) == (16, 4, 1, 500)
    assert res["seeds"]["seed0_repeat_ok"] and res["seeds"]["seed1_differs"]
    assert res["timing"]["small"]["steps"] == 200 and res["timing"]["large"]["num_envs"] == 256
    assert not list(out.glob("_*"))                       # stage files are cleaned up
    rollout = common.Rollout.load(out)
    act = rollout.phase("act")
    assert act.obs.shape == (501, 16, 4) and act.probe["pole_angle"].shape == (501, 16)
    assert act.obs.dtype == np.float32 and act.done.dtype == bool
    assert np.isnan(act.reward[0]).all() and np.isfinite(act.reward[1:]).all()
    assert np.abs(act.probe["pole_angle"]).max() < 0.2    # the script's feedback law holds the pole
    assert rollout.phase("rest").done.any()               # zero action: the pole falls and the world resets
    request = json.loads((task / "request.json").read_text())
    result = common.finish(common.generic_checks(rollout, request))
    assert result.level == 2, result.failed


def test_quick_and_in_process(tmp_path):
    code, res, out = _drive(_task(tmp_path), "--quick", "--in-process", "--num-envs", "4")
    assert code == 0 and res["quick"] and res["num_envs"] == 4
    assert res["timing"]["large"] is None and res["phases"]["random"]["steps"] == 40
    assert out.name == "out_quick" and common.Rollout.load(out).phase("rest").obs.shape == (21, 4, 4)


def test_policy_phase(tmp_path):
    code, res, out = _drive(_task(tmp_path, policy=True))
    assert code == 0 and res["phase_order"][-1] == "policy"
    assert res["policy"]["success_rate"] == 1.0
    result = common.finish(common.generic_checks(common.Rollout.load(out), {"policy": True}))
    assert result.level == 3, result.failed


def test_crash_keeps_what_was_recorded(tmp_path):
    code, res, out = _drive(_task(tmp_path), "--in-process", fault="crash")
    assert code == 1 and not res["completed"]
    assert "fake engine crash" in res["traceback"] and res["errors"][0]["phase"] == "rest"
    assert res["phases"]["rest"]["steps_done"] == 29
    rollout = common.Rollout.load(out)
    assert rollout.phase("rest").obs.shape == (30, 16, 4)
    assert common.finish(common.generic_checks(rollout, {})).level == 0


@pytest.mark.parametrize("fault, failed", [("nan", "finite"), ("shape", "shapes"), ("noseed", "seed_repeat")])
def test_faults_fail_the_right_generic_check(tmp_path, fault, failed):
    code, res, out = _drive(_task(tmp_path), "--quick", "--in-process", fault=fault)
    assert code == 1                                      # the driver itself names the problem to the agent
    result = common.finish(common.generic_checks(common.Rollout.load(out), {}))
    assert failed in result.failed and result.level == 0


def test_missing_env_writes_result(tmp_path):
    task = _task(tmp_path)
    (task / "env.py").unlink()
    code, res, _ = _drive(task, "--quick")
    assert code == 1 and not res["completed"] and "ModuleNotFoundError" in res["errors"][0]["error"]
