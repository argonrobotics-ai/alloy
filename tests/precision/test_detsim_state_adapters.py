# SPDX-License-Identifier: Apache-2.0
"""A DetSim's snapshot covers its solver's own state in a fresh process (PRC-13, 2026-11-09).

DetSim drives raw Newton solvers. The solver state adapters are registered by ``alloy.build`` (or by importing
alloy.coupling), so before the fix a DetSim snapshot held only Newton's State whenever nothing had imported them
first. That made R1 diverge at period 1 for SolverMuJoCo, whose warm start and velocities live in mjw_data:
PRC-13's MuJoCo 'requested' case, run alone, then reported a silent downgrade on main and on the Newton bump."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = """
import json, sys
from alloy.core import registry
from alloy._vendor.newton.solvers import SolverMuJoCo
from alloy.precision.repeat import repeat_test
from alloy.testing.precision import det_sim
before = any(issubclass(SolverMuJoCo, k) for k in registry._REG.state)  # registered by the imports alone?
make = lambda n: det_sim("mujoco", n, det=False, boxes=2)
keys = sorted(make(2).state_arrays())
r1 = repeat_test(make, "R1", steps=2, world_counts=(2,))
print(json.dumps({"before": before, "keys": keys, "r1": r1.equal[2]}))
"""


def _fresh_probe() -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "ALLOY_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=900, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_detsim_snapshot_covers_mujoco_warp_state_and_r1_holds_in_a_fresh_process():
    r = _fresh_probe()
    assert not r["before"], "the imports registered the adapters: the probe no longer starts unregistered"
    assert {"mjw.qacc_warmstart", "mjw.qacc", "mjw.cvel"} <= set(r["keys"]), r["keys"]
    assert r["r1"] is True
