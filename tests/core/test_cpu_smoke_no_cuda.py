# SPDX-License-Identifier: Apache-2.0
"""macOS on Apple chips (arm64) is a shipped and tested platform (user decision, coordinator 2026-11-08 01:22Z): with
no CUDA at all, the vendored Newton and MuJoCo Warp import and step on Warp's CPU device. The release CI's macos-14
job runs the ``cpu_route`` tests; on Linux the subprocess hides any GPU (CUDA_VISIBLE_DEVICES=""), so the test is
the same everywhere and skips nothing on a machine without one."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

CODE = """
import warp as wp
wp.init()
assert not wp.is_cuda_available(), "a CUDA device is visible"
import mujoco
import alloy
from alloy._vendor import mujoco_warp, newton

s = alloy.Scene(dt=0.01)
s.add_ground()
s.add_box("b", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.3), mass=1.0)
sim = alloy.build(s, device="cpu")
sim.step(3)
z = float(sim.state(world=0)["b.pos"][2])

m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='plane' size='1 1 .1'/>"
    "<body pos='0 0 0.5'><freejoint/><geom type='sphere' size='0.05'/></body></worldbody></mujoco>")
d = mujoco.MjData(m)
with wp.ScopedDevice("cpu"):
    mm, md = mujoco_warp.put_model(m), mujoco_warp.put_data(m, d)
    for _ in range(3):
        mujoco_warp.step(mm, md)
    qz = float(md.qpos.numpy()[0, 2])
print(newton.__version__, mujoco_warp.__version__, 0.0 < z < 0.3, 0.0 < qz < 0.5)
"""


@pytest.mark.cpu_route
@pytest.mark.timeout(900)   # a fresh machine compiles the CPU kernels of both engines once (cached afterwards)
def test_vendored_engines_step_on_the_cpu_without_cuda() -> None:
    from alloy._vendor import VERSIONS

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    env.pop("ALLOY_CPU_ROUTE", None)
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=880)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split("\n")[-2:][0].split() == [VERSIONS["newton"], VERSIONS["mujoco_warp"], "True", "True"], \
        out.stdout[-1000:]
