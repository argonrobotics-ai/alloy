# SPDX-License-Identifier: Apache-2.0
"""``alloy.envs.make`` and the ready-made scene functions print none of Warp's log lines.

Warp's start-up banner and its "Module ... load" lines were silenced only inside ``alloy.build``: PickPlace-v0 and
ClothFold-v0 vary their scene first, which starts Warp, so ``make`` printed the 6-line banner and 11-13 module lines
(install.md says they are silenced). ``ALLOY_VERBOSE=1`` keeps them. Each case runs in a fresh process, where Warp
has not started yet.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCENE = '''
import alloy
from alloy.envs import ActionSpec, Task
from alloy.envs.library_tasks import maybe_vary

SLIDE = """<mujoco><worldbody><body name="cart" pos="-1 0 0.05">
  <joint name="x" type="slide" axis="1 0 0" range="-0.1 0.1"/><geom type="box" size="0.05 0.05 0.05" mass="1"/>
  </body></worldbody><actuator><position joint="x" kp="100" kv="20"/></actuator></mujoco>"""


def scene_fn(worlds, seed):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.2, 0.0, 0.025), mass=0.2)
    scene.add_robot("cart", mjcf=SLIDE)
    return maybe_vary(scene, True, seed, pos=0.01, yaw_deg=10)       # as PickPlace-v0 and ClothFold-v0 do


def task_fn():
    return Task(obs=["block"], actions=ActionSpec(["cart/x"]), max_steps=5)
'''
MAKE = SCENE + '''
import alloy.envs
from alloy.envs.tasks import REGISTRY

REGISTRY["Quiet-v0"] = (scene_fn, task_fn)
alloy.envs.make("Quiet-v0", num_envs=1, device="cpu", backend="numpy").close()
'''
VARY = SCENE + '''
scene_fn(1, 0)
'''


def _warp_lines(code: str, verbose: bool = False) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != "ALLOY_VERBOSE"}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    if verbose:
        env["ALLOY_VERBOSE"] = "1"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600,
                         cwd=ROOT, check=True).stdout
    return [ln for ln in out.splitlines() if ln.startswith(("Warp ", "Module "))]


@pytest.mark.timeout(700)
def test_make_prints_no_warp_lines():
    assert _warp_lines(MAKE) == []                       # was: the banner and every module the variant loaded


@pytest.mark.timeout(700)
def test_a_varied_scene_function_prints_no_warp_lines():
    assert _warp_lines(VARY) == []


@pytest.mark.timeout(700)
def test_alloy_verbose_keeps_them():
    assert any(ln.startswith("Warp ") for ln in _warp_lines(VARY, verbose=True))
