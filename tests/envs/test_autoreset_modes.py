# SPDX-License-Identifier: Apache-2.0
"""autoreset="disabled" keeps an ended world ended, and "next_step" voids a restarting world's info (audit T1-001).

Before: with "disabled", a world past ``max_steps`` was reported truncated again at every later step (a new
``info["episode"]`` each time, its length 4, 5, ...) and a terminated world flipped back to not terminated; with
"next_step", the step that restarts a world carried the ended episode's ``info["success"]`` and targets.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, Task, VectorEnv
from alloy.envs import checks as C

SLIDER = """<mujoco><worldbody><body name="cart">
  <joint name="x" type="slide" axis="1 0 0" range="-0.5 0.5" damping="20"/>
  <geom type="box" size="0.02 0.02 0.02" mass="0.5"/></body></worldbody>
  <actuator><position joint="x" kp="2000" kv="100"/></actuator></mujoco>"""


def _env(autoreset: str, **task) -> VectorEnv:
    sc = alloy.Scene(dt=0.02)
    sc.add_ground()
    sc.add_robot("slider", mjcf=SLIDER, pos=(0.0, 0.0, 0.1))
    spec = ActionSpec(["slider/x"], mode="delta", delta=0.01)
    return VectorEnv(sc, Task(obs=["slider"], actions=spec, **task), worlds=2, device="cpu", backend="numpy",
                     autoreset=autoreset)


def _run(env: VectorEnv, steps: int) -> list[tuple]:
    env.reset(seed=0)
    out = []
    for _ in range(steps):
        _, _, term, trunc, info = env.step(np.zeros((2, 1), np.float32))
        out.append((np.array(term), np.array(trunc), np.array(info.get("_episode", np.zeros(2, bool))), info))
    return out


@pytest.mark.timeout(300)
def test_disabled_reports_a_truncated_episode_once_and_keeps_it_ended():
    env = _env("disabled", max_steps=3)
    steps = _run(env, 6)
    assert [s[2].tolist() for s in steps] == [[False, False]] * 2 + [[True, True]] + [[False, False]] * 3
    assert steps[2][3]["episode"]["l"].tolist() == [3, 3]
    assert all(s[1].all() for s in steps[2:])                     # still truncated until reset
    env.reset(options={"reset_mask": np.array([True, False])})
    term, trunc, ep, _ = _run_once(env)
    assert trunc.tolist() == [False, True] and not ep.any()       # world 0 runs again, world 1 still ended
    env.close()


@pytest.mark.timeout(300)
def test_disabled_never_turns_a_terminated_world_back():
    env = _env("disabled", max_steps=50, success=C.joint("slider/x") > -1.0)    # succeeds (and ends) at once
    steps = _run(env, 5)
    assert all(s[0].all() for s in steps) and steps[0][2].all() and not any(s[2].any() for s in steps[1:])
    env.close()


@pytest.mark.timeout(300)
def test_next_step_voids_the_restarting_worlds_info():
    env = _env("next_step", max_steps=50, success=C.joint("slider/x") > -1.0)
    steps = _run(env, 2)
    assert steps[0][0].all() and steps[0][3]["success"].all()     # ended with success
    restart = steps[1][3]                                         # the step that restarts both worlds
    assert not restart["success"].any() and not steps[1][0].any()
    w = env.writer
    assert np.array_equal(restart["targets"], np.asarray(env.xp.view(w.target)[w.rows]))
    env.close()


def _run_once(env: VectorEnv) -> tuple:
    _, _, term, trunc, info = env.step(np.zeros((2, 1), np.float32))
    return np.array(term), np.array(trunc), np.array(info.get("_episode", np.zeros(2, bool))), info
