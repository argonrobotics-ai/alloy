# SPDX-License-Identifier: Apache-2.0
"""A pour's drops still at the nozzle take no part in env state and checks, as in ``sim.state()`` (audit T2-005).

``scene.add_pour`` keeps every drop of the stream in ``particle_q`` from the start, parked at the nozzle until it is
poured, and ``sim.state()`` marks the poured ones in ``"<name>.active"``. The env's state had no ``.active``, its
``.center`` averaged the parked drops in, and ``C.fraction_inside`` counted them: a box around the nozzle read 100 %
of the water inside before a single drop was poured.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs import checks as C

ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="b" pos="0 0 0.05"><joint name="j" type="hinge" axis="0 0 1" range="-1 1"/>
  <geom type="sphere" size="0.02" mass="0.2"/></body></worldbody>
  <actuator><position joint="j" kp="50" kv="5"/></actuator></mujoco>"""
NOZZLE_BOX = ((-0.05, -0.05, 0.0), (0.05, 0.05, 0.20))         # the cup and the nozzle above it


@pytest.fixture(scope="module")
def env():
    scene = alloy.Scene(worlds=2, dt=0.02)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(-0.35, 0.0, 0.0))
    scene.add_open_box("cup", inner=(0.06, 0.06, 0.06), wall=0.008, pos=(0.0, 0.0, 0.0))
    scene.add_pour("stream", 8e-6, pos=(0.0, 0.0, 0.15), rate=16e-6, speed=0.4)
    e = VectorEnv(scene, Task(obs=["stream.center"], max_steps=40), worlds=2, device="cpu", backend="numpy")
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_nothing_poured_counts_nothing_inside(env):
    env.reset(seed=0)
    assert not np.asarray(env.state["stream.active"]).any()
    assert np.array_equal(env.state["stream.active"], env.sim.state(["stream"])["stream.active"])
    assert np.allclose(C.fraction_inside("stream", *NOZZLE_BOX)(env.state), 0.0)


@pytest.mark.compile_heavy
def test_center_and_fraction_follow_the_poured_drops(env):
    env.reset(seed=0)
    for _ in range(12):
        env.step(np.zeros((2, env.writer.size), np.float32))
    sim = env.sim.state(["stream"])
    active, pts = sim["stream.active"], sim["stream.points"]
    assert np.array_equal(env.state["stream.active"], active) and 0 < active[0].sum() < active.shape[1]
    poured = np.stack([pts[w][active[w]].mean(0) for w in range(2)])
    assert np.allclose(env.state["stream.center"], poured, atol=1e-5)
    lo, hi = (np.asarray(v) for v in NOZZLE_BOX)
    inside = (np.all((pts >= lo) & (pts <= hi), axis=-1) & active).mean(-1)
    assert np.allclose(C.fraction_inside("stream", *NOZZLE_BOX)(env.state), inside, atol=1e-6)
