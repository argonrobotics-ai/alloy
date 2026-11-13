# SPDX-License-Identifier: Apache-2.0
"""A rod's centre line in the env task state (Gemini audit T2-003, 0.1.2): ``"<rod>.points"`` is a state key, so
``Task(obs=["arm", "wire.points"])`` and ``C.fraction_inside("wire", ...)`` work on both rod routes, and read what
``sim.state()`` reads. Before, both raised EnvError, its fix pointed at ``wire.pos`` (the first segment only), and
the documented ``"<rod>/0.pos"`` workaround named no body (rod_vbd names its segments ``wire/wire_edge_body_<i>``)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.envs import Task, VectorEnv
from alloy.envs import checks as C
from alloy.envs.errors import EnvError

pytestmark = pytest.mark.timeout(1800)
ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="b" pos="0 0 0.05"><joint name="j" type="hinge" axis="0 0 1" range="-1 1"/>
  <geom type="sphere" size="0.02" mass="0.2"/></body></worldbody>
  <actuator><position joint="j" kp="50" kv="5"/></actuator></mujoco>"""
SEGMENTS = 6


def _sim(clamp: str | None) -> alloy.Sim:
    scene = alloy.Scene(worlds=2, dt=0.02)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(-0.3, 0.0, 0.0))
    scene.add_rod("wire", length=0.24, radius=0.004, pos=(0.0, 0.0, 0.15), direction=(1, 0, 0), segments=SEGMENTS,
                  clamp=clamp)
    return alloy.build(scene, device="cpu")


@pytest.mark.parametrize("clamp,route,backend", [("start", "rod_vbd", "numpy"), (None, "chain", "numpy"),
                                                  ("start", "rod_vbd", "torch")])
def test_a_rods_centre_line_is_an_observation_and_a_check(clamp, route, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    sim = _sim(clamp)
    try:
        assert sim.routes()["wire"]["route"] == route
        task = Task(obs=["arm", "wire.points"], max_steps=5,
                    reward={"in": (1.0, C.fraction_inside("wire", (-1, -1, -1), (1, 1, 1)))})
        env = VectorEnv(sim, task, backend=backend)
        obs, _ = env.reset(seed=0)
        env.step(env.xp.zeros((2, 1)))                              # the reward reads the line too
        sim.step(3)                                                 # the rod sags: the line is read, not cached
        s, np_ = env.state, env.xp.to_numpy
        want = sim.state(["wire"])["wire.points"]
        n = len(sim.state(["wire"], world=0)["wire.points"])
        assert n >= SEGMENTS + 1 and want.shape == (2, n, 3)
        assert np.allclose(np_(s["wire.points"]), want, atol=2e-6), (s["wire.points"], want)
        assert "wire.points" in s.keys() and tuple(obs.shape) == (2, 2 + 3 * n)
        assert np.allclose(np_(obs)[:, 2:].reshape(2, n, 3)[0], np_(obs)[:, 2:].reshape(2, n, 3)[1], atol=1e-3)
        assert np.allclose(np_(C.fraction_inside("wire", (-1, -1, -1), (1, 1, 1))(s)), 1.0)
        inside = np_(C.fraction_inside("wire", (-1, -1, -1), (0.1, 1, 1))(s))
        assert np.allclose(inside, np.mean(want[..., 0] <= 0.1, axis=-1))
        with pytest.raises(EnvError) as err:
            _ = s["wire.bogus"]
        assert "wire.points" in err.value.fix
    finally:
        sim.close()
