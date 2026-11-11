# SPDX-License-Identifier: Apache-2.0
"""RinseGlassware-v0's short options (CPU): ``glasses`` and ``coarse`` reach the port's task, the default stays the
task's six glasses, and one coarse glass runs through the env to the port's verdict with every action clipped to the
action space, as an RL wrapper does. The hand and tool action boxes sit about where the held bodies start (the rinse
stations are metres from the origin), so the expert's own action lies inside the space."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make
from alloy.envs.errors import EnvError
from alloy.envs.ports import rinsing
from alloy.envs.ports.movers import around_start


def test_glassware_options_set_the_ports_task() -> None:
    from alloy.tasks.rinse_glassware import Glassware  # noqa: PLC0415

    assert rinsing.glassware_task() == Glassware()                       # no option: the task's six glasses
    short = rinsing.glassware_task(glasses=1, coarse=True)
    assert (short.station.count, short.station.ds, short.station.n_theta) == (1, 4.0e-3, 16)
    assert (short.rinse_s, short.pace, short.rinse_dunks, short.sanitize_s) == (2.0, 0.6, 2, 34.0)
    two = rinsing.glassware_task(glasses=2)
    assert two.station.count == 2 and two.station.ds == Glassware().station.ds and two.pace == 1.0
    for bad in (0, 7, 1.5, True):
        with pytest.raises(EnvError) as e:
            rinsing.glassware_task(glasses=bad)
        assert e.value.field == "glasses" and "1 to 6" in e.value.message


def test_the_env_passes_the_options_to_the_run(monkeypatch) -> None:
    from alloy.tasks import rinse_glassware  # noqa: PLC0415

    seen = {}
    monkeypatch.setattr(rinse_glassware, "run", lambda task, **kw: seen.update(task=task, **kw))
    rinsing.run_glassware(2, "cpu", 0, glasses=3, coarse=True)
    assert seen["task"] == rinsing.glassware_task(3, True) and (seen["worlds"], seen["device"]) == (2, "cpu")


def test_position_bounds_sit_around_the_start_of_every_world() -> None:
    start = np.array([[[3.2, 0.1, 0.9], [2.0, -0.4, 1.0]], [[3.4, 0.0, 0.9], [2.1, -0.4, 1.1]]])   # (W=2, N=2, 3)
    low, high = around_start(start, 3.0)
    np.testing.assert_allclose(low, [0.2, -3.0, -2.1, -1.0, -3.4, -2.0], atol=1e-6)
    np.testing.assert_allclose(high, [6.4, 3.1, 3.9, 5.1, 2.6, 4.1], atol=1e-6)
    assert low.dtype == high.dtype == np.float32


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize(("name", "opts"), [("RinseSponge-v0", {}),
                                            ("RinseGlassware-v0", {"glasses": 1, "coarse": True})])
def test_the_experts_first_action_lies_inside_the_action_space(name: str, opts: dict) -> None:
    env = make(name, num_envs=1, device="cpu", autoreset="disabled", **opts)
    try:
        _, info = env.reset(seed=0)
        a, space = np.asarray(info["expert_action"]), env.single_action_space
        out = np.flatnonzero((a < space.low) | (a > space.high))
        assert out.size == 0, (name, out.tolist(), a.reshape(-1)[out % a.shape[-1]].tolist())
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
def test_one_coarse_glass_reaches_the_ports_verdict_through_the_env_cpu() -> None:
    env = make("RinseGlassware-v0", num_envs=1, device="cpu", autoreset="disabled", glasses=1, coarse=True)
    try:
        _, info = env.reset(seed=0)
        steps, done, space = 0, np.zeros(1, bool), env.single_action_space
        while not done.all() and steps < 6000:
            a = np.clip(info["expert_action"], space.low, space.high)     # as a wrapper clips to the space
            _, reward, done, _, info = env.step(a)
            steps += 1
        verdict = info["verdict"][0]
        assert done.all() and 1000 < steps < 6000, steps
        assert info["success"].tolist() == [True] and reward.tolist() == [1.0], verdict.failed
        assert verdict.glasses == 1 and verdict.judged[0][0] >= 1
    finally:
        env.close()
