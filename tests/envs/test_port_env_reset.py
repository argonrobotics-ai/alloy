# SPDX-License-Identifier: Apache-2.0
"""Gemini audit T1-003 / T1-007: a task port (``PortEnv``) follows Gymnasium's vector contract where it can and says
so where it cannot.

T1-003: ``metadata["autoreset_mode"]`` was a plain str (Gymnasium's RecordEpisodeStatistics raised TypeError),
``backend=`` reached the port's run as an unknown argument (TypeError), and the RL wrappers' error told the user to
call ``alloy.envs.make``, which had made the port. T1-007: ``reset(seed=S)`` on a fresh env (or straight after another
reset) kept the old seed and never reseeded ``action_space``, and ``reset_mask`` restarted every world. CPU."""
from __future__ import annotations

import numpy as np
import pytest

import alloy.envs
from alloy.envs.errors import EnvError
from alloy.envs.ports import PORTS
from alloy.envs.ports.env import check_options

BATCHED = "PaintClearCoat-v0"          # one simulation for all worlds
PER_WORLD = "CookReheatPorridge-v0"    # one simulation per world


@pytest.fixture(scope="module")
def batched():
    env = alloy.envs.make(BATCHED, num_envs=2, device="cpu", size="small", backend="numpy")
    yield env
    env.close()


def test_every_port_accepts_its_own_options_and_none_else():
    for spec in PORTS.values():
        check_options(spec, spec.options)                       # no false alarm on a shipped port
    with pytest.raises(EnvError) as err:
        check_options(PORTS[BATCHED], {"size": "small", "colour": "red"})
    assert "colour" in err.value.message and "size" in err.value.fix


@pytest.mark.compile_heavy
def test_metadata_and_unsupported_options_follow_gymnasium(batched):
    gym = pytest.importorskip("gymnasium")
    assert batched.metadata["autoreset_mode"] is gym.vector.AutoresetMode.SAME_STEP   # was: the str "same_step"
    gym.wrappers.vector.RecordEpisodeStatistics(batched)        # was: TypeError
    for kw, field in (({"backend": "torch"}, "backend"), ({"autoreset": "next_step"}, "autoreset")):
        with pytest.raises(EnvError) as err:
            alloy.envs.make(BATCHED, num_envs=2, device="cpu", size="small", **kw)
        assert err.value.field == field and err.value.fix


@pytest.mark.compile_heavy
def test_the_rl_wrappers_name_the_task_port_and_what_to_use(batched):
    from alloy.envs.wrappers._common import require_vector_env

    with pytest.raises(EnvError) as err:
        require_vector_env(batched, "rsl_rl")
    assert "task port" in err.value.message and "GripperLift-v0" in err.value.fix


@pytest.mark.compile_heavy
def test_a_batched_port_restarts_only_all_worlds_together(batched):
    batched.reset(seed=0)
    with pytest.raises(EnvError) as err:
        batched.reset(options={"reset_mask": np.array([True, False])})
    assert err.value.field == "reset_mask"
    batched.reset(options={"reset_mask": np.array([True, True])})    # all of them: fine


def _hob_and_temp(env, info) -> tuple[float, float]:
    sim = env._slots[0].run.sim
    return float(info["expert_action"][0, -1]), float(sim.plugins["particle_thermal"].temperature.numpy().mean())


@pytest.mark.compile_heavy
def test_a_seeded_reset_takes_effect_on_a_fresh_env_and_reseeds_the_action_space():
    late, early = (alloy.envs.make(PER_WORLD, num_envs=1, device="cpu"),
                   alloy.envs.make(PER_WORLD, num_envs=1, device="cpu", seed=42))
    try:
        got, want = _hob_and_temp(late, late.reset(seed=42)[1]), _hob_and_temp(early, early.reset(seed=42)[1])
        assert got == want                                       # was: the constructor's seed 0
        assert np.array_equal(late.action_space.sample(), early.action_space.sample())
        other = _hob_and_temp(late, late.reset(seed=99)[1])      # straight after a reset: was ignored
        assert other != want
    finally:
        late.close()
        early.close()


@pytest.mark.compile_heavy
def test_reset_mask_restarts_only_the_masked_worlds():
    env = alloy.envs.make(PER_WORLD, num_envs=2, device="cpu", seed=0)
    try:
        _, info = env.reset()
        obs, *_ = env.step(info["expert_action"])
        assert obs[1, -1] > 0.0
        after, _ = env.reset(options={"reset_mask": np.array([True, False])})
        assert after[0, -1] == 0.0 and after[1, -1] == obs[1, -1]   # was: world 1 back to t = 0
    finally:
        env.close()
