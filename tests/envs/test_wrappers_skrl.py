# SPDX-License-Identifier: Apache-2.0
"""The skrl adapter: tensors pass straight through, shapes follow skrl's rules, a grouped env gives a state."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("skrl")

from alloy.envs import VectorEnv, make  # noqa: E402
from alloy.envs.errors import EnvError  # noqa: E402
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task  # noqa: E402
from alloy.envs.wrappers.for_skrl import SkrlEnv  # noqa: E402
from alloy.envs.wrappers.train_skrl import train  # noqa: E402

GROUPS = {"policy": ["gripper", "action"], "critic": ["gripper", "cube", "action"]}


@pytest.fixture(scope="module")
def raw_env():
    e = make("GripperLift-v0", num_envs=4, device="cpu", backend="torch", max_steps=5)
    yield e
    e.close()


@pytest.fixture
def env(raw_env):
    return SkrlEnv(raw_env)


@pytest.mark.compile_heavy
def test_contract_shapes_dtypes_spaces(env):
    obs, info = env.reset()
    assert isinstance(obs, torch.Tensor) and obs.dtype == torch.float32 and obs.shape == (4, 22)
    assert env.num_envs == 4 and env.num_agents == 1 and str(env.device) == "cpu"
    assert env.observation_space.shape == (22,) and env.action_space.shape == (3,)
    assert env.state_space is None and env.state() is None
    obs, reward, term, trunc, info = env.step(env.zero_actions())
    assert obs.shape == (4, 22) and reward.shape == term.shape == trunc.shape == (4, 1)
    assert reward.dtype == torch.float32 and term.dtype == trunc.dtype == torch.bool
    obs2, _ = env.reset()                                    # a repeated reset keeps the running episode
    assert torch.equal(obs2, obs)
    assert env.unwrapped is env._env and env.render() is None


@pytest.mark.compile_heavy
def test_actions_pass_through_without_numpy(raw_env, env, monkeypatch):
    seen = []
    real = raw_env.step
    monkeypatch.setattr(raw_env, "step", lambda a: (seen.append(a), real(a))[1])
    actions = env.zero_actions()
    env.step(actions)
    assert seen[0] is actions and isinstance(seen[0], torch.Tensor)


@pytest.mark.compile_heavy
def test_truncation_at_max_steps_is_reported(env):
    env.reset()
    env.take_stats()
    truncated = []
    for _ in range(6):
        _, _, _, trunc, _ = env.step(env.zero_actions())
        truncated.append(bool(trunc.all()))
    assert any(truncated)
    assert env.tally.episodes >= 4
    assert env.take_stats()[0] >= 4


@pytest.mark.compile_heavy
def test_numpy_backend_is_refused():
    e = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy")
    try:
        with pytest.raises(EnvError, match="torch backend"):
            SkrlEnv(e)
    finally:
        e.close()


def test_imports_without_the_libraries():
    code = ("import sys; [sys.modules.__setitem__(m, None) for m in ('torch','skrl','gymnasium')]; "
            "import alloy.envs.wrappers.for_skrl as f, alloy.envs.wrappers.train_skrl\n"
            "try:\n    f.SkrlEnv(None)\nexcept Exception as e:\n    print(type(e).__name__, e.fix)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "EnvError" in out.stdout and "pip install" in out.stdout


@pytest.mark.compile_heavy
def test_train_smoke(tmp_path, capsys):
    summary = train("GripperLift-v0", num_envs=4, iterations=2, device="cpu", log_dir=str(tmp_path),
                    report_every=1, env_kwargs={"max_steps": 20})
    assert summary["samples"] == 2 * 32 * 4 and summary["checkpoint"].startswith(str(tmp_path))
    assert set(summary) == {"success_rate", "minutes", "samples", "checkpoint"}
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_agent_does_not_clip_value_updates(env):
    """skrl's default value_clip=0.2 froze the critic and GripperLift never learned (success 0.0, then NaN)."""
    from alloy.envs.wrappers.train_skrl import build_agent  # noqa: PLC0415

    agent = build_agent(env)
    assert agent.cfg.value_clip == 0 and agent.cfg.time_limit_bootstrap
    assert agent.cfg.entropy_loss_scale > 0                  # without it the action spread collapsed in some runs
    assert agent.policy._g_clip_actions is False if hasattr(agent.policy, "_g_clip_actions") else True


@pytest.fixture(scope="module")
def grouped_env():
    task = dataclasses.replace(gripper_lift_task(max_steps=20), obs=GROUPS)
    e = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu", backend="torch")
    yield e
    e.close()


@pytest.mark.compile_heavy
def test_critic_group_is_the_state(grouped_env):
    env = SkrlEnv(grouped_env)
    obs, _ = env.reset()
    state = env.state()
    assert env.has_critic and obs.shape[0] == state.shape[0] == 4 and state.shape[1] > obs.shape[1]
    assert env.state_space.shape == (state.shape[1],) and env.observation_space.shape == (obs.shape[1],)
    obs, *_ = env.step(env.zero_actions())
    assert env.state().shape == state.shape


@pytest.mark.compile_heavy
def test_grouped_env_trains_with_critic_in_value_model(tmp_path):
    task = dataclasses.replace(gripper_lift_task(max_steps=20), obs=GROUPS)
    e = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu", backend="torch")
    summary = train("GripperLift-v0", iterations=2, log_dir=str(tmp_path), env=e)
    assert summary["samples"] == 2 * 32 * 4
    agent = torch.load(summary["checkpoint"], weights_only=False)
    n_critic = e.single_observation_space["critic"].shape[0] if hasattr(e.single_observation_space, "spaces") else None
    first = agent["value"]["net.0.weight"]
    assert first.shape[1] > agent["policy"]["net.0.weight"].shape[1]
    assert n_critic is None or first.shape[1] == n_critic


@pytest.mark.compile_heavy
def test_time_limit_bootstrap_uses_the_ended_episodes_last_observation(grouped_env):
    """The env restarts a finished world inside step, so the returned observation is the next episode's first; the
    wrapper hands the ended episode's last observation (and critic state) to the agent for the bootstrap."""
    import torch

    from alloy.envs.wrappers.for_skrl import BOOTSTRAP_OBS, BOOTSTRAP_STATES
    from alloy.envs.wrappers.train_skrl import agent_class

    venv = SkrlEnv(grouped_env)
    venv.reset()
    info, steps = {}, 0
    while "final_obs" not in info:
        assert BOOTSTRAP_OBS not in info
        obs, _, _, truncated, info = venv.step(0.5 * torch.ones(venv.num_envs, 3))
        steps += 1
        assert steps <= grouped_env.task.max_steps
    done = info["_final_obs"]
    assert bool(done.all()) and bool(truncated.all())
    torch.testing.assert_close(info[BOOTSTRAP_OBS], info["final_obs"]["policy"])
    torch.testing.assert_close(info[BOOTSTRAP_STATES], info["final_obs"]["critic"])
    assert not torch.equal(info[BOOTSTRAP_OBS], obs)                 # the returned obs is the restarted world's

    seen = {}
    base = agent_class().__mro__[1]
    original = base.record_transition
    try:
        base.record_transition = lambda self, **kw: seen.update(kw)
        agent = agent_class().__new__(agent_class())                 # no models needed for this method
        agent.record_transition(next_observations=obs, next_states=venv.state(), infos=info)
    finally:
        base.record_transition = original
    assert seen["next_observations"] is info[BOOTSTRAP_OBS] and seen["next_states"] is info[BOOTSTRAP_STATES]
