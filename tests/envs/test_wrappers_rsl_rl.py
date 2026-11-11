# SPDX-License-Identifier: Apache-2.0
"""The rsl_rl adapter: interface contract, episode logging, refusal of NumPy envs and a two-iteration training."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from alloy.envs import VectorEnv, make  # noqa: E402
from alloy.envs.errors import EnvError  # noqa: E402
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task  # noqa: E402
from alloy.envs.wrappers.for_rsl_rl import RslRlVecEnv  # noqa: E402

WORLDS, MAX_STEPS = 4, 5


@pytest.fixture(scope="module")
def venv():
    env = make("GripperLift-v0", num_envs=WORLDS, device="cpu", backend="torch", max_steps=MAX_STEPS)
    wrapped = RslRlVecEnv(env)
    yield wrapped
    wrapped.close()


def _zeros(v):
    return torch.zeros((v.num_envs, v.num_actions))


@pytest.mark.compile_heavy
def test_interface_contract(venv):
    obs = venv.get_observations()
    assert list(obs.batch_size) == [WORLDS] and set(obs.keys()) == {"policy"}
    assert obs["policy"].dtype == torch.float32 and obs["policy"].shape[0] == WORLDS
    assert (venv.num_envs, venv.num_actions, venv.max_episode_length) == (WORLDS, 3, MAX_STEPS)
    assert str(venv.device) == "cpu" and venv.unwrapped is venv.env
    assert venv.cfg["num_envs"] == WORLDS and venv.default_obs_groups() == {"actor": ["policy"], "critic": ["policy"]}
    obs, rew, dones, extras = venv.step(_zeros(venv))
    assert rew.shape == dones.shape == (WORLDS,) and rew.dtype == torch.float32
    assert list(obs.batch_size) == [WORLDS] and extras["time_outs"].shape == (WORLDS,)


@pytest.mark.compile_heavy
def test_time_outs_follow_truncation_and_log_on_episode_end(venv):
    venv.reset()
    venv.take_stats()
    seen = []
    for _ in range(MAX_STEPS):
        _, _, dones, extras = venv.step(_zeros(venv))
        seen.append((dones.clone(), extras))
    dones, extras = seen[-1]
    assert bool(dones.all()) and bool(extras["time_outs"].all())
    assert not any(bool(e["time_outs"].any()) for _, e in seen[:-1])
    log = extras["log"]
    assert float(log["/episode_length"]) == MAX_STEPS and "/success_rate" in log and "/episode_return" in log
    assert any(k.startswith("Episode_Reward/") for k in log)
    assert "/success_rate" not in seen[0][1]["log"]
    episodes, _, mean_return = venv.take_stats()
    assert episodes == WORLDS and mean_return is not None


@pytest.mark.compile_heavy
def test_episode_length_buf_setter(venv):
    venv.reset()
    venv.episode_length_buf = torch.tensor([0, 1, 2, 3])
    assert venv.episode_length_buf.tolist() == [0, 1, 2, 3]
    assert venv.env.state["steps"].tolist() == [0, 1, 2, 3]


@pytest.mark.compile_heavy
def test_numpy_backend_is_refused():
    env = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy")
    try:
        with pytest.raises(EnvError) as err:
            RslRlVecEnv(env)
        assert err.value.field == "backend"
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_train_two_iterations(tmp_path, capsys):
    from alloy.envs.wrappers.train_rsl_rl import train

    summary = train("GripperLift-v0", num_envs=4, iterations=2, device="cpu", log_dir=str(tmp_path), report_every=1)
    assert summary["samples"] == 2 * 32 * 4 and (tmp_path / "model_2.pt").exists()
    assert summary["checkpoint"] == str(tmp_path / "model_2.pt")
    assert capsys.readouterr().out.count('"lib": "rsl_rl"') == 2


@pytest.mark.compile_heavy
def test_grouped_observations_reach_the_tensordict_and_train(tmp_path):
    from alloy.envs.wrappers.train_rsl_rl import train

    env = _grouped_env()
    try:
        obs = RslRlVecEnv(env).get_observations()
        assert set(obs.keys()) == {"policy", "critic"} and obs["critic"].shape[1] > obs["policy"].shape[1]
        assert RslRlVecEnv(env).default_obs_groups() == {"actor": ["policy"], "critic": ["critic"]}
    finally:
        env.close()
    summary = train("GripperLift-v0", iterations=2, log_dir=str(tmp_path), env=_grouped_env())
    assert summary["samples"] == 2 * 32 * WORLDS and (tmp_path / "model_2.pt").exists()


def _grouped_env():
    task = dataclasses.replace(gripper_lift_task(), obs={"policy": ["gripper", "action"],
                                                          "critic": ["gripper", "cube", "action"]})
    return VectorEnv(gripper_lift_scene(WORLDS), task, worlds=WORLDS, device="cpu", backend="torch")


def test_imports_without_the_libraries():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        for m in ("torch", "rsl_rl", "tensordict", "tensorboard"):
            sys.modules[m] = None
        import alloy.envs.wrappers.for_rsl_rl as w, alloy.envs.wrappers.train_rsl_rl
        from alloy.envs.wrappers import wrap
        from alloy.envs.errors import EnvError
        try:
            w.RslRlVecEnv(None)
        except EnvError as exc:
            assert "pip install" in (exc.fix or ""), exc
            print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr[-800:]
