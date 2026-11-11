# SPDX-License-Identifier: Apache-2.0
"""The Stable-Baselines3 adapter: the VecEnv contract, both backends, refusals, grouped observations and a
two-iteration training run. Skips where PyTorch or Stable-Baselines3 is missing."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from alloy.envs import make
from alloy.envs.errors import EnvError

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from alloy.envs.wrappers.for_sb3 import Sb3VecEnv  # noqa: E402
from alloy.envs.wrappers.train_sb3 import train  # noqa: E402


@pytest.fixture(scope="module")
def venv():
    v = Sb3VecEnv(make("GripperLift-v0", num_envs=4, device="cpu", max_steps=5))
    yield v
    v.close()


@pytest.mark.compile_heavy
def test_vecenv_contract_and_episode_info(venv):
    obs = venv.reset()
    assert isinstance(obs, np.ndarray) and obs.shape == (4, 22) and obs.dtype == np.float32
    assert venv.observation_space.shape == (22,) and venv.action_space.shape == (3,)
    snapshot = obs.copy()
    actions = np.zeros((4, 3), np.float32)
    for k in range(5):
        venv.step_async(actions)
        new, rew, done, infos = venv.step_wait()
    assert np.array_equal(obs, snapshot)                      # the returned obs is a copy, not a live view
    assert isinstance(rew, np.ndarray) and rew.shape == (4,) and done.dtype == bool and done.all()
    assert len(infos) == 4
    for info in infos:
        assert info["TimeLimit.truncated"] is True and info["episode"]["l"] == 5
        assert info["terminal_observation"].shape == (22,) and "is_success" in info
    assert venv.tally.episodes == 4


@pytest.mark.compile_heavy
def test_terminal_observation_is_the_envs_final_obs(venv):
    venv.reset()
    raw = venv.env
    for _ in range(5):
        venv.step_async(np.zeros((4, 3), np.float32))
        venv.step_wait()
    raw.reset()
    got = None
    for _ in range(5):
        obs, _, _, _, info = raw.step(raw.xp.zeros((4, 3)))
        if "final_obs" in info:
            got = raw.xp.to_numpy(info["final_obs"])
    assert got is not None and got.shape == (4, 22)
    assert venv.get_attr("num_envs") == [4] * 4 and venv.env_is_wrapped(object) == [False] * 4


@pytest.mark.compile_heavy
def test_numpy_backend_and_refusals():
    env = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy", max_steps=3)
    v = Sb3VecEnv(env)
    try:
        obs = v.reset()
        assert type(obs) is np.ndarray and not np.shares_memory(obs, env.xp.view(env.sim.engine.state.body_q))
        v.step_async(np.zeros((2, 3), np.float32))
        assert v.step_wait()[1].shape == (2,)
    finally:
        v.close()
    late = make("GripperLift-v0", num_envs=2, device="cpu", autoreset="next_step")
    try:
        with pytest.raises(EnvError) as err:
            Sb3VecEnv(late)
        assert err.value.field == "autoreset"
    finally:
        late.close()


@pytest.mark.compile_heavy
def test_grouped_env_trains_on_its_policy_group(tmp_path):
    from alloy.envs import VectorEnv
    from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

    groups = ["gripper", "cube", "action"]
    task = dataclasses.replace(gripper_lift_task(), obs={"policy": groups, "critic": groups}, max_steps=5)
    env = VectorEnv(gripper_lift_scene(4), task, worlds=4, device="cpu")
    out = train("grouped", iterations=2, env=env, log_dir=str(tmp_path / "g"))
    assert out["samples"] >= 2 * 32 * 4 and (tmp_path / "g" / "model.zip").exists()


@pytest.mark.compile_heavy
def test_train_two_iteration_smoke(tmp_path, capfd):
    out = train("GripperLift-v0", num_envs=4, iterations=2, device="cpu", log_dir=str(tmp_path / "run"),
                env_kwargs={"max_steps": 5})
    assert set(out) == {"success_rate", "minutes", "samples", "checkpoint"}
    assert out["samples"] == 2 * 32 * 4 and (tmp_path / "run" / "model.zip").exists()
    assert '"lib": "sb3"' in capfd.readouterr().out


def test_imports_without_the_libraries():
    code = ("import sys\nfor m in ('torch', 'stable_baselines3', 'gymnasium'): sys.modules[m] = None\n"
            "import alloy.envs.wrappers.for_sb3 as f, alloy.envs.wrappers.train_sb3\n"
            "from alloy.envs.errors import EnvError\n"
            "try:\n    f.Sb3VecEnv(None)\nexcept EnvError as e:\n    print(e.fix)\n    sys.exit(0)\nsys.exit(1)\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "pip install" in done.stdout
