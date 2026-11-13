# SPDX-License-Identifier: Apache-2.0
"""Gemini audit round 2, the RL wrappers (CPU):

- T1-013: rsl_rl and skrl took ``autoreset="disabled"`` / ``"next_step"`` silently and corrupted rollouts; every
  wrapper now raises ``EnvError(field="autoreset")`` as the Stable-Baselines3 one did.
- T1-018: ``RslRlVecEnv.seed(s)`` reseeded with an all-False reset mask, so the episode the runner reads next ran
  under the build seed; it now restarts every world from ``s``.
- T1-021: ``SkrlEnv.seed`` raised AttributeError, and after a space read ``reset()`` did nothing (counters and stale
  observations kept); ``seed`` restarts from the seed and ``reset`` restarts every world.
- T2-025: a camera-only task gave Stable-Baselines3 an empty "state" entry (a raw ValueError in its rollout buffer),
  and cameras under 36 x 36 crashed its default image encoder with a raw RuntimeError."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy.envs
from alloy.envs.errors import EnvError
from alloy.envs.wrappers import _common


def _make(seed: int, **kw):
    return alloy.envs.make("GripperLift-v0", num_envs=3, device="cpu", seed=seed, backend="torch", **kw)


@pytest.mark.parametrize("mode", ["disabled", "next_step"])
def test_every_wrapper_refuses_an_env_without_same_step_autoreset(mode):
    with pytest.raises(EnvError) as err:
        _common.require_same_step(SimpleNamespace(autoreset=mode), "rsl_rl")
    assert err.value.field == "autoreset" and "same_step" in err.value.fix
    _common.require_same_step(SimpleNamespace(autoreset="same_step"), "rsl_rl")


@pytest.mark.compile_heavy
@pytest.mark.parametrize("lib", ["rsl_rl", "skrl"])
def test_rsl_rl_and_skrl_wrappers_raise_on_next_step_autoreset(lib):
    pytest.importorskip(lib)
    from alloy.envs.wrappers import wrap

    env = _make(0, autoreset="next_step")
    try:
        with pytest.raises(EnvError) as err:                         # was: accepted, rollouts corrupted
            wrap(env, lib)
        assert err.value.field == "autoreset"
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_rsl_rl_seed_restarts_every_world_from_that_seed():
    pytest.importorskip("rsl_rl")
    from alloy.envs.wrappers.for_rsl_rl import RslRlVecEnv

    a, b = RslRlVecEnv(_make(0)), RslRlVecEnv(_make(0))
    try:
        a.step(torch_zeros(a))
        starts = {}
        for s in (10, 20, 10):
            a.seed(s)
            starts.setdefault(s, []).append(a.get_observations()["policy"].clone())
        assert bool((starts[10][0] == starts[10][1]).all())          # a seed repeats its start
        assert not bool((starts[10][0] == starts[20][0]).all())       # was: every seed read the same episode
        raw, _ = b.unwrapped.reset(seed=20)                         # what the env itself starts from seed 20
        assert bool((starts[20][0] == raw).all())
        assert a.unwrapped.stats.steps.tolist() == [0, 0, 0]         # the episode the runner reads starts at 0
    finally:
        a.close()
        b.close()


def torch_zeros(venv):
    import torch

    return torch.zeros((venv.num_envs, venv.num_actions))


@pytest.mark.compile_heavy
def test_skrl_seed_works_and_reset_restarts_after_a_space_read():
    pytest.importorskip("skrl")
    from alloy.envs.wrappers.for_skrl import SkrlEnv

    env = SkrlEnv(_make(0))
    try:
        assert env.observation_space.shape                           # a space read before any reset
        env.seed(42)                                                 # was: AttributeError
        first = env.reset()[0].clone()
        env.seed(7)
        assert not bool((env.reset()[0] == first).all())             # a new seed, a new start
        env.seed(42)
        assert bool((env.reset()[0] == first).all())
        for _ in range(3):
            env.step(env.zero_actions())
        env.reset()                                                  # was: a no-op, counters and observation kept
        assert env.unwrapped.stats.steps.tolist() == [0, 0, 0]
    finally:
        env.close()


def test_sb3_training_refuses_cameras_too_small_for_its_image_encoder():
    from alloy.envs.wrappers.train_sb3 import MIN_IMAGE_SIDE, check_image_sizes

    space = SimpleNamespace(spaces={"front": SimpleNamespace(shape=(32, 32, 3)),
                                    "top": SimpleNamespace(shape=(64, 64, 3))})
    with pytest.raises(EnvError) as err:                             # was: a raw RuntimeError inside NatureCNN
        check_image_sizes(SimpleNamespace(single_observation_space=space), ["front", "top"])
    assert "front" in err.value.message and "top" not in err.value.message and err.value.field == "cameras"
    big = SimpleNamespace(spaces={"front": SimpleNamespace(shape=(MIN_IMAGE_SIDE, MIN_IMAGE_SIDE, 3))})
    check_image_sizes(SimpleNamespace(single_observation_space=big), ["front"])


ARM = """<mujoco><worldbody><body name="b" pos="0 0 0.05">
  <joint name="j" type="slide" axis="1 0 0" range="-0.2 0.2"/>
  <geom type="box" size="0.03 0.03 0.03" mass="0.2"/>
</body></worldbody><actuator><position joint="j" kp="200" kv="20"/></actuator></mujoco>"""


@pytest.mark.compile_heavy
def test_a_camera_only_task_gives_sb3_only_the_images():
    pytest.importorskip("stable_baselines3")
    import alloy
    from alloy.camera import Camera
    from alloy.envs import Task, VectorEnv
    from alloy.envs import checks as C
    from alloy.envs.wrappers.for_sb3 import Sb3VecEnv

    scene = alloy.Scene(worlds=2, dt=0.02)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM)
    cam = Camera("front", look_at=((0.3, -0.3, 0.25), (0.0, 0.0, 0.05)), width=36, height=36, spp=1)
    task = Task(obs=[], cameras={"front": cam}, reward={"r": (1.0, C.joint("arm/j"))}, max_steps=4)
    venv = Sb3VecEnv(VectorEnv(scene, task, worlds=2, device="cpu", backend="numpy"))
    try:
        assert list(venv.observation_space.spaces) == ["front"]      # was: an extra empty "state" Box
        obs = venv.reset()
        assert list(obs) == ["front"] and np.asarray(obs["front"]).shape == (2, 36, 36, 3)
    finally:
        venv.close()
