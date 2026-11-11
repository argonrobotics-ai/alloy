# SPDX-License-Identifier: Apache-2.0
"""Observation groups: ``Task(obs={"policy": [...], "critic": [...]})`` returns one array per group; a list keeps the
one flat observation, unchanged."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.envs import EnvError, Task, VectorEnv
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task
from alloy.envs.vector import _obs_space

POLICY = ["gripper", "action"]
CRITIC = ["gripper", "cube", "action", ("cube_height", lambda s: s["cube.pos"][:, 2:3])]
W = 3


def _env(obs, **kwargs):
    task = dataclasses.replace(gripper_lift_task(max_steps=4), obs=obs)
    return VectorEnv(gripper_lift_scene(W), task, worlds=W, device="cpu", backend="numpy", **kwargs)


@pytest.fixture(scope="module")
def grouped():
    env = _env({"policy": POLICY, "critic": CRITIC})
    yield env
    env.close()


@pytest.fixture(scope="module")
def flat():
    env = _env(POLICY)
    yield env
    env.close()


@pytest.mark.compile_heavy
def test_groups_return_one_array_each(grouped):
    obs, _ = grouped.reset(seed=0)
    assert grouped.obs_groups == ("policy", "critic") and list(obs) == ["policy", "critic"]
    assert obs["policy"].shape == (W, 9) and obs["critic"].shape == (W, 23)
    assert obs["policy"].dtype == np.float32 and obs["critic"].dtype == np.float32
    space = grouped.single_observation_space
    assert space["policy"].shape == (9,) and space["critic"].shape == (23,)
    assert grouped.observation_space["critic"].shape == (W, 23)
    assert len(grouped.obs_names()) == 9 and grouped.obs_names("policy") == grouped.obs_names()
    names = grouped.obs_names("critic")
    assert len(names) == 23 and names[-1] == "cube_height" and "cube.pos[2]" in names
    cube_z = obs["critic"][:, names.index("cube.pos[2]")]
    np.testing.assert_array_equal(obs["critic"][:, -1], cube_z)             # the critic really sees the cube
    with pytest.raises(EnvError, match="no observation group named 'teacher'"):
        grouped.obs_names("teacher")


@pytest.mark.compile_heavy
def test_policy_group_equals_the_flat_observation_bit_for_bit(grouped, flat):
    """Same scene, seed and actions: the policy group is the flat env's observation, and rewards and ends match, so
    asking for groups changes what is returned, not what is simulated."""
    assert flat.obs_groups == () and not flat.observers
    og, _ = grouped.reset(seed=1)
    of, _ = flat.reset(seed=1)
    np.testing.assert_array_equal(og["policy"], of)
    rng = np.random.default_rng(0)
    for _ in range(6):                                                      # crosses the 4-step episode end
        a = rng.uniform(-1, 1, size=(W, 3)).astype(np.float32)
        og, rg, tg, ug, ig = grouped.step(a)
        of, rf, tf, uf, i_f = flat.step(a)
        np.testing.assert_array_equal(og["policy"], of)
        np.testing.assert_array_equal(rg, rf)
        np.testing.assert_array_equal(ug, uf)
        assert ("final_obs" in ig) == ("final_obs" in i_f)
        if "final_obs" in ig:
            assert set(ig["final_obs"]) == {"policy", "critic"}
            np.testing.assert_array_equal(ig["final_obs"]["policy"], i_f["final_obs"])
    assert isinstance(of, np.ndarray) and flat.single_observation_space.shape == (9,)


@pytest.mark.parametrize(("obs", "match"), [
    ({"critic": ["cube"]}, "no 'policy' group"),
    ({"policy": ["cube"], "state": ["cube"]}, "'state'"),
    ({"policy": "cube"}, "is the string 'cube'"),
    ("cube", "obs is the string"),
])
def test_bad_groups_are_refused_with_a_fix(obs, match):
    with pytest.raises(EnvError, match=match) as err:
        Task(obs=obs).validate()
    assert err.value.fix and err.value.field == "obs"


def test_group_named_like_a_camera_is_refused():
    with pytest.raises(EnvError, match="camera's name"):
        Task(obs={"policy": ["cube"], "front": ["cube"]}, cameras={"front": object()}).validate()


def test_space_of_groups_beside_camera_images():
    obs = {"policy": np.zeros((W, 5), np.float32), "critic": np.zeros((W, 8), np.float32),
           "front": np.zeros((W, 16, 16, 3), np.uint8)}
    space = _obs_space(obs, {"front"})
    assert space["policy"].shape == (5,) and space["critic"].shape == (8,)
    assert space["front"].shape == (16, 16, 3) and space["front"].dtype == np.uint8
    assert _obs_space(obs["policy"], set()).shape == (5,)
