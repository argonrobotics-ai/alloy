# SPDX-License-Identifier: Apache-2.0
"""Sim-to-real layers (``alloy.envs.realism``): off means unchanged, and each layer does what it says per world."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs import EnvError, VectorEnv
from alloy.envs.backend import NumpyBackend
from alloy.envs.realism import ActionDelay, ObsDelay, ObsHistory, ObsNoise, obs_names
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task

W = 4
GROUPS = {"policy": ["gripper", "action"], "critic": ["gripper", "cube", "action"]}
D = 9                                                        # the policy group's plain width


def _env(layers=None, backend="numpy", worlds=W):
    task = dataclasses.replace(gripper_lift_task(max_steps=4), obs=GROUPS)
    if layers is not None:
        task = dataclasses.replace(task, layers=layers)
    return VectorEnv(gripper_lift_scene(worlds), task, worlds=worlds, device="cpu", backend=backend)


def _fixture(layers):
    def make():
        env = _env(layers)
        yield env
        env.close()
    return pytest.fixture(scope="module")(make)


base = _fixture(None)
empty = _fixture([])
identity = _fixture([ObsNoise(std=0.0), ObsDelay(0), ActionDelay(0), ObsHistory(1)])
noisy = _fixture([ObsNoise({"gripper/lift.q": 0.02, "action": 0.05})])
delayed = _fixture([ObsDelay((0, 2))])
history = _fixture([ObsHistory(3)])
slowed = _fixture([ActionDelay((0, 2))])
combo = _fixture([ObsNoise(0.01), ObsDelay(1), ObsHistory(3)])


def _actions(n, worlds=W, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, worlds, 3)).astype(np.float32)


def _same(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    else:
        np.testing.assert_array_equal(a, b)


# --- off = unchanged -----------------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
@pytest.mark.parametrize("other", ["empty", "identity"])
def test_no_layers_or_identity_layers_change_nothing(base, empty, identity, other):
    """8 steps cross two autoresets (4-step episodes): obs, reward and ends match bit for bit."""
    env = {"empty": empty, "identity": identity}[other]
    assert base.layers == []
    shapes = [{k: sp.shape for k, sp in e.single_observation_space.spaces.items()} for e in (base, env)]
    assert shapes[0] == shapes[1]             # spaces without Gymnasium have no ==, so compare what they hold
    _same(base.reset(seed=2)[0], env.reset(seed=2)[0])
    for a in _actions(8):
        ob, rb, tb, ub, ib = base.step(a)
        oe, re, te, ue, ie = env.step(a)
        _same(ob, oe)
        _same((rb, tb, ub), (re, te, ue))
        _same(ib["targets"], ie["targets"])
        assert ("final_obs" in ib) == ("final_obs" in ie)
        if "final_obs" in ib:
            _same(ib["final_obs"], ie["final_obs"])


# --- noise ---------------------------------------------------------------------------------------------------------

def _record(env, acts, seed):
    out = [env.reset(seed=seed)[0]]
    for a in acts:
        obs, reward, *_ = env.step(a)
        out.append(obs)
    return out


@pytest.mark.compile_heavy
def test_noise_has_the_asked_size_only_on_named_columns_and_never_moves_the_simulation(base, noisy):
    acts = _actions(100)
    clean, rough = _record(base, acts, 3), _record(noisy, acts, 3)
    names = base.obs_names()
    sizes = {"gripper/lift.q": 0.02, "action[0]": 0.05, "action[1]": 0.05, "action[2]": 0.05}
    diff = np.stack([r["policy"] - c["policy"] for r, c in zip(rough, clean)])        # (steps, W, D)
    for col, name in enumerate(names):
        if name in sizes:
            ratio = diff[..., col].std() / sizes[name]
            assert 0.9 < ratio < 1.1, (name, ratio)
            assert abs(diff[..., col].mean()) < 0.2 * sizes[name]
        else:
            assert not diff[..., col].any(), name
    for r, c in zip(rough, clean):
        np.testing.assert_array_equal(r["critic"], c["critic"])
    np.testing.assert_array_equal(noisy.state["cube.pos"], base.state["cube.pos"])    # the same simulated world
    np.testing.assert_array_equal(noisy.state["steps"], base.state["steps"])


@pytest.mark.compile_heavy
def test_noise_repeats_with_the_same_seed_and_differs_with_another(noisy):
    acts = _actions(6)
    first, again, other = _record(noisy, acts, 7), _record(noisy, acts, 7), _record(noisy, acts, 8)
    for a, b in zip(first, again):
        _same(a, b)
    assert not np.array_equal(first[1]["policy"], other[1]["policy"])


# --- delay and history against a model of what each world should see -----------------------------------------------

def _lockstep(clean_env, env, expect, k_of, steps=11, masked_reset_at=5):
    """Step both envs with the same actions; ``expect(history, k)`` is the policy observation a world should return
    from its clean observations since its last reset. Crosses autoresets and one reset of world 1 alone."""
    acts = _actions(steps, seed=4)
    clean, got = clean_env.reset(seed=5)[0], env.reset(seed=5)[0]
    hist = [[clean["policy"][w]] for w in range(W)]
    for w in range(W):
        np.testing.assert_array_equal(got["policy"][w], expect(hist[w], k_of()[w]))
    for t, a in enumerate(acts):
        if t == masked_reset_at:
            before, mask = got["policy"].copy(), np.array([False, True, False, False])
            clean, got = clean_env.reset(options={"reset_mask": mask})[0], env.reset(options={"reset_mask": mask})[0]
            hist[1] = [clean["policy"][1]]
            np.testing.assert_array_equal(got["policy"][[0, 2, 3]], before[[0, 2, 3]])     # others: no new push
            np.testing.assert_array_equal(got["policy"][1], expect(hist[1], k_of()[1]))
        k = k_of().copy()
        clean, _, _, _, ci = clean_env.step(a)
        got, _, _, _, gi = env.step(a)
        np.testing.assert_array_equal(got["critic"], clean["critic"])
        done = ci["_final_obs"] if "final_obs" in ci else np.zeros(W, bool)
        for w in range(W):
            if not done[w]:
                hist[w].append(clean["policy"][w])
                np.testing.assert_array_equal(got["policy"][w], expect(hist[w], k[w]))
                continue
            hist[w].append(ci["final_obs"]["policy"][w])
            np.testing.assert_array_equal(gi["final_obs"]["policy"][w], expect(hist[w], k[w]))
            hist[w] = [clean["policy"][w]]
            np.testing.assert_array_equal(got["policy"][w], expect(hist[w], k_of()[w]))


@pytest.mark.compile_heavy
def test_obs_delay_returns_each_worlds_own_k_steps_old_observation(base, delayed):
    ks = delayed.layers[0].k
    assert len(set(ks.tolist())) > 1 and ks.max() > 0 and ks.min() >= 0       # worlds differ
    _lockstep(base, delayed, lambda h, k: h[max(len(h) - 1 - int(k), 0)], lambda: delayed.layers[0].k)
    assert delayed.layers[0].k.shape == (W,)


@pytest.mark.compile_heavy
def test_history_is_oldest_to_newest_filled_with_the_first_observation_after_a_reset(base, history):
    def expect(h, _k):
        return np.concatenate([h[max(len(h) - 3 + j, 0)] for j in range(3)])

    obs, _ = history.reset(seed=1)
    assert obs["policy"].shape == (W, 3 * D) and obs["critic"].shape == (W, 22)
    assert history.single_observation_space["policy"].shape == (3 * D,)
    np.testing.assert_array_equal(obs["policy"][:, :D], obs["policy"][:, 2 * D:])
    _lockstep(base, history, expect, lambda: np.zeros(W, int))
    names = obs_names(history)
    assert len(names) == 3 * D and names[0].endswith("@t-2") and names[-1] == "action[2]@t"
    assert obs_names(history, "critic") == base.obs_names("critic")
    assert history.obs_names() == base.obs_names()                           # the env's own labels stay per step


# --- action delay --------------------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
def test_action_delay_hands_the_writer_the_action_of_k_steps_ago(base, slowed):
    """A reference env fed the delayed actions (zeros first, cleared at a reset) gets the same targets."""
    layer = slowed.layers[0]
    acts = _actions(10, seed=6)
    slowed.reset(seed=9)
    base.reset(seed=9)
    assert len(set(layer.k.tolist())) > 1
    hist = [[] for _ in range(W)]
    for a in acts:
        k = layer.k.copy()
        delayed_a = np.zeros_like(a)
        for w in range(W):
            hist[w].append(a[w])
            delayed_a[w] = hist[w][-1 - int(k[w])] if len(hist[w]) > k[w] else 0.0
        _, _, _, _, info = slowed.step(a)
        _, _, _, _, ref = base.step(delayed_a)
        np.testing.assert_array_equal(info["targets"], ref["targets"])
        running = ~info["_episode"] if "_episode" in info else np.ones(W, bool)
        # the env's own "action" (state, observation, action-size rewards) is what the policy sent this step
        np.testing.assert_array_equal(np.asarray(slowed.state["action"])[running], np.clip(a, -1, 1)[running])
        if "_episode" in info:
            for w in np.nonzero(info["_episode"])[0]:
                hist[w] = []


def _stub(mode="delta", worlds=3):
    xp = NumpyBackend("cpu")
    xp.seed(0)
    spec = SimpleNamespace(mode=mode)
    return SimpleNamespace(xp=xp, num_envs=worlds, writer=SimpleNamespace(size=2, spec=spec), obs_groups=())


@pytest.mark.parametrize(("mode", "first"), [("delta", [0, 0, 1, 2]), ("absolute", [1, 1, 1, 2])])
def test_action_delay_before_k_actions_exist(mode, first):
    layer, env = ActionDelay(2), _stub(mode)
    layer.attach(env)
    out = [layer.on_action(np.full((3, 2), v, np.float32))[0, 0] for v in (1, 2, 3, 4)]
    assert out == first
    layer.on_reset(np.array([True, False, False]))                           # world 0 cleared, the others go on
    got = layer.on_action(np.full((3, 2), 9, np.float32))
    assert got[0, 0] == (0 if mode == "delta" else 9) and got[1, 0] == 3


# --- validation ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [lambda: ObsNoise(std=-1), lambda: ObsDelay((2, 1)), lambda: ObsDelay(1.5),
                                  lambda: ObsHistory(0), lambda: ActionDelay(-1), lambda: ObsDelay(1, groups=())])
def test_bad_arguments_raise_env_errors(make):
    with pytest.raises(EnvError) as err:
        make()
    assert err.value.field == "layers" and err.value.fix


def test_unknown_group_and_unknown_column_pattern_raise():
    env = _stub()
    with pytest.raises(EnvError, match="no observation group named 'teacher'"):
        ObsDelay(1, groups=("teacher",)).attach(env)
    env.obs_names, env.layers = (lambda group: ["a.q", "b[0]", "b[1]"]), []
    noise = ObsNoise({"c": 0.1})
    env.layers.append(noise)
    noise.attach(env)
    with pytest.raises(EnvError, match="matches no observation column"):
        noise.on_obs(np.zeros((3, 3), np.float32), None)


# --- composition and torch -----------------------------------------------------------------------------------------

@pytest.mark.compile_heavy
def test_noise_delay_history_in_order_leave_the_critic_alone(base, combo):
    got, clean = combo.reset(seed=1)[0], base.reset(seed=1)[0]
    assert got["policy"].shape == (W, 3 * D) and got["critic"].shape == (W, 22)
    prev = got["policy"]
    for a in _actions(3, seed=2):
        got, _, _, _, _ = combo.step(a)
        before = clean
        clean = base.step(a)[0]
        np.testing.assert_array_equal(got["critic"], clean["critic"])
        np.testing.assert_array_equal(got["policy"][:, :2 * D], prev[:, D:])             # one noise draw, then shifted
        noise = got["policy"][:, 2 * D:] - before["policy"]                          
        assert 0 < np.abs(noise).max() < 0.1
        prev = got["policy"]


@pytest.mark.compile_heavy
def test_layers_run_on_torch_and_repeat_with_the_seed():
    torch = pytest.importorskip("torch")
    env = _env([ObsNoise(0.01), ObsDelay(1), ObsHistory(2), ActionDelay((0, 2))], backend="torch")
    try:
        runs = []
        for _ in range(2):
            obs, _ = env.reset(seed=1)
            seen = [obs["policy"].clone()]
            for a in _actions(6):
                obs, *_ = env.step(torch.as_tensor(a))
                seen.append(obs["policy"].clone())
            runs.append(torch.stack(seen))
        assert runs[0].shape == (7, W, 2 * D) and torch.isfinite(runs[0]).all()
        assert obs["critic"].shape == (W, 22)
        assert torch.equal(runs[0], runs[1])
    finally:
        env.close()
