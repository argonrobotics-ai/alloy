# SPDX-License-Identifier: Apache-2.0
"""The recorder, replay and evaluation on a real environment (GripperLift on the CPU, NumPy backend).

Laws: (1) stepping through a ``Recorder`` gives bit for bit what stepping the env gives, and every recorded array is
what the env returned for that world and step; (2) a replay from the recorded start reproduces the recorded states
exactly on the same device, and a changed action shows up as a state error; (3) ``evaluate`` counts the same number
of episodes in every world."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import EnvError, make
from alloy.il import MemoryWriter, Recorder, evaluate, replay, wilson_interval

W, STEPS = 3, 60
pytestmark = pytest.mark.compile_heavy


def _make(**kwargs):
    return make("GripperLift-v0", num_envs=W, device="cpu", backend="numpy", max_steps=STEPS, **kwargs)


@pytest.fixture(scope="module")
def env():
    e = _make()
    yield e
    e.close()


def scripted(env):
    """Squeeze for 25 steps, then lift: succeeds within 60 steps."""
    def policy(obs):
        act = np.tile(np.array([0.0, 1.0, 1.0], np.float32), (env.num_envs, 1))
        act[np.asarray(env.state["steps"]) >= 25, 0] = 1.0
        return act
    return policy


def _record(env, policy, steps, **kwargs):
    writer = MemoryWriter()
    rec = Recorder(env, [writer], **kwargs)
    obs, _ = rec.reset(seed=0)
    outs = [(obs.copy(), None)]
    for _ in range(steps):
        act = policy(obs)
        obs, reward, term, trunc, info = rec.step(act)
        outs.append((obs.copy(), (act.copy(), reward.copy(), term.copy(), trunc.copy(), info)))
    return rec, writer, outs


def test_recorder_changes_nothing_and_records_what_the_env_returned(env):
    policy = scripted(env)
    obs, _ = env.reset(seed=0)
    plain = [obs.copy()]
    for _ in range(2 * STEPS):
        obs, *_ = env.step(policy(obs))
        plain.append(obs.copy())
    rec, writer, outs = _record(env, policy, 2 * STEPS, keep="all")
    for a, (b, _) in zip(plain, outs):
        assert np.array_equal(a, b)                              # law 1: bit for bit with and without the recorder
    assert rec.saved == 2 * W and rec.dropped == 0
    assert [e.world for e in writer.episodes] == [0, 1, 2, 0, 1, 2]
    assert [e.index for e in writer.episodes] == list(range(6))
    for ep in writer.episodes:
        first = 0 if ep.index < W else STEPS
        w = ep.world
        assert ep.length == STEPS and ep.success and ep.truncated[-1] and not ep.dones[:-1].any()
        for t in range(STEPS):
            before, (act, reward, term, trunc, info) = outs[first + t][0], outs[first + t + 1][1]
            assert np.array_equal(ep.obs["state"][t], before[w])
            assert np.array_equal(ep.actions[t], act[w]) and ep.rewards[t] == reward[w]
            after = info["final_obs"][w] if t == STEPS - 1 else outs[first + t + 1][0][w]
            assert np.array_equal(ep.next_obs["state"][t], after)
        assert ep.states.shape == (STEPS, len(rec.spec.state_names)) and ep.states.dtype == np.float32
        assert set(ep.start) >= {"joint_q", "joint_qd", "target.position"}
        assert np.array_equal(ep.states[0], np.concatenate([ep.start["joint_q"], ep.start["joint_qd"]]))
    spec = rec.spec
    assert spec.env == "GripperLift-v0" and spec.fps == pytest.approx(100.0) and spec.action_mode == "delta"
    assert spec.obs_shapes == {"state": (22,)} and spec.obs_names == env.obs_names()
    assert spec.action_names == env.action_names()
    rec.close()
    assert writer.closed


def test_keep_filter_and_max_episodes(env):
    idle = lambda obs: np.zeros((W, 3), np.float32)                                    # noqa: E731
    rec, writer, _ = _record(env, idle, STEPS, keep="success")
    assert rec.saved == 0 and rec.dropped == W and not writer.episodes                 # failures are dropped
    rec, writer, _ = _record(env, idle, STEPS, keep=lambda ep: ep.world == 1)
    assert [e.world for e in writer.episodes] == [1] and not writer.episodes[0].success
    rec, writer, _ = _record(env, scripted(env), 2 * STEPS, max_episodes=4)
    assert rec.saved == 4 and rec.full and rec.dropped == 2
    with pytest.raises(EnvError) as err:
        Recorder(env, [], keep="best")
    assert err.value.fix
    with pytest.raises(EnvError):
        Recorder(env, []).step(np.zeros((W, 3), np.float32))                           # step before reset


def test_finish_ends_the_running_episode(env):
    rec, writer, _ = _record(env, scripted(env), 10, keep="success")
    rec.finish(save=True)                                    # the operator keeps it whatever the filter says
    assert rec.saved == W and all(e.length == 10 and e.truncated[-1] and not e.success for e in writer.episodes)
    for _ in range(5):
        rec.step(np.zeros((W, 3), np.float32))
    rec.finish(save=False)
    assert rec.saved == W and rec.dropped == W
    rec.finish(save=True)                                    # nothing running: nothing written
    assert rec.saved == W


def test_replay_reproduces_the_recording_and_sees_a_changed_action(env):
    rec, writer, _ = _record(env, scripted(env), 2 * STEPS, keep="all")
    episodes = writer.episodes[:5]                           # 5 episodes in 3 worlds: two batches, one part-filled
    result = replay(env, episodes)
    assert result.matches.all() and result.success.all() and not result.ended_early.any()
    assert result.state_error.max() == 0.0 and result.obs_error.max() == 0.0          # law 2: exact on this device
    changed = writer.episodes[0]
    changed.actions = changed.actions.copy()
    changed.actions[30:, 0] = -1.0                           # lower instead of lift
    bad = replay(env, [changed])
    assert bad.state_error[0] > 1e-3 and not bad.success[0] and not bad.matches[0]
    changed.start = {}
    with pytest.raises(EnvError) as err:
        replay(env, [changed])
    assert "states=True" in err.value.fix


def test_replay_restores_mujoco_actuator_controls_in_the_env_that_recorded():
    """PandaReach drives MuJoCo-native servos (control.mujoco.ctrl) and starts every episode at jittered joints:
    the start must carry those controls, or a replay after the env's next reset servoes to the wrong pose."""
    panda = make("PandaReach-v0", num_envs=W, device="cpu", backend="numpy", max_steps=20)
    try:
        rng = np.random.default_rng(0)
        rec, writer, _ = _record(panda, lambda obs: rng.uniform(-1, 1, (W, panda.writer.size)).astype(np.float32),
                                 40, keep="all")
        assert len(writer.episodes) == 2 * W and "target.ctrl" in writer.episodes[0].start
        result = replay(panda, writer.episodes)              # the env has reset to other joints and targets since
        assert result.state_error.max() == 0.0 and result.obs_error.max() == 0.0 and result.matches.all()
        old = writer.episodes[:W]                            # a file written before the controls were recorded
        for ep in old:
            del ep.start["target.ctrl"]
        assert replay(panda, old).state_error.shape == (W,)  # still replays (controls as the env's reset left them)
    finally:
        panda.close()


def test_evaluate_counts_every_world_equally(env):
    good = evaluate(env, scripted(env), episodes=5, seed=1)                 # rounded up to 2 per world
    assert good.episodes == 6 and sorted(good.worlds.tolist()) == [0, 0, 1, 1, 2, 2]
    assert good.success_rate == 1.0 and (good.lengths == STEPS).all() and good.mean_return > 0
    assert good.interval == pytest.approx(wilson_interval(6, 6))
    bad = evaluate(env, lambda obs: np.zeros((W, 3), np.float32), episodes=3, seed=1)
    assert bad.success_rate == 0.0 and bad.summary()["successes"] == 0
    assert wilson_interval(45, 50) == pytest.approx((0.7864, 0.9565), abs=5e-4)        # textbook value
    assert wilson_interval(0, 0) == (0.0, 1.0)
    rec = Recorder(env, [MemoryWriter()], keep="all")
    assert evaluate(rec, scripted(env), episodes=3).episodes == 3 and rec.saved == 3    # rollouts recorded too
    with pytest.raises(EnvError):
        evaluate(env, scripted(env), episodes=0)


@pytest.mark.parametrize("mode", ["next_step", "disabled"])
def test_other_autoreset_modes(mode):
    env = _make(autoreset=mode)
    try:
        writer = MemoryWriter()
        rec = Recorder(env, [writer], keep="all")
        obs, _ = rec.reset(seed=0)
        policy, seen = scripted(env), []
        for k in range(2 * STEPS + 5):
            obs, _, term, trunc, _ = rec.step(policy(obs))
            seen.append(obs.copy())
            if mode == "disabled" and (term | trunc).any() and k % 7 == 0:             # reset late, by mask
                obs, _ = rec.reset(options={"reset_mask": term | trunc})
                seen[-1] = obs.copy()
        assert rec.saved >= W and all(e.length == STEPS and e.success for e in writer.episodes)
        if mode == "next_step":                              # the void step after an end is in no episode
            second = writer.episodes[W]
            assert np.array_equal(second.obs["state"][0], seen[STEPS][second.world])
            assert rec.saved == 2 * W
        with pytest.raises(EnvError):
            replay(env, writer.episodes[:1])                 # replay and evaluate need same_step
    finally:
        env.close()


def test_hdf5_round_trip_replays(env, tmp_path):
    pytest.importorskip("h5py")
    from alloy.il import HDF5Writer, load_hdf5

    path = tmp_path / "lift.hdf5"
    rec = Recorder(env, [HDF5Writer(path)], task="lift the cube", keep="all")
    obs, _ = rec.reset(seed=0)
    policy = scripted(env)
    while rec.saved < W:
        obs, *_ = rec.step(policy(obs))
    rec.close()
    spec, episodes = load_hdf5(path)
    assert spec.task == "lift the cube" and spec.env == "GripperLift-v0" and len(episodes) == W
    result = replay(env, episodes)
    assert result.matches.all() and result.state_error.max() == 0.0
