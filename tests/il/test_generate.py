# SPDX-License-Identifier: Apache-2.0
"""Laws of scripted demo generation (CPU, GripperLift, 3 worlds):

1. A ``PhaseScript`` gives each world the action of its own phase: equal to the hand-written script while the worlds
   run in step, and a world reset mid-episode starts again at the first phase while the others go on.
2. ``generate`` saves exactly the episodes asked for, counts every ended episode, and what it saved replays exactly.
3. A script that cannot succeed stops at ``max_attempts`` with ``complete`` False.
4. An episode whose world the engine froze (INVALID) shows in ``EvalResult`` and ``GenerateResult``, read per step
   from ``info["invalid"]``; it is never a success and ``keep="success"`` does not save it.
   That holds when the world was INVALID on an early step only and the last step passes the success check.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs import EnvError, make
from alloy.il import MemoryWriter, Phase, PhaseScript, demonstrator, evaluate, generate, replay

W, STEPS = 3, 60
pytestmark = pytest.mark.compile_heavy


@pytest.fixture(scope="module")
def env():
    e = make("GripperLift-v0", num_envs=W, device="cpu", backend="numpy", max_steps=STEPS)
    yield e
    e.close()


def by_hand(env):
    act = np.tile(np.array([0.0, 1.0, 1.0], np.float32), (W, 1))
    act[np.asarray(env.state["steps"]) >= 25, 0] = 1.0
    return act


def test_script_is_the_hand_written_policy_and_follows_each_world(env):
    script = demonstrator(env)
    env.reset(seed=0)
    for t in range(STEPS + 10):                              # past the episode's end: every world restarts
        act = script(None)
        assert np.array_equal(act, by_hand(env)), t
        env.step(act)
    assert script.names() == ["squeeze"] * W                 # 10 steps into the second episode
    for _ in range(30):
        env.step(script(None))
    assert script.names() == ["lift"] * W
    env.reset(options={"reset_mask": np.array([False, True, False])})
    act = script(None)                                       # world 1 squeezes again, the others still lift
    assert script.names() == ["lift", "squeeze", "lift"]
    assert act[:, 0].tolist() == [1.0, 0.0, 1.0]


def test_until_ends_a_phase_per_world_and_noise_is_seeded_and_clipped(env):
    high = lambda s: np.asarray(s["cube.pos"])[:, 2] > 0.06                     # noqa: E731
    phases = [Phase([0.0, 1.0, 1.0], steps=25, name="squeeze"), Phase([1.0, 1.0, 1.0], until=high, name="lift"),
              Phase(lambda s: np.tile(np.array([0.0, 1.0, 1.0], np.float32), (W, 1)), name="hold")]
    script = PhaseScript(env, phases)
    env.reset(seed=0)
    seen = []
    for _ in range(STEPS - 1):
        env.step(script(None))
        seen.append(script.names()[0])
    assert seen[0] == "squeeze" and seen[30] == "lift" and seen[-1] == "hold"
    z = np.asarray(env.state["cube.pos"])[:, 2]
    assert np.all(z > 0.06) and np.all(z < 0.12)             # the lift stopped when the cube was up, not at max_steps
    noisy = [PhaseScript(env, phases, noise=0.5, seed=s) for s in (1, 1, 2)]
    env.reset(seed=0)
    a, b, c = (n(None) for n in noisy)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.abs(a).max() <= 1.0 and np.abs(a - np.array([0.0, 1.0, 1.0])).max() <= 0.5 + 1e-6
    assert not np.array_equal(a, np.tile(np.array([0.0, 1.0, 1.0], np.float32), (W, 1)))


def test_script_errors_name_the_fix(env):
    with pytest.raises(EnvError) as err:
        PhaseScript(env, [Phase([1.0, 0.0])])
    assert "gripper/lift" in err.value.fix
    with pytest.raises(EnvError):
        PhaseScript(env, [])
    with pytest.raises(EnvError):
        PhaseScript(env, [Phase([0.0, 0.0, 0.0], steps=0)])
    bad = PhaseScript(env, [Phase(lambda s: np.zeros((W, 2), np.float32))])
    env.reset(seed=0)
    with pytest.raises(EnvError):
        bad(None)
    with pytest.raises(EnvError) as err:
        demonstrator(SimpleNamespace(task=SimpleNamespace(name="PandaReach-v0")))
    assert "PhaseScript" in err.value.fix and "GripperLift-v0" in err.value.fix


def test_generate_saves_what_was_asked_and_it_replays(env):
    writer, calls = MemoryWriter(), []
    result = generate(env, demonstrator(env), [writer], episodes=5, seed=0, task="lift the cube",
                      progress=lambda r: calls.append(r.saved))
    assert result.complete and result.saved == 5 == len(writer.episodes)
    assert result.attempted == 2 * W == result.successes and result.success_rate == 1.0      # two rounds of 3 worlds
    assert result.steps == 2 * STEPS and result.worlds == W and result.wall_s > 0
    assert calls == [3, 5] and writer.spec.task == "lift the cube"
    s = result.summary()
    assert s["episodes_per_s"] > 0 and s["world_steps_per_s"] > 0 and s["success_interval_95"][0] > 0.5
    assert all(ep.success and ep.length == STEPS for ep in writer.episodes)
    back = replay(env, writer.episodes)
    assert back.matches.all() and back.state_error.max() == 0.0
    noisy = MemoryWriter()
    generate(env, demonstrator(env, noise=0.2, seed=3), [noisy], episodes=3, seed=0, keep="all")
    assert not np.array_equal(noisy.episodes[0].actions, writer.episodes[0].actions)
    assert replay(env, noisy.episodes).state_error.max() == 0.0          # the noise is in the recorded actions


def test_generate_stops_a_script_that_never_succeeds(env):
    writer = MemoryWriter()
    idle = PhaseScript(env, [Phase([0.0, 0.0, 0.0])])
    result = generate(env, idle, [writer], episodes=2, seed=0, max_attempts=W)
    assert not result.complete and result.saved == 0 and result.attempted == W and result.successes == 0
    assert result.success_rate == 0.0 and writer.episodes == []
    with pytest.raises(EnvError):
        generate(env, idle, [writer], episodes=0)


# --- INVALID episodes (TA-7 finding 5): counted per step from info["invalid"], never a success ----------------------


def _throws_cube(env, policy, at: int, world: int = 2):
    """``policy``, but on call ``at`` the cube of ``world`` gets 500 m/s upward: the engine freezes that world."""
    calls = {"n": 0}

    def act(obs):
        if calls["n"] == at:
            eng = env.sim.engine
            qd = eng.state.joint_qd.numpy()
            qd[env.layout.world_of["qd"].tolist().index(world) + 2] = 500.0
            eng.state.joint_qd.assign(qd)
        calls["n"] += 1
        return policy(obs)
    return act


def test_evaluate_shows_a_forced_invalid_world_and_health_alone_does_not(env):
    result = evaluate(env, _throws_cube(env, demonstrator(env), at=10), episodes=2 * W, seed=0)
    assert result.invalid.shape == result.success.shape and result.summary()["invalid"] == 1
    assert result.worlds[result.invalid].tolist() == [2] and not result.success[result.invalid].any()
    assert env.sim.health()["invalid_worlds"] == []          # the reset cleared the record: only the result has it
    clean = evaluate(env, demonstrator(env), episodes=W, seed=0)
    assert clean.summary()["invalid"] == 0 and clean.success.all()


def test_generate_counts_a_forced_invalid_episode_and_does_not_save_it(env):
    sink = MemoryWriter()
    result = generate(env, _throws_cube(env, demonstrator(env), at=10), [sink], episodes=2 * W - 1, seed=0)
    assert result.invalid == 1 and result.summary()["invalid"] == 1
    assert result.successes == result.attempted - 1 and result.complete
    assert not any(ep.invalid for ep in sink.episodes)
    kept = MemoryWriter()
    generate(env, _throws_cube(env, demonstrator(env), at=10), [kept], episodes=W, seed=0, keep="all")
    flagged = [ep for ep in kept.episodes if ep.invalid]
    assert len(flagged) == 1 and flagged[0].world == 2 and not flagged[0].success


class _FrozenAtSuccess:
    """Two worlds, 2-step episodes, the task's check always True; world 1 is INVALID on the first step of each."""

    autoreset, num_envs, xp = "same_step", 2, SimpleNamespace(to_numpy=np.asarray)
    task = SimpleNamespace(max_steps=2)

    def reset(self, seed=None):
        self.t = 0
        return None, {}

    def step(self, actions):
        self.t += 1
        info = {"invalid": np.array([False, self.t % 2 == 1])}
        if self.t % 2 == 0:
            info["_episode"] = np.array([True, True])
            info["episode"] = {"success": np.array([True, True]), "r": np.zeros(2), "l": np.full(2, 2)}
        return None, None, None, None, info


def test_an_episode_invalid_at_any_step_is_not_a_success_even_when_the_check_passes():
    result = evaluate(_FrozenAtSuccess(), lambda obs: None, episodes=4)
    assert result.invalid.tolist() == [False, True, False, True] and result.success.tolist() == [True, False] * 2
    assert result.success_rate == 0.5 and result.summary()["successes"] == 2


class _InvalidEarly:
    """``env``, but ``info["invalid"]`` is True for ``world`` on step ``at`` of every episode and on no other step:
    the physics goes on, so the episode still passes the task's success check on its last step."""

    def __init__(self, env, world: int, at: int) -> None:
        self._env, self._world, self._at = env, world, at

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        mark = int(np.asarray(self._env.state["steps"])[self._world]) + 1 == self._at
        obs, reward, terminated, truncated, info = self._env.step(actions)
        bad = np.array(info["invalid"], bool)
        bad[self._world] |= mark
        return obs, reward, terminated, truncated, {**info, "invalid": bad}


def test_recorder_and_generate_drop_an_episode_invalid_mid_way_that_passes_the_success_check(env):
    wrapped, sink = _InvalidEarly(env, world=1, at=5), MemoryWriter()
    result = generate(wrapped, demonstrator(env), [sink], episodes=2, seed=0)        # one round: worlds 0 and 2
    assert result.attempted == W and result.invalid == 1 and result.successes == W - 1
    assert sorted(ep.world for ep in sink.episodes) == [0, 2] and not any(ep.invalid for ep in sink.episodes)
    kept = MemoryWriter()
    generate(wrapped, demonstrator(env), [kept], episodes=W, seed=0, keep="all")
    assert [(ep.invalid, ep.success) for ep in sorted(kept.episodes, key=lambda e: e.world)] == [
        (False, True), (True, False), (False, True)]
