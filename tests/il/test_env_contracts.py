# SPDX-License-Identifier: Apache-2.0
"""Laws of ``alloy.il`` against env set-ups the first laws did not cover (CPU, small scenes):

1. Observation groups (``Task(obs={...})``): every group is recorded with its own labels, replays with zero error.
2. Teleoperation: only what the operator approves is saved; a timed-out attempt is dropped and counted.
3. ``generate`` works with ``autoreset="disabled"`` (it resets the ended worlds) and counts real episodes only.
4. An env that is not a ``VectorEnv`` (a port env, a Gymnasium wrapper) gets an ``EnvError`` with a fix.
5. Absolute actions are recorded as the env clipped them.
6. ``evaluate(Recorder)`` saves the counted episodes and no others.
7. ``obs`` and ``next_obs`` of an episode share one array per key.
8. ``ObsHistory``: one label per column, zero replay error; an actuator model or a rod on the rod solver: replay
   refuses with the reason.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

import alloy
from alloy.envs import ActionSpec, EnvError, Task, VectorEnv, make
from alloy.envs import checks as C
from alloy.envs.actuators import ActuatorSpec, motorized
from alloy.envs.realism import ObsHistory
from alloy.envs.realism import obs_names as realism_obs_names
from alloy.il.replay import _idle_actions
from alloy.envs.tasks import GRIPPER_MJCF, gripper_lift_scene, gripper_lift_task
from alloy.il import (KeyboardTeleop, MemoryWriter, Recorder, ScriptKeys, demonstrator, evaluate, generate, replay,
                      teleop_loop)

pytestmark = pytest.mark.compile_heavy
COLUMNS = ["gripper/lift", "gripper/close_left", "gripper/close_right"]
SLIDER = """<mujoco><worldbody><body name="b" pos="0 0 0.1">
  <joint name="j" type="slide" axis="1 0 0" range="-1 1" damping="5"/><geom type="sphere" size="0.02" mass="0.1"/>
  </body></worldbody><actuator><position joint="j" kp="300" kv="25"/></actuator></mujoco>"""


def _record(env, steps: int, action=None, **kwargs) -> tuple[MemoryWriter, Recorder]:
    sink = MemoryWriter()
    with Recorder(env, [sink], keep="all", **kwargs) as rec:
        rec.reset(seed=0)
        act = np.zeros((env.num_envs, env.writer.size), np.float32) if action is None else action
        for _ in range(steps):
            rec.step(act)
    return sink, rec


# --- 1. observation groups ------------------------------------------------------------------------------------------


def test_observation_groups_are_recorded_with_their_labels_and_replay_exactly():
    task = Task(name="Grouped", obs={"policy": ["gripper", "action"], "critic": ["gripper", "cube", "action"]},
                actions=ActionSpec(COLUMNS, mode="delta"), end_on_success=False, max_steps=5)
    env = VectorEnv(gripper_lift_scene(2), task, worlds=2, device="cpu", backend="numpy")
    try:
        sink, rec = _record(env, 5, np.full((2, 3), 0.5, np.float32))
        spec = rec.spec
        assert spec.state_key == "policy" and set(spec.obs_key_names) == {"policy", "critic"}
        for group in ("policy", "critic"):
            assert spec.obs_key_names[group] == env.obs_names(group)
            assert len(spec.obs_key_names[group]) == spec.obs_shapes[group][0] == sink.episodes[0].obs[group].shape[1]
        assert spec.obs_names == env.obs_names("policy")
        result = replay(env, sink.episodes)
        assert result.obs_error.max() == 0.0 and result.state_error.max() == 0.0
        sink.episodes[0].next_obs["critic"][2, 0] += 0.25             # the critic group is compared too
        assert replay(env, sink.episodes).obs_error[0] == pytest.approx(0.25, abs=1e-6)
    finally:
        env.close()


# --- 2. teleoperation -----------------------------------------------------------------------------------------------


def _teleop(env, keys: str, **kwargs) -> tuple[dict[str, int], MemoryWriter, Recorder]:
    sink = MemoryWriter()
    rec = Recorder(env, [sink], **kwargs)
    device = KeyboardTeleop(env.action_names(), dt=env.sim.dt, keys=ScriptKeys.parse(keys))
    rec.reset(seed=0)
    stats = teleop_loop(rec, device, realtime=False)
    rec.close()
    return stats, sink, rec


def test_teleop_saves_only_what_the_operator_approves():
    env = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy", max_steps=12)
    try:
        stats, sink, rec = _teleop(env, ".*16 esc", keep="all")       # one timeout, then quit: nothing approved
        assert sink.episodes == [] and rec.saved == 0 and not rec.manual
        assert stats == {"steps": 16, "saved": 0, "discarded": 4}     # 2 worlds timed out, 2 running at quit
        stats, sink, rec = _teleop(env, ".*5 backspace .*5 enter", keep="all")
        assert stats == {"steps": 10, "saved": 2, "discarded": 2} and [ep.length for ep in sink.episodes] == [5, 5]
        stats, sink, rec = _teleop(env, ".*5 enter", keep=lambda ep: ep.success)      # a keep function decides
        assert stats == {"steps": 5, "saved": 0, "discarded": 2} and sink.episodes == []
        stats, sink, rec = _teleop(env, ".*5 enter", keep="success")  # Enter approves an unfinished attempt
        assert stats["saved"] == 2 and not any(ep.success for ep in sink.episodes)
    finally:
        env.close()


def test_a_recorder_outside_teleop_still_saves_timeouts_with_keep_all():
    env = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy", max_steps=6)
    try:
        sink, rec = _record(env, 6)
        assert rec.saved == 2 and not sink.episodes[0].success
        manual, _ = _record(env, 6, manual=True)
        assert manual.episodes == []
    finally:
        env.close()


# --- 3. generate and autoreset --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["disabled", "next_step"])
def test_generate_resets_the_ended_worlds_in_every_autoreset_mode(mode):
    env = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy", max_steps=60, autoreset=mode)
    try:
        sink = MemoryWriter()
        result = generate(env, demonstrator(env), [sink], episodes=4, max_attempts=8, seed=0)
        assert result.complete and result.saved == 4 == len(sink.episodes)
        assert result.attempted == 4 and result.successes == 4
        assert [ep.length for ep in sink.episodes] == [60] * 4
        assert sorted(ep.world for ep in sink.episodes) == [0, 0, 1, 1]
    finally:
        env.close()


# --- 4. envs alloy.il cannot drive ----------------------------------------------------------------------------------


class _Wrapper:
    """What a Gymnasium vector wrapper looks like from outside: the env in ``.env``, none of its attributes."""

    def __init__(self, env) -> None:
        self.env, self.num_envs = env, env.num_envs


class _PortLike:
    num_envs, autoreset = 1, "same_step"


@pytest.mark.parametrize("call", ["Recorder", "evaluate", "generate", "replay"])
def test_an_env_that_is_not_a_vector_env_raises_with_a_fix(call):
    env = make("GripperLift-v0", num_envs=1, device="cpu", backend="numpy", max_steps=4)
    run = {"Recorder": lambda e: Recorder(e, [MemoryWriter()]), "evaluate": lambda e: evaluate(e, None, 1),
           "generate": lambda e: generate(e, None, [MemoryWriter()], 1), "replay": lambda e: replay(e, [])}[call]
    try:
        with pytest.raises(EnvError, match="needs an alloy.envs.VectorEnv") as wrapped:
            run(_Wrapper(env))
        assert "_Wrapper.env" in wrapped.value.fix
        with pytest.raises(EnvError, match="got a _PortLike") as port:
            run(_PortLike())
        assert "your own loop" in port.value.fix
    finally:
        env.close()


# --- 5. absolute actions --------------------------------------------------------------------------------------------


def test_absolute_actions_are_recorded_as_the_env_clipped_them():
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_robot("gripper", mjcf=GRIPPER_MJCF, pos=(0.0, 0.0, 0.075))
    task = Task(name="Abs", obs=["gripper", "action"], end_on_success=False, max_steps=3,
                actions=ActionSpec(COLUMNS, mode="absolute", low=[-0.03, 0.0, 0.0], high=[0.20, 0.05, 0.05]))
    env = VectorEnv(scene, task, worlds=1, device="cpu", backend="numpy")
    try:
        sink = MemoryWriter()
        with Recorder(env, [sink], keep="all") as rec:
            rec.reset(seed=0)
            for _ in range(3):
                info = rec.step(np.array([[10.0, -5.0, 99.0]], np.float32))[4]
        assert np.allclose(sink.episodes[0].actions, [[0.20, 0.0, 0.05]] * 3)
        assert np.allclose(sink.episodes[0].actions[0], np.asarray(info["targets"])[0])
        assert np.allclose(sink.episodes[0].next_obs["state"][:, -3:], [[0.20, 0.0, 0.05]] * 3)   # obs["action"]
        assert replay(env, sink.episodes).state_error.max() == 0.0
    finally:
        env.close()


# --- 6. evaluate with a recorder ------------------------------------------------------------------------------------


def test_evaluate_with_a_recorder_saves_the_counted_episodes_only():
    scene = alloy.Scene(worlds=4, dt=0.01)
    scene.add_ground()
    scene.add_robot("bot", mjcf=SLIDER)
    task = Task(name="QuickFail", obs=["bot"], actions=ActionSpec(["bot/j"], mode="delta", delta=0.05),
                success=C.joint("bot/j") > 0.05, failure=C.joint("bot/j") < -0.02, end_on_success=False,
                max_steps=12)
    env = VectorEnv(scene, task, worlds=4, device="cpu", backend="numpy", seed=0)
    policy = lambda obs: np.array([[-1.0], [-1.0], [1.0], [1.0]], np.float32)      # noqa: E731
    try:
        for kwargs in ({}, {"max_episodes": 8}):
            sink = MemoryWriter()
            with Recorder(env, [sink], keep="all", **kwargs) as rec:
                result = evaluate(rec, policy, episodes=8, seed=0)
                assert rec.world_quota is None                       # the quota lasts for the evaluation only
            assert result.episodes == 8 and result.success_rate == 0.5
            assert sorted(ep.world for ep in sink.episodes) == [0, 0, 1, 1, 2, 2, 3, 3]
            assert np.mean([ep.success for ep in sink.episodes]) == 0.5
    finally:
        env.close()


# --- 7. memory ------------------------------------------------------------------------------------------------------


def test_obs_and_next_obs_of_an_episode_share_one_array():
    env = make("GripperLift-v0", num_envs=2, device="cpu", backend="numpy", max_steps=6)
    try:
        sink, _ = _record(env, 6, np.full((2, 3), 0.3, np.float32))
        ep = sink.episodes[0]
        assert np.shares_memory(ep.obs["state"], ep.next_obs["state"])
        assert np.array_equal(ep.obs["state"][1:], ep.next_obs["state"][:-1])
        assert not np.array_equal(ep.next_obs["state"][-1], ep.obs["state"][-1])
    finally:
        env.close()


# --- 8. layers, actuator models, rods -------------------------------------------------------------------------------


def test_obs_history_gets_a_label_per_column_and_replays_exactly():
    task = dataclasses.replace(gripper_lift_task(max_steps=10), layers=[ObsHistory(3)])
    env = VectorEnv(gripper_lift_scene(2), task, worlds=2, device="cpu", backend="numpy", seed=0)
    try:
        sink, rec = _record(env, 10, np.full((2, 3), 0.5, np.float32))
        width = rec.spec.obs_shapes["state"][0]
        assert width == 3 * len(env.obs_names()) == len(rec.spec.obs_names)
        assert realism_obs_names(env) == rec.spec.obs_names
        assert rec.spec.obs_names[0].endswith("@t-2") and rec.spec.obs_names[-1].endswith("@t")
        result = replay(env, sink.episodes)
        assert result.obs_error.max() == 0.0 and result.state_error.max() == 0.0
    finally:
        env.close()


def test_replay_refuses_an_actuator_model_and_the_recorder_warns(caplog):
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_robot("gripper", mjcf=motorized(GRIPPER_MJCF), pos=(0, 0, 0.075))
    spec = ActuatorSpec(COLUMNS, kp=[3000, 400, 400], kd=[150, 10, 10], mode="delta", low=[-0.03, 0.0, 0.0],
                        high=[0.2, 0.05, 0.05], delta=[0.004, 0.002, 0.002], delay_steps=(0, 2))
    sim = alloy.build(scene, plugins=[spec.plugin()], device="cpu", seed=0)
    env = VectorEnv(sim, Task(name="Act", obs=["gripper"], actions=spec, max_steps=4, end_on_success=False),
                    device="cpu", backend="numpy", seed=0)
    try:
        with caplog.at_level(logging.WARNING, logger="alloy.il"):
            sink, _ = _record(env, 4)
        assert "actuator model" in caplog.text and len(sink.episodes) == 2
        with pytest.raises(EnvError, match="actuator model") as err:
            replay(env, sink.episodes)
        assert "evaluate" in err.value.fix
    finally:
        env.close()
        sim.close()


def test_replay_refuses_a_rod_on_the_rod_solver():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_robot("bot", mjcf=SLIDER)
    scene.add_rod("cable", length=0.30, radius=0.012, segments=5, material="silicone_40a", pos=(0.05, 0.0, 0.25),
                  clamp="start")
    task = Task(name="Cable", obs=["bot"], actions=ActionSpec(["bot/j"], mode="delta", delta=0.02), max_steps=2,
                end_on_success=False)
    env = VectorEnv(scene, task, worlds=2, device="cpu", backend="numpy", seed=0)
    try:
        assert env.sim.routes()["cable"]["route"] == "rod_vbd"
        sink, _ = _record(env, 2)
        with pytest.raises(EnvError, match="rod solver") as err:
            replay(env, sink.episodes)
        assert "cable" in err.value.message and "evaluate" in err.value.fix
    finally:
        env.close()


# --- 9. randomized worlds, INVALID in a replay ----------------------------------------------------------------------


def _randomized_env(when: str, worlds: int):
    scene = gripper_lift_scene(worlds=worlds, seed=0)
    scene.randomize("gripper", "kp_scale", 0.05, 1.5, when=when)
    return VectorEnv(scene, gripper_lift_task(max_steps=80), worlds=worlds, device="cpu", backend="numpy", seed=0)


def test_replay_runs_each_episode_in_the_world_that_recorded_it():
    env = _randomized_env("build", 4)
    try:
        sink = MemoryWriter()
        generate(env, demonstrator(env), [sink], episodes=6, seed=0, max_attempts=24)
        worlds = [ep.world for ep in sink.episodes]
        assert worlds != sorted(set(worlds)) and len(sink.episodes) == 6      # not one episode per world in order
        assert "random.gripper.kp_scale" in sink.episodes[0].start
        result = replay(env, sink.episodes)
        assert result.matches.all() and result.state_error.max() == 0.0 and result.obs_error.max() == 0.0
        moved = dataclasses.replace(sink.episodes[0], world=(worlds[0] + 1) % 4)
        with pytest.raises(EnvError, match="kp_scale") as err:                # another world's servo gains
            replay(env, [moved])
        assert "world count and seed" in err.value.fix
    finally:
        env.close()


def test_replay_refuses_values_drawn_at_every_reset():
    env = _randomized_env("reset", 1)
    try:
        sink = MemoryWriter()
        generate(env, demonstrator(env), [sink], episodes=2, seed=0, max_attempts=10, keep="all")
        with pytest.raises(EnvError, match="recorded with") as err:
            replay(env, sink.episodes)
        assert "when='reset'" in err.value.fix and "evaluate" in err.value.fix
    finally:
        env.close()


def test_replay_reports_an_invalid_world_and_leaves_finished_worlds_alone():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_robot("gripper", mjcf=GRIPPER_MJCF, pos=(0.0, 0.0, 0.075))
    task = Task(name="Abs", obs=["gripper"], end_on_success=False, max_steps=6,
                actions=ActionSpec(COLUMNS, mode="absolute", low=[-0.03, 0.0, 0.0], high=[0.20, 0.05, 0.05]))
    env = VectorEnv(scene, task, worlds=2, device="cpu", backend="numpy")
    try:
        sink, _ = _record(env, 6, np.array([[0.1, 0.02, 0.02], [0.05, 0.01, 0.01]], np.float32))
        short = dataclasses.replace(sink.episodes[0], actions=sink.episodes[0].actions[:2],
                                    states=sink.episodes[0].states[:2])
        sent = []
        step = env.step
        env.step = lambda a: (sent.append(np.asarray(a).copy()), step(a))[1]
        result = replay(env, [short, sink.episodes[1]])
        del env.step
        assert result.state_error.max() == 0.0 and not result.invalid.any() and result.summary()["invalid"] == 0
        assert np.allclose(sent[1][0], short.actions[1])
        env.reset()
        assert np.allclose(_idle_actions(env), 0.0)                # a reset world: its start targets
        env.step(np.array([[0.1, 0.02, 0.02], [0.05, 0.01, 0.01]], np.float32))
        assert np.allclose(_idle_actions(env), [[0.1, 0.02, 0.02], [0.05, 0.01, 0.01]])     # then what it holds
        flagged = dataclasses.replace(sink.episodes[1], invalid=True)
        result = replay(env, [flagged])
        assert result.recorded_invalid.tolist() == [True] and not result.matches[0]     # it did not freeze again
    finally:
        env.close()
