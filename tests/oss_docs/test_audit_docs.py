# SPDX-License-Identifier: Apache-2.0
"""Docs findings of the 0.1.2 outside audit, kept fixed: each test is the finding's own check, run on this tree.

- The checkpoint page counted every episode a world finished in a fixed window, so worlds that ended early were
  counted several times and the success rate was skewed. It now uses ``alloy.il.evaluate`` (one episode per world).
- The imitation guides said teleoperation was joint-space only; hand-motion actions are driven the same way.
- The viewer guides denied the ``viewer`` extra and the opt-in joint sliders and mouse dragging.
"""
from __future__ import annotations

import re
import tomllib

import numpy as np
import pytest

import alloy
from alloy.testing.oss_docs import REPO

GUIDE, PUBLIC = REPO / "docs" / "guide", REPO / "docs" / "public"
LIFT, HOLD, LOWER = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]


def _text(path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _half_succeed_env(early: str):
    """Four gripper worlds of which exactly two succeed. ``early="success"``: the two that succeed end at once;
    ``early="failure"``: the two that fail end at once and the others succeed at the time limit."""
    from alloy.envs import Task, VectorEnv, checks as C, tasks

    base = tasks.gripper_lift_task(max_steps=20)
    lifted = C.joint("gripper/lift") > 0.002
    if early == "success":
        task = Task(obs=base.obs, actions=base.actions, reward=base.reward, success=lifted, max_steps=20)
        rows = [LIFT, LIFT, HOLD, HOLD]
    else:
        task = Task(obs=base.obs, actions=base.actions, reward=base.reward, success=C.joint("gripper/lift") < -0.002,
                    failure=lifted, end_on_success=False, max_steps=20)
        rows = [LIFT, LIFT, LOWER, LOWER]
    env = VectorEnv(tasks.gripper_lift_scene(4), task, worlds=4, device="cpu", backend="numpy")
    return env, np.asarray(rows, np.float32)


@pytest.mark.parametrize("early", ["success", "failure"])
def test_evaluate_counts_one_episode_per_world(early) -> None:
    """Worlds that end early restart at once; the evaluation still counts each world once (2 of 4, not 12 of 14)."""
    from alloy.il import evaluate

    env, actions = _half_succeed_env(early)
    try:
        result = evaluate(env, lambda obs: actions, episodes=4, seed=0)
    finally:
        env.close()
    assert result.episodes == 4 and sorted(result.worlds.tolist()) == [0, 1, 2, 3]
    assert int(result.success.sum()) == 2


def test_checkpoint_page_evaluates_one_episode_per_world() -> None:
    page = (PUBLIC / "how-to" / "evaluate-checkpoint.md").read_text(encoding="utf-8")
    assert "evaluate(env" in page, "the page must count with alloy.il.evaluate"
    assert "take_stats" not in page and "max_episode_length" not in page, "a fixed window counts early enders twice"


def test_imitation_guides_do_not_deny_hand_motion_teleoperation() -> None:
    for path in (GUIDE / "imitation.md", PUBLIC / "guides" / "imitation.md", REPO / "AGENTS.md"):
        text = _text(path)
        stale = re.search(r"no Cartesian|no end-effector control|joint-space only|Joint-space teleoperation only",
                          text)
        assert stale is None, f"{path.name}: {stale.group(0)!r}, but EEActionSpec columns are driven like any other"
    assert "EEActionSpec" in _text(GUIDE / "imitation.md") and "EEActionSpec" in _text(PUBLIC / "guides/imitation.md")


def test_keyboard_drives_the_hand_of_a_seven_joint_arm() -> None:
    """The claim the guides now make: keys move the hand of a Panda through its hand-motion action columns."""
    from alloy.envs.diffik import ee_pose, panda_reach_ee
    from alloy.il import KeyboardTeleop, MemoryWriter, Recorder, ScriptKeys, teleop_loop

    env = panda_reach_ee(num_envs=1, device="cpu", seed=0, max_steps=50)
    try:
        assert env.action_names() == ["ee_dx", "ee_dy", "ee_dz"]
        memory = MemoryWriter()
        recorder = Recorder(env, [memory], keep="all")
        recorder.reset(seed=0)
        start = np.asarray(env.xp.to_numpy(ee_pose(env)[0]))[0].copy()
        keys = KeyboardTeleop(env.action_names(), dt=env.sim.dt, keys=ScriptKeys.parse("w*10 q*10 enter"))
        stats = teleop_loop(recorder, keys, realtime=False)
        recorder.close()
        env.reset(seed=0)
        for action in memory.episodes[0].actions:
            env.step(env.xp.asarray(action[None, :]))
        moved = np.asarray(env.xp.to_numpy(ee_pose(env)[0]))[0] - start
    finally:
        env.close()
    assert stats["saved"] == 1 and moved[0] > 0.015 and moved[2] > 0.015, moved


def test_viewer_guides_name_the_extra_and_the_tools() -> None:
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert any(req.startswith("viser") for req in extras["viewer"])
    viewer, export = _text(GUIDE / "viewer.md"), _text(GUIDE / "frame-export.md")
    assert "with or without extras" not in viewer and "alloy-engine[viewer]" in viewer
    assert "no mouse dragging" not in export and "joint sliders" in export


def test_viewer_ships_joint_sliders_and_mouse_dragging() -> None:
    import alloy.viewer

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.03), mass=0.15)
    sim = alloy.build(scene, device="cpu")
    try:
        with alloy.viewer.launch(sim, backend="null", frames=5, joints=True, drag=True) as view:
            tools = [type(tool).__name__ for tool in view.tools]
    finally:
        sim.close()
    assert tools == ["JointSliders", "MouseDrag"]


def test_locomotion_guide_matches_commands_and_skrl() -> None:
    """The guide said ``commands.set`` refuses a tensor and that skrl drops the critic group; neither is so."""
    text = _text(GUIDE / "locomotion.md")
    assert "tensor is refused" not in text and "skrl and Stable-Baselines3 take the policy group" not in text
    from alloy.envs.loco import Go2VelocityConfig, go2_velocity

    env = go2_velocity(num_envs=2, device="cpu", backend="numpy", config=Go2VelocityConfig(pushes=None))
    try:
        env.reset(seed=0)
        rows = np.asarray([[0.6, -0.2, 0.3], [0.4, 0.1, -0.5]], np.float32)
        env.commands.set(rows)
        assert np.allclose(env.xp.to_numpy(env.commands.value), rows)
        env.commands.set((0.8, 0.0, 0.0))
        assert np.allclose(env.xp.to_numpy(env.commands.value), [[0.8, 0.0, 0.0]] * 2)
        assert set(env.obs_groups) >= {"policy", "critic"}
    finally:
        env.close()


def test_skrl_value_network_reads_the_critic_group() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("skrl")
    from alloy.envs.loco import Go2VelocityConfig, go2_velocity
    from alloy.envs.wrappers.for_skrl import SkrlEnv
    from alloy.envs.wrappers.train_skrl import build_agent

    env = go2_velocity(num_envs=2, device="cpu", backend="torch", config=Go2VelocityConfig(pushes=None))
    env.reset(seed=0)
    env.commands.set(torch.tensor([[0.6, -0.2, 0.3], [0.4, 0.1, -0.5]], dtype=torch.float32))   # a tensor is taken
    wrapped = SkrlEnv(env)
    try:
        agent = build_agent(wrapped)
        policy_in = int(agent.models["policy"].net[0].in_features)
        value_in = int(agent.models["value"].net[0].in_features)
    finally:
        wrapped.close()
    assert wrapped.has_critic and value_in > policy_in, (policy_in, value_in)

