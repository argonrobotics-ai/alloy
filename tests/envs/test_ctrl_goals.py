# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-native actuators (kind "ctrl"), goal points, a fixed start pose for a joint, free bodies inside a robot
file, and trimmed robot files. CPU, NumPy."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import alloy.api as alloy
from alloy.envs import ActionSpec, Task, VectorEnv, jitter_body, jitter_joint
from alloy.envs import checks as C
from alloy.envs.mjcf import without_bodies

SERVO = ('<mujoco><worldbody>'
         '<body name="carriage" pos="0 0 0.3"><joint name="slide" type="slide" axis="0 0 1" range="-0.2 0.2" '
         'damping="20"/><geom type="box" size="0.02 0.02 0.02" mass="0.2" contype="0" conaffinity="0"/></body>'
         '<body name="prop" pos="0.3 0 0.05"><freejoint name="prop_free"/>'
         '<geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body></worldbody>'
         '<actuator><general name="lift" joint="slide" gainprm="400" biastype="affine" biasprm="0 -400 -40" '
         'ctrlrange="-0.2 0.2" ctrllimited="true"/></actuator></mujoco>')


@pytest.fixture(scope="module")
def env():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", mjcf=SERVO)
    task = Task(obs=["r", "target"], actions=ActionSpec(mode="absolute"), max_steps=50,
                reward={"reach": (-1.0, C.distance("r/carriage", "target"))},
                goals={"target": ((0.0, 0.0, 0.2), (0.0, 0.0, 0.4))},
                randomize=[jitter_joint("r/slide", at=0.1), jitter_body("r/prop", pos=(0.05, 0.0, 0.0))])
    e = VectorEnv(scene, task, worlds=2, device="cpu", backend="numpy", autoreset="disabled")
    yield e
    e.close()


def _ctrl(env: VectorEnv) -> np.ndarray:
    return env.sim.engine.control.mujoco.ctrl.numpy()


@pytest.mark.compile_heavy
def test_ctrl_actions_and_start_pose(env):
    assert env.writer.kind == "ctrl" and env.action_names() == ["r/lift"]
    assert env.single_action_space.low.tolist() == pytest.approx([-0.2]) and "r/prop" in env.layout.free
    env.reset(seed=0)
    s = env.state
    assert np.allclose(s["r/slide.q"], 0.1, atol=1e-6) and np.allclose(_ctrl(env), 0.1)   # start pose + servo held
    assert np.all(np.abs(s["r/prop.pos"][:, 0] - 0.3) <= 0.05) and np.ptp(s["r/prop.pos"][:, 0]) > 1e-4
    for _ in range(40):
        obs, r, term, trunc, info = env.step(np.full((2, 1), 0.15))
    assert np.allclose(_ctrl(env), 0.15) and np.allclose(env.state["r/slide.q"], 0.15, atol=0.01)


@pytest.mark.compile_heavy
def test_reset_puts_an_unjittered_ctrl_actuator_back():
    """A reset world starts at its build-time control again, a masked reset leaves the other worlds' alone. It kept
    the last episode's: the layout lost the worlds of the ``mujoco.ctrl`` rows, so the writer restored none of them
    (found by a replay that started each episode at the old targets, 2026-11-10)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("r", mjcf=SERVO)
    task = Task(obs=["r"], actions=ActionSpec(mode="absolute"), max_steps=50)       # no jitter rewrites the control
    e = VectorEnv(scene, task, worlds=2, device="cpu", backend="numpy", autoreset="disabled")
    try:
        assert e.writer.kind == "ctrl" and e.layout.world_of["ctrl"].tolist() == [0, 1]
        e.reset(seed=0)
        build = _ctrl(e).copy()
        for _ in range(3):
            e.step(np.full((2, 1), 0.15))
        assert np.allclose(_ctrl(e), 0.15) and not np.allclose(build, 0.15)
        e.reset(options={"reset_mask": np.array([True, False])})
        assert _ctrl(e)[0] == build[0] and np.isclose(_ctrl(e)[1], 0.15)
        e.reset()
        np.testing.assert_array_equal(_ctrl(e), build)
    finally:
        e.close()


@pytest.mark.compile_heavy
def test_goals_redrawn_per_reset_and_observed(env):
    env.reset(seed=1)
    s = env.state
    target = s["target.pos"].copy()
    assert np.allclose(target[:, :2], 0.0) and np.all((target[:, 2] >= 0.2) & (target[:, 2] <= 0.4))
    assert np.ptp(target[:, 2]) > 1e-4 and "target.pos[2]" in env.obs_names()
    obs, r, *_ = env.step(np.zeros((2, 1)))
    assert np.allclose(obs[:, env.obs_names().index("target.pos[2]")], target[:, 2])
    assert np.allclose(r, -np.linalg.norm(env.state["r/carriage.pos"] - target, axis=1), atol=1e-5)
    env.reset(options={"reset_mask": np.array([False, True])})
    after = env.state["target.pos"]
    assert np.allclose(after[0], target[0]) and not np.allclose(after[1], target[1])


def test_without_bodies(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path))
    from alloy.cards.library import default_root

    src = default_root() / "robots" / "emb_e2_panda" / "scene.xml"
    if not src.is_file():
        pytest.skip("library robot not present")
    out = without_bodies(src, ["box", "cylinder", "mug"])
    assert out.parent == tmp_path / "envs" and without_bodies(src, ["box", "cylinder", "mug"]) == out
    root = ET.parse(out).getroot()
    names = {b.get("name") for b in root.iter("body")}
    assert {"box", "cylinder", "mug"}.isdisjoint(names) and "hand" in names and root.find("keyframe") is None
    files = [m.get("file") for m in root.iter("mesh")]
    assert files and all(Path(f).is_absolute() and Path(f).is_file() for f in files)
    from alloy.envs import EnvError

    with pytest.raises(EnvError, match="no top-level bodies"):
        without_bodies(src, ["nope"])


def test_two_copies_of_one_robot_file_get_their_own_trimmed_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))
    xml = ('<mujoco><asset><mesh name="m" file="part.stl"/></asset>'
           '<worldbody><body name="a"><geom type="mesh" mesh="m"/></body><body name="b"/></worldbody></mujoco>')
    outs = []
    for name in ("one", "two"):                                    # the same text in two checkouts
        (tmp_path / name).mkdir()
        (tmp_path / name / "robot.xml").write_text(xml)
        outs.append(without_bodies(tmp_path / name / "robot.xml", ["b"]))
    assert outs[0] != outs[1]                                      # deleting one checkout leaves the other working
    for name, out in zip(("one", "two"), outs):
        assert ET.parse(out).getroot().find("asset/mesh").get("file") == str((tmp_path / name / "part.stl").resolve())
