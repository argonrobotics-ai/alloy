# SPDX-License-Identifier: Apache-2.0
"""AUDIT-STACK majors in the public API (2026-11-09, the rc8 wheel run as a user; CPU): each case is the auditor's
repro, pinned at what the docs say.

- AGENTS-1 / API-2 / WORLDS-1 / WORLDS-2: ``scene.randomize`` refuses a card number the engine cannot draw per world
  when it is called, with the fields it can (restitution and poisson failed the build);
- AGENTS-3 / ERR-4: ``set_targets`` on a robot without actuators raises, naming the MJCF line;
- AGENTS-4: ``set_controls`` on an MJCF ``<position>`` actuator moves the joint;
- API-3: a position target outside the joint's range stops at the limit, per world;
- API-1: ``reset()`` returns to the settled state in a scene with ``scene.randomize``;
- ERR-5: a malformed robot MJCF is a ``SceneError`` that names the robot and the place;
- ERR-6: ``add_liquid`` of a solid material is refused;
- INSTALL-2: a missing optional extra is a ``FeatureUnavailableError`` with the install command;
- PHYSB-2: a robot joint named like its body keeps its ``.q``;
- TOOLS-1: ``Scene.save`` to a relative folder writes a robot path the saved card loads from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.api import errors as E


def _arm(actuator: str = "", joint: str = "hinge", body: str = "link") -> str:
    return f"""<mujoco><compiler angle="radian"/><worldbody><body name="{body}" pos="0 0 0.1">
      <joint name="{joint}" type="hinge" axis="0 0 1" range="-1 1" damping="2"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
      </body></worldbody>{actuator}</mujoco>"""


POSITION = '<actuator><position name="act" joint="hinge" kp="300" kv="20"/></actuator>'


def _build(mjcf: str, worlds: int = 1, name: str = "arm"):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_robot(name, mjcf=mjcf, pos=(0, 0, 0.5))
    return alloy.build(scene, device="cpu")


@pytest.fixture(scope="module")
def servo():
    sim = _build(_arm(POSITION), worlds=2)
    yield sim
    sim.close()


def _q(sim, joint: str = "hinge") -> np.ndarray:
    return sim.state([joint])[f"{joint}.q"]


# --- randomize ---------------------------------------------------------------------------------------------------

def test_randomize_refuses_what_the_engine_cannot_draw_per_world() -> None:
    scene = alloy.Scene(worlds=4)
    scene.add_sphere("ball", 0.03, pos=(0, 0, 0.5), material="steel_carbon")
    with pytest.raises(alloy.SceneError, match="one restitution for every world") as info:
        scene.randomize("ball", "material.surface.restitution", 0.1, 0.5)
    assert "restitution=" in info.value.message and "material.surface.mu_dynamic" in info.value.fix
    with pytest.raises(alloy.SceneError, match="cannot draw this card number per world") as info:
        scene.randomize("ball", "material.elastic.poisson", 0.2, 0.3)
    assert all(f in info.value.fix for f in ("density", "friction", "youngs"))
    with pytest.raises(alloy.SceneError, match="one contact stiffness for every world") as info:
        scene.randomize("ball", "material.surface.contact_stiffness", 2e5, 8e5)      # was offered; the build failed
    assert "contact_stiffness" not in info.value.fix
    assert scene.items["ball"]["randomize"] == []
    scene.randomize("ball", "material.surface.mu_dynamic", 0.2, 0.3)          # the docs' example path
    scene.randomize("ball", "friction", 0.4, 0.6)
    assert [r["field"] for r in scene.items["ball"]["randomize"]] == ["material.surface.mu_dynamic",
                                                                       "material.surface.mu_static"]


# --- targets and controls ----------------------------------------------------------------------------------------

def test_set_targets_without_any_actuator_raises_with_the_line_to_add() -> None:
    sim = _build(_arm())
    try:
        assert sim.names()["actuators"] == []
        with pytest.raises(alloy.SimulationError, match="has no actuator") as info:
            sim.set_targets({"hinge": 0.8})
        assert '<position joint="..."' in info.value.fix and info.value.field == "hinge"
        with pytest.raises(alloy.SimulationError, match="has no actuator"):
            sim.run(0.02, targets={"hinge": (0.0, 0.8)})
    finally:
        sim.close()


def test_a_position_target_past_the_range_stops_at_the_limit_per_world(servo) -> None:
    servo.reset()
    servo.run(0.1, record=[], targets={"hinge": (0.0, 3.0)})                   # 2 rad past the +1 rad stop
    servo.step(40)
    assert np.allclose(_q(servo), 1.0, atol=0.02)                              # was 1.32 rad, health OK
    servo.set_targets({"hinge": np.array([-3.0, 0.5])})
    servo.step(45)
    assert np.allclose(_q(servo), [-1.0, 0.5], atol=0.02)


def test_the_clamp_keeps_a_wider_ctrlrange_and_leaves_unlimited_joints_and_raw_controls(servo) -> None:
    """WP13's review: a servo whose ctrlrange is wider than its joint's range squeezes by aiming past the stop, so the
    wider range holds; a joint without a range and sim.set_controls are not held at all."""
    two = """<mujoco><compiler angle="radian"/><worldbody><body name="a" pos="0 0 0.1">
      <joint name="grip" type="hinge" axis="0 0 1" range="-1 1" damping="2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
      <body name="b" pos="0.2 0 0"><joint name="spin" type="hinge" axis="0 0 1" damping="2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/></body></body></worldbody>
      <actuator><position joint="grip" kp="300" kv="20" ctrlrange="-1.5 1.5"/>
      <position joint="spin" kp="300" kv="20"/></actuator></mujoco>"""
    sim = _build(two)
    try:
        sim.set_targets({"grip": 3.0, "spin": 7.0})
        target = sim.engine.control.joint_target_q.numpy()
        assert target[sim._io.target_start[sim.index.joint("grip")]] == pytest.approx(1.5)     # not the joint's 1.0
        assert target[sim._io.target_start[sim.index.joint("spin")]] == pytest.approx(7.0)     # no range: untouched
    finally:
        sim.close()
    servo.reset()
    servo.set_controls({"arm/act": 3.0})                                       # raw ctrl: past the +1 rad stop
    rows = servo._io.target_start[servo.index.per_world(servo.index.joint("hinge"), "joint")]
    assert np.allclose(servo.engine.control.joint_target_q.numpy()[rows], 3.0)
    servo.set_targets({"hinge": 0.0})


def test_set_controls_on_a_position_actuator_moves_its_joint(servo) -> None:
    servo.reset()
    servo.set_targets({"hinge": 0.0})
    assert servo.names()["actuators"] == ["arm/act", "act"]
    servo.set_controls({"arm/act": np.array([0.4, -0.2])})
    servo.step(40)
    assert np.allclose(_q(servo), [0.4, -0.2], atol=0.02)                      # was 0.0: the ctrl row is not read


# --- settle, then reset, with randomize --------------------------------------------------------------------------

def test_reset_returns_to_the_settled_state_with_randomize() -> None:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.2)
    scene.randomize("block", "x", -0.05, 0.05)
    sim = alloy.build(scene, device="cpu", seed=0)
    try:
        start = sim.state(["block"])["block.pos"].copy()
        sim.settle(0.4)
        settled = sim.state(["block"])
        assert np.all(settled["block.pos"][:, 2] < 0.03) and np.allclose(settled["block.pos"][:, 0], start[:, 0],
                                                                         atol=1e-3)
        sim.set_pose("block", pos=(0.3, 0.0, 0.4))
        sim.step(5)
        sim.reset()
        again = sim.state(["block"])
        for key in ("block.pos", "block.quat_wxyz", "block.vel"):
            assert np.array_equal(again[key], settled[key]), key              # was the build start, z = 0.1
        assert sim.check()["ok"]
    finally:
        sim.close()


# --- scene errors ------------------------------------------------------------------------------------------------

def test_a_malformed_robot_mjcf_is_a_scene_error_naming_the_robot(tmp_path: Path) -> None:
    bad = "<mujoco><worldbody><body name='a'><geom type='sphere' size='0.1'></body></worldbody>"
    scene = alloy.Scene()
    with pytest.raises(alloy.SceneError, match="robot 'arm'.*not well-formed XML.*line 1") as info:
        scene.add_robot("arm", mjcf=bad)
    assert info.value.field == "mjcf" and "arm" not in scene.items
    path = tmp_path / "bad.xml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(alloy.SceneError, match="not well-formed XML") as info:
        scene.add_robot("arm", path)
    assert info.value.field == "path"
    scene.add_robot("arm", mjcf=_arm())                                        # a good one is taken as before
    assert scene.items["arm"]["robot_kind"] == "mjcf"


def test_add_liquid_refuses_a_solid_material() -> None:
    scene = alloy.Scene()
    with pytest.raises(alloy.SceneError, match="'steel_carbon' is a solid material") as info:
        scene.add_liquid("water", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.03), material="steel_carbon")
    assert "add_box" in info.value.fix and info.value.field == "material" and "water" not in scene.items
    scene.add_material("my_plastic", density=900.0)
    with pytest.raises(alloy.SceneError, match="solid material"):
        scene.add_pour("drip", 1e-5, pos=(0, 0, 0.2), material="my_plastic")
    for k, material in enumerate(("water", "honey", "mayonnaise", "sand_silica_dry")):      # these flow
        scene.add_liquid(f"l{k}", size=(0.04, 0.04, 0.04), pos=(0.1 * k, 0, 0.03), material=material)
    with pytest.raises(alloy.SceneError, match="unknown material"):
        scene.add_liquid("x", size=(0.04, 0.04, 0.04), material="no_such_material")


def test_a_missing_extra_is_feature_unavailable_with_the_command() -> None:
    from alloy.core.errors import AssemblyError

    exc = AssemblyError("c: tetrahedralising a mesh needs the optional [mesh] extra (pytetwild), not installed")
    err = E.wrap(exc, "build")
    assert isinstance(err, alloy.FeatureUnavailableError) and 'pip install "alloy-engine[mesh]"' in err.fix
    assert "install.sh" not in err.fix and "pip install -e" not in err.fix and "pytetwild" in err.message
    assert 'alloy-engine[usd]' in E.remedy_for(ModuleNotFoundError("No module named 'pxr'"))
    assert "alloy-engine[usd]" not in E.remedy_for(ModuleNotFoundError("No module named 'warp'"))


# --- names -------------------------------------------------------------------------------------------------------

def test_a_joint_named_like_its_body_keeps_its_q() -> None:
    sim = _build(_arm(POSITION.replace('joint="hinge"', 'joint="arm"'), joint="arm", body="arm"), name="clip")
    try:
        assert "clip/arm" in sim.names()["joints"] and "clip/arm" in sim.names()["bodies"]
        state = sim.state(world=0)
        assert {"clip/arm.q", "clip/arm.pos", "clip/arm.quat_wxyz", "clip/arm.vel"} <= set(state)
        assert set(sim.state(["clip/arm"])) == {"t", "clip/arm.q", "clip/arm.pos", "clip/arm.quat_xyzw",
                                                "clip/arm.quat_wxyz", "clip/arm.vel"}
        traj = sim.run(0.3, record=["clip/arm"], targets={"clip/arm": (0.0, 0.3)})
        assert traj["clip/arm.q"].shape == (31, 1) and traj["clip/arm.q"][-1, 0] > 0.1
    finally:
        sim.close()


# --- save --------------------------------------------------------------------------------------------------------

def test_save_to_a_relative_folder_writes_a_robot_path_that_loads(tmp_path: Path, monkeypatch) -> None:
    import yaml

    monkeypatch.chdir(tmp_path)
    scene = alloy.Scene()
    scene.add_ground()
    scene.add_robot("arm", mjcf=_arm(POSITION), pos=(0, 0, 0.6))
    card = scene.save("saved_scene").resolve()
    path = Path(yaml.safe_load(card.read_text(encoding="utf-8"))["robots"][0]["path"])
    assert path.is_absolute() and path == (tmp_path / "saved_scene" / "robots" / "arm.xml").resolve()
    monkeypatch.chdir(tmp_path.parent)                                         # the card loads from anywhere
    assert "arm" in alloy.Scene.load(card).names()
