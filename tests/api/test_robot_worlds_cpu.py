# SPDX-License-Identifier: Apache-2.0
"""Robots in the Scene API: their shapes' contact gap, and free-base robots placed per world (CPU).

Robot files used to keep Newton's importer gap (0.1 m) while every scene object gets 5 mm, so packed robot scenes
filled MuJoCo Warp's contact buffer: a carton among six boxes made 53 rigid contacts at 0.1 m, 28 at 5 mm (CPU,
2026-11-07). A robot whose root body has a free joint (a carton with a flap here) starts per world, is randomized and
teleported with all its links.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

import alloy.api as alloy
from alloy.api._materialize import materialize

CARTON = """<mujoco><worldbody><body name="box"><freejoint/>
  <geom type="box" size="0.05 0.04 0.03" mass="0.1"/>
  <geom name="tagged" type="sphere" size="0.005" pos="0 0 0.035" gap="0.002" mass="0.001"/>
  <body name="flap" pos="0.05 0 0.03"><joint name="hinge" type="hinge" axis="0 1 0" range="-1.6 0.2" damping="0.01"/>
    <geom type="box" size="0.03 0.04 0.002" pos="0.03 0 0" mass="0.01"/></body>
  </body></worldbody></mujoco>"""
ARM = """<mujoco><worldbody><body name="link"><joint name="j" type="hinge" axis="0 1 0" range="-1 1"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/></body></worldbody></mujoco>"""
ROWS = [[0.0, 0.0, 0.031], [0.2, 0.0, 0.031], [0.0, 0.2, 0.031]]
UP_YAW90 = (0.7071068, 0.0, 0.0, 0.7071068)


def shape_gaps(sim, name: str) -> dict[str, float]:
    """Contact gap of each of ``name``'s shapes by its label's last part."""
    m = sim.engine.model
    labels, gaps = [str(x) for x in m.shape_label], m.shape_gap.numpy()
    return {labels[i].rsplit("/", 1)[-1]: round(float(gaps[i]), 4) for i in sim.built.handles[name].template.shape}


@pytest.mark.compile_heavy
def test_robot_shapes_take_the_scene_contact_gap_with_an_opt_out():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("carton", mjcf=CARTON, pos=(0, 0, 0.031))
    scene.add_robot("arm", mjcf=ARM, pos=(0.3, 0, 0.3), contact_gap=0.1)        # the importer's old default
    sim = alloy.build(scene, device="cpu")
    assert shape_gaps(sim, "carton") == {"box_geom_0": 0.005, "tagged": 0.002, "flap_geom_0": 0.005}  # own gap kept
    assert shape_gaps(sim, "arm") == {"link_geom_0": 0.1}
    with pytest.raises(alloy.SimulationError, match="no free base") as err:
        sim.set_pose("arm", pos=(0, 0, 1))                                          # welded to the world
    assert "<freejoint/>" in err.value.fix
    sim.close()


@pytest.mark.compile_heavy
def test_free_base_robot_starts_per_world_is_randomized_teleported_and_reset():
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_robot("carton", mjcf=CARTON, pos=ROWS)
    scene.randomize("carton", "x", -0.02, 0.02)
    sim = alloy.build(scene, device="cpu", seed=3)
    s0 = sim.state(["carton", "carton/flap"])
    x = s0["carton.pos"][:, 0]
    assert np.all(np.abs(x) <= 0.02) and len(set(np.round(x, 6))) == 3              # drawn per world
    assert np.allclose(s0["carton.pos"][:, 1:], np.array(ROWS)[:, 1:], atol=1e-6)    # y, z from the rows
    assert np.allclose(s0["carton/flap.pos"] - s0["carton.pos"], [0.05, 0, 0.03], atol=1e-6)  # the link follows
    sim.run(0.3)
    target = np.array([[0.0, 0.0, 0.25], [0.3, 0.0, 0.25], [0.0, 0.3, 0.25]])     # +0.24 J: not an ENERGY_JUMP
    hinge = sim.state(["carton/hinge"])["carton/hinge.q"]
    sim.set_pose("carton", pos=target, quat_wxyz=UP_YAW90)
    s = sim.state(["carton", "carton/flap", "carton/hinge"])
    assert np.allclose(s["carton.pos"], target, atol=1e-6) and np.allclose(s["carton.vel"], 0.0)
    assert np.allclose(s["carton/hinge.q"], hinge)                                   # joint angles kept
    flap = s["carton/flap.pos"] - target                                             # yawed 90 deg: +x -> +y
    assert np.allclose(flap[:, :2], [0.0, 0.05], atol=1e-5)
    sim.run(0.5)
    assert np.allclose(sim.state(["carton"])["carton.pos"][:, :2], target[:, :2], atol=1e-3) and sim.check()["ok"]
    sim.reset()
    assert np.allclose(sim.state(["carton"])["carton.pos"], s0["carton.pos"])        # back to the drawn starts
    sim.close()
    again = alloy.build(scene, device="cpu", seed=3)
    assert np.array_equal(again.state(["carton"])["carton.pos"], s0["carton.pos"])   # same seed, same draw
    again.close()


@pytest.mark.compile_heavy
def test_welded_robot_with_per_world_positions_is_a_build_error():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_robot("arm", mjcf=ARM, pos=[[0, 0, 0.3], [0.1, 0, 0.3]])
    with pytest.raises(alloy.BuildError, match="no free base"):
        alloy.build(scene, device="cpu")


def test_robot_scene_checks_and_card(tmp_path):
    scene = alloy.Scene(dt=0.01)
    with pytest.raises(alloy.SceneError, match="contact_gap"):
        scene.add_robot("arm", mjcf=ARM, contact_gap=-0.001)
    assert "arm" not in scene.items
    scene.add_robot("arm", mjcf=ARM, contact_gap=0.002)
    scene.add_robot("carton", mjcf=CARTON)
    with pytest.raises(alloy.SceneError, match="base position"):
        scene.randomize("carton", "density", 100, 200)
    scene.randomize("carton", "y", -0.01, 0.01)
    back = alloy.Scene.from_dict(scene.to_dict())
    assert back.items["arm"]["contact_gap"] == 0.002 and back.items["carton"]["randomize"][0]["field"] == "pose.1"
    card = yaml.safe_load(materialize(scene, tmp_path).scene_file.read_text())
    robots = {r["name"]: r for r in card["robots"]}
    assert robots["arm"]["contact_gap"] == 0.002 and "contact_gap" not in robots["carton"]  # default: card unchanged
