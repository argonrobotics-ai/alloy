# SPDX-License-Identifier: Apache-2.0
"""A kinematic fixture is one kinematic body (alloy.scene.emitters.rigid.emit_part, FIX-LIST item 6), CPU.

An outside tester's kinematic ``add_open_cylinder("cup")`` beside a 2-dof robot made 152 dofs: 25 parts, each its own
6-dof kinematic body. Now the root part (the floor) is the body and the other parts are shapes on it.
- dofs: the cup beside the 2-dof arm is 8 dofs (2 + 6, was 152) and one body for the cup (+ a free block: 14).
- names: "cup" and "cup_floor" are that body (same state); a wall part has no state of its own and its error names
  the fixture; sim.names() lists no wall as a body or as static.
- parts: every part keeps its shape label and its sim.routes() row, and each shape sits where the part's own box sat
  (within 1e-6 m), in every world of a per-world start.
- mass: the body carries the whole fixture: its mass is the parts' summed density x volume (1e-6 relative) and its
  centre of mass their mass-weighted centre (1e-6 m), so after the Newton bump's soft weld on kinematic roots it steps
  with the fixture's inertia, not the floor's alone; readback (run at build) checks the root against the sum.
- drive: sim.drive("cup") moves the one body (0.05 m/s for 0.2 s: +10 mm), walls with it. With no pivot a fixture
  still turns about its parts' centre: an open box (0.1 x 0.1 x 0.08 at z 0.2) at 90 deg/s about y for 1 s puts its
  floor at (-0.0336, 0, 0.2356), as on main before this change (base 97998ba9d, same script).
- a static fixture is unchanged: no body, its parts static shapes (tilt-pour laws and bitwise runs: the SPH and API
  suites, and the gate's static-fixture tests).
- a rider part (not the floor) takes no randomized start and is no attachment endpoint: the build says to use the
  floor; randomizing the floor's x moves the whole fixture in each world. No part takes a density draw (the one body
  carries every part's mass, and a draw would scale one part's object only).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.api.fixtures import kinematic_groups

CUP = {"inner_radius": 0.03, "depth": 0.06, "wall": 0.004}
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="l1" pos="0.3 0 0.1">
  <joint name="j1" type="hinge" axis="0 0 1" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.5"/>
  <body name="l2" pos="0.1 0 0"><joint name="j2" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.5"/></body></body></worldbody>
  <actuator><position joint="j1" kp="50"/><position joint="j2" kp="50"/></actuator></mujoco>"""


def _cup(worlds: int = 1, pos=(0.0, 0.0, 0.0), **kw) -> tuple[alloy.Scene, list[str]]:
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    parts = sc.add_open_cylinder("cup", **CUP, pos=pos, **kw)
    sc.add_box("block", size=(0.03, 0.03, 0.03), pos=(0.6, 0.0, 0.015), mass=0.1)     # something to step
    return sc, parts


def _shape_world(sim: alloy.Sim) -> np.ndarray:
    """(shapes, 7) world poses (x, y, z, qx, qy, qz, qw) of every shape at the current state."""
    import warp as wp

    m, st = sim.engine.model, sim.engine.state
    body, xf, bq = m.shape_body.numpy(), m.shape_transform.numpy(), st.body_q.numpy()
    out = np.zeros((len(body), 7))
    for s, b in enumerate(body):
        t = wp.transform(wp.vec3(*xf[s][:3]), wp.quat(*xf[s][3:]))
        if b >= 0:
            t = wp.transform_multiply(wp.transform(wp.vec3(*bq[b][:3]), wp.quat(*bq[b][3:])), t)
        out[s] = [*t.p, *t.q]
    return out


def _same_pose(a: np.ndarray, b: np.ndarray) -> bool:
    return np.abs(a[:3] - b[:3]).max() < 1e-6 and min(np.abs(a[3:] - b[3:]).max(), np.abs(a[3:] + b[3:]).max()) < 1e-6


def test_a_kinematic_cup_beside_a_two_dof_arm_is_one_body_and_8_dofs():
    sc, parts = _cup(kinematic=True)
    sc.add_robot("arm", mjcf=ARM)
    sim = alloy.build(sc, device="cpu")
    m = sim.engine.model
    assert (m.body_count, m.joint_dof_count) == (4, 14), (m.body_count, m.joint_dof_count)  # +the block; was 28, 158
    labels = [str(x) for x in m.shape_label]
    assert all(p in labels for p in parts)
    rows = sim.routes()
    assert all(any("kinematic fixture 'cup'" in r for r in rows[p]["reasons"]) for p in parts), rows[parts[1]]
    assert sim.check()["ok"], sim.check()["problems"]


def test_the_body_carries_the_whole_fixtures_mass_and_centre():
    sc, parts = _cup(kinematic=True)
    sim = alloy.build(sc, device="cpu")
    m, body = sim.engine.model, sim.index.body("cup")
    objs = {t.name: t for t in sim.built._plan.objects}
    half = {p: sc.items[p]["shape"]["primitive"]["size"] for p in parts}                 # card sizes: half edges
    mass = np.array([objs[p].obj.material.density.value * 8.0 * float(np.prod(half[p]))
                     for p in parts])
    centres = np.array([sc.items[p]["pose"][:3] for p in parts])
    assert float(m.body_mass.numpy()[body]) == pytest.approx(float(mass.sum()), rel=1e-6, abs=0)
    com = sim.state(["cup"], world=0)["cup.pos"] + m.body_com.numpy()[body]          # the floor's frame is unrotated
    assert com == pytest.approx(mass @ centres / mass.sum(), abs=1e-6, rel=0), com


def test_cup_and_cup_floor_name_the_body_and_a_wall_points_at_the_fixture():
    sc, parts = _cup(kinematic=True)
    sim = alloy.build(sc, device="cpu")
    s = sim.state(["cup", "cup_floor"], world=0)
    assert np.array_equal(s["cup.pos"], s["cup_floor.pos"]) and np.array_equal(s["cup.quat_xyzw"],
                                                                                s["cup_floor.quat_xyzw"])
    names = sim.names()
    assert "cup" in names["bodies"] and not set(parts[1:]) & set(names["bodies"] + names["static"]), names
    with pytest.raises(alloy.UnknownNameError) as e:
        sim.state([parts[1]])
    assert "kinematic fixture 'cup'" in e.value.message and "'cup'" in e.value.fix


@pytest.mark.parametrize("kinematic", [True, False])
def test_every_part_sits_where_its_own_box_sat_in_every_world(kinematic):
    starts = [[0.0, 0.0, 0.0], [0.25, -0.1, 0.0]] if kinematic else [0.0, 0.0, 0.0]
    sc, parts = _cup(worlds=2, pos=starts, kinematic=kinematic)
    sim = alloy.build(sc, device="cpu")
    m = sim.engine.model
    labels, world = [str(x) for x in m.shape_label], m.shape_world.numpy()
    poses = _shape_world(sim)
    for p in parts:
        want = np.asarray(sc.items[p]["per_world_pos"] or [sc.items[p]["pose"][:3]] * 2, float)
        quat = np.asarray(sc.items[p]["pose"][3:], float)                  # x, y, z, w
        for w in range(2):
            s = next(i for i, lab in enumerate(labels) if lab == p and int(world[i]) in (w, -1))
            assert _same_pose(poses[s], np.r_[want[w], quat]), (p, w, poses[s], want[w], quat)
    if not kinematic:
        assert m.body_count == 2 and set(parts) <= set(sim.names()["static"])          # the block, 2 worlds
        assert not kinematic_groups(sc.items)


def test_a_rider_part_refuses_a_randomized_start_and_an_attachment():
    sc, parts = _cup(kinematic=True)
    sc.randomize(parts[3], "x", -0.01, 0.01)
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(sc, device="cpu")
    assert e.value.field == parts[3] and "'cup_floor'" in e.value.fix
    sc, parts = _cup(kinematic=True)
    sc.extra["attachments"] = [{"a": parts[5], "b": "block", "kind": "weld"}]
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(sc, device="cpu")
    assert e.value.field == "attachments" and "'cup_floor'" in e.value.fix


@pytest.mark.parametrize("part", [0, 4])
def test_no_part_of_a_kinematic_fixture_takes_a_density_draw(part):
    sc, parts = _cup(kinematic=True)
    sc.randomize(parts[part], "density", 900.0, 1200.0)
    with pytest.raises(alloy.BuildError) as e:
        alloy.build(sc, device="cpu")
    assert e.value.field == parts[part] and "one body" in e.value.message, e.value.message


def test_randomizing_the_floor_moves_the_whole_fixture_per_world():
    sc, parts = _cup(worlds=2, kinematic=True)
    sc.randomize("cup_floor", "x", 0.1, 0.3)
    sim = alloy.build(sc, device="cpu", seed=3)
    m = sim.engine.model
    labels, world = [str(x) for x in m.shape_label], m.shape_world.numpy()
    poses, x = _shape_world(sim), sim.state(["cup"])["cup.pos"][:, 0]
    assert np.all((x > 0.1) & (x < 0.3)) and abs(float(x[0] - x[1])) > 1e-4, x
    for w in range(2):
        floor = next(i for i, lab in enumerate(labels) if lab == parts[0] and int(world[i]) == w)
        wall = next(i for i, lab in enumerate(labels) if lab == parts[9] and int(world[i]) == w)
        rel = sc.items[parts[9]]["pose"][0] - sc.items[parts[0]]["pose"][0]
        assert poses[wall, 0] - poses[floor, 0] == pytest.approx(rel, abs=1e-6, rel=0)


@pytest.mark.parametrize("name", ["bin", "bin_floor"])
def test_a_fixture_driven_without_a_pivot_turns_about_its_parts_centre(name):
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("bin", inner=(0.1, 0.1, 0.08), pos=(0, 0, 0.2), kinematic=True)
    sc.add_box("block", size=(0.03, 0.03, 0.03), pos=(0.6, 0.0, 0.015), mass=0.1)
    sim = alloy.build(sc, device="cpu")
    sim.drive(name, angular_velocity=(0, math.radians(90), 0), duration=1.0)      # either name: the one body
    sim.step(100)
    floor = sim.state(["bin_floor"], world=0)["bin_floor.pos"]
    assert floor == pytest.approx([-0.0336, 0.0, 0.2356], abs=1e-4, rel=0), floor


def test_driving_the_cup_moves_its_one_body_and_every_wall():
    sc, parts = _cup(kinematic=True)
    sim = alloy.build(sc, device="cpu")
    labels = [str(x) for x in sim.engine.model.shape_label]
    wall = labels.index(parts[7])
    p0, w0 = sim.state(["cup"], world=0)["cup.pos"].copy(), _shape_world(sim)[wall, :3].copy()
    sim.drive("cup", velocity=(0.05, 0.0, 0.0), duration=0.2)
    sim.step(30)
    p1, w1 = sim.state(["cup"], world=0)["cup.pos"], _shape_world(sim)[wall, :3]
    assert p1 - p0 == pytest.approx([0.01, 0.0, 0.0], abs=1e-6, rel=0), p1 - p0
    assert w1 - w0 == pytest.approx([0.01, 0.0, 0.0], abs=1e-6, rel=0), w1 - w0
    assert sim.check()["ok"], sim.check()["problems"]
