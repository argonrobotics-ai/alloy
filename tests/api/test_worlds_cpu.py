# SPDX-License-Identifier: Apache-2.0
"""Agent friction trial items (2026-11-07) on CPU: per-world starts for every kind that moves (F7), STL meshes (F6)
and cloth self-contact (F11).

F7: a (worlds, 3) ``pos`` worked only for free bodies; static boxes and cloth failed at build with a raw
AttributeError, fixtures with "pos needs 3 finite numbers", liquids' randomized x with an AssemblyError. Now free and
kinematic bodies, kinematic fixtures, robots, cloth and liquids start per world, and static things are refused when
added, with the fix (kinematic=True). F6: ``add_mesh("x.stl")`` was accepted, then assembly refused .stl.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api import meshes

ROWS = [[0.0, 0.0, 0.1], [0.3, 0.0, 0.1]]


def vbd_self_contact(sim) -> list[bool]:
    """The self-contact switch of every VBD solver in the sim."""
    st = sim.engine.stepper
    solvers = [getattr(st, "solver", None), *(getattr(e, "solver", None) for e in getattr(st, "_entries", {}).values())]
    return [bool(s.particle_enable_self_contact) for s in solvers if hasattr(s, "particle_enable_self_contact")]


def test_static_things_refuse_per_world_positions_when_added():
    scene = alloy.Scene(worlds=2)
    calls = {"box": lambda: scene.add_box("box", size=(0.1,) * 3, pos=ROWS, static=True),
             "bin": lambda: scene.add_open_box("bin", inner=(0.05, 0.05, 0.05), pos=ROWS),
             "peg": lambda: scene.add_object("peg", "peg_d12", pos=ROWS, static=True),
             "pour": lambda: scene.add_pour("pour", 1e-5, pos=ROWS)}
    for name, call in calls.items():
        with pytest.raises(alloy.SceneError, match="same in every world") as err:
            call()
        assert err.value.field == "pos" and name not in scene.items and not any(k.startswith(name) for k in scene.items)
        assert ("kinematic=True" in err.value.fix) == (name != "pour")
    scene.add_box("table", size=(0.1,) * 3, pos=(0, 0, 0.05), static=True)
    with pytest.raises(alloy.SceneError, match="same in every world"):
        scene.randomize("table", "x", -0.01, 0.01)
    scene.add_liquid("water", size=(0.02, 0.02, 0.02), pos=(0, 0, 0.3))
    scene.randomize("water", "x", -0.01, 0.01)                       # a liquid's start: drawn by the API at build
    assert scene.items["water"]["start_random"] == [{"field": "pose.0", "lo": -0.01, "hi": 0.01}]
    assert not scene.items["water"]["randomize"]


@pytest.mark.compile_heavy
def test_kinematic_bodies_fixtures_and_cloth_start_per_world():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("pusher", size=(0.02,) * 3, pos=[[0, 0.2, 0.01], [0.3, 0.2, 0.01]], kinematic=True)
    scene.add_open_box("jug", inner=(0.05, 0.05, 0.05), pos=[[0, -0.2, 0.0], [0.3, -0.2, 0.0]], kinematic=True)
    scene.add_cloth("towel", size=(0.2, 0.2), pos=ROWS, resolution=8, mass=0.02, self_contact=False)
    scene.randomize("towel", "y", -0.05, 0.05)
    sim = alloy.build(scene, device="cpu", seed=1)
    s = sim.state(["pusher", "jug_floor", "towel"])
    assert np.allclose(s["pusher.pos"][:, 0], [0.0, 0.3], atol=1e-6)
    assert np.allclose(s["jug_floor.pos"][:, :2], [[0.0, -0.2], [0.3, -0.2]], atol=1e-6)
    c = s["towel.points"].mean(axis=1)
    assert np.allclose(c[:, [0, 2]], [[0.0, 0.1], [0.3, 0.1]], atol=1e-5)
    assert np.all(np.abs(c[:, 1]) <= 0.05) and abs(c[0, 1] - c[1, 1]) > 1e-4         # y drawn per world
    assert vbd_self_contact(sim) == [False]                                             # one cloth, switched off
    sim.run(0.2)
    assert sim.check()["ok"]
    sim.reset()
    assert np.allclose(sim.state(["towel"])["towel.points"].mean(axis=1), c, atol=1e-6)
    sim.close()


@pytest.mark.compile_heavy
def test_cloth_self_contact_switch_reaches_the_solver():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_cloth("towel", size=(0.2, 0.2), pos=ROWS, resolution=8, mass=0.02, self_contact=True)
    sim = alloy.build(scene, device="cpu")                          # no bodies at all: was an AttributeError (F7)
    assert vbd_self_contact(sim) == [True]
    assert np.allclose(sim.state(["towel"])["towel.points"].mean(axis=1), ROWS, atol=1e-5)
    sim.close()


def _stl(path, v, f) -> None:
    """A binary STL: 80-byte header, triangle count, then per triangle a normal, 3 vertices, 2 spare bytes."""
    tri = np.asarray(v, np.float32)[np.asarray(f)]
    rec = b"".join(struct.pack("<12fH", 0, 0, 0, *t.ravel(), 0) for t in tri)
    path.write_bytes(b"\0" * 80 + struct.pack("<I", len(tri)) + rec)


@pytest.mark.compile_heavy
def test_stl_mesh_builds_and_rests(tmp_path):
    v, f = meshes.box((0.04, 0.04, 0.02))                           # bottom at z = 0
    _stl(tmp_path / "block.stl", v, f)
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_mesh("block", tmp_path / "block.stl", pos=(0, 0, 0.05), mass=0.05)
    sim = alloy.build(scene, device="cpu")
    z = sim.run(0.6, record=["block"]).world(0)["block.pos"][-1, 2]
    assert z == pytest.approx(0.0, abs=1e-3) and sim.check()["ok"]   # the origin is the bottom face's centre
    sim.close()
