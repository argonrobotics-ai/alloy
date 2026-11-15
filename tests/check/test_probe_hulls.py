# SPDX-License-Identifier: Apache-2.0
"""OSS-7: the check's contact probe collides every mesh MuJoCo collides as MuJoCo does, as its convex hull. Newton's
model keeps an MJCF mesh geom as a triangle mesh (a robot link, a RoboCasa object's hull colliders): collided that way,
1,024 worlds of RoboCasa kitchen objects overflowed the probe's buffers, and its depths were not the simulation's.

CPU proxy: an MJCF mesh of two separate pillars, whose hull spans the gap between them, and a free box in that gap.
As triangles the pillars never touch the box; MuJoCo's hull holds it 19.5 mm deep, and so does the probe.

L4 row (variety_1024's kitchen at 1,024 worlds, its five plain boxes swapped for RoboCasa objects, 11.5 s of physics,
no camera, then sim.check()): on main 88f9ca79e (the probe that grows from its budget) Newton dropped candidates once,
one rebuild held them, 12.0 GB peak; with the hulls, 141,312 mesh shapes collided as convex hulls, no drop and no
rebuild, 11.4 GB peak. Both runs reported the same findings (a start overlap of two swapped objects, PEN_WARN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.check import contacts as cc
from alloy.check import probe_view as pv
from alloy.check.objects import ObjectIndex

GAP_DEPTH_M = 0.0195     # the box's shortest way out of the hull: down, its top (2.05 cm) to the hull floor (1 mm)


def _cube(lo: tuple[float, float, float], hi: tuple[float, float, float], first: int) -> tuple[list, list]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return v, [(a + first, b + first, c + first) for a, b, c in faces]


def _pillars(folder: Path) -> Path:
    va, fa = _cube((-0.07, -0.02, 0.0), (-0.03, 0.02, 0.04), 0)
    vb, fb = _cube((0.03, -0.02, 0.0), (0.07, 0.02, 0.04), 8)
    lines = [f"v {x} {y} {z}" for x, y, z in va + vb] + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in fa + fb]
    (folder / "pillars.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    xml = folder / "pillars.xml"
    xml.write_text('<mujoco model="pillars"><compiler meshdir="."/><asset><mesh name="pillars" file="pillars.obj"/>'
                   '</asset><worldbody><body name="base" pos="0 0 0.001"><geom type="mesh" mesh="pillars"/></body>'
                   '</worldbody></mujoco>', encoding="utf-8")
    return xml


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    scene = alloy.Scene(worlds=cc.BUDGET_MIN_WORLDS, dt=0.0025)
    scene.add_ground()
    scene.add_robot("pillars", _pillars(tmp_path_factory.mktemp("pillars")))
    scene.add_box("box", size=(0.02, 0.02, 0.02), pos=(0, 0, 0.0105), mass=0.05)
    return alloy.build(scene, device="cpu")


def _box_pair(probe, sim):
    rows = probe.depths(sim.engine.state, ObjectIndex.build(sim.engine.built))
    return [p for p in rows if "box" in (p.a, p.b) and "ground" not in (p.a, p.b)]


def test_meshes_collide_as_mujocos_hulls(sim):
    built = sim.engine.built
    assert sim.engine.native == "mujoco" and pv.hull_shapes(built, "mujoco").size == cc.BUDGET_MIN_WORLDS
    probe = cc.ContactProbe(built, native=sim.engine.native)
    assert pv.hulled(probe.view) == cc.BUDGET_MIN_WORLDS and built.model is not probe.view
    (pair,) = _box_pair(probe, sim)
    assert pair.depth == pytest.approx(GAP_DEPTH_M, abs=1e-3)
    assert probe.need()["triangle_pairs"] == 0
    assert int(built.model.shape_type.numpy()[pv.hull_shapes(built, "mujoco")][0]) == int(
        alloy._vendor.newton.GeoType.MESH)                                  # the shared model keeps its meshes


def test_the_depth_is_the_one_mujoco_reports(sim):
    import alloy.viewer

    probe = cc.ContactProbe(sim.engine.built, native=sim.engine.native)
    (pair,) = _box_pair(probe, sim)
    sim.step(1)                                                            # two 1.25 ms steps: MuJoCo's contacts
    c = alloy.viewer.contacts(sim, world=0)
    pillar = np.array(["box" in (a, b) and any(n.startswith("pillars") for n in (a, b))
                       for a, b in zip(c["body_a"], c["body_b"])])
    assert pillar.any()
    assert float(np.max(c["depth"][pillar])) == pytest.approx(pair.depth, rel=0.1)


def test_newton_contacts_keep_triangle_meshes(sim):
    built = sim.engine.built
    assert pv.hull_shapes(built, "newton").size == 0 and pv.collider_view(built, "newton") is built.model
    probe = cc.ContactProbe(built, native="newton")
    assert _box_pair(probe, sim) == []                                     # the pillars' triangles never reach it
    probe.use_native("mujoco")
    (pair,) = _box_pair(probe, sim)
    assert pair.depth == pytest.approx(GAP_DEPTH_M, abs=1e-3)


def test_particle_scenes_keep_triangle_meshes(sim):
    """Particle contacts need the triangle surface (cloth over a mesh), so a scene with particles keeps its meshes."""
    from types import SimpleNamespace

    built = sim.engine.built
    model = SimpleNamespace(particle_count=8, shape_count=built.model.shape_count)
    assert pv.hull_shapes(SimpleNamespace(model=model, entries=built.entries), "mujoco").size == 0
