# SPDX-License-Identifier: Apache-2.0
"""A scene of only static and kinematic parts builds and steps: nothing in it needs a free body.

Laws (CPU): a scene of a table (static), a tray and an open bin (kinematic) builds; ``sim.set_pose`` puts the tray
where it says, a drive moves it as far as its velocity times its duration, ``sim.reset`` brings it back, and the
check is ok. Its driven path is the one the same scene takes with a free ball beside it (the old workaround), frame
for frame. A scene of static parts alone builds and its clock runs. A kinematic part too small for MuJoCo's inertia
floor at its default density (a 0.37 mm thread segment) builds in such a scene, since nothing in it reaches MuJoCo.
Before, each of these failed with "scene has no dynamic entry to step", so posed parts drawn frame by frame (a clip
redraw) needed a ball out of frame and a raised density.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy

#: a thread segment, 0.37 mm across and 0.56 mm long, as a closed box mesh: at a default density its smallest
#: principal moment is about 2e-15 kg m^2, under the 1e-14 MuJoCo takes for a full inertia tensor
SEG = (0.00037, 0.00037, 0.00056)
POSED = {"features": {"min_wall": {"value": 0.00037, "source": "model", "note": "posed with set_pose, never stepped"}}}


def _box_mesh(size: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = (0.5 * s for s in size)
    v = np.array([(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)])
    f = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7),
                  (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    return v, f


def _posed(ball: bool = False, kinematic: bool = True) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    if kinematic:
        scene.add_box("tray", size=(0.2, 0.2, 0.02), pos=(0, 0, 0.5), kinematic=True)
        scene.add_open_box("bin", inner=(0.1, 0.1, 0.05), wall=0.005, pos=(0.2, 0, 0.41), kinematic=True)
    if ball:
        scene.add_sphere("ball", 0.01, pos=(1.0, 1.0, 0.01), mass=0.01)     # out of the way, resting on the ground
    return scene


def _drive(sim: alloy.Sim) -> np.ndarray:
    sim.drive("tray", velocity=(0.0, 0.1, 0.0), angular_velocity=(0.0, 0.0, 0.5), duration=0.5)
    return sim.run(1.0, record=["tray"])["tray.pos"][:, 0]


@pytest.mark.compile_heavy
def test_static_and_kinematic_parts_alone_build_and_move_as_told():
    sim = alloy.build(_posed(), device="cpu")
    try:
        assert set(sim.names()["bodies"]) >= {"tray", "bin"}
        sim.set_pose("tray", pos=(0.1, 0.0, 0.6))
        sim.step(1)
        assert np.allclose(sim.state(world=0)["tray.pos"], (0.1, 0.0, 0.6), atol=1e-6)
        path = _drive(sim)
        assert np.allclose(path[-1], (0.1, 0.05, 0.6), atol=1e-5)          # 0.1 m/s for 0.5 s, then held
        assert sim.check()["ok"]
        sim.reset()
        assert np.allclose(sim.state(world=0)["tray.pos"], (0.0, 0.0, 0.5), atol=1e-6)
    finally:
        sim.close()


@pytest.mark.compile_heavy
def test_the_driven_path_is_the_one_beside_a_free_ball():
    paths = []
    for ball in (False, True):
        sim = alloy.build(_posed(ball=ball), device="cpu")
        try:
            paths.append(_drive(sim))
        finally:
            sim.close()
    assert np.allclose(paths[0], paths[1], atol=1e-6)


def test_static_parts_alone_build_and_their_clock_runs():
    sim = alloy.build(_posed(kinematic=False), device="cpu")
    try:
        sim.step(3)
        assert sim.t == pytest.approx(0.03) and sim.check()["ok"]
    finally:
        sim.close()


def test_a_part_under_mujocos_inertia_floor_builds_when_nothing_is_free():
    v, f = _box_mesh(SEG)
    scene = alloy.Scene(dt=0.01)
    scene.add_box("worktop", size=(0.4, 0.4, 0.02), pos=(0, 0, 0.01), static=True)
    scene.add_mesh("thread", vertices=v, faces=f, pos=(0, 0, 0.05), kinematic=True, card_extra=POSED)
    sim = alloy.build(scene, device="cpu")
    try:
        sim.set_pose("thread", pos=(0.01, 0.0, 0.05))
        sim.step(1)
        assert np.allclose(sim.state(world=0)["thread.pos"], (0.01, 0.0, 0.05), atol=1e-6)
    finally:
        sim.close()
