# SPDX-License-Identifier: Apache-2.0
"""``add_robot(..., pos=(worlds, 3))`` moves the whole robot file in each world: its free base, its own floor and the
parts it welds to the world, as a ``pos`` tuple does.

Laws (CPU, on both CPU routes: ``mjc``, one MuJoCo C model per world, what users get, and ``warp``; solver asserted),
for a free-base walker whose file brings a floor plane and a welded block: with rows of different heights (each
order) and of different x, every world starts clear of its floor (no start overlap, health OK) and moves as world 0
does, offset by its row (1e-4 m); the block stands at its file pose plus the row; ``sim.set_pose`` moves the base of
such a file; world 0 steps bit for bit as a build at its row as a tuple, whatever world 1's row (and equal rows as
that tuple). On a GPU (``gpu`` mark) the same rows, within float32. Before, only the base moved: the floor stayed at
the first world's row, so a lower world started 100 mm inside it (INVALID) and a higher one fell; on ``mjc`` a raised
floor then held no contacts (MuJoCo's compile-time frame flag and bounding tree kept world 0's floor); a file with a
welded part raised "no free base".
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy

WALKER = """<mujoco model="walker"><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="0 0 0.05"/>
  <body name="base" pos="0 0 0.12"><freejoint/>
    <geom type="box" size="0.1 0.06 0.03" mass="2"/>
    <geom type="sphere" size="0.02" pos="0.08 0.05 -0.08" mass="0.1"/>
    <geom type="sphere" size="0.02" pos="0.08 -0.05 -0.08" mass="0.1"/>
    <geom type="sphere" size="0.02" pos="-0.08 0.05 -0.08" mass="0.1"/>
    <geom type="sphere" size="0.02" pos="-0.08 -0.05 -0.08" mass="0.1"/>
    <geom type="capsule" fromto="0.08 0.05 -0.03 0.08 0.05 -0.08" size="0.01" mass="0.05"/>
    <geom type="capsule" fromto="-0.08 -0.05 -0.03 -0.08 -0.05 -0.08" size="0.01" mass="0.05"/>
  </body>
  <body name="block" pos="0.6 0 0.05"><geom type="box" size="0.05 0.05 0.05" mass="1"/></body>
</worldbody></mujoco>"""
SOLVER = {"mjc": "MjcCoreSolver", "warp": "SolverMuJoCo"}      # the rigid solver each CPU route builds
ROWS = {"z 0 / 0.1": [[0, 0, 0], [0, 0, 0.1]], "z 0.1 / 0": [[0, 0, 0.1], [0, 0, 0]], "x 0 / 0.5": [[0, 0, 0], [0.5, 0, 0]]}


@pytest.fixture(params=sorted(SOLVER))
def route(request, monkeypatch) -> str:
    """Both CPU routes, whatever the suite pinned."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", request.param)
    return request.param


def _sim(route: str, rows) -> alloy.Sim:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_robot("walker", mjcf=WALKER, pos=np.asarray(rows, np.float64))
    sim = alloy.build(scene, device="cpu")
    assert SOLVER[route] in [type(s).__name__ for s in sim.engine.solvers], route
    return sim


@pytest.mark.compile_heavy
@pytest.mark.parametrize("rows", sorted(ROWS))
def test_each_world_starts_on_its_own_floor_and_moves_as_world_0(route: str, rows: str):
    row = np.asarray(ROWS[rows], np.float64)
    sim = _sim(route, row)
    block = sim.state(["walker/block"])["walker/block.pos"]
    assert np.allclose(block, np.array([0.6, 0, 0.05]) + row, atol=1e-6)        # was world 0's place in each world
    traj = sim.run(0.5, record=["walker/base"])["walker/base.pos"]            # (frames, worlds, 3)
    assert sim.check()["ok"], sim.check()["problems"]                          # was INVALID: 100 mm start overlap
    assert all(w.get("start_overlap") is None for w in sim.health()["worlds"])
    assert np.allclose(traj[:, 1] - row[1], traj[:, 0] - row[0], atol=1e-4)   # mjc z 0 / 0.1: sank, no contacts
    assert traj[-1, 0, 2] - row[0, 2] == pytest.approx(0.1, abs=0.01)           # standing on its feet
    sim.close()


@pytest.mark.compile_heavy
def test_set_pose_moves_the_base_of_a_file_with_a_welded_part(route: str):
    sim = _sim(route, ROWS["x 0 / 0.5"])                                        # was BuildError "no free base"
    sim.set_pose("walker", pos=(0.0, 0.0, 0.3))
    assert np.allclose(sim.state(["walker/base"])["walker/base.pos"], [0.0, 0.0, 0.3], atol=1e-6)
    sim.close()


@pytest.mark.compile_heavy
def test_world_0_is_bit_for_bit_the_tuple_build(route: str):
    """World 0 of per-world rows steps as a build at its row as a tuple does, bit for bit, also when world 1's row
    differs (only a world that differs from world 0 gets its frame data again); equal rows step as the tuple."""
    def run(pos) -> np.ndarray:
        scene = alloy.Scene(worlds=2, dt=0.01)
        scene.add_robot("walker", mjcf=WALKER, pos=pos)
        sim = alloy.build(scene, device="cpu")
        assert SOLVER[route] in [type(s).__name__ for s in sim.engine.solvers], route
        out = sim.run(0.3, record=["walker/base"])["walker/base.pos"]
        sim.close()
        return out

    tuple_run = run((0.0, 0.0, 0.1))
    assert np.array_equal(run(np.array([[0, 0, 0.1], [0, 0, 0.1]])), tuple_run)
    assert np.array_equal(run(np.array([[0, 0, 0.1], [0.5, 0, 0]]))[:, 0], tuple_run[:, 0])


@pytest.mark.gpu
@pytest.mark.parametrize("rows", sorted(ROWS))
def test_each_world_starts_on_its_own_floor_on_a_gpu(rows: str):
    row = np.asarray(ROWS[rows], np.float64)
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_robot("walker", mjcf=WALKER, pos=row)
    sim = alloy.build(scene, device="cuda:0")
    assert np.allclose(sim.state(["walker/block"])["walker/block.pos"], np.array([0.6, 0, 0.05]) + row, atol=1e-6)
    traj = sim.run(0.5, record=["walker/base"])["walker/base.pos"]
    assert sim.check()["ok"], sim.check()["problems"]
    assert np.allclose(traj[:, 1] - row[1], traj[:, 0] - row[0], atol=1e-3)    # MuJoCo Warp: float32
    assert traj[-1, 0, 2] - row[0, 2] == pytest.approx(0.1, abs=0.01)
    sim.close()
