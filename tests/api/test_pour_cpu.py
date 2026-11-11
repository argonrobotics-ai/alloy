# SPDX-License-Identifier: Apache-2.0
"""A pour on CPU keeps the health monitor's energy check on: the poured energy is booked as external work."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api import _liquid, _pour


def _pour_sim(export: bool) -> alloy.Sim:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_open_cylinder("cup", inner_radius=0.03, depth=0.05, wall=0.004)
    scene.add_pour("water", 6e-6, pos=(0, 0, 0.12), rate=2e-5)          # 100 particles over 0.3 s
    sim = alloy.build(scene, device="cpu")
    if not export:
        sim.engine._observers[:] = [o for o in sim.engine._observers if not isinstance(o, _pour.FluxEnergy)]
    return sim


def _flags(sim: alloy.Sim) -> set[str]:
    return {f for w in sim.health()["worlds"] for f in w.get("flags", [])}


@pytest.mark.compile_heavy
def test_pour_books_its_energy_and_stays_healthy():
    sim = _pour_sim(export=True)
    assert sum(isinstance(o, _pour.FluxEnergy) for o in sim.engine._observers) == 1
    assert sim.engine.health._energy_on                     # the energy check is not skipped for a pour
    sim.run(0.6)
    assert sim.check()["ok"] and "ENERGY_JUMP" not in _flags(sim)
    assert sim.state(["water"])["water.active"].sum(axis=1).tolist() == [100, 100]
    assert sim.health()["retired_particles"] == [0, 0]
    sim.close()


@pytest.mark.compile_heavy
def test_without_the_export_the_pour_reads_as_an_energy_jump():
    sim = _pour_sim(export=False)                           # control: the check is live, so the export matters
    sim.run(0.6)
    assert "ENERGY_JUMP" in _flags(sim) and not sim.check()["ok"]
    sim.close()


def _cup_pour(scene: alloy.Scene) -> alloy.Scene:
    scene.add_ground()
    scene.add_open_cylinder("cup", inner_radius=0.03, depth=0.06, wall=0.004)
    scene.add_pour("water", 40e-6, pos=(0, 0, 0.15), rate=1e-4)              # 648 particles over 0.4 s
    scene.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.2, 0, 0.02), mass=0.1)
    return scene


@pytest.mark.compile_heavy
def test_pour_beside_a_free_body_reads_its_live_flags():
    sim = alloy.build(_cup_pour(alloy.Scene(dt=0.01)), device="cpu")       # SPH runs on the coupler's model view
    active = sim.run(0.6, record=["water"], every=20).world(0)["water.active"].sum(axis=1)
    assert active[0] == 0 and active[-1] == 648 and np.all(np.diff(active) >= 0)
    sim.close()


def test_liquid_solvers_include_one_way_followers():
    class Sph:                                                               # what liquid_solvers looks for
        add_emitter = report = None

    sph, rigid = Sph(), object()
    leaf = SimpleNamespace(leaf=SimpleNamespace(solver=sph))
    engine = SimpleNamespace(solvers=(rigid, sph), stepper=SimpleNamespace(_leaves=[leaf]))
    assert _liquid.liquid_solvers(engine) == [sph]                           # once, though listed twice
    engine = SimpleNamespace(solvers=(rigid,), stepper=SimpleNamespace(_leaves=[leaf]))   # cloth + rigid + liquid
    assert _liquid.liquid_solvers(engine) == [sph]


def test_mpm_grids_are_copies_from_every_solver():
    grid = {"grid_type": "dense", "budget_cells": 4096, "fallback": None}
    mpm, rigid = SimpleNamespace(alloy_grid=grid), object()
    leaf = SimpleNamespace(leaf=SimpleNamespace(solver=mpm))
    engine = SimpleNamespace(solvers=(rigid,), stepper=SimpleNamespace(_leaves=[leaf]))
    got = _liquid.mpm_grids(engine)
    grid.update(grid_type="sparse", fallback={"cells": 5000, "budget_cells": 4096})   # a later fallback
    assert got == [{"grid_type": "dense", "budget_cells": 4096, "fallback": None}]
    assert _liquid.mpm_grids(SimpleNamespace(solvers=(rigid,))) == []


def _jug_of_water(scene: alloy.Scene) -> alloy.Scene:
    scene.add_ground()
    scene.add_open_box("jug", inner=(0.05, 0.05, 0.05), wall=0.004, pos=(0, 0, 0.1), kinematic=True)
    scene.add_liquid("water", size=(0.044, 0.044, 0.02), pos=(0, 0, 0.115))
    return scene


@pytest.mark.compile_heavy
def test_kinematic_container_beside_a_free_body_holds_liquid():
    scene = _jug_of_water(alloy.Scene(dt=0.01))
    scene.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.2, 0, 0.02), mass=0.1)
    sim = alloy.build(scene, device="cpu")
    assert _liquid.unseen_kinematic(sim.engine) == 0          # the rigid entry carries the jug
    sim.run(0.3)
    assert np.median(sim.state(["water"], world=0)["water.points"][:, 2]) > 0.105    # still in the jug
    sim.close()


@pytest.mark.compile_heavy
def test_kinematic_container_with_cloth_and_no_rigid_solver_is_carried():
    scene = _jug_of_water(alloy.Scene(dt=0.01))
    scene.add_cloth("sheet", size=(0.1, 0.1), pos=(0.4, 0, 0.3), resolution=5, mass=0.01)
    sim = alloy.build(scene, device="cpu")                    # VBD + SPH only: the kinematic carrier entry owns the jug
    assert _liquid.unseen_kinematic(sim.engine) == 0          # was 1: no entry owned it and the water fell through
    sim.close()                                               # (the laws: tests/solvers/sph/test_kinematic_no_rigid.py)


@pytest.mark.compile_heavy
def test_two_pours_in_one_world_each_pour_their_own_reservoir():
    """WP9's find, 2026-11-11: two add_pour calls in one world split all its spares evenly, so a 20 ml and a 60 ml
    pour delivered 38 ml each. Law: each emitter pools exactly its own pour's particles, and each pour lands all of
    them under its own nozzle (bound: exact ids and counts)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("a_box", inner=(0.08, 0.08, 0.06), wall=0.005, pos=(-0.12, 0, 0))
    scene.add_open_box("b_box", inner=(0.1, 0.1, 0.08), wall=0.005, pos=(0.12, 0, 0))
    scene.add_pour("a", 6e-6, pos=(-0.12, 0, 0.12), rate=1e-4)               # 100 particles
    scene.add_pour("b", 20e-6, pos=(0.12, 0, 0.12), rate=1e-4)               # 294 particles
    sim = alloy.build(scene, device="cpu")
    em = _liquid.liquid_solvers(sim.engine)[0].emitters
    ids, begin, size = em.dev.pool_ids.numpy(), em.dev.pool_begin.numpy(), em.dev.pool_size.numpy()
    for e, name in enumerate("ab"):                          # was 197 each: a's 100 and 97 of b's, then 197 of b's
        assert ids[begin[e]:begin[e] + size[e]].tolist() == list(sim.index.particle_range(name))
    sim.run(0.5)
    st = sim.state(["a", "b"], world=0)
    for name, side, n in (("a", -1.0, 100), ("b", 1.0, 294)):
        landed = st[f"{name}.points"][st[f"{name}.active"]]
        assert len(landed) == n and bool(np.all(np.sign(landed[:, 0]) == side))
    sim.close()
