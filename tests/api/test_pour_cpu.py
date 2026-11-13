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
    scene.add_pour("water", 6e-6, pos=(0, 0, 0.12), rate=2e-5)          # 94 particles over 0.3 s
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
    assert sim.state(["water"])["water.active"].sum(axis=1).tolist() == [94, 94]
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
    scene.add_pour("water", 40e-6, pos=(0, 0, 0.15), rate=1e-4)              # 625 particles over 0.4 s
    scene.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.2, 0, 0.02), mass=0.1)
    return scene


@pytest.mark.compile_heavy
def test_pour_beside_a_free_body_reads_its_live_flags():
    sim = alloy.build(_cup_pour(alloy.Scene(dt=0.01)), device="cpu")       # SPH runs on the coupler's model view
    active = sim.run(0.6, record=["water"], every=20).world(0)["water.active"].sum(axis=1)
    assert active[0] == 0 and active[-1] == 625 and np.all(np.diff(active) >= 0)
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
def test_a_rolled_back_period_releases_its_layer_again():
    """GAP-LC's side finding, 2026-11-11: the engine made its rollback snapshot at build, before the pour's emitter
    existed, so a rolled-back period put the particles' flags back but not the emitter's cursor, and each re-run that
    had released a layer skipped it for good (a 20 ml pour onto cloth: three rollbacks as the stream landed, 244 of
    294 particles poured). Law: restoring the period snapshot puts the cursor back with the flags, for the rolled-back
    world only (WP4's review: the other world keeps its layer), and the pour still releases every particle (bound:
    exact counts)."""
    sim = alloy.build(_cup_pour(alloy.Scene(worlds=2, dt=0.01)), device="cpu")    # SPH on the coupler: it rolls back
    eng, em = sim.engine, _liquid.liquid_solvers(sim.engine)[0].emitters
    flags = _liquid.liquid_solvers(sim.engine)[0].model.particle_flags
    begin, size, ids = em.dev.pool_begin.numpy(), em.dev.pool_size.numpy(), em.dev.pool_ids.numpy()
    assert em.dev.world.numpy().tolist() == [0, 1] and not eng._good.whole_arrays     # a per-world snapshot

    def books() -> list[tuple[int, int]]:                    # (cursor, poured) per world
        f, c = flags.numpy(), em.dev.cursor.numpy()
        return [(int(c[e]), int(np.count_nonzero(f[ids[begin[e]:begin[e] + size[e]]] & 1))) for e in range(2)]

    sim.step(5)
    for _ in range(4):                                       # a layer leaves about every 10 ms: one period
        eng._good.save(eng.state)                            # a period start, as save_period_start makes it
        before = books()
        sim.step(1)
        if books()[0][0] > before[0][0]:
            break
    after = books()
    assert all(a[0] > b[0] and a[1] > b[1] for a, b in zip(after, before))
    eng._good.restore(eng.state, eng._mask(np.array([1, 0])))     # a rollback of world 0 only, as _roll_back does
    assert books() == [before[0], after[1]]
    sim.step(60)
    assert books() == [(625, 625)] * 2
    assert sim.state(["water"])["water.active"].sum(axis=1).tolist() == [625, 625]
    sim.close()


@pytest.mark.compile_heavy
def test_two_pours_in_one_world_each_pour_their_own_reservoir():
    """WP9's find, 2026-11-11: two add_pour calls in one world split all its spares evenly, so a 20 ml and a 60 ml
    pour delivered 38 ml each. Law: each emitter pools exactly its own pour's particles, and each pour lands all of
    them under its own nozzle (bound: exact ids and counts)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("a_box", inner=(0.08, 0.08, 0.06), wall=0.005, pos=(-0.12, 0, 0))
    scene.add_open_box("b_box", inner=(0.1, 0.1, 0.08), wall=0.005, pos=(0.12, 0, 0))
    scene.add_pour("a", 6e-6, pos=(-0.12, 0, 0.12), rate=1e-4)               # 94 particles
    scene.add_pour("b", 20e-6, pos=(0.12, 0, 0.12), rate=1e-4)               # 312 particles
    sim = alloy.build(scene, device="cpu")
    em = _liquid.liquid_solvers(sim.engine)[0].emitters
    ids, begin, size = em.dev.pool_ids.numpy(), em.dev.pool_begin.numpy(), em.dev.pool_size.numpy()
    for e, (name, n) in enumerate((("a", 94), ("b", 312))):  # the even split gave each half of all the spares
        r = sim.index.particle_range(name)
        assert ids[begin[e]:begin[e] + size[e]].tolist() == list(range(r.start, r.start + n))
    sim.run(0.5)
    st = sim.state(["a", "b"], world=0)
    for name, side, n in (("a", -1.0, 94), ("b", 1.0, 312)):
        landed = st[f"{name}.points"][st[f"{name}.active"]]
        assert len(landed) == n and bool(np.all(np.sign(landed[:, 0]) == side))
    sim.close()


def test_a_pour_holds_the_asked_volume_within_half_a_particle():
    """Physics ruling 2026-11-11: add_pour's n x n x m block was up to 11 % off the asked volume (20 ml poured 18.8
    ml), and super re-rounded the default block. Law: from 5 ml to 2 l at both liquid accuracies, the pour item as
    built (add_pour, then at_spacing) holds the ask within half a particle, in a block that fits it with sides within
    2x (bound: 0.5 particle)."""
    for accuracy, s in (("default", 4e-3), ("super", 2e-3)):
        for v in np.geomspace(5e-6, 2e-3, 300):
            scene = alloy.Scene(liquid_accuracy=accuracy)
            scene.add_pour("water", float(v), pos=(0, 0, 0.3), rate=2e-4)
            p = _pour.at_spacing(scene, s).items["water"]["pour"]
            (a, b, m), n = _pour.reservoir(float(v), s)
            assert p["count"] == n and abs(n - v / s ** 3) <= 0.5 and p["volume"] == n * s ** 3
            assert a * b * m >= n and max(a, b, m) <= 2 * min(a, b, m)


@pytest.mark.compile_heavy
def test_a_super_pour_pools_its_count_and_liquid_volume_reads_it():
    """The built side of the law above, at super (2 mm): the emitter pools exactly the count, the rest of the block
    is retired at build (never poured, not counted as escaped) and belongs to no name, so "water" is the count's
    particles and sim.liquid_volume reads it (bound: exact)."""
    scene = alloy.Scene(dt=0.01, liquid_accuracy="super")
    scene.add_ground()
    scene.add_pour("water", 5e-6, pos=(0, 0, 0.12), rate=2e-5)               # 625 particles of 2 mm
    sim = alloy.build(scene, device="cpu")
    em = _liquid.liquid_solvers(sim.engine)[0].emitters
    assert em.dev.pool_size.numpy()[:1].tolist() == [625] and len(sim.index.particle_range("water")) == 625
    assert sim.liquid_volume("water") == 625 * sim.particle_volume("water")
    assert sim.health()["retired_particles"] == [0]
    sim.close()


@pytest.mark.compile_heavy
def test_a_tiny_pour_far_off_its_volume_warns(caplog):
    """AUDIT-STACK's nit, 2026-11-11: the build's asked-vs-simulated check compared a pour with its rounded volume, so
    0.1 ml (2 particles, 0.128 ml: +28 %) never warned. Law: the check reads the ask and warns."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_pour("water", 1e-7, pos=(0, 0, 0.12), rate=1e-6)
    with caplog.at_level("WARNING"):
        sim = alloy.build(scene, device="cpu")
    assert any("not the 0.1 ml you asked for" in r.getMessage() for r in caplog.records)
    sim.close()


@pytest.mark.compile_heavy
def test_a_saved_pour_loads_as_a_pour_not_a_falling_block(tmp_path, caplog):
    """API-19 (AUDIT-STACK, 2026-11-11): scene.save() wrote only a pour's reservoir block, so Scene.load gave liquid
    falling from the nozzle at t 0, with no warning. Law: the saved card holds the pour as an emitter and no
    reservoir, the loaded scene pours the same count through the same nozzle at the same rate (nothing active at t 0),
    and save warns that a card keeps no pour name (bound: exact; the ask within 1e-12 of itself)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_pour("water", 20e-6, pos=(0, 0, 0.12), rate=2e-4)
    with caplog.at_level("WARNING"):
        path = scene.save(tmp_path / "cards")
    assert any("<material>_pour_<k>" in r.getMessage() for r in caplog.records)
    loaded = alloy.Scene.load(str(path))
    pours = {n: it["pour"] for n, it in loaded.items.items() if it.get("pour")}
    assert list(pours) == ["water_pour_0"] and "water" not in loaded.items
    a, b = scene.items["water"]["pour"], pours["water_pour_0"]
    for key in ("count", "rate", "half_extents", "velocity"):
        assert b[key] == a[key], key
    assert list(map(float, b["pos"])) == list(map(float, a["pos"]))
    assert abs(b["asked"] - a["asked"]) <= 1e-12 * a["asked"]
    for ml in (50.4, 68.0, 212.0):                           # asks on a half particle: the count survives the trip
        one = alloy.Scene(dt=0.01)
        one.add_pour("water", ml * 1e-6, pos=(0, 0, 0.3), rate=1e-4)
        back = alloy.Scene.load(str(one.save(tmp_path / f"ml{ml}")))
        assert back.items["water_pour_0"]["pour"]["count"] == one.items["water"]["pour"]["count"], ml
    sim = alloy.build(loaded, device="cpu")
    assert int(sim.state(["water_pour_0"])["water_pour_0.active"].sum()) == 0      # a pour, not a block at t 0
    assert len(sim.index.particle_range("water_pour_0")) == 312
    sim.close()
