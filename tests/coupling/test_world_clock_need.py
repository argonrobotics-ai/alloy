# SPDX-License-Identifier: Apache-2.0
"""Gap #10b: each world's own exchange level (alloy/coupling/world_clock_need.py) from per-world bounds, and the
contact band's per-world rows behind its shared maxima (alloy/coupling/contact_band.py). Reading them changes no step:
the shared clock still takes the pack-wide bound."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.coupling import contact_band as CB
from alloy.coupling.world_clock_need import AttemptLevels, ClockGrid, per_world_rungs, saving, shared_level

GAP = 0.003                    # m: the full-surface band
GRID = ClockGrid(control_dt=0.01, dt_x_max=math.inf, j_max=10)
TOP, BOX_KG = 0.05, 1.0        # m, kg: the drop rig's table top and box
FAST, PERIODS = 2, 17          # the world whose box lands inside the run; periods run


# ---------------------------------------------------------------------------- the formula (host only)
def test_each_world_takes_the_level_of_its_own_tighter_bound() -> None:
    """Interface 1/16, 1/16, 1/4 of the period; band 1/64 in world 1 only: levels 4, 6, 2, bound by the interface, the
    band and the interface. The shared clock would run all three at 6: the worlds' own levels run 84 of 192."""
    need = per_world_rungs(np.array([0.01 / 16, 0.01 / 16, 0.01 / 4]), np.array([math.inf, 0.01 / 64, math.inf]), GRID)
    assert need.j.tolist() == [4, 6, 2] and need.terms() == ["interface", "band", "interface"]
    assert shared_level(need) == 6 and saving(need) == pytest.approx((16 + 64 + 4) / (3 * 64), rel=1e-12, abs=0)


def test_the_floor_the_attempts_halvings_and_its_cap_apply_to_every_world() -> None:
    bounds = np.array([0.01 / 16, 0.01 / 2, math.inf])
    floor = per_world_rungs(bounds, math.inf, GRID, AttemptLevels(floor=3))
    assert floor.j.tolist() == [4, 3, 3] and floor.terms() == ["interface", "floor", "floor"]
    extra = per_world_rungs(bounds, math.inf, GRID, AttemptLevels(floor=3, extra=2, j_attempt=5))
    assert extra.j.tolist() == [5, 5, 5] and set(extra.terms()) == {"attempt"}       # 4 + 2 capped at the attempt's 5
    free = per_world_rungs(bounds, math.inf, GRID)
    assert free.j.tolist() == [4, 1, 0] and free.terms()[2] == "none"


def test_a_scalar_band_is_the_shared_bound_for_every_world() -> None:
    """Without per-world rows the band binds every world alike, as the census counted it before."""
    need = per_world_rungs(np.array([0.01 / 4, math.inf]), 0.01 / 32, GRID)
    assert need.j.tolist() == [5, 5] and need.terms() == ["band", "band"] and saving(need) == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_the_finest_world_is_the_shared_level(seed: int) -> None:
    """max over worlds of j_world is the level of the minimum bound (what ``ClockController.decide`` grids)."""
    from alloy.coupling.clocks import grid_level

    rng = np.random.default_rng(seed)
    interface, band = 10.0 ** rng.uniform(-5, -2, 16), 10.0 ** rng.uniform(-5, -2, 16)
    need = per_world_rungs(interface, band, GRID, AttemptLevels(floor=int(rng.integers(0, 4))))
    j_shared = grid_level(float(min(interface.min(), band.min())), GRID.control_dt, GRID.j_max)[0]
    assert shared_level(need) == j_shared                         # every floor drawn here is below the bound's level
    assert 0.0 < saving(need) <= 1.0


# ---------------------------------------------------------------------------- the band's per-world rows (stand-ins)
def _route(qd, indices, body_vel, count, world, mass):
    contacts = SimpleNamespace(
        soft_contact_max=len(indices), soft_contact_count=wp.array([count], dtype=wp.int32, device="cpu"),
        soft_contact_indices=wp.array(np.asarray(indices, np.int32), dtype=wp.vec3i, device="cpu"),
        soft_contact_body_vel=wp.array(np.asarray(body_vel, np.float32), dtype=wp.vec3, device="cpu"))
    factory = SimpleNamespace(built=[(None, SimpleNamespace(soft_contact_gap=GAP), contacts, None)])
    entries = {"cloth": SimpleNamespace(
        view=SimpleNamespace(particle_world=wp.array(np.asarray(world, np.int32), dtype=wp.int32, device="cpu")),
        state_1=SimpleNamespace(particle_qd=wp.array(np.asarray(qd, np.float32), dtype=wp.vec3, device="cpu")),
        solver=SimpleNamespace(model=SimpleNamespace(
            particle_mass=wp.array(np.asarray(mass, np.float32), dtype=float, device="cpu"))))}
    return factory, entries


def _three_worlds():
    """World 0: a record at 1 m/s relative, and a free particle at 2 m/s. World 1: no record, a particle at 5 m/s.
    World 2: a record at 0.25 m/s. A record past the count (world 1, 50 m/s relative) is ignored."""
    qd = [(3.0, 0, 0), (0, 2.0, 0), (0, 0, 5.0), (0.25, 0, 0), (0, 0, 0)]
    world = [0, 0, 1, 2, 1]
    indices = [(0, -1, -1), (3, -1, -1), (4, -1, -1)]
    body_vel = [(2.0, 0, 0), (0, 0, 0), (50.0, 0, 0)]
    return _route(qd, indices, body_vel, count=2, world=world, mass=np.ones(5))


def test_the_bands_per_world_rows_sit_behind_the_same_shared_maximum() -> None:
    """Law (ii) on stand-ins. The shared maximum is 5 m/s, world 1's free particle (the pipeline holds a record, so
    every particle counts: today's rule). World by world under that rule: 3, 5, 0.25 m/s, whose maximum is the shared
    one bit for bit. Each world's own need: world 0 its fastest particle (it holds a record), world 1 nothing (no
    record of its own), world 2 its record's 0.25 m/s."""
    factory, entries = _three_worlds()
    band = CB.ContactBandBound([(factory, "cloth")], "cpu")
    band.keep_world_rows()
    band.ensure_worlds(3)
    band.accumulate(entries)
    bound = band.end_period()
    rule, need = band.world_speed(need=False)[0], band.world_speed(need=True)[0]
    assert np.float32(rule.max()) == np.float32(band.last_speed[0]) == np.float32(5.0)
    assert rule.tolist() == [3.0, 5.0, 0.25] and need.tolist() == [3.0, 0.0, 0.25]
    assert float(band.world_bounds(need=False).min()) == bound == CB.band_bound(GAP, 5.0)      # the shared bound
    assert band.world_bounds(need=True).tolist() == [CB.band_bound(GAP, 3.0), math.inf, CB.band_bound(GAP, 0.25)]
    assert float(band.vmax.numpy().max()) == 0.0                              # the read clears every row
    assert band.end_period() == math.inf and band.world_bounds().tolist() == [math.inf] * 3


def test_the_shared_maximum_is_the_same_with_and_without_the_rows() -> None:
    """The rows are extra writes, kept only when asked (``keep_world_rows``): the shared maximum and bound are bit for
    bit those of a band without them (never asked, with or without an ``ensure_worlds``), and a band without rows
    answers None. Off is today's work: the array is one slot per route and the kernels get no row to write."""
    out = []
    for rows in (True, False, None):
        factory, entries = _three_worlds()
        band = CB.ContactBandBound([(factory, "cloth")], "cpu")
        if rows:
            band.keep_world_rows()
        if rows is not None:
            band.ensure_worlds(3)
        views = (entries["cloth"].view, None)
        assert (band._world_inputs(views, 0, 0, 5)[3] == CB.NO_ROWS) == (not rows)
        assert int(band.vmax.shape[0]) == (1 + 3 * 3 if rows else 1)
        band.accumulate(entries)
        out.append((band.end_period(), band.last_speed.tobytes(), band.worlds, band.world_bounds() is None))
    assert out[0][:2] == out[1][:2] == out[2][:2]
    assert [o[2:] for o in out] == [(3, False), (0, True), (0, True)]


def test_a_parked_world_writes_no_row_and_no_shared_speed() -> None:
    """One signature for the rows and the parked mask: world 1 parked, its 5 m/s particle leaves the shared maximum
    (3 m/s, world 0's) and its own rows stay zero; the other worlds' rows are as unparked."""
    factory, entries = _three_worlds()
    band = CB.ContactBandBound([(factory, "cloth")], "cpu", 3)
    band.keep_world_rows()
    band.ensure_worlds(3)
    band.park(np.array([False, True, False]))
    band.accumulate(entries)
    assert band.end_period() == CB.band_bound(GAP, 3.0)
    assert band.world_speed(need=False)[0].tolist() == [3.0, 0.0, 0.25]
    assert band.world_speed(need=True)[0].tolist() == [3.0, 0.0, 0.25]


def test_a_route_without_a_particle_world_map_keeps_only_the_shared_maximum() -> None:
    factory, entries = _three_worlds()
    entries["cloth"].view = None
    band = CB.ContactBandBound([(factory, "cloth")], "cpu")
    band.keep_world_rows()
    band.ensure_worlds(3)
    band.accumulate(entries)
    assert band.end_period() == CB.band_bound(GAP, 5.0) and not band.world_speed().any()


# ---------------------------------------------------------------------------- the drop rig (CPU, Warp route)
def _scene(drops) -> alloy.Scene:
    """A 1 kg steel-like box above a sheet on a table, one drop height [m] per world."""
    side = (BOX_KG / 8000.0) ** (1.0 / 3.0)
    scene = alloy.Scene(worlds=len(drops), dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, TOP), pos=(0, 0, TOP / 2), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, TOP + 0.002), resolution=(11, 11), mass=0.03,
                    thickness=0.004, self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=np.array([[0, 0, TOP + 0.004 + d + side / 2] for d in drops]),
                  mass=BOX_KG)
    return scene


def _run(drops, periods: int, world_rows: bool = True) -> dict:
    sim = alloy.build(_scene(drops), device="cpu")
    st = getattr(sim.engine.stepper, "base", sim.engine.stepper)
    st.keep_world_rows(world_rows)                  # the rows are off unless asked
    rows = []
    for _ in range(periods):
        sim.step(1)
        rep, band = sim.engine._last_coupling, st._band
        rows.append({"dt_x": rep.dt_x, "census": dict(rep.notes["census"]), "shared": band.last_speed.copy(),
                     "rule": band.world_speed(need=False).copy(), "need": band.world_speed(need=True).copy(),
                     "bound": st._band_bound})
    state = sim.state()
    return {"rows": rows, "state": {k: np.asarray(v).tobytes() for k, v in state.items()},
            "rollbacks": sim.engine.rollbacks, "ok": sim.check()["ok"]}


@pytest.fixture(scope="module")
def staggered():
    """Four worlds; world FAST's box falls 10 cm and lands inside the run, the others' 50 cm and do not."""
    return _run([0.5, 0.5, 0.1, 0.5], PERIODS)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1700)
def test_the_rows_change_no_step(staggered) -> None:
    """Law (i): with the per-world rows off, the default (the old arrays and kernel writes), the same run ends in the
    same bits and took the same dt_x every period; its census says it read the shared band."""
    plain = _run([0.5, 0.5, 0.1, 0.5], PERIODS, world_rows=False)
    assert {r["census"]["band_rule"] for r in plain["rows"]} == {"shared"}
    assert [r["dt_x"] for r in plain["rows"]] == [r["dt_x"] for r in staggered["rows"]]
    assert [r["bound"] for r in plain["rows"]] == [r["bound"] for r in staggered["rows"]]
    assert all(r["shared"].tobytes() == s["shared"].tobytes() for r, s in zip(plain["rows"], staggered["rows"]))
    assert plain["state"] == staggered["state"] and plain["rollbacks"] == staggered["rollbacks"]
    assert not any(r["rule"].size for r in plain["rows"])                      # off: no rows were kept


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(900)
def test_only_the_landing_world_needs_the_finer_level(staggered) -> None:
    """Laws (ii) and (iii). Every period the maximum over worlds of the per-world speeds (the shared rule) is the
    shared maximum bit for bit. Once world FAST's box lands, the band binds the shared clock finer than the interface
    level; the census gives that level to world FAST alone (term ``band``), the others keep the interface's."""
    assert staggered["ok"]
    for r in staggered["rows"]:
        assert np.array_equal(r["rule"].max(axis=1).astype(np.float32), r["shared"].astype(np.float32))
    base = staggered["rows"][0]["census"]["j_world"][0]
    landed = [r["census"] for r in staggered["rows"] if r["census"]["term_world"][FAST] == "band"]
    assert landed, "world FAST's band never bound its level"
    finest = max(landed, key=lambda c: c["j_world"][FAST])
    others = [j for w, j in enumerate(finest["j_world"]) if w != FAST]
    print(f"interface level {base}; at its finest world {FAST} needs {finest['j_world'][FAST]}, the others {others}; "
          f"per-world exchanges {finest['per_world_exchanges']} of {finest['world_exchanges']}")
    assert finest["j_world"][FAST] > base and others == [base] * 3
    assert all(t != "band" for w, t in enumerate(finest["term_world"]) if w != FAST)
    assert finest["per_world_exchanges"] < finest["world_exchanges"] and finest["drift"] is False
    need = np.stack([r["need"][0] for r in staggered["rows"]])
    assert need[:, [0, 1, 3]].max() < 0.01 and need[:, FAST].max() > 1.0       # m/s: the others only settle (period 0)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(900)
def test_identical_worlds_all_need_the_shared_level() -> None:
    """Law (iv): two identical worlds, both landing: every period both worlds' levels are equal, and equal the level
    the shared clock ran (no saving to claim)."""
    run = _run([0.1, 0.1], PERIODS)
    seen_band = False
    for r in run["rows"]:
        c = r["census"]
        j_ran = int(round(math.log2(0.01 / r["dt_x"])))
        assert c["j_world"][0] == c["j_world"][1] and c["drift"] is False
        assert c["per_world_exchanges"] <= c["world_exchanges"]
        seen_band = seen_band or "band" in c["term_world"]
        if c["term_world"][0] in ("interface", "band", "attempt"):
            assert c["j_world"][0] <= j_ran
    assert seen_band and run["ok"]
