# SPDX-License-Identifier: Apache-2.0
"""The film-pool exchange (supply_pool.py) on the stub film, CPU.

- A pool falls past a vertical plate: the film under its water joins the pool. Each row that leaves the water keeps
  the Landau-Levich film for its exit speed, sqrt(2·g·depth). Every m³ is booked, and no cell stays flooded.
- A static cup half under a pool's water: the flooded cells are exactly those whose centre is in the water.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow.charts import PlaneChart, RevolutionChart
from alloy.plugins.surface_flow.supply import Pool
from alloy.plugins.surface_flow.supply_stub import StubFilm

G = 9.81
RHO, NU, SIGMA = 998.0, 1.0e-6, 0.0728
S45 = math.sqrt(0.5)
LEVEL0, H0 = 0.15, 1.0e-4                      # the falling pool's surface (world z) at the start; the plate's film


L_C = math.sqrt(SIGMA / (RHO * G))                     # capillary length of water


def _lld(u: np.ndarray) -> np.ndarray:
    """Landau-Levich, 0.94·l_c·Ca^(2/3) with Ca = μU/σ, bounded by the gravity-viscous film sqrt(μU/(ρg))."""
    return np.minimum(0.94 * L_C * (RHO * NU * u / SIGMA) ** (2 / 3), np.sqrt(RHO * NU * u / (RHO * G)))


def _sink(volume: float, area: float = 0.16) -> dict:
    return {"nodes": [{"name": "sink", "medium": "tank", "volume": volume, "area": area}], "edges": []}


def _falling_pool(b) -> None:
    """A static vertical plate (box top turned to face -y: chart rows along x, columns up z from 0 to 0.2 m) and a
    free basin body whose pool (floor 0.3 m below it, surface 0.25 m above the floor) falls past the plate."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.1, (S45, 0.0, 0.0, S45)), hx=0.1, hy=0.1, hz=0.005, label="wall")
    basin = b.add_body(xform=body_xform(0.0, 0.1, LEVEL0 + 0.05), label="basin")
    b.add_shape_sphere(basin, radius=0.01)


def test_a_falling_pool_takes_the_film_and_leaves_landau_levich_films():
    pool = Pool("sink", floor=(0.0, 0.0, -0.3), half=(0.2, 0.2), body="basin", flood=True)
    p = StubFilm(charts=(PlaneChart("wall", cells=(10, 40)),), network={**_sink(0.25 * 0.16), "pools": [pool]})

    def init(sim) -> None:
        p.h.assign(np.full(p.h.shape, H0))
    sim = build_simulation(TestScene(build=_falling_pool, init=init, ground=False, worlds=2), [p], "cpu")
    sim.step_period()
    flooded = p.flooded.numpy()[0].reshape(10, 40)
    z = (np.arange(40) + 0.5) * 0.005                                     # column centres, world z
    under = z < LEVEL0
    assert (flooded[:, under] == 0).all() and (flooded[:, ~under] == -1).all()
    for _ in range(11):                                                    # 0.24 s: the surface falls below z = 0
        sim.step_period()
    r = p.supply.read()
    assert (p.flooded.numpy() == -1).all()                                 # nothing stays flooded
    h = p.h.numpy()[0].reshape(10, 40)
    assert np.allclose(h[:, ~under], H0, atol=0.0)                                   # never under: the film stays
    want = _lld(np.sqrt(2.0 * G * (LEVEL0 - z[under])))
    got = h[:, under]
    deep = z[under] < LEVEL0 - 0.01                                        # a row near the start line exits at ~0
    assert np.allclose(got[:, deep], want[deep], rtol=0.03, atol=0.0), (got[0, deep][:5], want[deep][:5])
    area = p.tables.area_np
    taken = H0 * area.sum() * under.mean()
    assert np.allclose(r["pooled"], taken, rtol=1e-12, atol=0.0)                     # the plate's film under the water
    assert np.allclose(r["withdrawn"], p.supplied.numpy().sum(axis=1), rtol=1e-12, atol=0.0)
    assert np.allclose(p.pooled_total.numpy().sum(axis=1), r["pooled"], rtol=1e-12, atol=0.0)
    v = r["amount"][:, 0]                                                  # the tank: V0 + pooled - withdrawn
    assert np.allclose(v, 0.25 * 0.16 + r["pooled"] - r["withdrawn"], rtol=1e-12, atol=0.0)


CUP = ((0.006, 0.0), (0.035, 0.0), (0.035, 0.14), (0.038, 0.14), (0.038, 0.0))     # floor, wall, rim, outside


def test_a_cup_half_under_water_floods_exactly_the_cells_in_the_water():
    """A static cup (its frame at z = 0.05) in a static pool whose surface is at z = 0.12 (inside and outside both
    flood below it: part 1 has no cavities)."""
    cup = RevolutionChart(profile=CUP, n_theta=24, ds=2e-3, frame=(0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0))
    pool = Pool("sink", floor=(0.0, 0.0, 0.0), half=(0.2, 0.2), flood=True)
    p = StubFilm(charts=(PlaneChart("slab", cells=(4, 4)), cup), network={**_sink(0.12 * 0.16), "pools": [pool]})

    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(1.0, 0.0, 0.01), hx=0.1, hy=0.1, hz=0.01, label="slab")
        dummy = b.add_body(xform=body_xform(-1.0, 0.0, 0.5), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)
    sim = build_simulation(TestScene(build=build, ground=False, worlds=2), [p], "cpu")
    sim.step_period()
    cells = p.tables.cells_of(1)
    zc = p.tables.cpos.numpy()[cells][:, 2] + 0.05
    flooded = p.flooded.numpy()[:, cells]
    assert ((flooded == 0) == (zc <= 0.12)[None, :]).all()
    assert (p.flooded.numpy()[:, p.tables.cells_of(0)] == -1).all()       # the slab is off the pool's footprint


def _flat_plate(b) -> None:
    """A static horizontal plate (100 x 100 mm, top at z = 0.1) and the falling basin."""
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.095), hx=0.05, hy=0.05, hz=0.005, label="plate")
    basin = b.add_body(xform=body_xform(0.0, 0.0, LEVEL0 + 0.05), label="basin")
    b.add_shape_sphere(basin, radius=0.01)


@pytest.mark.parametrize("angle", [None, 40.0])
def test_a_horizontal_plate_leaving_the_water_all_at_once_is_reset(angle):
    """A static horizontal plate (top at z = 0.1) under the falling pool's surface: the whole face leaves the water in
    one exchange, so the chart drops out of the pool's box at once. Only last exchange's box resets it. It moves
    along its normal, so it drags no Landau-Levich film: without a contact angle it comes out dry. With one (40°) it
    keeps the static puddle 2·l_c·sin(θ/2), well under the 50 mm of water it had over it."""
    pool = Pool("sink", floor=(0.0, 0.0, -0.3), half=(0.2, 0.2), body="basin", flood=True)
    p = StubFilm(charts=(PlaneChart("plate", cells=(8, 8), contact_angle=angle),),
                 network={**_sink(0.25 * 0.16), "pools": [pool]})
    sim = build_simulation(TestScene(build=_flat_plate, ground=False, worlds=2), [p], "cpu")
    sim.step_period()
    assert (p.flooded.numpy() == 0).all()
    for _ in range(7):                                                     # the surface passes z = 0.1 at 0.10 s
        sim.step_period()
    assert (p.flooded.numpy() == -1).all()
    want = 0.0 if angle is None else 2.0 * L_C * math.sin(math.radians(angle) / 2.0)
    assert np.allclose(p.h.numpy(), want, rtol=1e-5, atol=1e-12), (p.h.numpy().min(), p.h.numpy().max(), want)


def _film_run(skip: bool, periods: int = 8) -> dict:
    """WP15's film on a 96 x 96 horizontal plate (6 x 6 tiles, top at z = 0.1) under the falling pool: a trace film
    (below H_DRY: its tiles are dry and skipped) on one side and a puddle on the other. The pool's surface passes
    the plate at 0.10 s; the plate (30° contact angle) keeps its puddle."""
    from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415

    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.095), hx=0.096, hy=0.096, hz=0.005, label="plate")
        basin = b.add_body(xform=body_xform(0.0, 0.0, LEVEL0 + 0.05), label="basin")
        b.add_shape_sphere(basin, radius=0.01)
    pool = Pool("sink", floor=(0.0, 0.0, -0.3), half=(0.2, 0.2), body="basin", flood=True)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(96, 96), contact_angle=30.0),), material="water",
                       skip_dry=skip,
                       network={**_sink(0.25 * 0.16), "pools": [pool]})

    def init(sim) -> None:
        x = flow.cell_centres(0)[..., 0]
        flow.set_depth(np.where(x < -0.05, 5.0e-7, np.where(x > 0.07, 5.0e-4, 0.0)), chart=0)
    sim = build_simulation(TestScene(build=build, init=init, ground=False, worlds=2), [flow], "cpu")
    for _ in range(periods):
        sim.step_period()
    return flow.read()


def test_the_real_film_gives_its_water_to_the_pool_exactly_and_skips_exactly():
    """On WP15's film: the dry-tile skip matches the dense step bit for bit with flooding (a flooded trace film in a
    skipped tile is marked wet, so it reaches the pool). The film's pooled volume is what the tanks got, and the cross
    ledger closes with the withdrawal films."""
    dense, skip = _film_run(False), _film_run(True)
    for k in ("h", "u", "q", "ledger", "pooled_total_volume", "supply_pooled", "supply_withdrawn", "pending_volume"):
        assert np.array_equal(dense[k], skip[k]), (k, np.abs(dense[k] - skip[k]).max())
    r = skip
    assert (r["supply_pooled"] > 0).all() and (r["supply_withdrawn"] > 0).all()
    assert np.allclose(r["pooled_total_volume"], r["supply_pooled"], rtol=1e-12, atol=0.0)
    cross = r["supply_sent"] + r["supply_swept"] + r["supply_withdrawn"] - (
        r["supplied_volume"] + r["pending_volume"] + r["supply_to_tank"] + r["supply_lost"] + r["supply_overspray"]
        + r["supply_mist"] + r["supply_caught"] + r["supply_lumped"])
    assert np.abs(cross).max() <= 1e-12 * r["supply_withdrawn"].max(), cross
    v = r["supply_amount"][:, 0]                                           # the tank: V0 + pooled - withdrawn
    assert np.allclose(v, 0.25 * 0.16 + r["supply_pooled"] - r["supply_withdrawn"], rtol=1e-12, atol=0.0)


def test_surface_flow_with_a_flooding_pool_passes_the_harness_hooks_on_cpu():
    """Contract, crash, determinism, identical and isolated worlds, graph: a static pool 2.5 mm over the check scene's
    plate (top at z = 0) drains through a wide waste orifice, so the plate floods and comes out during the run."""
    import alloy.plugins as ap  # noqa: PLC0415
    from alloy.plugins.fluid import P_ATM  # noqa: PLC0415
    from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415

    net = _sink(0.0525 * 0.16)
    net["edges"] = [{"name": "waste", "kind": "orifice", "a": "sink", "reservoir": P_ATM, "cd_area": 4e-3,
                     "dp_lam": 1.0}]
    net["pools"] = [Pool("sink", floor=(0.0, 0.0, -0.05), half=(0.2, 0.2), flood=True)]
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50), edges=("open",) * 4),), material="water",
                       network=net)
    report = ap.check(flow, device="cpu", quick=True, laws=False)
    assert report.passed, report.summary()


def test_a_pool_short_of_water_grants_what_its_tank_holds():
    """The falling pool with a tank of 0.1 cm² (2.5 mL for its 0.25 m depth) under the flat plate with a 40° contact
    angle: the plate's puddle (1.86 mm over 100 cm², 18.6 mL) is asked for in one exchange, so the pool grants what its
    tank holds and books the rest as refused. Its tank ends empty, never below; what it gave is what the film got."""
    pool = Pool("sink", floor=(0.0, 0.0, -0.3), half=(0.2, 0.2), body="basin", flood=True)
    v0 = 0.25 * 1.0e-5
    p = StubFilm(charts=(PlaneChart("plate", cells=(8, 8), contact_angle=40.0),),
                 network={**_sink(v0, area=1.0e-5), "pools": [pool]})
    sim = build_simulation(TestScene(build=_flat_plate, ground=False, worlds=2), [p], "cpu")
    for _ in range(8):
        sim.step_period()
    r = p.supply.read()
    v = r["amount"][:, 0]
    asked = 2.0 * L_C * math.sin(math.radians(20.0)) * 0.01                # the puddle over the plate
    assert (v >= 0.0).all() and np.allclose(v, 0.0, atol=1e-15), v
    assert np.allclose(r["withdrawn"], v0, rtol=1e-12, atol=0.0)
    assert np.allclose(r["withdrawn"] + r["refused"], asked, rtol=1e-5, atol=0.0)
    assert np.allclose(p.supplied.numpy().sum(axis=1), v0, rtol=1e-12, atol=0.0)
