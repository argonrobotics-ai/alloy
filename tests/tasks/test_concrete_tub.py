# SPDX-License-Identifier: Apache-2.0
"""The pad tasks' tub fill (alloy.tasks.concrete_tub): the scripted policy finds each seed's low corner where their
start puts it, takes its carry cap from the tub and lays it on the low's east flank (CPU, fast)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import concrete_pad as CP
from alloy.tasks import concrete_tub as TUB
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR


def _sw(f):
    ix, iy = TUB.interior(f)
    return ix - INTERIOR[0] / 2, iy - INTERIOR[1] / 2


def test_the_low_spot_is_where_their_start_puts_the_low_corner():
    """Even seeds: within (0.15-0.25, 0.15-0.25) m of the south-west corner; odd: (1.019, 0.18) m (layout.py:248)."""
    task = CP.PadDarby(factor=5)
    for seed in (0, 2):
        f = CP.start(task, seed)
        x, y, deficit = TUB.low_spot(f)
        u, v = x - _sw(f)[0], y - _sw(f)[1]
        assert 0.12 < u < 0.28 and 0.12 < v < 0.28, (seed, u, v)
        assert 1.0e-4 < deficit < 4.0e-4                                    # a 5 mm, sigma 80 mm dip: ~0.2 L
    f = CP.start(task, 1)
    x, y, _ = TUB.low_spot(f)
    assert abs(x - _sw(f)[0] - (INTERIOR[0] - 0.2)) < 0.03 and abs(y - _sw(f)[1] - 0.18) < 0.03


def test_the_lifted_blade_keeps_their_stick_limit_and_trips_cover_the_low():
    """GA's carrying knife keeps 0.5·A·tau/(rho·g·6) lifted (coupling.py:933-936): 3.83e-5 m³ of 5 min concrete
    (their tau_b 1641 Pa and density) on the 127 x 50.8 mm blade, under the 1.5e-4 m³ carry cap (WP15's numbers)."""
    from alloy.tasks import concrete_materials as CM

    rho = CM.Concrete().rho
    assert TUB.gob(1641.0, rho) == pytest.approx(0.5 * 0.127 * 0.0508 * 1641.0 / (rho * 9.81 * 6.0), rel=1e-12)
    assert TUB.gob(1641.0, rho) == pytest.approx(3.83e-5, rel=2e-3) and TUB.gob(1e9, rho) == 1.5e-4
    store = TUB.store()
    assert (store.capacity, store.split, store.drip, store.held, store.scrape, store.fill) == (1.5e-4, 1.0, 0.0, 1.0,
                                                                                               True, "knife")
    assert store.stick_area == pytest.approx(0.127 * 0.0508, rel=1e-12)
    assert TUB.loads(0.0, 3.8e-5) == 1 and TUB.loads(2.2e-4, 3.8e-5) == 6 and TUB.loads(1.0, 3.8e-5) == TUB.MAX_LOADS


def test_the_trowel_dips_in_the_tub_and_lets_go_over_the_deepest_deficit():
    f = CP.start(CP.PadDarby(factor=5), 0)
    x0, y0, _ = TUB.low_spot(f)
    points = TUB.drops(f, 6, 3.83e-5)
    assert TUB.cone_radius(3.83e-5, 0.01) == pytest.approx(0.04) and TUB.cone_radius(1e-6, 0.01) == pytest.approx(0.02)
    m = np.asarray(f["paste_mask"]) > 0.5
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    near = m & (np.hypot(X - x0, Y - y0) < TUB.LOW_RADIUS)
    deepest = np.unravel_index(np.argmax(np.where(near, -np.asarray(f["h"]), -np.inf)), X.shape)
    assert points[0] == (float(f["x"][deepest[0]]), float(f["y"][deepest[1]]))      # the deepest cell, unsmoothed
    assert all(np.hypot(x - x0, y - y0) < TUB.LOW_RADIUS for x, y in points) and len(set(points)) == len(points)
    keys = TUB.trowel_program(f, points).keys
    tray = TUB.tub(f)
    level = tray.center[2] + TUB.TUB_VOLUME / (4 * tray.half[0] * tray.half[1])
    under = TUB.TROWEL["half"][2]
    dips = [k for k in keys if k.pos[2] - under < level]
    assert len(dips) == 2 * len(points)
    assert all(abs(k.pos[0] - tray.center[0]) < tray.half[0] and abs(k.pos[1] - tray.center[1]) < tray.half[1]
               for k in dips)                                               # in the heap, inside the tub
    lets = [k for k in keys if k.release > 0.0]
    assert [k.pos[:2] for k in lets] == points and all(k.pos[2] == FORM_H + TUB.DROP_H + under for k in lets)
    assert keys[0].pos == keys[-1].pos == TUB.home(f)
    ix, iy = TUB.interior(f)
    assert tray.center[:2] == pytest.approx((ix + 1.40, iy + 1.34), abs=1e-12)          # where theirs stands


def test_one_trowel_trip_books_the_tub_the_blade_and_the_film_exactly():
    """One trip on a small slab: the blade picks up toward the cap in the tub, sheds what its stick limit can't hold
    back into the tub's pool on lifting, and lets the rest go over the slab, where it lands as film. Tray, store,
    tank and film books close to 1e-12."""
    from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
    from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
    from alloy.plugins.surface_flow.applicators import Applicator
    from alloy.tasks.concrete_tools import ToolCarriage, add_tool_body

    f = {"x": np.array([-0.1, 0.1]), "y": np.array([-0.1, 0.1]), "paste_mask": np.ones((2, 2))}
    tray, net = TUB.tub(f), TUB.network(f)
    from alloy.tasks import concrete_materials as CM

    rho = CM.Concrete().rho
    film = FilmMaterial("concrete_paste", density=rho, viscosity=5.0, yield_stress=1641.0, surface_tension=0.0)
    trowel = Applicator(body="mtrowel", shape="rect", size=TUB.TROWEL["half"][:2],
                        center=(0.0, 0.0, -TUB.TROWEL["half"][2]), stiffness=1e5, implicit=True, store=TUB.store())
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(30, 30)),), material=film, inertia=False, max_depth=0.5,
                       tools=(trowel,), tool_speed=0.2, trays=(tray,), network=net)
    prog = TUB.trowel_program(f, [(0.0, 0.0)], speed=3.0)                # fast flights: a short test
    carriage = ToolCarriage(programs=(prog,), flow=flow)

    def build(b):
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, FORM_H - 0.025), hx=0.15, hy=0.15, hz=0.025, label="slab")
        add_tool_body(b, "mtrowel", TUB.TROWEL["half"], TUB.TROWEL["mass"], prog.keys[0].pos, collide=False)

    sim = build_simulation(TestScene(build=build, ground=False, worlds=1, dt=2.0e-3), [carriage, flow], "cpu")
    for _ in range(int(prog.duration / 0.02) + 10):
        sim.step_period()
    r = flow.read()
    picked, released = float(r["tool_picked"].sum()), float(r["tool_released"].sum())
    shed, landed = float(r["supply_to_tank"][0]), float(r["supplied_volume"][0])
    stick = TUB.gob(1641.0, rho)
    assert picked >= 1.5e-4 * (1 - np.exp(-5.0 * TUB.DIP_S)) * 0.98   # (in the tub the stick rule sheds as it picks:
                                                                       # WP15, 11-07; the books hold either way)
    assert released == pytest.approx(stick, rel=1e-6, abs=0)           # (the engine's yield and density: float32)
    assert landed == pytest.approx(released, rel=1e-12, abs=0)
    assert shed == pytest.approx(picked - released, rel=1e-12, abs=0)           # back into the tub's pool
    assert float(r["film_volume"][0]) == pytest.approx(landed, rel=1e-12, abs=0) and float(r["supply_lost"][0]) == 0
    assert abs(float(r["store_ledger"][0])) <= 1e-12 * picked
    assert float(r["tray_ledger"][0]) == pytest.approx(TUB.TUB_VOLUME, rel=1e-12, abs=0)
