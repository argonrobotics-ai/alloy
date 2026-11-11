# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped bodies: the chart under a body along gravity, and drips that start in a pool's water
(lumped_drip.py; WP15's third review of the lumped films: probes q8, q12, q5b and the M3, M4 mutants). CPU, harness."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, RevolutionChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_specs import Lumped
from alloy.plugins.surface_flow.supply import Pool

R, DT, G, RD = 0.014, 2.0e-3, (0.0, 0.0, -9.81), 1.5e-3
DROP = 4.0 / 3.0 * math.pi * RD ** 3
HOLD = 5.0e-5 * 0.01                                     # Lumped(area=0.01): its clinging film


def _fixed(b, label: str, xf) -> int:
    body = b.add_link(xform=xf, label=label)
    b.add_articulation([b.add_joint_fixed(-1, body, parent_xform=xf)])
    return body


def _dummy(b) -> None:
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(d, radius=0.01)


def _sim(flow: SurfaceFlow, build, init, worlds: int = 1):
    return build_simulation(TestScene(build=build, init=init, ground=False, gravity=G, worlds=worlds, dt=DT,
                                      control_dt=DT), [flow], "cpu")


def _soap(flow: SurfaceFlow, q: float, c: float):
    """A stream of q m³ per exchange at c kg/m³ of soap into every lumped body."""
    def init(sim) -> None:
        inp = flow.lumped.inputs
        inp.vol.fill_(q)
        sp = inp.sp.numpy().copy()
        sp[..., 0] = c * q
        inp.sp.assign(sp)
    return init


# --- q8: a box resting on a cup's floor with its lowest corner by the cup's vertical wall ---------------------------

BOX = (0.01, 0.01, 0.005)
CUP = ((0.012, 0.0), (0.06, 0.0), (0.06, 0.06))           # floor out to r 60 mm, then a vertical wall
TILT = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -1.0, 0.0)), -0.05)    # the (+x, +y, -z) corner dips


def _cup_corner(b) -> None:
    rot = np.array(wp.quat_to_matrix(TILT), np.float64).reshape(3, 3)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64) * BOX
    low = corners[np.argmin((corners @ rot.T)[:, 2])] @ rot.T
    rc = (0.06 - 2.0e-4) / math.sqrt(2.0)                  # that corner 0.2 mm off the wall, 0.5 mm over the floor
    xf = wp.transform(wp.vec3(rc - low[0], rc - low[1], 5.0e-4 - low[2]), TILT)
    b.add_shape_box(_fixed(b, "brick", xf), hx=BOX[0], hy=BOX[1], hz=BOX[2])
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_box_resting_by_a_cup_wall_wets_the_cup_floor():
    """WP15's must-fix (probe q8): the chart under a body is found along gravity. A box resting 0.5 mm over a cup's
    floor with its lowest corner 0.2 mm off the cup's vertical wall (2 mm cells, 2.7 mm drops: the wall is nearer
    than the floor along the normal) gives its shed soapy water to the floor cell under it: no drops, none floored,
    shed = wet + waiting and its soap = the wet soap + the waiting soap (1e-12), at the stream's concentration
    (1e-12). Before: nothing wet, every drop started under the floor."""
    flow = SurfaceFlow(charts=(RevolutionChart(profile=CUP, n_theta=24, ds=2.0e-3),), material="water",
                       max_depth=5.0e-3, species=("soap",), drops={"drop_radius": 2.7e-3, "floor_z": -0.5},
                       lumped=(Lumped("brick", area=0.01),))
    q, c = 1.0e-7, 4.0
    sim = _sim(flow, _cup_corner, _soap(flow, q, c))
    for _ in range(10):
        sim.step_period()
    r, d = flow.lumped.read(), flow.drops.read()
    shed, wet = r["lump_shed"][0, 0], d["lump_wet_volume"][0]
    assert shed > 0.0 and wet > 0.0 and d["lump_drops_volume"][0] == 0.0 and d["floored_volume"][0] == 0.0
    assert shed == pytest.approx(wet + d["lump_pending_volume"][0], rel=1e-12, abs=0.0)
    sp_wet = r["lump_species_wet"][0, 0, 0]
    assert r["lump_species_shed"][0, 0, 0] == pytest.approx(sp_wet + r["lump_species_drip_pending"][0, 0, 0],
                                                            rel=1e-12, abs=0.0)
    assert sp_wet == pytest.approx(c * wet, rel=1e-12, abs=0.0)


# --- q12: a burst of drips over a steep board near its upper edge ----------------------------------------------------

TH, HX, HZ, ABOVE = math.radians(60.0), 0.1, 0.005, 0.04
ROT60 = np.array(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), TH)), np.float64).reshape(3, 3)


def _steep_board(b) -> None:
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), TH)),
                    hx=HX, hy=0.1, hz=HZ, label="board")
    foot = ROT60 @ np.array([-HX + 0.01, 0.0, HZ])         # 10 mm inside the board's upper edge, on its top
    b.add_shape_sphere(_fixed(b, "tomato", body_xform(*(foot + np.array([0.0, 0.0, ABOVE + R])))), radius=R)
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_burst_over_a_steep_board_lands_on_it():
    """WP15's must-fix (probe q12): a tomato 40 mm straight above a 60° board, 10 mm inside its upper edge (its
    normal foot is off the board), sheds 30 drops' worth at once: the board along gravity limits the first column to
    the 13 drops that start a radius over it (r + 2r·q down to 2.5 mm over it), and all 30 land on it, none floored
    (17 of 30 floored before)."""
    flow = SurfaceFlow(charts=(PlaneChart("board", cells=(20, 20)),), material="water", max_depth=5.0e-3,
                       drops={"drop_radius": RD, "floor_z": -0.5}, lumped=(Lumped("tomato", area=0.01),))
    sim = _sim(flow, _steep_board, lambda sim: flow.lumped.set_water(HOLD + 30.2 * DROP))
    n_top = ROT60 @ np.array([0.0, 0.0, 1.0])
    sim.step_period()                                         # shed in the 1st exchange, the first column in the 2nd
    sim.step_period()
    alive = flow.drops.pool.alive.numpy()[0] == 1
    first = flow.drops.pool.pos.numpy()[0][alive]
    assert 0 < len(first) <= 13 and ((first @ n_top - HZ) / n_top[2] > RD).all()     # a radius over the board, along z
    for _ in range(78):
        sim.step_period()
    d = flow.drops.read()
    assert d["lump_drops_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0)
    assert d["landed_volume"][0] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0) and d["floored_volume"][0] == 0.0


# --- M3, M4: a shelf over a body is not under it; drops carry their species -----------------------------------------

def _under_shelf(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -0.01), hx=0.2, hy=0.2, hz=0.01, label="slab")
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.02 + 2.0 * R + 0.01), hx=0.1, hy=0.1, hz=0.002, label="shelf")
    b.add_shape_sphere(_fixed(b, "tomato", body_xform(0.0, 0.0, 0.02 + R)), radius=R)
    b.add_shape_sphere(_fixed(b, "plum", body_xform(0.35, 0.0, 0.02 + R)), radius=R)         # off the slab
    _dummy(b)


@pytest.mark.compile_heavy
def test_drops_under_a_shelf_land_below_with_their_soap():
    """WP15's M3, M4 and round-5 S1, S2. In two worlds, a soapy tomato 20 mm over a slab under a shelf (its chart
    10 mm over the tomato's top) sheds 30 drops' worth and a plum off the slab's edge 10: a surface above a body is not
    under it, so the shelf stays dry; the tomato's 30 land on the slab, the plum's 10 fall below floor_z. The drops
    take their share of the waiting soap (6 and 2 kg/m³ in the tomatoes, 4 and 1 in the plums), booked directly:
    each body's dropped soap = its drops' water at its concentration and shed = wet + waiting + dropped + tanks;
    per world the landed soap = the slab film's (supplied + pending), at the tomato's concentration in every wet cell,
    and the floored soap = the plum's dropped (all 1e-12). The soap is host-set (``set_species`` per world and body,
    JOB3's ask): the books start from it (ledger 1e-15); bad bodies, shapes, names, strings, NaN and negatives are
    refused (PluginError) and change nothing, bools too."""
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(10, 10)), PlaneChart("shelf", cells=(10, 10))),
                       material="water", max_depth=5.0e-3, species=("soap",),
                       drops={"drop_radius": RD, "floor_z": -0.05},
                       lumped=(Lumped("tomato", area=0.01), Lumped("plum", area=0.01)))
    c = np.array([[6.0, 4.0], [2.0, 1.0]])
    water = np.array([HOLD + 30.2 * DROP, HOLD + 10.2 * DROP])

    def init(sim) -> None:
        flow.lumped.set_water(np.broadcast_to(water, (2, 2)))
        flow.lumped.set_species(c * water, "soap")
    sim = _sim(flow, _under_shelf, init, worlds=2)
    for _ in range(80):
        sim.step_period()
    r, d, f = flow.lumped.read(), flow.drops.read(), flow.read()
    h, mass, A = flow.h.numpy(), flow.sp.mass.numpy()[:, :, 0], flow.tables.area_np
    for w in range(2):
        assert d["lump_wet_volume"][w] == 0.0 and d["in_flight_volume"][w] == 0.0
        assert d["lump_drops_volume"][w] == pytest.approx(40 * DROP, rel=1e-12, abs=0.0)
        assert d["landed_volume"][w] == pytest.approx(30 * DROP, rel=1e-12, abs=0.0)
        assert d["floored_volume"][w] == pytest.approx(10 * DROP, rel=1e-12, abs=0.0)
        assert h[w][flow.tables.cells_of(1)].max() == 0.0                           # the shelf stays dry
        sp = {k: r[f"lump_species_{k}"][w, :, 0] for k in ("shed", "wet", "drip_pending", "dropped", "tanks", "set")}
        np.testing.assert_allclose(sp["dropped"], c[w] * np.array([30.0, 10.0]) * DROP, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(sp["shed"], sp["wet"] + sp["drip_pending"] + sp["dropped"] + sp["tanks"],
                                   rtol=1e-12, atol=0.0)
        film_soap = f["species_supplied"][w].sum(axis=0)[0] + float((flow.sp.src.numpy()[w, :, 0] * A).sum())
        assert d["species_landed"][w, 0] == pytest.approx(sp["dropped"][0], rel=1e-12, abs=0.0)
        assert film_soap == pytest.approx(d["species_landed"][w, 0], rel=1e-12, abs=0.0)
        assert d["species_floored"][w, 0] == pytest.approx(sp["dropped"][1], rel=1e-12, abs=0.0)
        wet = h[w] > 0.0
        assert wet.any()
        np.testing.assert_allclose(mass[w][wet] / h[w][wet], c[w, 0], rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(r["lump_species_set"][:, :, 0], c * water)
    assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * (c * water).max()
    bad = ((-1.0, "soap", None), (float("nan"), "soap", None), (1.0, "salt", None), ("1e-6", "soap", None),
           (1.0, "soap", -1), (1.0, "soap", 2), (np.ones(3), "soap", None), (np.ones((2, 2)), "soap", 0),
           (True, "soap", None), (1.0, "soap", True))
    for values, name, body in bad:
        with pytest.raises(PluginError):
            flow.lumped.set_species(values, name, body)
    np.testing.assert_array_equal(flow.lumped.read()["lump_species_set"][:, :, 0], c * water)


# --- q5b: a body partly under a pool's level drips into it ----------------------------------------------------------

def _half_under(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5), hx=0.1, hy=0.1, hz=0.01, label="plate")
    b.add_shape_sphere(_fixed(b, "tomato", body_xform(0.0, 0.0, -0.251)), radius=R)
    _dummy(b)


@pytest.mark.compile_heavy
def test_a_body_partly_under_a_pools_level_drips_into_its_tank():
    """WP15's should-fix 1 (probe q5b): a tomato whose centre is 1 mm under a 2 L bowl's level (not immersed: its
    overflow is shed) under a soapy stream: its drips would start under the water, so they and their soap join the
    bowl's tank (DC5's pool_add): nothing dropped, wet or floored; shed = tanks + waiting (1e-12); the supply's
    lump_pool = tanks (1e-12); the bowl holds 2 L + tanks and the tanks' soap (1e-12)."""
    net = {"nodes": [{"name": "bowl", "medium": "tank", "volume": 2.0e-3, "area": 0.04, "species": (0.0,)}],
           "edges": [], "species": ("soap",), "pools": [Pool("bowl", floor=(0.0, 0.0, -0.3), half=(0.1, 0.1))]}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", species=("soap",), network=net,
                       drops={"drop_radius": RD, "floor_z": -1.0}, lumped=(Lumped("tomato", area=0.01),))
    q, c = 1.0e-7, 3.0
    sim = _sim(flow, _half_under, _soap(flow, q, c))
    for _ in range(20):
        sim.step_period()
    r, d, s = flow.lumped.read(), flow.drops.read(), flow.supply.read()
    shed, tanks = r["lump_shed"][0, 0], d["lump_tanks_volume"][0]
    assert flow.lumped.state.pool_of.numpy()[0, 0] == -1 and r["lump_pooled"][0, 0] == 0.0     # not immersed
    assert tanks > 0.0
    assert d["lump_drops_volume"][0] == 0.0 and d["lump_wet_volume"][0] == 0.0 and d["floored_volume"][0] == 0.0
    assert shed == pytest.approx(tanks + d["lump_pending_volume"][0], rel=1e-12, abs=0.0)
    assert s["lump_pool"][0] == pytest.approx(tanks, rel=1e-12, abs=0.0)
    assert s["amount"][0, 0] == pytest.approx(2.0e-3 + tanks, rel=1e-12, abs=0.0)
    sp_tanks = r["lump_species_tanks"][0, 0, 0]
    assert sp_tanks == pytest.approx(c * tanks, rel=1e-12, abs=0.0)
    assert r["lump_species_shed"][0, 0, 0] == pytest.approx(sp_tanks + r["lump_species_drip_pending"][0, 0, 0],
                                                            rel=1e-12, abs=0.0)
    assert flow.supply.fluid.species.numpy()[0, :, 0].sum() == pytest.approx(sp_tanks, rel=1e-12, abs=0.0)
