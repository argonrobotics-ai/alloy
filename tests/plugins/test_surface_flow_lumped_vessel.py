# SPDX-License-Identifier: Apache-2.0
"""Lumped bodies in a vessel and drops into pools (WP8's rinse-tomatoes: a colander over a sink; WP9). CPU, harness:
a tomato resting in a holed colander sends its overflow to the colander's tank (its holes rain it into the sink), and
drops falling into a sink's water join its tank, so neither falls through to the floor."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_specs import Lumped
from alloy.plugins.surface_flow.supply import Pool

R, DT, G = 0.014, 2.0e-3, (0.0, 0.0, -9.81)
CYL = ((0.05, 0.0), (0.05, 0.08))                              # a 100 mm colander, 80 mm deep, floor at z = 0
HOLD = 5.0e-5 * 4.0 * np.pi * R ** 2


def _scene(z_tomato: float):
    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.0), hx=0.05, hy=0.05, hz=0.01, label="far")
        xf = body_xform(0.0, 0.0, z_tomato)
        t = b.add_link(xform=xf, label="tomato")
        b.add_articulation([b.add_joint_fixed(-1, t, parent_xform=xf)])
        b.add_shape_sphere(t, radius=R)
        d = b.add_body(xform=body_xform(-2.0, 0.0, 1.0), label="dummy")
        b.add_shape_sphere(d, radius=0.01)
    return build


def _flow(pools: list, nodes: list, pool: int = 256) -> SurfaceFlow:
    net = {"nodes": nodes, "edges": [], "species": ("soap",), "pools": pools}
    return SurfaceFlow(charts=(PlaneChart("far", cells=(2, 2)),), material="water", species=("soap",), network=net,
                       drops={"drop_radius": 1.5e-3, "floor_z": -1.0, "pool": pool}, lumped=(Lumped("tomato"),))


def _stream(flow: SurfaceFlow, q: float, c=0.0):
    """q m³ per exchange into the tomato at c kg/m³ of soap (a scalar, or one per world)."""
    def init(sim) -> None:
        inp = flow.lumped.inputs
        inp.vol.fill_(q)
        sp = inp.sp.numpy().copy()
        sp[:, 0, 0] = np.asarray(c, np.float64) * q
        inp.sp.assign(sp)
    return init


@pytest.mark.compile_heavy
def test_a_tomato_in_a_colander_sends_its_overflow_to_the_colander():
    """WP8's gap 1: a tomato resting in an empty colander (its lowest point inside the vessel: DC5's in_cavity) under a
    soapy stream sends what is over its hold into the colander's tank with its soap (pool_add, booked pooled), never as
    drops; the colander's floor hole rains it into the sink. Books: the colander + the sink hold what the tomato sent
    and its soap (1e-12), the supply's lump_pool = -the tomato's pooled; nothing shed, dropped or floored."""
    nodes = [{"name": "colander", "medium": "tank", "volume": 0.0, "area": np.pi * 0.05 ** 2, "species": (0.0,)},
             {"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16, "species": (0.0,)}]
    flow = _flow([Pool("colander", cavity=CYL, holes=((0.0, 1.0e-4, 0.0),), holes_to="sink")], nodes)
    q, c = 2.0e-8, 3.0
    sim = build_simulation(TestScene(build=_scene(R + 2.0e-3), init=_stream(flow, q, c), ground=False, gravity=G,
                                     worlds=1, dt=DT, control_dt=DT), [flow], "cpu")
    n = 50
    for _ in range(n):
        sim.step_period()
    r, d, s = flow.lumped.read(), flow.drops.read(), flow.supply.read()
    sent = n * q - r["lump_water"][0, 0]
    assert r["lump_water"][0, 0] == pytest.approx(HOLD, rel=1e-6, abs=0.0) and sent > 0.0
    assert r["lump_shed"][0, 0] == 0.0 and d["lump_drops_volume"][0] == 0.0 and d["floored_volume"][0] == 0.0
    assert -r["lump_pooled"][0, 0] == pytest.approx(sent, rel=1e-12, abs=0.0)
    assert s["lump_pool"][0] == pytest.approx(sent, rel=1e-12, abs=0.0)
    assert s["amount"][0].sum() == pytest.approx(sent, rel=1e-9, abs=0.0) and s["amount"][0, 1] > 0.0
    soap = flow.supply.fluid.species.numpy()[0, :, 0].sum()
    assert soap == pytest.approx(c * n * q - r["lump_species"][0, 0, 0], rel=1e-9, abs=0.0)
    assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * c * n * q


@pytest.mark.compile_heavy
def test_drops_fall_into_a_sinks_water_with_their_soap():
    """WP8's gap 2: a tomato 0.15 m over a sink holding 1 L sheds drops; they cross the sink's free surface inside its
    footprint and join its tank (pool_add, booked to_tanks), none floored: drops made = to_tanks + in flight, the
    sink grew by to_tanks and the supply's lump_pool = to_tanks (1e-12). The drops carry the tomato's soap (WP15's
    rounds 5 and 6), here 6 and 2 kg/m³ in two worlds with an 8-drop pool, full on some exchanges (refused requests
    keep their soap): per world the sink's soap = species_to_tanks = c·to_tanks and in flight c·in flight; the drops'
    soap stays in the tomato's dropped book (= to_tanks + in flight), its tanks book is for drips that start in the
    water, as its water's (0 here); shed = wet + waiting + dropped + tanks (1e-12)."""
    nodes = [{"name": "sink", "medium": "tank", "volume": 1.0e-3, "area": 0.04, "species": (0.0,)}]
    flow = _flow([Pool("sink", floor=(0.0, 0.0, 0.0), half=(0.1, 0.1))], nodes, pool=8)
    q, c = 2.0e-8, np.array([6.0, 2.0])
    sim = build_simulation(TestScene(build=_scene(0.15), init=_stream(flow, q, c), ground=False, gravity=G, worlds=2,
                                     dt=DT, control_dt=DT), [flow], "cpu")
    for _ in range(200):
        sim.step_period()
    d, s, r = flow.drops.read(), flow.supply.read(), flow.lumped.read()
    assert d["pool_full"].min() > 0
    soap = flow.supply.fluid.species.numpy()[:, :, 0].sum(axis=1)
    for w in range(2):
        made, tanks, fly = d["lump_drops_volume"][w], d["to_tanks_volume"][w], d["in_flight_volume"][w]
        assert tanks > 0.0 and d["floored_volume"][w] == 0.0 and d["landed_volume"][w] == 0.0
        assert made == pytest.approx(tanks + fly, rel=1e-12, abs=0.0)
        assert s["amount"][w, 0] == pytest.approx(1.0e-3 + tanks, rel=1e-12, abs=0.0)
        assert s["lump_pool"][w] == pytest.approx(tanks, rel=1e-12, abs=0.0)
        sp = {k: r[f"lump_species_{k}"][w, 0, 0] for k in ("shed", "wet", "drip_pending", "dropped", "tanks")}
        assert d["species_to_tanks"][w, 0] == pytest.approx(c[w] * tanks, rel=1e-12, abs=0.0)
        assert soap[w] == pytest.approx(d["species_to_tanks"][w, 0], rel=1e-12, abs=0.0)
        assert d["species_in_flight"][w, 0] == pytest.approx(c[w] * fly, rel=1e-12, abs=0.0)
        assert sp["tanks"] == 0.0 and r["lump_drip_tanks"][w, 0] == 0.0
        assert sp["dropped"] == pytest.approx(d["species_to_tanks"][w, 0] + d["species_in_flight"][w, 0], rel=1e-12,
                                              abs=0.0)
        assert sp["shed"] == pytest.approx(sp["wet"] + sp["drip_pending"] + sp["dropped"] + sp["tanks"], rel=1e-12,
                                           abs=0.0)
