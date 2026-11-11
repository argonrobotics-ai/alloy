# SPDX-License-Identifier: Apache-2.0
"""Species through the supply's transfers on WP15's film (CPU): a jet's sweep carries the film's species with the film
it moves to the ring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.supply import Nozzle, Pool

H0, C0 = 1.0e-3, 10.0                                   # a 1 mm film with 10 kg/m³ of soap


def _faucet() -> dict:
    """Mains -> valve -> line -> spout, about 0.1 L/s."""
    return {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4}],
            "edges": [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + 2e5, "cd_area": 5e-6,
                       "dp_lam": 1.0},
                      {"name": "spout", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 1e-4,
                       "dp_lam": 1.0}]}


def test_a_jet_rinsing_a_soapy_film_carries_the_soap_with_the_swept_film():
    """A faucet on a walled plate under 1 mm of soapy water: the film the sheet sweeps to the jump ring takes its soap
    along (before, the soap stayed under the sheet and reached 28x its concentration there). The soap mass stays
    exactly what it was, the species ledger closes, and no cell is more concentrated than the film was."""
    jet = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=1e-4)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50)),), material="water", species=("soap",),
                       nozzles=(jet,), network=_faucet(), max_depth=1.0e-2)

    def init(sim) -> None:
        flow.set_depth(H0)
        flow.set_species(C0 * H0, "soap")
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), init=init, ground=False, worlds=2), [flow], "cpu")
    r0 = flow.read()
    for _ in range(10):
        sim.step_period()
    r = flow.read()
    mass0 = r0["species_mass"][:, 0]
    assert (r["supply_sent"] > 1e-5).all() and (r["supply_swept_species"][:, 0] > 0.0).all()
    assert np.allclose(r["species_mass"][:, 0], mass0, rtol=1e-12, atol=0.0)        # walls: nothing leaves
    assert np.abs(r["species_ledger"] - r0["species_ledger"]).max() <= 1e-12 * mass0.max()
    h, m = r["h"], r["species"][..., 0]
    wet = h > 1.0e-5
    assert (m[wet] / h[wet]).max() <= C0 * (1.0 + 1e-9), (m[wet] / h[wet]).max()


def test_a_drain_takes_the_soap_into_its_tank():
    """A soapy film on a plate with a free drain into a sink tank that carries soap (network species named): the
    soap the film drains is the soap the tank gains, every kg is booked, and the film's species ledger closes."""
    from alloy.plugins.surface_flow.supply import Drain  # noqa: PLC0415
    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16, "species": (0.0,)}],
           "edges": [], "species": ("soap",)}
    grate = Drain("grate", chart=0, center=(0.05, 0.0, 0.0), radius=0.03, to="sink")
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 40)),), material="water", species=("soap",),
                       drains=(grate,), network=net)

    def init(sim) -> None:
        flow.set_depth(H0)
        flow.set_species(C0 * H0, "soap")
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), init=init, ground=False, worlds=2), [flow], "cpu")
    r0 = flow.read()
    for _ in range(5):
        sim.step_period()
    r = flow.read()
    tank = flow.supply.fluid.species.numpy()[:, 0, 0]
    assert (tank > 0.0).all() and np.allclose(r["supply_species_tanks"][:, 0], tank, rtol=1e-12, atol=0.0)
    assert np.allclose(r["species_mass"][:, 0] + tank, r0["species_mass"][:, 0], rtol=1e-12, atol=0.0)
    assert np.abs(r["species_ledger"] - r0["species_ledger"]).max() <= 1e-12 * r0["species_mass"].max()
    assert (r["supply_species_away"] == 0.0).all()


def test_withdrawal_films_carry_the_pools_concentration():
    """A pool of soapy water (10 kg/m³) falls past a dry vertical plate: every film it leaves carries its
    concentration, the film that runs back into it carries its soap back, and the soap is conserved."""
    from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply import Pool  # noqa: PLC0415
    s45 = 0.5 ** 0.5

    def build(b) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.1, (s45, 0.0, 0.0, s45)), hx=0.1, hy=0.1, hz=0.005,
                        label="wall")
        basin = b.add_body(xform=body_xform(0.0, 0.1, 0.2), label="basin")
        b.add_shape_sphere(basin, radius=0.01)
    v0 = 0.25 * 0.16
    net = {"nodes": [{"name": "sink", "medium": "tank", "volume": v0, "area": 0.16, "species": (C0,)}],
           "edges": [], "species": ("soap",),
           "pools": [Pool("sink", floor=(0.0, 0.0, -0.3), half=(0.2, 0.2), body="basin", flood=True)]}
    flow = SurfaceFlow(charts=(PlaneChart("wall", cells=(10, 40)),), material="water", species=("soap",),
                       network=net)
    sim = build_simulation(TestScene(build=build, ground=False, worlds=2), [flow], "cpu")
    for _ in range(12):
        sim.step_period()
    r = flow.read()
    tank = flow.supply.fluid.species.numpy()[:, 0, 0]
    assert (r["supply_withdrawn"] > 0.0).all()
    assert np.allclose(r["supply_species_withdrawn"][:, 0], C0 * r["supply_withdrawn"], rtol=1e-9, atol=0.0)
    assert np.allclose(r["species_mass"][:, 0] + tank, C0 * v0, rtol=1e-12, atol=0.0)            # soap conserved
    h, m = r["h"], r["species"][..., 0]
    wet = h > 1.0e-6
    assert wet.any() and np.allclose(m[wet] / h[wet], C0, rtol=1e-6, atol=0.0)   # the pool's concentration


@pytest.mark.parametrize("how", ["jet", "spray", "node"])
def test_nozzles_bring_their_species(how):
    """A dry walled plate under a faucet or a spray: with ``Nozzle(species={"soap": 5})``, or (``node``) a faucet
    whose line carries soap at 5 kg/m³ (the network's species, named), the film's soap is what the nozzle sent onto
    it, C times the volume that landed, and the film's concentration is C."""
    c = 5.0
    net = _faucet()
    fixed = None if how == "node" else {"soap": c}
    if how == "node":
        net["nodes"][0]["species"] = (c,)
        net["edges"][0]["species"] = (c,)                  # the mains bring soapy water
        net["species"] = ("soap",)
    pattern = {"pattern": "spray", "angle": math.radians(30.0), "rays": 16, "efficiency": 0.8} if how == "spray" else {}
    noz = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=1e-4 if how != "spray" else 2e-5,
                 species=fixed, **pattern)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50)),), material="water", species=("soap",),
                       nozzles=(noz,), network=net, max_depth=1.0e-2)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2), [flow], "cpu")
    for _ in range(6):
        sim.step_period()
    r = flow.read()
    landed = (r["supply_sent"] - r["supply_mist"] - r["supply_overspray"] - r["supply_lost"] - r["supply_to_tank"]
              - r["supply_caught"])
    assert (landed > 0.0).all()
    assert np.allclose(r["supply_species_sent"][:, 0], c * landed, rtol=1e-9, atol=0.0)
    for col in ("mist", "overspray", "lost", "to_tank", "caught"):        # every share of the water has its species
        assert np.allclose(r[f"supply_species_{col}"][:, 0], c * r[f"supply_{col}"], rtol=1e-9, atol=0.0), col
    if how == "spray":
        assert (r["supply_species_mist"][:, 0] > 0.0).all()
    assert np.allclose(r["species_mass"][:, 0], r["supply_species_sent"][:, 0], rtol=1e-9, atol=0.0)
    h, m = r["h"], r["species"][..., 0]
    wet = h > 1.0e-5
    assert np.allclose(m[wet] / h[wet], c, rtol=1e-6, atol=0.0)


def test_a_fixed_soap_nozzle_filling_a_sink_brings_its_soap():
    """WP15's follow-up: a faucet with ``Nozzle(species={"soap": 5})`` on clean mains pours into a sink's pool (the
    network carries soap, named): the sink holds 5 kg/m³ times the water that landed in it, as the supply's species
    ledger books (before, it took the line's clean water's concentration)."""
    from alloy.plugins.surface_flow.supply_stub import faucet_network, slab_scene  # noqa: PLC0415
    net = faucet_network(0.0, pool=True)
    for node in net["nodes"]:
        node["species"] = (0.0,)
    net["edges"][0]["species"] = (0.0,)                                   # clean mains
    net["species"] = ("soap",)
    noz = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=1e-4, species={"soap": 5.0})
    flow = SurfaceFlow(charts=(PlaneChart("slab", cells=(10, 10)),), material="water", species=("soap",),
                       nozzles=(noz,), network=net)
    sim = build_simulation(TestScene(build=slab_scene(slab_x=1.0, crate=(-1.0, 0.0)), ground=True, worlds=2),
                           [flow], "cpu")
    for _ in range(4):
        sim.step_period()
    r = flow.read()
    tank = flow.supply.fluid.species.numpy()[:, 1, 0]
    assert (r["supply_to_tank"] > 0.0).all() and np.allclose(tank, 5.0 * r["supply_to_tank"], rtol=1e-9, atol=0.0)
    assert np.allclose(r["supply_species_to_tank"][:, 0], tank, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("pattern", ["fan", "cone"])
def test_a_fan_or_cone_partly_missing_books_the_species_of_every_ray(pattern):
    """WP15's sf9 review (must-fix): a 120° fan or cone 0.23 m over a 0.2 m plate sends most rays past it. Every ray's
    soap is booked where its water went, so each species column is C times its volume column, and the film holds
    what the landed rays sent."""
    c = 5.0
    noz = Nozzle("wide", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=2e-5, pattern=pattern, rays=16,
                 angle=math.radians(120.0), species={"soap": c})
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(40, 40)),), material="water", species=("soap",),
                       nozzles=(noz,), network=_faucet(), max_depth=1.0e-2)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2), [flow], "cpu")
    for _ in range(3):
        sim.step_period()
    r = flow.read()
    assert (r["supply_lost"] > 0.0).all() and (r["supply_lost"] < r["supply_sent"]).all()        # some rays missed
    landed = r["supply_sent"] - r["supply_mist"] - r["supply_overspray"] - r["supply_lost"] - r["supply_to_tank"]
    assert np.allclose(r["supply_species_sent"][:, 0], c * landed, rtol=1e-12, atol=0.0)
    for col in ("lost", "mist", "overspray", "to_tank"):
        assert np.allclose(r[f"supply_species_{col}"][:, 0], c * r[f"supply_{col}"], rtol=1e-12, atol=0.0), col
    assert np.allclose(r["species_mass"][:, 0], r["supply_species_sent"][:, 0], rtol=1e-9, atol=0.0)


SENT_COLUMNS = ("sent", "mist", "overspray", "lost", "to_tank", "caught", "lumped")


def test_the_species_in_flight_close_the_networks_book():
    """JOB3's ask (the can's dump): the network takes a nozzle's species from its node one exchange before the supply
    books them; supply.read()["species_in_flight"] is that exchange's share (Q·dt at the nozzle's concentration), so
    every period the soap the spout edge carried out of the line equals what the supply booked plus what is in
    flight (1e-12), on a faucet whose mains bring 5 kg/m³ of soap (the network's, named) into a line of clean water
    (its concentration rising, the nozzle's the line's)."""
    net = _faucet()
    net["nodes"][0]["species"] = (0.0,)
    net["edges"][0]["species"] = (5.0,)
    net["species"] = ("soap",)
    noz = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=1e-4)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20)),), material="water", species=("soap",),
                       nozzles=(noz,), network=net, max_depth=1.0e-2)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2), [flow], "cpu")
    s = flow.supply
    e = s.net.edge_index("spout")
    for _ in range(4):
        sim.step_period()
        r = s.read()
        out = -s.fluid.species_ledger.numpy()[:, e, 0]
        booked = sum(r[f"species_{c}"][:, 0] for c in SENT_COLUMNS)
        flying = r["species_in_flight"][:, 0]
        c_line = s.fluid.species.numpy()[:, 0, 0] / r["amount"][:, 0]
        assert (c_line < 4.9).all() and np.allclose(flying, -r["discharge"][:, e] * s.dt_x * c_line, rtol=1e-12, atol=0.0)
        assert np.allclose(out, booked + flying, rtol=1e-12, atol=0.0)




def test_the_concentration_a_nozzle_carried_is_state():
    """The concentration each node's outflows carried in the last network step (out_conc, what the supply books the
    water in flight at) is network state: on the faucet whose line grows soapier every step, the hooks run twice from
    one saved state give the same books (a snapshot, a reset or a rollback keeps it), per world and graph-safe."""
    from alloy.plugins.harness.hooks import hook_checks  # noqa: PLC0415
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    net = _faucet()
    net["nodes"][0]["species"] = (0.0,)
    net["edges"][0]["species"] = (5.0,)
    net["species"] = ("soap",)
    noz = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.25), exit_area=1e-4)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20)),), material="water", species=("soap",),
                       nozzles=(noz,), network=net, max_depth=1.0e-2)
    run = run_scene(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2, periods=3), [flow], "cpu")
    assert run.error is None, run.error
    results = hook_checks(run)
    assert results and all(r.passed for r in results), [(r.name, r.value, r.detail) for r in results]

def _tipped_glass(deg: float):
    """The plate and a glass welded with its floor at z = 0.2, tipped ``deg`` about y (its mouth toward +x, down)."""
    h = 0.5 * math.radians(deg)
    q = (0.0, math.sin(h), 0.0, math.cos(h))

    def build(b) -> None:
        sl.plate_scene(0.4, 0.4)(b)
        link = b.add_link(xform=body_xform(0.0, 0.0, 0.2, q), label="glass")
        b.add_shape_sphere(link, xform=body_xform(0.0, 0.0, -0.05), radius=0.005)
        b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(0.0, 0.0, 0.2, q))])
    return build


def test_a_glass_poured_dry_books_its_last_parcel():
    """JOB3's ask (the can's dump): a glass holding 5 mm of 3 kg/m³ soapy water, tipped 110°, pours over its rim into a
    sink of clean water until it is empty (exactly, from period 23). The network step that empties it sends its last soap at c' = M/(dt·q) and leaves the
    tank with no water and no soap, where M/amount reads 0; the supply books each parcel at the concentration its
    edge carried (the network's out_conc), so every period the pour edge's soap equals what the supply booked plus
    what is in flight (1e-12, the dry periods included; before, 31 % of the can's last parcel was never booked), and
    at the end the sink's tank holds all the glass's soap (into a tank at that concentration too)."""
    r_in, depth, c0 = 0.03, 0.005, 3.0
    v0 = math.pi * r_in * r_in * depth
    net = {"nodes": [{"name": "glass", "medium": "tank", "volume": v0, "area": math.pi * r_in * r_in,
                      "species": (c0,)},
                     {"name": "sink", "medium": "tank", "volume": 0.4 * 0.4 * 0.02, "area": 0.16}],
           "edges": [{"name": "pour", "kind": "source", "a": "glass", "flow": -1.0e-3}],
           "pools": [Pool("glass", body="glass", cavity=((r_in, 0.0), (r_in, 0.1)), spill="pour"),
                     Pool("sink", floor=(0.0, 0.0, 0.0), half=(0.2, 0.2))], "species": ("soap",)}
    lip = Nozzle("lip", edge="pour", rim_of="glass", body="glass", exit_area=1e-4)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(20, 20)),), material="water", species=("soap",),
                       nozzles=(lip,), network=net, max_depth=1.0e-2)
    sim = build_simulation(TestScene(build=_tipped_glass(110.0), ground=False, worlds=2), [flow], "cpu")
    s = flow.supply
    dry = 0
    for _ in range(26):
        sim.step_period()
        r = s.read()
        out = -s.fluid.species_ledger.numpy()[:, 0, 0]
        booked = sum(r[f"species_{c}"][:, 0] for c in SENT_COLUMNS)
        assert np.allclose(out, booked + r["species_in_flight"][:, 0], rtol=1e-12, atol=0.0)
        dry += bool((r["amount"][:, 0] == 0.0).all())
    held = s.fluid.species.numpy()[:, :, 0]
    assert dry >= 3 and (r["lost"] == 0.0).all(), (dry, r["lost"])                    # all of it into the sink
    assert np.allclose(r["species_to_tank"][:, 0], c0 * v0, rtol=1e-12, atol=0.0), r["species_to_tank"]
    assert np.allclose(held[:, 1], c0 * v0, rtol=1e-12, atol=0.0) and (held[:, 0] == 0.0).all(), held
