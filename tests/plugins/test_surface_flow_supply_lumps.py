# SPDX-License-Identifier: Apache-2.0
"""Jets on lumped bodies (WP9's lumped films, supply_lump.py), CPU, one exchange a period: a faucet on a tomato hands
its water (exactly the ledger's ``lumped``), species, impact, footprint share and sheet shear to the tomato's lump and
its momentum to the tomato; sprays on a lump and jets on other shapes are caught; misshaped arrays are refused."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow import laws as sl
from alloy.plugins.surface_flow.supply import Nozzle
from alloy.plugins.surface_flow.supply_kernels import LUMP
from alloy.plugins.surface_flow.supply_lump import lumps_of
from alloy.plugins.surface_flow.supply_stub import (CHART, FAUCET, StubFilm, faucet_network, ledger_sides, lump_arrays,
                                                     run_stub, slab_scene)

R_T = 0.03                                               # a tomato of 30 mm radius, fixed 0.12 m up under the faucet
Z_T = 0.12
DT = 0.02                                                # one exchange a period
AREA = 4.0 * math.pi * R_T * R_T
SPRAY = Nozzle("spray", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=2e-5, pattern="spray", angle=math.radians(10.0),
               rays=8)


def _tomato(x: float = 0.0, base: object = None):
    """``base`` (the stub's slab by default) and a tomato on a fixed link at (x, 0, Z_T)."""
    def build(b) -> None:
        (base or slab_scene())(b)
        t = b.add_link(xform=body_xform(x, 0.0, Z_T), label="tomato")
        b.add_shape_sphere(t, radius=R_T)
        b.add_articulation([b.add_joint_fixed(-1, t, parent_xform=body_xform(x, 0.0, Z_T))])
    return build


def _rays(p) -> dict[str, np.ndarray]:
    """The faucet's rays this exchange (the only nozzle): hit points and velocities [W, R, 3] and weights [R]."""
    wt = p.supply.ray_wt.numpy()
    R = len(wt)
    return {"p": p.supply.hit["p"].numpy()[:, :R].astype(np.float64),
            "w": p.supply.hit["w"].numpy()[:, :R].astype(np.float64), "frac": wt[:R] / wt[:R].sum()}


def _expected(p, q: np.ndarray, x: float, area: float) -> dict[str, np.ndarray]:
    """The lump's fields from the cast's hits (supply_lump.lump_deposit in float64): ½ρ·w_n² with the normal from the
    centre, Σ q/w_n over the area (at most 1), and the largest sheet shear."""
    h = _rays(p)
    n = h["p"] - np.array([x, 0.0, Z_T])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    wn = np.maximum(-(h["w"] * n).sum(-1), 0.0)
    speed = np.linalg.norm(h["w"], axis=-1)
    qr = q[:, None] * h["frac"]
    rho, nu = FAUCET.density, FAUCET.viscosity
    d = np.maximum(np.sqrt(qr / (math.pi * speed)), math.sqrt(area / (4.0 * math.pi)))
    tau = 4.0 / 3.0 * 0.332 * rho * speed * speed / np.sqrt(speed * d / nu)
    return {"p": 0.5 * rho * (wn * wn).max(axis=1), "share": np.minimum((qr / wn).sum(axis=1) / area, 1.0),
            "tau": tau.max(axis=1), "wn_over_w": (wn / speed).min()}


@pytest.mark.parametrize("x", [0.0, 0.012])
def test_a_faucet_on_a_tomato_hands_its_water_to_the_lump(x):
    """The faucet on the tomato's top (centred, or 12 mm off so the normal from its centre tilts 24°): every ray is a
    LUMP hit; the last exchange's ``lump_in`` is exactly that exchange's ``lumped`` and q·dt; ``lumped`` is all that was
    sent; the pressure, share and shear are lump_deposit's from the hits; the tomato feels ρ·q·w at the hits."""
    p = StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network(), lumps=(("tomato", AREA),))
    run = run_scene(TestScene(build=_tomato(x), ground=True, worlds=2, periods=3, dt=DT, control_dt=DT), [p], "cpu")
    assert run.error is None, run.error
    p = run.sim.plugins[p.label]
    r0 = {k: np.array(v, copy=True) for k, v in p.supply.read().items()}     # CPU reads are views
    run.sim.step_period()
    r = p.supply.read()
    q = -r0["discharge"][:, 1]                                       # the flow this exchange's jets carried
    lump_in = p.lump_in.numpy()[:, 0]
    assert (p.supply.hit["kind"].numpy()[:, :len(_rays(p)["frac"])] == LUMP).all()
    assert np.allclose(lump_in, r["lumped"] - r0["lumped"], rtol=1e-12, atol=0.0)
    assert np.allclose(lump_in, q * DT, rtol=1e-12, atol=0.0)
    assert np.allclose(r["lumped"], r["sent"], rtol=1e-12, atol=0.0)
    assert (r["caught"] == 0.0).all() and (r["lost"] == 0.0).all()
    lhs, rhs, spout, sent = ledger_sides(p)
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0) and np.allclose(spout, sent, rtol=1e-12, atol=0.0)
    e = _expected(p, q, x, AREA)
    if x > 0.0:
        assert e["wn_over_w"] < 0.95                                 # the normal is not the jet's
    assert np.allclose(p.lump_p.numpy()[:, 0], e["p"], rtol=1e-6, atol=0.0)
    assert np.allclose(p.lump_share.numpy()[:, 0], e["share"], rtol=1e-5, atol=0.0)
    assert np.allclose(p.lump_tau.numpy()[:, 0], e["tau"], rtol=1e-5, atol=0.0)
    f = run.probe(run.sim.state.body_q.numpy())                      # the jets' momentum at the current flow
    h, qn = _rays(p), -r["discharge"][:, 1]
    for w, b in enumerate(p.ctx.bodies("tomato").numpy()):
        mom = FAUCET.density * qn[w] * h["frac"][:, None] * h["w"][w]
        arm = h["p"][w] - np.array([x, 0.0, Z_T])
        want = np.concatenate([mom.sum(axis=0), np.cross(arm, mom).sum(axis=0)])
        assert np.allclose(f[b], want, rtol=1e-5, atol=1e-6 * abs(want[2])), (f[b], want)


def test_a_tiny_lump_is_covered_at_most_once():
    """A lump whose area is under the jets' footprint q/w_n: its share is 1, not their ratio."""
    p = run_stub(StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network(), lumps=(("tomato", 1e-6),)),
                 periods=2, build=_tomato())
    assert (p.lump_share.numpy() == 1.0).all() and (p.supply.read()["lumped"] > 0.0).all()


def _flow(nozzle: Nozzle) -> SurfaceFlow:
    """WP15's film with soap, ``nozzle`` fed by mains through a line (about 0.1 L/s)."""
    net = {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4}],
           "edges": [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + 2e5, "cd_area": 5e-6,
                      "dp_lam": 1.0},
                     {"name": "spout", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 1e-4,
                      "dp_lam": 1.0}]}
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(50, 50)),), material="water", species=("soap",),
                       nozzles=(nozzle,), network=net, max_depth=1.0e-2)


@pytest.mark.parametrize("case", ["jet_on_lump", "spray_on_lump", "jet_on_shape"])
def test_species_go_where_the_water_goes(case):
    """WP15's film, a soapy faucet (5 kg/m³) over a fixed tomato, its lump arrays registered after the build (a lumped
    stage that sets up after the film: the supply binds them at its first apply, inside the captured period): a jet
    on the lump hands it C·lump_in of soap and books C·lumped; a spray on it is caught with C·caught; a jet on a shape
    that is no lump (here the lump is the dummy body) is caught, with its soap, and nothing is lost. (The species
    ``sent`` column is what landed on the film: none here.)"""
    c = 5.0
    lump = "dummy" if case == "jet_on_shape" else "tomato"
    pattern = {"pattern": "spray", "angle": math.radians(10.0), "rays": 8} if case == "spray_on_lump" else {}
    noz = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=1e-4 if not pattern else 2e-5,
                 species={"soap": c}, **pattern)
    flow = _flow(noz)
    sim = build_simulation(TestScene(build=_tomato(base=sl.plate_scene(0.2, 0.2)), ground=False, worlds=2, dt=DT,
                                     control_dt=DT), [flow], "cpu", graph=True)
    for name, a in lump_arrays(flow.ctx, ((lump, AREA),), n_film=1).items():
        setattr(flow, name, a)
    for _ in range(3):
        sim.step_period()
    r = flow.read()
    assert flow.supply.n_lumps == 1
    for col in ("lumped", "caught", "lost"):
        assert np.allclose(r[f"supply_species_{col}"][:, 0], c * r[f"supply_{col}"], rtol=1e-12, atol=0.0), col
    assert np.allclose(r["supply_sent"], r["supply_lumped"] + r["supply_caught"], rtol=1e-12, atol=0.0)
    assert (r["supply_lost"] == 0.0).all()                  # none on the film: a hidden spray's soap leaves "sent"
    assert (np.abs(r["supply_species_sent"][:, 0]) <= 1e-12 * c * r["supply_sent"]).all()     # as it came (rounding)
    if case == "jet_on_lump":
        assert (r["supply_lumped"] > 0.0).all() and (r["supply_caught"] == 0.0).all()
        assert np.allclose(flow.lump_sp_in.numpy()[:, 0, 0], c * flow.lump_in.numpy()[:, 0], rtol=1e-12, atol=0.0)
    else:
        assert (r["supply_caught"] > 0.0).all() and (r["supply_lumped"] == 0.0).all()
        assert not flow.lump_in.numpy().any() and not flow.lump_sp_in.numpy().any()


@pytest.mark.parametrize("bad", ["missing", "area_f32", "species_k", "shape_count"])
def test_misshaped_lump_arrays_are_refused(bad):
    """The supply checks the film's lump arrays when it binds them: all seven, with WP9's shapes and types."""
    p = run_stub(StubFilm(charts=(CHART,), nozzles=(FAUCET,), network=faucet_network()), periods=1, build=_tomato())
    a = lump_arrays(p.ctx, (("tomato", AREA),))
    W = p.ctx.world_count
    if bad == "missing":
        del a["lump_tau"]
    elif bad == "area_f32":
        a["lump_area"] = p.ctx.array(np.full(1, AREA, np.float32), wp.float32)
    elif bad == "species_k":
        a["lump_sp_in"] = p.ctx.zeros((W, 1, 2), wp.float64)
    else:
        a["lump_of_shape"] = p.ctx.full((W, 1), -1, wp.int32)
    with pytest.raises(PluginError, match="lump_tau" if bad == "missing" else "lumped films"):
        lumps_of(p.ctx, SimpleNamespace(**a), 0)
    assert lumps_of(p.ctx, SimpleNamespace(**lump_arrays(p.ctx, (("tomato", AREA),))), 0) is not None


def test_a_lump_stage_adds_to_and_takes_from_a_pool():
    """WP9's pool exchange (a sponge in a bowl), one exchange a period: the supply publishes the bowl's volume
    (pool_avail); an add of 0.1 L with 0.3 g of soap reaches the tank and its soap; a swap (add 0.2 L of clean water,
    take 0.2 L) leaves the volume and takes soap at the concentration after the add; a take beyond the bowl empties it
    and books the rest refused. pool_took / pool_took_sp confirm each take; the requests are consumed."""
    from alloy.plugins.harness.scene import TestScene, build_simulation  # noqa: PLC0415
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow import laws as sl  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply import Pool  # noqa: PLC0415
    net = {"nodes": [{"name": "bowl", "medium": "tank", "volume": 2.0e-3, "area": 0.04, "species": (5.0,)}],
           "edges": [], "species": ("soap",), "pools": [Pool("bowl", floor=(0.0, 0.0, -0.3), half=(0.1, 0.1))]}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", species=("soap",), network=net)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2, dt=0.02, control_dt=0.02),
                           [flow], "cpu")
    s = flow.supply
    sim.step_period()

    def state():
        return s.read()["amount"][:, 0].copy(), s.fluid.species.numpy()[:, 0, 0].copy()      # numpy(): a view on CPU

    def ask(add=0.0, add_sp=0.0, take=0.0):
        s.pool_add.assign(np.full((2, 1), add))
        s.pool_add_sp.assign(np.full((2, 1, 1), add_sp))
        s.pool_take.assign(np.full((2, 1), take))
        sim.step_period()
        return s.pool_took.numpy()[:, 0].copy(), s.pool_took_sp.numpy()[:, 0, 0].copy()
    v0, m0 = state()
    assert np.allclose(s.pool_avail.numpy()[:, 0], v0, rtol=1e-12, atol=0.0)
    took, took_sp = ask(add=1.0e-4, add_sp=3.0e-4)
    v1, m1 = state()
    assert np.allclose(v1, v0 + 1.0e-4, rtol=1e-12, atol=0.0) and np.allclose(m1, m0 + 3.0e-4, rtol=1e-12, atol=0.0)
    assert not took.any() and not took_sp.any()
    assert not (s.pool_add.numpy().any() or s.pool_add_sp.numpy().any() or s.pool_take.numpy().any())
    took, took_sp = ask(add=2.0e-4, take=2.0e-4)                      # the sponge's swap: clean water in, bowl out
    v2, m2 = state()
    assert np.allclose(v2, v1, rtol=1e-12, atol=0.0) and np.allclose(took, 2.0e-4, rtol=1e-12, atol=0.0)
    assert np.allclose(took_sp, 2.0e-4 * m1 / (v1 + 2.0e-4), rtol=1e-9, atol=0.0)
    assert np.allclose(m2, m1 - took_sp, rtol=1e-12, atol=0.0)
    refused0 = s.read()["refused"]
    took, took_sp3 = ask(take=1.0)
    v3, m3 = state()
    assert np.allclose(v3, 0.0, atol=1e-15) and np.allclose(m3, 0.0, atol=1e-15)          # 1e-12 of the bowl
    assert np.allclose(took, v2, rtol=1e-12, atol=0.0) and np.allclose(took_sp3, m2, rtol=1e-12, atol=0.0)
    led = s.read()["species_lump_pool"][:, 0]
    assert np.allclose(led, 3.0e-4 - took_sp - took_sp3, rtol=1e-12, atol=0.0)          # the species ledger
    assert np.allclose(s.read()["refused"] - refused0, 1.0 - v2, rtol=1e-12, atol=0.0)
    assert np.allclose(s.read()["lump_pool"], 1.0e-4 - v2, rtol=1e-12, atol=0.0)


def _bowl_flow(species=("soap",), pools=True, faucet=False):
    """WP15's film over the plate, a bowl tank (2 L, 5 kg/m³ soap) whose pool lies off the plate at x = 0.5, and
    with ``faucet`` clean mains through a line onto it (about 0.1 L/s)."""
    from alloy.plugins.fluid import P_ATM  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply import Pool  # noqa: PLC0415
    nodes = [{"name": "bowl", "medium": "tank", "volume": 2.0e-3, "area": 0.04, "species": (5.0,)}]
    edges, nozzles = [], ()
    if faucet:
        nodes.append({"name": "line", "medium": "liquid", "volume": 1e-4, "species": (0.0,)})
        edges = [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + 2e5, "cd_area": 5e-6,
                  "dp_lam": 1.0, "species": (0.0,)},
                 {"name": "spout", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 1e-4, "dp_lam": 1.0}]
        nozzles = (Nozzle("faucet", edge="spout", pos=(0.5, 0.0, 0.25), exit_area=1e-4),)
    net = {"nodes": nodes, "edges": edges, "species": ("soap",),
           "pools": [Pool("bowl", floor=(0.5, 0.0, -0.3), half=(0.1, 0.1))] if pools else []}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", species=species,
                       nozzles=nozzles, network=net)
    sim = build_simulation(TestScene(build=sl.plate_scene(0.2, 0.2), ground=False, worlds=2, dt=DT, control_dt=DT),
                           [flow], "cpu")
    return flow, sim


def test_a_take_sees_what_came_in_this_exchange():
    """WP15's probe: a clean jet into the bowl in the same exchange as a take of V0 + ½·jet: the take leaves at the
    concentration after everything booked this exchange, m0/(V0 + jet) (it used m0/V0 and took more soap than the
    bowl held), and the bowl keeps m0 minus that."""
    flow, sim = _bowl_flow(faucet=True)
    s = flow.supply
    for _ in range(3):
        sim.step_period()
    r = {k: np.array(v, copy=True) for k, v in s.read().items()}
    v0, m0 = r["amount"][:, 0], s.fluid.species.numpy()[:, 0, 0].copy()
    jet = -r["discharge"][:, s.net.edge_index("spout")] * DT                 # what lands in the bowl this exchange
    s.pool_take.assign((v0 + 0.5 * jet)[:, None])
    sim.step_period()
    took, took_sp = s.pool_took.numpy()[:, 0].copy(), s.pool_took_sp.numpy()[:, 0, 0].copy()
    m1 = s.fluid.species.numpy()[:, 0, 0]
    assert np.allclose(took, v0 + 0.5 * jet, rtol=1e-12, atol=0.0)
    assert np.allclose(took_sp, took * m0 / (v0 + jet), rtol=1e-9, atol=0.0)
    assert (m1 > 0.0).all() and np.allclose(m1, m0 - took_sp, rtol=1e-9, atol=0.0)


def test_a_species_no_tank_carries_goes_away_once():
    """WP15's probe: a film species the network does not carry (salt), added with water, is booked ``away`` and not
    also ``lump_pool``; the carried soap is booked ``lump_pool``."""
    flow, sim = _bowl_flow(species=("soap", "salt"))
    s = flow.supply
    sim.step_period()
    s.pool_add.assign(np.full((2, 1), 1.0e-4))
    s.pool_add_sp.assign(np.tile(np.array([3.0e-4, 1.0e-3]), (2, 1, 1)))
    sim.step_period()
    r = s.read()
    assert np.allclose(r["species_away"][:, 1], 1.0e-3, rtol=1e-12, atol=0.0)
    assert (r["species_lump_pool"][:, 1] == 0.0).all() and (r["species_away"][:, 0] == 0.0).all()
    assert np.allclose(r["species_lump_pool"][:, 0], 3.0e-4, rtol=1e-12, atol=0.0)


def test_bad_pool_requests_are_refused():
    """A negative add or take is refused (booked ``refused``, nothing moves), a negative species add ignored; a scene
    without pools refuses and clears what a stage asks."""
    flow, sim = _bowl_flow()
    s = flow.supply
    sim.step_period()
    v0, m0 = s.read()["amount"][:, 0].copy(), s.fluid.species.numpy()[:, 0, 0].copy()
    s.pool_add.assign(np.full((2, 1), -1.0e-4))
    s.pool_add_sp.assign(np.full((2, 1, 1), -3.0e-4))
    s.pool_take.assign(np.full((2, 1), -2.0e-4))
    sim.step_period()
    r = s.read()
    assert np.allclose(r["refused"], 3.0e-4, rtol=1e-12, atol=0.0) and (r["lump_pool"] == 0.0).all()
    assert np.allclose(r["amount"][:, 0], v0, rtol=1e-12, atol=0.0)
    assert np.allclose(s.fluid.species.numpy()[:, 0, 0], m0, rtol=1e-12, atol=0.0)
    bare, sim = _bowl_flow(pools=False)
    sim.step_period()
    bare.supply.pool_add.assign(np.full((2, 1), 1.0e-4))
    bare.supply.pool_take.assign(np.full((2, 1), 2.0e-4))
    sim.step_period()
    assert np.allclose(bare.supply.read()["refused"], 3.0e-4, rtol=1e-12, atol=0.0)
    assert not bare.supply.pool_add.numpy().any() and not bare.supply.pool_take.numpy().any()


def _species_take(s, sim, take_sp, add_sp=0.0, take=0.0):
    """One exchange of ``pool_take_sp`` (with ``pool_add_sp`` and a water take): (taken_sp, took, took_sp) [W, K]."""
    K = s.pool_take_sp.shape[2]
    s.pool_take_sp.assign(np.broadcast_to(np.asarray(take_sp, np.float64), (2, 1, K)).copy())
    s.pool_add_sp.assign(np.broadcast_to(np.asarray(add_sp, np.float64), (2, 1, K)).copy())
    s.pool_take.assign(np.full((2, 1), take))
    sim.step_period()
    return (s.pool_taken_sp.numpy()[:, 0].copy(), s.pool_took.numpy()[:, 0].copy(),
            s.pool_took_sp.numpy()[:, 0].copy())


def test_walls_take_soap_alone_from_a_pool():
    """SCR-WP8-10 (WP9: a wall under the water adsorbs the pool's soap): a species take of 0.4 of the bowl's soap
    leaves it m0 - x and its water as it was, confirmed in pool_taken_sp, booked lump_pool out (1e-12); with an add
    of soap and a water take in the same exchange the walls take after the add and the lumps take at
    (m + add - x)/V; a take beyond the bowl takes all it holds with that exchange's add and the rest is species
    refused."""
    flow, sim = _bowl_flow()
    s = flow.supply
    sim.step_period()
    v0, m0 = s.read()["amount"][:, 0].copy(), s.fluid.species.numpy()[:, 0, 0].copy()
    led0 = s.read()["species_lump_pool"][:, 0].copy()
    taken, _, _ = _species_take(s, sim, (0.4 * m0)[:, None, None])
    m1 = s.fluid.species.numpy()[:, 0, 0].copy()
    assert np.allclose(taken[:, 0], 0.4 * m0, rtol=1e-12, atol=0.0) and np.allclose(m1, 0.6 * m0, rtol=1e-12, atol=0.0)
    assert np.allclose(s.read()["amount"][:, 0], v0, rtol=1e-12, atol=0.0)
    assert np.allclose(s.read()["species_lump_pool"][:, 0] - led0, -0.4 * m0, rtol=1e-12, atol=0.0)
    assert not s.pool_take_sp.numpy().any() and (s.read()["species_refused"] == 0.0).all()
    taken, took, took_sp = _species_take(s, sim, (0.5 * m1)[:, None, None], add_sp=1.0e-3, take=5.0e-4)
    want = 5.0e-4 * (m1 + 1.0e-3 - 0.5 * m1) / v0
    assert np.allclose(taken[:, 0], 0.5 * m1, rtol=1e-12, atol=0.0) and np.allclose(took, 5.0e-4, rtol=1e-12, atol=0.0)
    assert np.allclose(took_sp[:, 0], want, rtol=1e-12, atol=0.0)
    m2 = s.fluid.species.numpy()[:, 0, 0].copy()
    assert np.allclose(m2, m1 + 1.0e-3 - 0.5 * m1 - want, rtol=1e-12, atol=0.0)
    taken, _, _ = _species_take(s, sim, 1.0, add_sp=2.0e-4)                  # beyond the bowl and this add
    assert np.allclose(taken[:, 0], m2 + 2.0e-4, rtol=1e-12, atol=0.0)
    assert np.allclose(s.fluid.species.numpy()[:, 0, 0], 0.0, atol=1e-15)
    assert np.allclose(s.read()["species_refused"][:, 0], 1.0 - m2 - 2.0e-4, rtol=1e-12, atol=0.0)


def test_bad_species_takes_are_refused():
    """A negative species take, one of a species the network does not carry (salt) and one with no pool are species
    refused (their size) and cleared; nothing moves."""
    flow, sim = _bowl_flow(species=("soap", "salt"))
    s = flow.supply
    sim.step_period()
    m0 = s.fluid.species.numpy()[:, 0, 0].copy()
    taken, _, _ = _species_take(s, sim, np.array([-2.0e-4, 3.0e-4]))
    assert not taken.any() and np.allclose(s.fluid.species.numpy()[:, 0, 0], m0, rtol=1e-12, atol=0.0)
    assert np.allclose(s.read()["species_refused"], [[2.0e-4, 3.0e-4]] * 2, rtol=1e-12, atol=0.0)
    assert not s.pool_take_sp.numpy().any()
    bare, sim = _bowl_flow(pools=False)
    sim.step_period()
    bare.supply.pool_take_sp.assign(np.full((2, 1, 1), 4.0e-4))
    sim.step_period()
    assert np.allclose(bare.supply.read()["species_refused"][:, 0], 4.0e-4, rtol=1e-12, atol=0.0)
    assert not bare.supply.pool_take_sp.numpy().any()
