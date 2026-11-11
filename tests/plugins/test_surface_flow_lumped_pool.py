# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped bodies in DC5's pools (design note §4.9, Inflow 2; WP9): a sponge in a bowl fills from it and
swaps water with it. CPU, harness, one exchange a period, no gravity (the sponge stays where it is put)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_specs import Lumped, Pores
from alloy.plugins.surface_flow.supply import Pool

DT = 0.02
C_BOWL = 5.0                                       # kg/m³ of soap in the bowl
V_BOWL, A_BOWL = 2.0e-3, 0.04                      # 2 L over 0.2 x 0.2 m: the level 50 mm over the floor at z = -0.3
STARTS = ((0.0, -0.28, False, 0.0),               # world 0: immersed, empty: fills
          (0.0, -0.28, True, 0.0),                 # world 1: immersed, full of clean water, at rest: swaps at k_mix0
          (0.0, -0.28, True, 0.1),                 # world 2: the same, moving at 0.1 m/s: k_mix0 + k_mix1·0.1
          (0.0, -0.20, False, 0.0))                # world 3: above the level: nothing


def _scene(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5), hx=0.1, hy=0.1, hz=0.01, label="plate")    # the chart
    sponge = b.add_body(xform=body_xform(0.0, 0.0, -0.28), label="sponge")
    b.add_shape_box(sponge, hx=0.0555, hy=0.033, hz=0.006)                                         # GA's sponge


def _flow() -> SurfaceFlow:
    net = {"nodes": [{"name": "bowl", "medium": "tank", "volume": V_BOWL, "area": A_BOWL, "species": (C_BOWL,)}],
           "edges": [], "species": ("soap",), "pools": [Pool("bowl", floor=(0.0, 0.0, -0.3), half=(0.1, 0.1))]}
    return SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", species=("soap",), network=net,
                       lumped=(Lumped("sponge", pores=Pores()),))


def _sim(flow: SurfaceFlow):
    def init(sim) -> None:
        q, qd = sim.state.joint_q.numpy().reshape(-1, 7), sim.state.joint_qd.numpy().reshape(-1, 6)
        for w, (x, z, _, v) in enumerate(STARTS):
            q[w, 0], q[w, 2], qd[w] = x, z, (v, 0.0, 0.0, 0.0, 0.0, 0.0)
        sim.state.joint_q.assign(q.reshape(-1))
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
        cap = Pores().free
        flow.lumped.set_water(np.asarray([[cap if full else 0.0] for _, _, full, _ in STARTS]))
    return build_simulation(TestScene(build=_scene, init=init, ground=False, gravity=(0.0, 0.0, 0.0),
                                      worlds=len(STARTS), dt=DT, control_dt=DT), [flow], "cpu")


def _swap(cap: float, k_x: float, n: int) -> tuple[float, float]:
    """The sponge's soap after n exchanges, and its pending take, by the exchange's own steps (a full sponge at rest
    in the bowl: add vx = cap·k_x·dt at its concentration, take vx at the bowl's after the add, booked next)."""
    vb, mb, m, pend = V_BOWL, V_BOWL * C_BOWL, 0.0, 0.0
    vx = cap * min(1.0, k_x * DT)
    for _ in range(n):
        m += pend
        out = m * vx / cap
        m, vb, mb = m - out, vb + vx, mb + out
        pend = vx * mb / vb
        vb, mb = vb - vx, mb - pend
    return m, pend


@pytest.mark.compile_heavy
def test_a_sponge_in_a_bowl_fills_and_swaps_its_soap():
    """World 0: an empty sponge under the bowl's level fills at k_fill·cap per second: water + the take awaiting its
    booking = min(cap, n·k_fill·cap·dt) exactly, never over capacity. Worlds 1-2: a full sponge of clean water swaps
    min(water, pool)·min(1, k_x·dt) each exchange, k_x = k_mix0 + k_mix1·|v|, so its soap follows the exchange's
    closed recurrence (1e-12) toward the bowl's 5 kg/m³, faster when it moves. World 3, above the level: nothing.
    Ledgers: the sponge's (water = pooled + what the host set) exact; pooled + pending = -the supply's lump_pool; bowl + sponge soap
    conserved."""
    flow = _flow()
    sim = _sim(flow)
    lu, s = flow.lumped, flow.supply
    cap, spec = Pores().free, flow.config.lumped[0]
    m0 = s.fluid.species.numpy()[:, 0, 0].copy()
    done = 0
    for n in (10, 40):
        for _ in range(n - done):
            sim.step_period()
        done = n
        r = lu.read()
        water, pend = r["lump_water"][:, 0], r["lump_pool_pending"][:, 0]
        assert water[0] + pend[0] == pytest.approx(min(cap, n * spec.k_fill * cap * DT), rel=1e-12, abs=0.0)
        assert (water <= cap * (1.0 + 1e-12)).all() and water[3] == 0.0 and pend[3] == 0.0
        for w, v in ((1, 0.0), (2, float(np.float32(0.1)))):                 # the body's float32 speed
            m, p = _swap(cap, spec.k_mix0 + spec.k_mix1 * v, n)
            assert r["lump_species"][w, 0, 0] == pytest.approx(m, rel=1e-12, abs=0.0)
            assert r["lump_species_pool_pending"][w, 0, 0] == pytest.approx(p, rel=1e-12, abs=0.0)
        start = np.asarray([cap if full else 0.0 for _, _, full, _ in STARTS])      # set_water: the host's
        np.testing.assert_array_equal(r["lump_water_set"][:, 0], start)
        assert np.abs(r["lump_ledger"][:, 0]).max() <= 1e-15 * cap
        assert np.allclose(r["lump_pooled"][:, 0] + pend, -s.read()["lump_pool"], rtol=1e-12, atol=0.0)
        soap = s.fluid.species.numpy()[:, 0, 0] + r["lump_species"][:, 0, 0] + r["lump_species_pool_pending"][:, 0, 0]
        assert np.allclose(soap, m0, rtol=1e-12, atol=0.0)
        assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * m0.max()   # species = ... + pooled (net)
    c1, c2 = (r["lump_species"][w, 0, 0] / r["lump_water"][w, 0] for w in (1, 2))
    assert 0.1 * C_BOWL < c1 < c2 < C_BOWL                                # the moving sponge is nearer the bowl's


def _fixed_sponge(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5), hx=0.1, hy=0.1, hz=0.01, label="plate")
    xf = body_xform(0.0, 0.0, -0.28)
    s = b.add_link(xform=xf, label="sponge")
    b.add_articulation([b.add_joint_fixed(-1, s, parent_xform=xf)])
    b.add_shape_box(s, hx=0.0555, hy=0.033, hz=0.006)
    d = b.add_body(xform=body_xform(0.5, 0.0, 1.0), label="dummy")
    b.add_shape_sphere(d, radius=0.01)


@pytest.mark.compile_heavy
def test_an_immersed_sponge_overflows_into_its_pool_not_as_drops():
    """WP15's must-fix 2 (probe q5): a fixed sponge 30 mm under the bowl's level at 1.5x its capacity (as a jet or a
    squeeze leaves it), gravity and the drop stage on: its excess goes into the bowl (pool_add), booked pooled; nothing
    is shed, no drop leaves under the water; pooled + pending = -the supply's lump_pool and the soap books close."""
    net = {"nodes": [{"name": "bowl", "medium": "tank", "volume": V_BOWL, "area": A_BOWL, "species": (C_BOWL,)}],
           "edges": [], "species": ("soap",), "pools": [Pool("bowl", floor=(0.0, 0.0, -0.3), half=(0.1, 0.1))]}
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", species=("soap",), network=net,
                       drops={"drop_radius": 1.5e-3, "floor_z": -1.0}, lumped=(Lumped("sponge", pores=Pores()),))
    cap = Pores().free
    sim = build_simulation(TestScene(build=_fixed_sponge, init=lambda sim: flow.lumped.set_water(1.5 * cap),
                                     ground=False, gravity=(0.0, 0.0, -9.81), worlds=1, dt=DT, control_dt=DT),
                           [flow], "cpu")
    v0 = flow.supply.read()["amount"][0, 0]
    sim.step_period()
    r, d = flow.lumped.read(), flow.drops.read()
    assert r["lump_shed"][0, 0] == 0.0 and d["lump_drops_volume"][0] == 0.0 and d["lump_pending_volume"][0] == 0.0
    assert r["lump_water"][0, 0] <= cap and -r["lump_pooled"][0, 0] >= 0.5 * cap
    for _ in range(20):
        sim.step_period()
    r, d, s = flow.lumped.read(), flow.drops.read(), flow.supply.read()
    assert r["lump_shed"][0, 0] == 0.0 and d["lump_drops_volume"][0] == 0.0 and d["floored_volume"][0] == 0.0
    assert r["lump_pooled"][0, 0] + r["lump_pool_pending"][0, 0] == pytest.approx(-s["lump_pool"][0], rel=1e-12,
                                                                                   abs=0.0)
    assert s["amount"][0, 0] + r["lump_water"][0, 0] + r["lump_pool_pending"][0, 0] == pytest.approx(
        v0 + 1.5 * cap, rel=1e-12, abs=0.0)
    assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * V_BOWL * C_BOWL


def _two_sponges(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5), hx=0.1, hy=0.1, hz=0.01, label="plate")
    for i, x in enumerate((-0.05, 0.05)):
        s = b.add_body(xform=body_xform(x, 0.0, -0.28), label=f"sponge{i}")
        b.add_shape_box(s, hx=0.02, hy=0.02, hz=0.002)


@pytest.mark.compile_heavy
def test_two_sponges_in_one_bowl_split_its_take_by_their_asks():
    """WP15's test gap (probe q7): two empty sponges in one bowl, filling at k_fill 2/s and 4/s: the supply confirms
    one take for the bowl and each books its share (its ask over the bowl's asks), so each holds water + pending =
    min(cap, n·k_fill·cap·dt) (1e-12; share 1 fails), pooled + pending = -the supply's lump_pool, and the soap books
    close."""
    net = {"nodes": [{"name": "bowl", "medium": "tank", "volume": V_BOWL, "area": A_BOWL, "species": (C_BOWL,)}],
           "edges": [], "species": ("soap",), "pools": [Pool("bowl", floor=(0.0, 0.0, -0.3), half=(0.1, 0.1))]}
    pores = Pores(area=0.04 * 0.04, height=0.004, q_stop=0.003)
    flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(4, 4)),), material="water", species=("soap",), network=net,
                       lumped=(Lumped("sponge0", pores=pores, k_fill=2.0), Lumped("sponge1", pores=pores, k_fill=4.0)))
    sim = build_simulation(TestScene(build=_two_sponges, ground=False, gravity=(0.0, 0.0, 0.0), worlds=1, dt=DT,
                                     control_dt=DT), [flow], "cpu")
    n = 10
    for _ in range(n):
        sim.step_period()
    r, s = flow.lumped.read(), flow.supply.read()
    w, p = r["lump_water"][0], r["lump_pool_pending"][0]
    for b, kf in enumerate((2.0, 4.0)):
        assert w[b] + p[b] == pytest.approx(min(pores.free, n * kf * pores.free * DT), rel=1e-12, abs=0.0)
    assert r["lump_pooled"][0].sum() + p.sum() == pytest.approx(-s["lump_pool"][0], rel=1e-12, abs=0.0)
    assert np.abs(r["lump_species_ledger"]).max() <= 1e-15 * V_BOWL * C_BOWL

