# SPDX-License-Identifier: Apache-2.0
"""Two-phase chambers of the lumped-fluid core (``alloy.plugins.fluid.mixed``, SCR-WP8-11 Part A): closed forms of
laws G-1..G-6 at the core level (the owner's volume driven directly; the syringe plug-in's laws add bodies), the port
level geometry, the books, and the refusals (a plain gas-liquid edge is still refused). CPU, float64."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.fluid import P_ATM, Edge, Network, Node, stored_energy
from alloy.plugins.fluid.actuators import Cylinder, validate
from alloy.plugins.fluid.mixed import P_VAPOUR, volume_above
from alloy.plugins.library.syringe import needle

R3 = 8.59e-3 / 2                        # m, a 3 mL syringe's bore radius (BD, 8.59 mm)
RV = 10.75e-3                           # m, a 10 mL vial's inner radius
N25 = needle(0.26e-3, 0.016)            # 25 G x 16 mm
N18 = needle(0.84e-3, 0.038)            # 18 G x 38 mm blunt fill
K = 2.2e9                               # Pa, water
DT = 1e-3


@wp.kernel
def _above(v: wp.array[wp.float64], r: wp.array[wp.float64], d: wp.array[wp.float64], up: wp.array[wp.float64],
           out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = volume_above(v[i], r[i], d[i], up[i])


def _above_host(v, r, d, up) -> np.ndarray:
    a = [wp.array(np.asarray(x, np.float64), dtype=wp.float64, device="cpu") for x in (v, r, d, up)]
    out = wp.zeros(len(v), dtype=wp.float64, device="cpu")
    wp.launch(_above, dim=len(v), inputs=a, outputs=[out], device="cpu")
    return out.numpy()


def _above_numeric(v: float, r: float, d: float, up: float, n: int = 400) -> float:
    """Monte-Carlo-free midpoint quadrature over the cylinder: points above the plane through the port."""
    ell = v / (math.pi * r * r)
    d = min(max(d, 0.0), ell)                                # a port past the far end sits at it
    s = math.sqrt(max(1.0 - up * up, 0.0))
    z = (np.arange(n) + 0.5) / n * ell                       # depth below the head
    y = (np.arange(n) + 0.5) / n * 2 * r - r                 # across, along the projection of up
    zz, yy = np.meshgrid(z, y, indexing="ij")
    chord = 2.0 * np.sqrt(np.maximum(r * r - yy * yy, 0.0))  # the slice's width at y
    above = (d - zz) * up + yy * s > 0.0
    return float(np.sum(chord * above) * (ell / n) * (2 * r / n))


def test_the_level_geometry_is_the_cylinders():
    """Closed form against quadrature for heads up, tilted, horizontal and down, ports at the head and inside; and
    the ungula (2/3)·r³·tan θ above a head-centred port."""
    rng = np.random.default_rng(3)
    cases = [(3.1e-6, R3, 0.0, 1.0), (3.1e-6, R3, 0.0, -1.0), (3.1e-6, R3, 0.0, 0.0), (10e-6, RV, 0.005, -0.8)]
    cases += [(float(rng.uniform(1e-6, 1e-5)), float(rng.uniform(3e-3, 1.1e-2)), float(rng.uniform(0, 0.01)),
               float(rng.uniform(-1, 1))) for _ in range(12)]
    got = _above_host(*zip(*cases, strict=True))
    for (v, r, d, up), g in zip(cases, got, strict=True):
        assert g == pytest.approx(_above_numeric(v, r, d, up), rel=2e-3, abs=2e-3 * v), (v, r, d, up)
    th = np.radians([5.0, 30.0, 60.0])
    got = _above_host([3.1e-6] * 3, [R3] * 3, [0.0] * 3, np.cos(th))
    np.testing.assert_allclose(got, 2.0 / 3.0 * R3 ** 3 * np.tan(th), rtol=1e-12)


def _closed_form_p(v: float, liq: float, a: float, c: float, n: float) -> float:
    """Host bisection of L·(1 - (P - P_atm)/K) + a·(c/P)^(1/n) = V."""
    lo, hi = 1e3, 1e9
    for _ in range(200):
        p = 0.5 * (lo + hi)
        f = liq * (1 - (p - P_ATM) / K) + a * (c / p) ** (1 / n) - v
        lo, hi = (p, hi) if f > 0 else (lo, p)
    return 0.5 * (lo + hi)


def _syringe(v: float, liq: float, n: float = 1.0, up: float = 1.0, edges=(), worlds: int = 1):
    net = Network([Node("s", medium="mixed", volume=v, liquid=liq, radius=R3, pressure=P_ATM, polytropic=n,
                        bulk_modulus=K, up=up)], list(edges))
    return net.compile("cpu", worlds)


def _drive(fs, node: int, v_from: float, v_to: float, steps: int, hold: int = 0, each=None):
    for k in range(steps + hold):
        v = fs.volume.numpy()
        v[:, node] = v_from + (v_to - v_from) * min((k + 1) / steps, 1.0)
        fs.volume.assign(v)
        fs.step(DT)
        if each is not None:
            each(fs)


@pytest.mark.parametrize("n", [1.0, 1.4])
def test_g1_a_capped_syringe_is_boyles(n):
    """G-1: 2.8 mL of water and 0.3 mL of air, capped; the plunger takes 0.2 mL: P = P_atm·(0.3/0.1)^n (water's
    compression included), the same whatever path the plunger took."""
    v0, liq = 3.1e-6, 2.8e-6
    fs = _syringe(v0, liq, n)
    _drive(fs, 0, v0, v0 - 0.2e-6, 50, hold=10)
    want = _closed_form_p(v0 - 0.2e-6, liq, P_ATM * 0.3e-6, P_ATM ** (1 - n), n)
    assert fs.pressure.numpy()[0, 0] == pytest.approx(want, rel=1e-9)
    assert want == pytest.approx(P_ATM * 3.0 ** n, rel=1e-2)               # bare Boyle, water's 0.4-0.7 % aside


def _purge(up: float):
    v0 = 3.1e-6
    fs = _syringe(v0, 2.8e-6, up=up, edges=[Edge("nz", "port", "s", reservoir=P_ATM, **N25)])
    rows = []
    _drive(fs, 0, v0, 2.0e-6, 2200, hold=3500, each=lambda f: rows.append((f.mixed.air_out_ml("nz")[0],
                                                                -f.ledger.numpy()[0, 0] * 1e6,
                                                                f.pressure.numpy()[0, 0],
                                                                f.mixed.gas_volume()[0, 0])))
    return fs, np.asarray(rows)


def test_g2_nozzle_up_purges_the_air_first():
    """G-2: head up, the air leaves first, all 0.3 mL (at P_atm), then only liquid; tilted 30°, the air stops with
    the ungula (2/3)·r³·tan θ left above the nozzle plane."""
    fs, rows = _purge(1.0)
    first = int(np.argmax(rows[:, 1] > 0))
    assert rows[first, 0] == pytest.approx(0.3, rel=1e-9) and np.ptp(rows[first:, 0]) == 0.0   # split at the crossing
    assert rows[-1, 1] == pytest.approx(1.1 - 0.3, rel=1e-6)
    q = 1.1e-6 / 2.2                                     # m³/s, the plunger's displacement rate
    assert rows[first + 100, 2] - P_ATM == pytest.approx(q / N25["conductance"], rel=1e-2)        # Poiseuille, water
    p, vg = rows[first // 2, 2], rows[first // 2, 3]      # air: the step's mean pressure drives the flow, the
    assert p - P_ATM == pytest.approx(q / N25["gas_conductance"] - 0.5 * p * q * DT / vg, rel=2e-2)   # end reads ½ jump
    assert q / N25["gas_conductance"] == pytest.approx(1291.0, rel=1e-3)                        # less: 1124 Pa here
    th = math.radians(30.0)
    fs, rows = _purge(math.cos(th))
    ungula = 2.0 / 3.0 * R3 ** 3 * math.tan(th)
    assert rows[-1, 0] == pytest.approx(0.3 - ungula * 1e6, rel=1e-6)
    assert fs.mixed.gas_volume()[0, 0] == pytest.approx(ungula, rel=1e-6)


def test_g4_nozzle_down_passes_no_air():
    """G-4: head down, liquid covers the nozzle all the way: the air ledger stays exactly 0."""
    fs, rows = _purge(-1.0)
    assert np.all(rows[:, 0] == 0.0) and rows[-1, 1] == pytest.approx(1.1, rel=1e-3)


def _vial_net(edges_kw: dict, dead: float = 0.07e-6, worlds: int = 1):
    nodes = [Node("syr", medium="mixed", volume=dead + 2.5e-6, liquid=0.0, radius=R3, pressure=P_ATM, bulk_modulus=K),
             Node("vial", medium="mixed", volume=10e-6, liquid=5e-6, radius=RV, pressure=P_ATM, bulk_modulus=K)]
    return Network(nodes, [Edge("ndl", "port", "syr", "vial", depth_b=0.005, **edges_kw)]).compile("cpu", worlds)


def test_g3_the_vial_draw_follows_the_equal_volume_rule():
    """G-3: 2.5 mL of air into an upright vial with a 5 mL headspace: P = P_atm·7.57/5.07 (with the syringe's 0.07 mL
    dead space); inverted, drawing 2.5 mL of liquid brings the vial back to P_atm; the dead-space air stays in the
    syringe (its nozzle is up)."""
    dead = 0.07e-6
    fs = _vial_net(N18, dead)
    _drive(fs, 0, dead + 2.5e-6, dead, 2400, hold=600)
    p = fs.pressure.numpy()[0]
    want = _closed_form_p(10e-6 + dead, 5e-6, P_ATM * (7.5e-6 + dead), 1.0, 1.0)
    assert p[1] == pytest.approx(want, rel=1e-6) and p[0] == pytest.approx(p[1], rel=1e-6)
    up = fs.mixed.up.numpy()
    up[:, 1] = -1.0
    fs.mixed.up.assign(up)
    _drive(fs, 0, dead, dead + 2.5e-6, 2400, hold=600)
    assert fs.pressure.numpy()[0, 1] == pytest.approx(P_ATM, rel=1e-9)
    assert fs.mixed.gas.numpy()[0, 0] == pytest.approx(want * dead, rel=1e-6)


def test_g5_books_balance_every_step_in_every_world():
    """G-5: two worlds driven differently (one pushes air into the vial, one draws liquid from the inverted vial):
    liquid and gas amounts plus their ledgers are each constant to 1e-12 every step."""
    fs = _vial_net(N18, worlds=2)
    up = fs.mixed.up.numpy()
    up[1, 1] = -1.0
    fs.mixed.up.assign(up)
    liq0, gas0 = fs.mixed.liquid.numpy().sum(axis=1), fs.mixed.gas.numpy().sum(axis=1)
    worst = [0.0, 0.0]

    def books(f):
        liq, gas = f.mixed.liquid.numpy().sum(axis=1), f.mixed.gas.numpy().sum(axis=1)
        worst[0] = max(worst[0], float(np.max(np.abs(liq / liq0 - 1))))
        worst[1] = max(worst[1], float(np.max(np.abs(gas / gas0 - 1))))
    for k in range(1500):
        v = fs.volume.numpy()
        v[0, 0] = 0.07e-6 + 2.5e-6 * (1 - min((k + 1) / 1000, 1.0))
        v[1, 0] = 0.07e-6 + 2.5e-6 + 1.5e-6 * min((k + 1) / 1000, 1.0)
        fs.volume.assign(v)
        fs.step(DT)
        books(fs)
    assert worst[0] <= 1e-12 and worst[1] <= 1e-12, worst
    led = fs.ledger.numpy()[:, 0]                                  # liquid into the syringe = what left the vial
    np.testing.assert_allclose(fs.mixed.liquid.numpy()[:, 0], led, rtol=1e-12, atol=1e-18)


def _exergy(a: np.ndarray, p: np.ndarray, vg: np.ndarray) -> np.ndarray:
    """An isothermal pocket's exergy against the atmosphere: a·ln(P/P_atm) - a + P_atm·V_g [J]."""
    return np.where(a > 0, a * np.log(np.maximum(p, 1.0) / P_ATM) - a + P_ATM * vg, 0.0)


def test_g6_plunger_work_is_stored_plus_throttled():
    """G-6 (n = 1): the work done on the contents, -∫(P - P_atm)·dV, equals the stored exergy (pocket and water)
    plus the ports' throttling, within 1 %: a capped push, then the cap off (air out through a 25 G needle)."""
    v0 = 3.1e-6
    fs = _syringe(v0, 2.8e-6, edges=[Edge("nz", "port", "s", reservoir=P_ATM, **N25)])
    fs.set_opening("nz", wp.array([0.0], dtype=wp.float32, device="cpu"))

    def stored(f):
        p = f.pressure.numpy()[0, 0]
        water = f.mixed.liquid.numpy()[0, 0] * (p - P_ATM) ** 2 / (2 * K)
        return float(_exergy(f.mixed.gas.numpy()[0, 0], p, f.mixed.gas_volume()[0, 0]) + water)
    _drive(fs, 0, v0, v0 - 0.15e-6, 400)
    work = -fs.node_work.numpy()[0, 0]
    assert work == pytest.approx(stored(fs), rel=1e-2) and work > 1e-3
    fs.set_opening("nz", wp.array([1.0], dtype=wp.float32, device="cpu"))
    _drive(fs, 0, v0 - 0.15e-6, v0 - 0.25e-6, 400, hold=200)
    work = -fs.node_work.numpy()[0, 0]
    assert work == pytest.approx(stored(fs) + fs.work.numpy()[0, 0], rel=1e-2)


def test_refusals():
    """A plain gas-liquid edge is still refused (lifted only for ports between mixed nodes); a mixed network takes no
    gas, liquid or tank node, no non-port edge and no species; liquid that does not fit is refused."""
    gas, liq = Node("g"), Node("l", medium="liquid")
    mixed = Node("s", medium="mixed", volume=3e-6, liquid=1e-6, radius=R3)
    with pytest.raises(ValueError, match="joins a gas and a liquid node"):
        Network([gas, liq], [Edge("x", "linear", "g", "l", conductance=1e-9)])
    with pytest.raises(ValueError, match="holds only mixed nodes"):
        Network([mixed, gas], [Edge("x", "port", "s", "g", **N25)])
    with pytest.raises(ValueError, match="edges are ports"):
        Network([mixed], [Edge("x", "linear", "s", conductance=1e-9)])
    with pytest.raises(ValueError, match="holds only mixed nodes"):
        Network([liq], [Edge("x", "port", "l", **N25)])
    with pytest.raises(ValueError, match="no dissolved species"):
        Network([mixed], [Edge("x", "port", "s", reservoir_medium="liquid", species=(1.0,), **N25)])
    with pytest.raises(ValueError, match="does not fit"):
        Network([Node("s", medium="mixed", volume=1e-6, liquid=2e-6, radius=R3)], []).compile("cpu", 1)


def test_the_cores_one_amount_readers_refuse_a_mixed_network():
    """amount() and stored_energy() have one amount per node; a mixed node has two (WP13's and DC5's reviews), and a
    cylinder's force would read the pocket alone: all three refuse, naming where to look."""
    fs = _syringe(3.1e-6, 2.8e-6)
    with pytest.raises(ValueError, match="mixed.liquid and .gas"):
        fs.amount(fs.pressure.numpy(), fs.volume.numpy())
    with pytest.raises(ValueError, match="mixed network"):
        stored_energy(fs.net, fs.pressure.numpy(), fs.volume.numpy())
    with pytest.raises(ValueError, match="syringe plug-in"):
        validate((Cylinder("c", joint="j", a="s", bore=0.01),), fs.net)


def test_an_owners_inflow_is_liquid_booked_and_consumed():
    """FluidState's inflow contract holds for a mixed node: inflow [W, n] (m³/s at p_ref) is added to its liquid,
    booked in inflow_total and zeroed by the step."""
    fs = _syringe(3.1e-6, 2.8e-6, worlds=2)
    inflow = fs.inflow.numpy()
    inflow[1, 0] = 1e-7
    fs.inflow.assign(inflow)
    fs.step(DT)
    assert fs.mixed.liquid.numpy()[:, 0] == pytest.approx([2.8e-6, 2.8e-6 + 1e-10], rel=1e-12)
    assert fs.inflow_total.numpy()[:, 0] == pytest.approx([0.0, 1e-10], rel=1e-12)
    assert not fs.inflow.numpy().any()


@pytest.mark.parametrize("air", [0.0, 1e-10])
def test_a_cavity_or_a_small_pocket_refills_without_overshoot(air):
    """An all-liquid barrel (or one with a 0.1 µL pocket) withdrawn at 0.5 mL/s through an 18 G port from a liquid
    reservoir (a vein at +1.1 kPa): each step's draw is more than the water's compression down to the vapour floor (a
    cavity at the step start) or 5x the pocket. Neither is linear compliance: the refill settles at Poiseuille's,
    P_res - q/G, and never overshoots the reservoir (a cavity linearised as gas at P_VAPOUR, or the expanded pocket's
    tangent, overfilled them: 0.9 MPa spikes, and a growing plunger-pressure loop in the flush tasks)."""
    v0, q, p_res = 2.8e-6, 5e-7, P_ATM + 1.1e3
    fs = _syringe(v0 + air, v0, edges=[Edge("lk", "port", "s", reservoir=p_res, reservoir_medium="liquid", **N18)])
    p = []
    _drive(fs, 0, v0 + air, v0 + air + 0.2e-6, int(0.2e-6 / q / DT), each=lambda f: p.append(f.pressure.numpy()[0, 0]))
    p = np.asarray(p)
    assert q * DT >= max(v0 * (P_ATM - P_VAPOUR) / K, 5 * air)      # the step does start past either
    assert p.max() <= p_res and p[-1] == pytest.approx(p_res - q / N18["conductance"], rel=1e-4)
    assert fs.ledger.numpy()[0, 0] == pytest.approx(0.2e-6, rel=1e-3)


def test_a_cavity_the_port_cannot_fill_stays_at_the_floor():
    """2 mL/s through 25 G: the cavity outlasts every step, and the port refills it at Poiseuille's from the floor."""
    v0, p_res = 2.8e-6, P_ATM + 1.1e3
    fs = _syringe(v0, v0, edges=[Edge("nz", "port", "s", reservoir=p_res, reservoir_medium="liquid", **N25)])
    _drive(fs, 0, v0, v0 + 0.2e-6, 100)
    assert fs.pressure.numpy()[0, 0] == P_VAPOUR
    assert fs.ledger.numpy()[0, 0] == pytest.approx(N25["conductance"] * (p_res - P_VAPOUR) * 0.1, rel=1e-3)
