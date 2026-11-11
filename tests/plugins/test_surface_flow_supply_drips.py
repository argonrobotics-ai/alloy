# SPDX-License-Identifier: Apache-2.0
"""Drips from a colander's holes (supply_holes.py, holes_to=None; WP8's rinse-tomatoes) on the stub film, CPU, one
exchange a period: where the drips leave (equal shares of the holes' flow round the wet rings), how fast (Torricelli
along the wall's normal, plus the vessel's velocity), the water and its species through the ledgers with one exchange
in flight, and the push on the vessel."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Pool
from alloy.plugins.surface_flow.supply_holes import hole_tables
from alloy.plugins.surface_flow.supply_kernels import CHART as HIT_CHART
from alloy.plugins.surface_flow.supply_stub import CHART, RHO, StubFilm, slab_scene

G = 9.81
R, HT = 0.03, 0.1                                       # a straight colander: 30 mm inner radius, 100 mm deep
A = math.pi * R * R
CYL = ((R, 0.0), (R, HT))
Z0 = 0.25                                               # its floor, 230 mm above the slab
DT = 0.02                                               # one exchange a period
TANK_RTOL = 1e-9          # a tank's volume is κ·(P - P_ref), P absolute: a small head keeps its rounding
FAR = PlaneChart("far", cells=(2, 2))


def _far(b) -> None:
    b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.01), hx=0.05, hy=0.05, hz=0.01, label="far")
    dummy = b.add_body(xform=body_xform(-2.0, 0.0, 0.5), label="dummy")           # the solver needs a free body
    b.add_shape_sphere(dummy, radius=0.01)


def _colander(base=None):
    """``base`` (the stub's slab by default) and the colander on a fixed link, its floor at (0, 0, Z0), with a base
    plate just under its floor holes (its own shape: the drips pass it)."""
    def build(b) -> None:
        (base or slab_scene())(b)
        link = b.add_link(xform=body_xform(0.0, 0.0, Z0), label="colander")
        b.add_shape_box(link, xform=body_xform(0.0, 0.0, -0.003), hx=0.04, hy=0.04, hz=0.002)   # its base, under
        b.add_articulation([b.add_joint_fixed(-1, link, parent_xform=body_xform(0.0, 0.0, Z0))])
    return build


def _net(h0: float, holes, rays: int = 12, soap: float | None = None, sink: bool = False) -> dict:
    sp = {"species": (soap,)} if soap is not None else {}
    nodes = [{"name": "colander", "medium": "tank", "volume": A * h0, "area": A, **sp}]
    pools = [Pool("colander", body="colander", cavity=CYL, holes=holes, drip_rays=rays)]
    if sink:
        nodes.append({"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16,
                      **({"species": (0.0,)} if soap is not None else {})})
        pools.append(Pool("sink", floor=(0.0, 0.0, 0.05), half=(0.2, 0.2)))
    net = {"nodes": nodes, "edges": [], "pools": pools}
    if soap is not None:
        net["species"] = ("soap",)
    return net


def _sim(p: StubFilm, build, gravity=(0.0, 0.0, -G)):
    scene = TestScene(build=build, ground=True, gravity=gravity, worlds=2, dt=DT, control_dt=DT)
    return build_simulation(scene, [p], "cpu")


def _read(p: StubFilm) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in p.supply.read().items()}      # CPU reads are views


def _drips(p: StubFilm) -> tuple[np.ndarray, np.ndarray, slice]:
    """The drip nozzle's rays (the only nozzle): exit points and velocities [W, N, 3] (body frame), and their slice."""
    n = int(p.supply.drip_ray.numpy()[0][1])
    r0 = int(p.supply.drip_ray.numpy()[0][0])
    rays = slice(r0, r0 + n)
    return p.supply.ray_p0.numpy()[:, rays].astype(np.float64), p.supply.ray_dir.numpy()[:, rays].astype(np.float64), \
        rays


def test_a_floor_ring_defaults_to_the_rms_radius():
    """The docstring's rule (it fell back to the floor's edge radius): floor holes without r are a ring at r/√2; side
    holes at the wall; a wall's outward normal is radial, the floor's down."""
    _, rings, normals, _ = hole_tables((Pool("c", body="c", cavity=CYL, holes=((0.0, 1e-5), (0.05, 1e-5))),),
                                       SimpleNamespace(nodes=(SimpleNamespace(name="c", medium="tank"),)))
    assert rings[0][2] == pytest.approx(R / math.sqrt(2.0), rel=1e-6) and rings[1][2] == pytest.approx(R, rel=1e-6)
    assert np.allclose(normals, [[0.0, -1.0], [1.0, 0.0]], rtol=0.0, atol=0.0)


def test_an_upright_colander_drips_evenly_at_torricelli_speed_and_its_water_closes():
    """60 mm of water over a floor ring (Cd·A 20 mm²): the last exchange's drips leave from 12 points evenly round the
    ring (r/√2, at (i + ½)·30°), straight down at sqrt(2·g·h), carrying Cd·A·sqrt(2·g·h) (h the level when they left);
    they land on the slab (nothing lost); what the holes took is what the drips sent plus the exchange in flight, and
    the cross ledger closes."""
    p = StubFilm(charts=(CHART,), network=_net(0.06, ((0.0, 2.0e-5),)))
    sim = _sim(p, _colander())
    for _ in range(5):
        sim.step_period()
    h_prev = _read(p)["amount"][:, 0] / A
    sim.step_period()
    r = _read(p)
    pos, vel, rays = _drips(p)
    q = p.supply.drip_q.numpy()[:, 0]
    assert np.allclose(q, 2.0e-5 * np.sqrt(2.0 * G * h_prev), rtol=1e-6, atol=0.0)
    phi = (np.arange(12) + 0.5) * 2.0 * math.pi / 12.0
    rr = R / math.sqrt(2.0)
    for w in range(2):
        assert np.allclose(pos[w], np.stack([rr * np.cos(phi), rr * np.sin(phi), np.zeros(12)], 1), rtol=0.0,
                           atol=1e-6)
        assert np.allclose(vel[w], [[0.0, 0.0, -math.sqrt(2.0 * G * h_prev[w])]] * 12, rtol=1e-5, atol=1e-6)
    assert (p.supply.hit["kind"].numpy()[:, rays] == HIT_CHART).all()
    assert (r["lost"] == 0.0).all() and (r["sent"] > 0.0).all()
    assert np.allclose(r["holes"], r["sent"] + q * DT, rtol=1e-12, atol=0.0)          # one exchange in flight
    assert np.allclose(r["amount"][:, 0] + r["holes"], A * 0.06, rtol=TANK_RTOL, atol=0.0)
    lhs = r["sent"] + r["swept"] + r["withdrawn"]
    rhs = p.supplied.numpy().sum(axis=1) + r["to_tank"] + r["lost"] + r["overspray"] + r["mist"] + r["caught"] + \
        r["lumped"]
    assert np.allclose(lhs, rhs, rtol=1e-12, atol=0.0)


def _ring_share(phi: np.ndarray, level: float, mid: float, amp: float, a: float, n: int = 20_000) -> np.ndarray:
    """The ring's flow from its dry edge a to each φ over its whole wet arc, by n midpoints (an independent quadrature)."""
    psi = a + (np.arange(n) + 0.5) * (2.0 * (math.pi - a) / n)
    dens = np.sqrt(2.0 * G * np.maximum(level - (mid + amp * np.cos(psi)), 0.0))
    cum = np.concatenate([[0.0], np.cumsum(dens)])
    edges = a + np.arange(n + 1) * (2.0 * (math.pi - a) / n)
    return np.interp(phi, edges, cum) / cum[-1]


def test_a_tipped_colander_drips_from_its_low_side():
    """Tipped 30°, 60 mm of water, a side ring at 55 mm whose low side alone is wet: the 12 drips leave at equal shares
    of the ring's flow (an independent 2·10⁴-point quadrature from the dry edge: (i + ½)/12, to 3e-3), all on its wet
    arc and so on its low side, outward along the wall at sqrt(2·g·head) of their own point."""
    th, h0, z = math.radians(30.0), 0.06, 0.055
    p = StubFilm(charts=(FAR,), network=_net(h0, ((z, 2.0e-4),)))
    _sim(p, _colander(_far), (G * math.sin(th), 0.0, -G * math.cos(th))).step_period()
    pos, vel, _ = _drips(p)
    level, mid, amp = math.cos(th) * h0, math.cos(th) * z, R * math.sin(th)       # heights along the up
    a = math.acos((level - mid) / amp)
    e = np.array([-1.0, 0.0, 0.0])                                                   # body-frame up tilts to -x
    for w in range(2):
        rad = pos[w] - np.array([0.0, 0.0, z])
        assert np.allclose(np.linalg.norm(rad, axis=1), R, rtol=1e-6, atol=0.0)
        rad /= R
        phi = np.mod(np.arctan2(rad[:, 1] * e[0] - rad[:, 0] * e[1], rad @ e), 2.0 * math.pi)
        head = level - (mid + amp * np.cos(phi))
        assert (head > 0.0).all() and (rad @ e < 0.0).all()                          # wet, on the low side
        share = _ring_share(phi, level, mid, amp, a)
        assert np.allclose(np.sort(share), (np.arange(12) + 0.5) / 12.0, rtol=0.0, atol=3e-3), share
        assert np.allclose(vel[w], rad * np.sqrt(2.0 * G * head)[:, None], rtol=1e-4, atol=1e-6)


def test_a_moving_colander_carries_its_drips_and_they_push_it():
    """The colander moving at 0.3 m/s along x as it drips: each drip leaves with the vessel's velocity added (its
    impact velocity's x is 0.3 m/s; straight-down drips have no other), and the drips push the colander up by
    ρ·Q·sqrt(2·g·h) at the holes (besides its water's weight)."""
    p = StubFilm(charts=(CHART,), network=_net(0.06, ((0.0, 2.0e-5),)))
    sim = _sim(p, _colander())
    sim.step_period()
    st = sim.model.state()
    b = int(p.ctx.bodies("colander").numpy()[0])
    qd = st.body_qd.numpy()
    qd[b] = (0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
    st.body_qd.assign(qd)
    forces = SimpleNamespace(body_f=wp.zeros(sim.model.body_count, dtype=wp.spatial_vector, device="cpu"))
    p.supply.apply(st, forces, DT)
    _, vel, rays = _drips(p)
    hw = p.supply.hit["w"].numpy()[0, rays]
    assert np.allclose(hw[:, 0], 0.3, rtol=1e-6, atol=0.0) and np.allclose(hw[:, 1], 0.0, rtol=0.0, atol=1e-6)
    q = float(p.supply.drip_q.numpy()[0, 0])
    f = forces.body_f.numpy()[b].astype(np.float64)
    weight = RHO * float(_read(p)["amount"][0, 0]) * G                                # it carries its water (sf12)
    assert f[2] + weight == pytest.approx(RHO * q * np.linalg.norm(vel[0], axis=1).mean(), rel=1e-4)
    assert np.allclose(f[:2], 0.0, rtol=0.0, atol=1e-6 * f[2])


def test_drips_into_a_sink_carry_the_colanders_soap():
    """A soapy colander (10 kg/m³) dripping into a sink's pool: the sink gets its soap at the colander's
    concentration (to_tank), and the soap is conserved, the drips in flight carrying theirs."""
    p = StubFilm(charts=(FAR,), network=_net(0.06, ((0.0, 2.0e-5),), soap=10.0, sink=True))
    sim = _sim(p, _colander(_far))
    for _ in range(6):
        sim.step_period()
    r = _read(p)
    soap = p.supply.fluid.species.numpy()[:, :, 0]
    flight_v = p.supply.drip_q.numpy()[:, 0] * DT
    flight_m = flight_v * p.supply.ns.drip.numpy()[:, 0, 0]
    assert (r["to_tank"] > 0.0).all() and np.allclose(r["to_tank"], r["sent"], rtol=1e-12, atol=0.0)
    assert np.allclose(r["amount"][:, 1], r["to_tank"], rtol=TANK_RTOL, atol=0.0)
    assert np.allclose(soap[:, 1], 10.0 * r["amount"][:, 1], rtol=TANK_RTOL, atol=0.0)
    assert np.allclose(soap.sum(axis=1) + flight_m, 10.0 * A * 0.06, rtol=1e-12, atol=0.0)
    assert np.allclose(flight_m, 10.0 * flight_v, rtol=1e-9, atol=0.0)
