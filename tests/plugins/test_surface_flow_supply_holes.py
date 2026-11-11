# SPDX-License-Identifier: Apache-2.0
"""Holes in a cavity pool's vessel (supply_holes.py, WP8's colander) on the stub film, CPU: Torricelli drainage
through a floor hole against its closed form, a side ring that stops draining at its height, and the water and its
species booked into the tank below."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Pool
from alloy.plugins.surface_flow.supply_stub import StubFilm

G = 9.81
R, HT = 0.03, 0.1
A = math.pi * R * R
CYL = ((R, 0.0), (R, HT))
FAR = PlaneChart("far", cells=(2, 2))
TANK_RTOL = 1e-9          # a tank's volume is κ·(P - P_ref), P absolute: a small head keeps its rounding


def _far(b) -> None:
    b.add_shape_box(-1, xform=body_xform(2.0, 0.0, 0.01), hx=0.05, hy=0.05, hz=0.01, label="far")
    dummy = b.add_body(xform=body_xform(-2.0, 0.0, 0.5), label="dummy")
    b.add_shape_sphere(dummy, radius=0.01)


def _colander(h0: float, holes, species: bool = False, gravity=(0.0, 0.0, -G), dt: float = 2.0e-3):
    soap = {"species": (10.0,)} if species else {}
    net = {"nodes": [{"name": "colander", "medium": "tank", "volume": A * h0, "area": A, **soap},
                     {"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16, **({"species": (0.0,)} if
                                                                                       species else {})}],
           "edges": [], "pools": [Pool("colander", cavity=CYL, holes=holes, holes_to="sink")]}
    p = StubFilm(charts=(FAR,), network=net)
    scene = TestScene(build=_far, ground=False, gravity=gravity, worlds=2, dt=dt, control_dt=0.02)
    return p, build_simulation(scene, [p], "cpu")


def test_a_floor_hole_drains_a_colander_as_torricelli():
    """A 1 cm² (Cd·A) hole at the centre of the floor drains 80 mm of water: sqrt(h) falls linearly,
    sqrt(h) = sqrt(h0) - (Cd·A/A)·sqrt(g/2)·t; after 1 s the level is that to 2e-3, and the sink holds the rest."""
    p, sim = _colander(0.08, ((0.0, 1.0e-4, 0.0),))
    for _ in range(50):
        sim.step_period()
    r = p.supply.read()
    v = r["amount"]
    h = math.sqrt(0.08) - 1.0e-4 / A * math.sqrt(G / 2.0) * 50 * 0.02                    # 50 periods of 20 ms
    assert np.allclose(v[:, 0] / A, h * h, rtol=2e-3, atol=0.0), (v[:, 0] / A, h * h)
    assert np.allclose(v.sum(axis=1), A * 0.08, rtol=TANK_RTOL, atol=0.0)
    assert np.allclose(r["holes"], v[:, 1], rtol=TANK_RTOL, atol=0.0)


def test_a_side_ring_drains_down_to_its_height_and_carries_the_soap():
    """A ring of side holes at 50 mm drains a colander filled to 80 mm down to the ring and no further (the head
    there is gone), into the sink with the soap at the colander's concentration (10 kg/m³)."""
    p, sim = _colander(0.08, ((0.05, 2.0e-4),), species=True)
    for _ in range(100):
        sim.step_period()
    v = p.supply.read()["amount"]
    level = v[:, 0] / A
    assert (level >= 0.05 * (1.0 - 1e-6)).all() and (level < 0.0505).all(), level
    soap = p.supply.fluid.species.numpy()[:, :, 0]
    assert np.allclose(soap[:, 1], 10.0 * v[:, 1], rtol=TANK_RTOL, atol=0.0)
    assert np.allclose(soap.sum(axis=1), 10.0 * A * 0.08, rtol=1e-12, atol=0.0)


def _ring_mean(head_mid: float, amp: float, n: int = 10_000) -> float:
    """(1/2π)∮ sqrt(2·g·max(head_mid + amp·cos φ, 0)) dφ by n midpoints: the ring's mean Torricelli speed."""
    phi = (np.arange(n) + 0.5) * (2.0 * math.pi / n)
    return float(np.sqrt(2.0 * G * np.maximum(head_mid + amp * np.cos(phi), 0.0)).mean())


@pytest.mark.parametrize("z_ring", [0.055, 0.09])
def test_a_tipped_colander_drains_from_its_low_side(z_ring):
    """Tipped 30° (gravity tilted), filled to 60 mm along its axis: the surface stands (V/A)·cos θ up along gravity,
    and a side ring at z (radius R) has heads cos θ·(V/A - z) + R·sin θ·cos φ round it. At 55 mm only its low side is
    wet: the first exchange drains Cd·A times the ring's mean of sqrt(2·g·head) (an independent quadrature, 10⁴
    points), to 1e-3; at 90 mm it is above the water everywhere and drains nothing."""
    th, h0, cda = math.radians(30.0), 0.06, 2.0e-4
    p, sim = _colander(h0, ((z_ring, cda),), gravity=_tilted(th), dt=0.02)       # one exchange a period
    sim.step_period()
    r = p.supply.read()
    head_mid, amp = math.cos(th) * (h0 - z_ring), R * math.sin(th)
    if z_ring > 0.06:
        assert head_mid < -amp and (r["holes"] == 0.0).all() and (r["amount"][:, 1] == 0.0).all()
        return
    assert abs(head_mid) < amp                                                   # part of the ring is dry
    want = cda * _ring_mean(head_mid, amp) * 0.02
    assert np.allclose(r["holes"], want, rtol=1e-3, atol=0.0), (r["holes"], want)
    assert np.allclose(r["amount"][:, 1], r["holes"], rtol=TANK_RTOL, atol=0.0)


def _tilted(th: float) -> tuple[float, float, float]:
    return (G * math.sin(th), 0.0, -G * math.cos(th))   # tilting gravity tips the colander relative to it


def test_holes_need_a_cavity_and_a_ring_inside_it():
    from alloy.plugins.errors import PluginError  # noqa: PLC0415
    with pytest.raises(PluginError, match="hole ring"):
        _colander(0.05, ((0.2, 1.0e-4),))
