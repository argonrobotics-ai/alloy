# SPDX-License-Identifier: Apache-2.0
"""Liquid in a held or tilted kinematic container beside a free rigid body (gaps-now #1, 2026-11-08), CPU.

A free body makes the scene coupled (MuJoCo + SPH). Since WP4's b377182b8 the unowned kinematic container joins the
rigid entry, and its proxy carries the container's pose and velocity into SPH every exchange (still and lifted trays:
tests/integration/test_stepper_kinematic_walls.py). A tilt still leaked: the coupler re-ran the tilt's first period
three times, and each rollback left sim.drive's own clock at the discarded attempt's end, so the jug jumped 30 ms
(3 deg) ahead in one exchange and its walls flung corner particles out (5 of 484 here; the free-ball pour put 0.3 %
on the floor before it poured, CPU and L4). Drives now go back with the period (test_drive_rollback.py here).
The static-sphere reference is one SPH solver with no coupling and no rollbacks.
- A jug tilted 30 deg about its lip in 0.3 s keeps every particle inside it, on its floor or above, and the water's
  height above the floor matches the reference within 1 mm (fixed: 0 outside, 9.01 vs 9.17 mm; before: 5 outside).
- Slow: examples/api/06_tilt_pour.py's 90 deg pour into a cup lands the same share in the cup as the reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy

IN, WALL, X, Z = 0.05, 0.004, 0.0, 0.10
LIP = (X + IN / 2 + WALL, 0.0, Z + WALL + IN)          # the jug's +x rim, outer top edge


def _local(p: np.ndarray, pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Points in the frame of a body at ``pos`` with xyzw quaternion ``q``."""
    x, y, z, w = (float(c) for c in q)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return (p - pos) @ r


def _tilted(free: bool) -> tuple[np.ndarray, alloy.Sim]:
    """Water points in the jug floor's frame (floor top at z = WALL / 2) after a 30 deg tilt about the lip."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(X, 0, Z), kinematic=True)
    sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(X, 0, Z + WALL + 0.009))
    sc.add_sphere("ball", 0.02, pos=(0.2, 0.0, 0.02), **({"mass": 0.1} if free else {"static": True}))
    sim = alloy.build(sc, device="cpu")
    assert sim.routes()["ball"]["route"] == ("rigid" if free else "static")
    sim.step(5)
    sim.drive("jug", angular_velocity=(0, math.radians(100), 0), pivot=LIP, duration=0.3)
    sim.step(40)
    s = sim.state(["water", "jug_floor"], world=0)
    return _local(s["water.points"], s["jug_floor.pos"], s["jug_floor.quat_xyzw"]), sim


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_tilted_kinematic_jug_holds_its_water_beside_a_free_body():
    loc, sim = _tilted(free=True)
    assert float(np.degrees(2 * math.asin(abs(float(sim.state(["jug_floor"], world=0)["jug_floor.quat_xyzw"][1]))))) \
        == pytest.approx(30.0, abs=0.5)
    inside = (np.abs(loc[:, 0]) < IN / 2 + 1e-3) & (np.abs(loc[:, 1]) < IN / 2 + 1e-3)
    assert inside.all(), int((~inside).sum())
    assert float(loc[:, 2].min()) > WALL / 2 - 1.5e-3, float(loc[:, 2].min())       # on the floor, never through it
    assert sim.check()["ok"], sim.check()["problems"]
    ref, _ = _tilted(free=False)
    assert float(np.median(loc[:, 2])) == pytest.approx(float(np.median(ref[:, 2])), abs=1.0e-3, rel=0)


def _pour_share(free: bool) -> float:
    jug_x, jug_z = -0.06, 0.12
    lip = (jug_x + IN / 2 + WALL, 0.0, jug_z + WALL + IN)
    cup_x, cup_r, cup_depth = lip[0] + 0.04, 0.035, 0.06
    sc = alloy.Scene(dt=0.005)
    sc.add_ground()
    sc.add_open_box("jug", inner=(IN, IN, IN), wall=WALL, pos=(jug_x, 0, jug_z), kinematic=True)
    sc.add_liquid("water", size=(0.044, 0.044, 0.02), pos=(jug_x, 0, jug_z + WALL + 0.011))
    sc.add_open_cylinder("cup", inner_radius=cup_r, depth=cup_depth, wall=WALL, pos=(cup_x, 0, 0))
    sc.add_sphere("ball", 0.02, pos=(0.0, 0.15, 0.02), **({"mass": 0.1} if free else {"static": True}))
    sim = alloy.build(sc, device="cpu")
    sim.settle(0.2)
    sim.drive("jug", angular_velocity=(0, math.radians(40), 0), pivot=lip, duration=2.25)
    sim.run(3.5)
    p = sim.state(["water"], world=0)["water.points"]
    assert sim.check()["ok"], sim.check()["problems"]
    return float(((np.hypot(p[:, 0] - cup_x, p[:, 1]) < cup_r) & (p[:, 2] < WALL + cup_depth)).mean())


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_tilt_pour_beside_a_free_body_matches_the_static_reference():
    free, ref = _pour_share(True), _pour_share(False)
    print(f"in the cup: free ball {free:.4f}, static ball {ref:.4f}")
    assert ref > 0.9, ref
    assert free == pytest.approx(ref, abs=0.02, rel=0)
