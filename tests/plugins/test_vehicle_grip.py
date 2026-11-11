# SPDX-License-Identifier: Apache-2.0
"""Vehicle plug-in, transient tyre (SCR-WP8-12 F1, WP13 review): grade climb and hold, a locked slide, and the two
low-speed release laws, on GA's H2.5A truck at the forklift task's step, against the closed forms in laws_grip."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.vehicle import laws as vl
from alloy.plugins.vehicle import laws_grip as lg

pytestmark = pytest.mark.law
UPHILL_X = (-1.0, 0.0)                 # gravity tilts toward -x: the truck (facing +x) has its front uphill
DOWNHILL_X = (1.0, 0.0)


def _world_y(force: np.ndarray, steer: float, quat: np.ndarray) -> np.ndarray:
    """``[K]`` the tyres' summed force along world y: each wheel's (F_x, F_y) turned by the truck's yaw and its steer
    angle (front 0, rear -steer on the axle)."""
    from scipy.spatial.transform import Rotation

    yaw = Rotation.from_quat(quat).as_euler("ZYX")[:, 0][:, None]
    ang = np.array([0.0, 0.0, -steer, -steer])[None, :] + yaw
    return (force[..., 0] * np.sin(ang) + force[..., 1] * np.cos(ang)).sum(axis=-1)


def _grade(tan: float, downhill, **kw) -> lg.Tilt:
    return lg.Tilt(math.atan(tan), downhill, **kw)


@pytest.mark.parametrize("factor, climbs", [(0.96, True), (1.04, False)])
def test_r1_front_drive_climbs_below_the_traction_limit_and_spins_back_above(factor, climbs):
    cfg = {**lg.TRUCK, "rolling_resistance": 0.0}         # c_rr holds a wheel at rest: no dead band at 0
    tilt = _grade(factor * lg.traction_limit(cfg["mu"], cfg["rolling_resistance"]), UPHILL_X)
    theta = tilt.angle
    need = 1.02 * 3990.0 * lg.G * (math.sin(theta) + cfg["rolling_resistance"] * math.cos(theta))
    throttle = need * lg.R_FRONT / cfg["gear"] / cfg["motor_torque"]
    go = lg.tilt_done(tilt) + 0.5
    r = lg.run(cfg, go + 2.0, {"brake": 1.0}, tilt, schedule={go: {"brake": 0.0, "throttle": throttle}})
    after = r["t"] >= go
    moved = r["q"][after][-1, 0] - r["q"][after][0, 0]
    assert (moved > 0.02) if climbs else (moved < -0.02), moved


def test_r1_full_throttle_climbs_the_ramp_at_the_motor_line_speed():
    cfg = {**lg.TRUCK, "motor_torque": 400.0, "motor_free_speed": 130.0}   # motor line time constant 0.5 s
    tilt = _grade(lg.GRADE, UPHILL_X)
    r = lg.run(cfg, 5.0, {"throttle": 1.0}, tilt)
    got = float(r["qd"][r["t"] > 4.0, 0].mean())
    want = lg.climb_speed(cfg, tilt.angle)
    assert got == pytest.approx(want, rel=0.02), (got, want)


@pytest.mark.parametrize("downhill, steer", [(DOWNHILL_X, 0.0), (UPHILL_X, 0.0), (DOWNHILL_X, lg.FULL_LOCK)])
def test_r2_braked_truck_holds_on_the_ramp(downhill, steer):
    """Also at full lock (WP13): the steer axle's tread then runs across the axle, where the pin makes its mobility
    ~30x larger; the low-speed damper must use it there (steered 2x2 mobility) or it goes unstable. Both wheels at 80°
    (parallel steering, Ackermann off)."""
    tilt = _grade(lg.GRADE, downhill)
    start = lg.tilt_done(tilt) + 1.0
    r = lg.run({**lg.TRUCK, "ackermann": (0.0,)}, start + 10.0, {"brake": 1.0, "steer": steer}, tilt)
    x = r["q"][:, :2]
    creep = float(np.linalg.norm(x[-1] - x[np.argmax(r["t"] >= start)]))
    assert creep <= 1.0e-3, creep
    assert np.abs(r["rate"][r["t"] >= start + 9.0]).max() < 1.0e-3            # treads settled, no oscillation


def test_r2_locked_front_slides_at_the_closed_form_acceleration():
    mu = 0.4
    cfg = {**lg.TRUCK, "mu": mu}
    tan_hold = mu * lg.B_CG / (lg.L - mu * lg.H_CG)
    tilt = _grade(1.15 * tan_hold, DOWNHILL_X)
    r = lg.run(cfg, lg.tilt_done(tilt) + 3.0, {"brake": 1.0}, tilt)
    slid = r["q"][:, 0] - r["q"][0, 0]
    win = (slid >= 4.0 * cfg["relaxation_x"]) & (r["t"] >= lg.tilt_done(tilt))   # tread past 98 % of locked slip
    assert win.sum() >= 25, slid[-1]
    got = float(np.polyfit(r["t"][win], r["qd"][win, 0], 1)[0])
    want = lg.slide_accel(mu, tilt.angle, cfg["rolling_resistance"])
    assert np.abs(r["omega"][win, :2]).max() < 1.0e-6                     # the front wheels stay locked
    assert got == pytest.approx(want, rel=0.03), (got, want)


@pytest.mark.parametrize("steer", [0.0, lg.FULL_LOCK])
def test_g1_a_sideslip_stops_and_the_tyres_carry_the_static_share(steer):
    """WP13: after a slide, the stored tread deflection releases within σ_y and the truck does not creep uphill; also
    with the steer wheels at full lock and every wheel braked, so the steer axle's treads hold along the axle, where
    the pin makes the mobility largest (unbraked, they would roll down the slope). Both steer wheels at 80°: the law's
    parallel steering (Ackermann off; with it the outer wheel would stand at 49.5°)."""
    cfg = {**lg.TRUCK, "mu": 0.25, "ackermann": (0.0,), **({"brake_torque": (3000.0,)} if steer else {})}
    tilt = lg.Tilt(math.radians(18.0), (0.0, -1.0), dwell=1.0, end=math.radians(8.0))
    back = lg.tilt_done(tilt) + tilt.dwell + (tilt.angle - tilt.end) / tilt.rate
    r = lg.run(cfg, back + 7.0, {"brake": 1.0, "steer": steer}, tilt)
    slid = float(r["q"][np.argmax(r["t"] >= back), 1] - r["q"][0, 1])
    after = r["t"] >= back
    moving = np.flatnonzero(after & (np.abs(r["qd"][:, 1]) > 5.0e-3))
    tail = r["t"] >= r["t"][moving[-1]] + 1.0                               # 1 s after it last moved (stops, rings)
    lateral = float(_world_y(r["force"][tail], steer, r["q"][tail, 3:]).mean())
    want = 3990.0 * lg.G * math.sin(tilt.end)
    y = r["q"][:, 1]
    uphill = float(y[-1] - y[after].min())                                  # the stored tread's release
    assert slid < -0.2                                                      # it really slid downhill
    assert tail.sum() >= 25, "never came to rest"
    assert lateral == pytest.approx(want, rel=0.03), (lateral, want)
    assert uphill <= cfg["relaxation_y"], uphill
    assert abs(float(y[-1] - y[tail][0])) <= 1.0e-3                         # and holds
    assert np.abs(r["rate"][tail]).max() < 1.0e-3


def test_g2_wheels_spun_at_standstill_keep_no_drive_force_once_stopped():
    """WP13: the spun tread's deflection is bounded by the peak slip once the spin stops, and the free wheel
    unwinds it: the rear tyres' force falls to rolling resistance."""
    cfg = {"mu": 0.3, "brake_torque": (3000.0, 3000.0, 0.0, 0.0), "motor_torque": 50.0, "motor_free_speed": 1.0e6}
    from alloy.plugins.vehicle import Vehicle

    c = Vehicle(**cfg).config
    r = lg.run(cfg, 4.0, {"brake": 1.0, "throttle": 1.0}, schedule={1.0: {"throttle": 0.0}},
               build=vl.cart(c), chassis=c.chassis, bodies=(c.chassis,))
    rear = r["omega"][:, 2:]
    assert rear[r["t"] <= 1.0].max() > 20.0                                  # it spun
    stopped = np.flatnonzero((r["t"] > 1.0) & (np.abs(rear).max(axis=1) < 0.05))
    assert stopped.size, "the rear wheels never stopped"
    tail = r["t"] >= r["t"][stopped[0]] + 0.3
    fx, fz = r["force"][tail, 2:, 0], r["force"][tail, 2:, 2]
    assert np.abs(fx).max() <= 1.5 * c.rolling_resistance * fz.max() + 0.01 * c.mu * fz.max(), np.abs(fx).max()
    after = r["q"][stopped[0]:, :2]           # the free wheels unwind it; the cart barely moves (σ_x·κ_peak: 42 mm)
    assert float(np.linalg.norm(after[-1] - after[0])) <= 3.0e-3
