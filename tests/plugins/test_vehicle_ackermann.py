# SPDX-License-Identifier: Apache-2.0
"""Ackermann steering (SCR-WP8-12 F5): GA's truck steers its axle's wheels square to one turning centre on the drive
axle's line (GA: an Ackermann equality on the knuckles, F/steering.py:2). Before, both rear wheels turned by the same
angle and scrubbed, and the truck turned 1.6x wider than its steering angle says at full lock (1.49 m for 0.95 m),
too wide for GA's right-angle turn into a rack face."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.plugins.vehicle import Vehicle
from alloy.plugins.vehicle import laws_axle as ax
from alloy.plugins.vehicle import laws_grip as lg
from alloy.plugins.vehicle.kernels import steer_angle
from alloy.plugins.vehicle.wheels import Wheel, wheel_array, wheel_specs

pytestmark = pytest.mark.law
WHEELBASE = ax.FRONT[0] - ax.REAR[0]


def _circle(xy: np.ndarray) -> float:
    """Radius of the least-squares circle through the points."""
    a = np.c_[2.0 * xy, np.ones(len(xy))]
    c = np.linalg.lstsq(a, (xy**2).sum(axis=1), rcond=None)[0]
    return float(math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2))


def _hold(speed: float):
    def control(t: float, last: dict) -> dict[str, float]:
        u = 0.2 + 2.0 * (speed - float(np.linalg.norm(last["qd"][:2])))
        return {"throttle": float(np.clip(u, 0.0, 1.0)), "brake": 0.0}
    return control


@wp.kernel
def _steer_angles(wheels: wp.array[Wheel], steer: wp.float32, out: wp.array[wp.float32]):
    i = wp.tid()
    out[i] = steer_angle(wheels[i], steer)


def _angles(delta: float) -> np.ndarray:
    """The plug-in's own steering angles (``kernels.steer_angle``) of GA's four wheels for the input ``delta``."""
    config = Vehicle(**lg.TRUCK).config
    wheels = wheel_array(wheel_specs(config), config, "cpu")
    out = wp.zeros(wheels.shape[0], dtype=wp.float32, device="cpu")
    wp.launch(_steer_angles, dim=wheels.shape[0], inputs=[wheels, float(delta)], outputs=[out], device="cpu")
    return out.numpy().astype(np.float64)


def test_inner_wheel_turns_further_than_the_outer():
    """At the operator's full lock (1.1 rad at the centreline) the plug-in turns the inner rear wheel 80.6° (GA's
    axle: about 80°) and the outer 49.5° the other way (rear steer), each square to the centre L / tan δ beside the
    front axle; the front wheels stay straight."""
    delta = 1.1
    a = _angles(delta)
    centre = WHEELBASE / math.tan(delta)
    left, right = a[2], a[3]                                 # rear wheels at y = +0.5685 (inner) and -0.5685
    assert a[0] == 0.0 and a[1] == 0.0
    assert math.degrees(-left) == pytest.approx(80.6, abs=0.1) and math.degrees(-right) == pytest.approx(49.5, abs=0.1)
    for angle, y in ((left, ax.REAR[1]), (right, -ax.REAR[1])):
        assert math.tan(-angle) == pytest.approx(WHEELBASE / (centre - y), rel=1e-5)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("delta", [0.6, 1.05])
def test_truck_turns_at_the_radius_its_steering_sets(delta: float):
    """At 0.5 m/s with the rear axle steered by δ, the front axle's centre circles at L / tan δ within 1 % (the slip
    angles at 0.5 m/s are small)."""
    r = lg.run(lg.TRUCK, 10.0, {"steer": delta, "brake": 0.0}, control=_hold(0.5))
    turning = r["t"] >= 5.0
    radius = _circle(r["q"][turning, :2].astype(np.float64))
    want = WHEELBASE / math.tan(delta)
    assert radius == pytest.approx(want, rel=0.01), (radius, want)
