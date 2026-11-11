# SPDX-License-Identifier: Apache-2.0
"""Vehicle plug-in (Gym-Anything forklift gap): harness laws vs closed forms, braking, traction, steering, config."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.vehicle import Vehicle
from alloy.plugins.vehicle import laws as vl

pytestmark = pytest.mark.law
G = 9.81


@pytest.mark.compile_heavy
def test_harness_quick_check_passes():
    report = ap.check(Vehicle(), quick=True)
    assert report.passed, report.summary()
    assert {"straight_accel", "coast_down", "cornering", "airborne"} <= {c.name for c in report.checks}


@pytest.mark.parametrize("bad", [{"driven": (False,)}, {"radius": (0.3, 0.3, 0.3)}, {"radius": (-0.3,)},
                                 {"mu": 0.0}, {"gear": math.nan}, {"positions": ((0.0, 0.0),)}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(PluginError):
        Vehicle(**bad)


def _run(car: Vehicle, init, periods: int = 50):
    run = run_scene(TestScene(build=vl.cart(car.config), init=init, ground=False, periods=periods, worlds=1),
                    [car], "cpu")
    assert run.error is None, run.error
    b = int(run.bodies(car.config.chassis)[0])
    return run, b


def test_full_brake_locks_without_reversing_and_stops_within_the_friction_band():
    """Full brake from 5 m/s: the wheels lock (never spin backwards) and the cart slides to rest at a deceleration
    between the locked-wheel friction (65 % of the peak at C = 1.65) and the peak."""
    car = Vehicle()
    def init(sim):
        vl.start(speed=5.0)(sim)
        sim.plugins[car.label].set_input("brake", 1.0)
    run, b = _run(car, init, periods=60)
    v = run.trace.body_qd[:, b, 0].astype(np.float64)
    omega = np.asarray(run.trace.plugin(car.label, "omega"))[:, 0]
    moving = v > 0.1
    assert omega[moving].min() >= 0.0                          # no reversal while it slides
    assert omega.min() > -0.1 and abs(v[-1]) < 0.05            # at rest: a settling flutter of mm/s at the rim
    stop = int(np.argmax(v < 0.05))
    decel = (v[0] - v[stop]) / (run.trace.t[stop] - run.trace.t[0])
    mu = car.config.mu
    assert 0.6 * mu * G < decel < 1.02 * mu * G, decel


def test_wheelspin_caps_the_acceleration_at_the_traction_limit():
    """On μ = 0.2 a strong motor spins the rear wheels: the cart accelerates no faster than μ·F_z(rear)/m, and no
    slower than the spinning tyre's grip (sin(1.65·π/2) = 0.52 of the peak at large slip, less the front wheels)."""
    car = Vehicle(mu=0.2, motor_torque=400.0, motor_free_speed=1.0e6)
    run, b = _run(car, vl.start(throttle=1.0), periods=40)
    kappa = np.asarray(run.trace.plugin(car.label, "slip"))[-1, 0, 2:, 0]
    fz_rear = np.asarray(run.trace.plugin(car.label, "force"))[-1, 0, 2:, 2].sum()
    t, v = np.asarray(run.trace.t), run.trace.body_qd[:, b, 0].astype(np.float64)
    a = float(np.polyfit(t[10:], v[10:], 1)[0])
    m = float(run.model.body_mass.numpy()[b])
    assert kappa.min() > 0.2                                    # spinning
    assert a <= 1.02 * 0.2 * fz_rear / m
    assert a >= 0.4 * 0.2 * fz_rear / car.effective_mass(m)


def test_rear_steering_turns_the_cart_the_input_way():
    """Rear wheels steering opposite to the input (a forklift): a positive input still yaws the cart left."""
    car = Vehicle(steer=(0.0, 0.0, -1.0, -1.0), rolling_resistance=0.0)
    run, b = _run(car, vl.start(speed=3.0, steer=0.1), periods=40)
    yaw = run.trace.body_qd[-10:, b, 5].mean()
    assert yaw == pytest.approx(3.0 * 0.1 / vl.WHEELBASE, rel=0.05)


def test_stiff_tyres_are_refused():
    car = Vehicle(radial_stiffness=(1.0e9,))
    sim = build_simulation(vl.drive_scene(car.config), [car], "cpu", worlds=1)
    with pytest.raises(PluginError, match="tyre springs"):
        sim.step_period()


def test_the_step_check_reads_one_world_whatever_the_world_count():
    """100k red (rc-1109): the tread load summed every world's wheel bodies, so the tread checks grew with
    sqrt(worlds) (3.7 at 512 worlds, 5.3 at 1,024 for task_forklift_pick) and refused a stable vehicle. They read one
    world's load: the same at 1 and 8 worlds."""
    reads = []
    for worlds in (1, 8):
        car = Vehicle()
        sim = build_simulation(vl.drive_scene(car.config), [car], "cpu", worlds=worlds)
        sim.step_period()                                  # the host checks the step here (check_dt)
        reads.append({**car._tread_checks(1.0e-3), **car._surface_checks(1.0e-3)})
    assert reads[0].keys() == reads[1].keys() and reads[0]
    for name, v in reads[0].items():
        assert reads[1][name] == pytest.approx(v, rel=1e-6), name


@pytest.mark.gpu
def test_harness_full_check_on_gpu():
    """Laws, hooks, determinism, worlds, graph replay and speed with real CUDA graphs (3 worlds: a light smoke)."""
    report = ap.check(Vehicle(), device="cuda:0")
    print(report.summary())
    assert report.passed, report.summary()
