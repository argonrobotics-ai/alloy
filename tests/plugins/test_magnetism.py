# SPDX-License-Identifier: Apache-2.0
"""Magnetism plug-in on CPU: closed forms in field.py, config, the drive input, and the harness's quick check (laws)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import fresh
from alloy.plugins.harness.run import run_scene
from alloy.plugins.magnetism import Magnetism
from alloy.plugins.magnetism import field as F
from alloy.plugins.magnetism import laws as ML


def test_field_closed_forms():
    d, m = 0.04, 0.5
    x = np.array([1.0, 0.0, 0.0])
    coax = F.force(np.zeros(3), m * x, d * x, m * x)
    assert coax == pytest.approx([-3 * F.MU0 * m * m / (2 * math.pi * d ** 4), 0, 0], rel=1e-12)
    y = np.array([0.0, 1.0, 0.0])
    side = F.force(np.zeros(3), m * y, d * x, m * y)
    assert side == pytest.approx([3 * F.MU0 * m * m / (4 * math.pi * d ** 4), 0, 0], rel=1e-12)
    p0, m0, p1, m1 = np.zeros(3), np.array([0.3, -0.2, 0.5]), np.array([0.02, 0.03, -0.01]), np.array([-0.1, 0.4, 0.2])
    eps = 1e-7                                                   # F = -∇U (target moved, moments fixed)
    grad = [(F.energy(p0, m0, p1 + eps * e, m1) - F.energy(p0, m0, p1 - eps * e, m1)) / (2 * eps) for e in np.eye(3)]
    assert F.force(p0, m0, p1, m1) == pytest.approx(-np.array(grad), rel=1e-6)
    assert F.torque(p0, m0, p1, m1) == pytest.approx(np.cross(m1, F.field(p0, m0, p1)))


def test_plate_image():
    n, h, m = np.array([0.0, 0.0, 1.0]), 0.01, 0.2
    p = np.array([0.0, 0.0, h])
    for mom, factor in ((np.array([0.0, 0.0, m]), 1.0), (np.array([m, 0.0, 0.0]), 0.5)):
        pi, mi = F.image(p, mom, np.zeros(3), n, 1.0)
        assert pi == pytest.approx([0, 0, -h]) and mi[2] == pytest.approx(mom[2]) and mi[0] == pytest.approx(-mom[0])
        assert F.force(pi, mi, p, mom)[2] == pytest.approx(-factor * F.plate_force_normal(m, h), rel=1e-12)
    assert F.kappa(1.0) == 0.0 and F.kappa(3.0) == 0.5
    assert math.degrees(F.stirrer_lag(1.0, 1.0, 0.5, 1.0)) == pytest.approx(30.0) and math.isnan(
        F.stirrer_lag(1.0, 1.0, 2.0, 1.0))


def test_registered_and_validated():
    assert "magnetism" in [p.name for p in ap.available()]
    with pytest.raises(PluginError, match="one moment per body"):
        Magnetism(bodies=("a", "b", "c"), moments=(1.0, 2.0))
    with pytest.raises(PluginError, match="drive_moments"):
        Magnetism(drives=((0, 0, 0),), drive_moments=())
    with pytest.raises(PluginError, match="plate_mu_r"):
        Magnetism(plates=((0, 0, 0, 0, 0, 1),), plate_mu_r=(0.5,))
    assert Magnetism(drives=(), plates=()).internal and not Magnetism().internal


def test_drive_turns_with_its_input_and_probes_do_not_turn_it():
    mag = fresh(Magnetism(), ML.STIRRER)
    run = run_scene(ML.default_scene(), [mag], "cpu", periods=5)
    assert run.error is None, run.error
    phase = run.trace.plugin("magnetism", "phase")[:, :, 0]
    omega = ML.RPM_LAG[0] * 2 * math.pi / 60
    assert np.diff(np.unwrap(phase[:, 0])) == pytest.approx(np.full(5, omega * 0.02), rel=1e-4)
    before = mag.phase.numpy().copy()
    run.probe(run.trace.body_q[-1])
    assert np.array_equal(mag.phase.numpy(), before)
    mag.set_input("rpm", 0.0)
    run.sim.step_period()
    assert np.array_equal(mag.phase.numpy(), before)


@pytest.mark.law
def test_quick_check_passes_every_law():
    report = ap.check("magnetism", quick=True)
    assert report.passed, report.summary()
