# SPDX-License-Identifier: Apache-2.0
"""Heat v1 (Thermal): areas, validation and the contact export (CPU). The laws run in the harness quick check
(tests/plugins/test_plugins_library.py runs every built-in, thermal included)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.thermal import Thermal
from alloy.plugins.thermal.sensor import BOX, CAPSULE, CYLINDER, SPHERE, shape_area


def test_shape_areas():
    assert shape_area(BOX, np.array([0.01, 0.02, 0.03])) == pytest.approx(8 * (2e-4 + 6e-4 + 3e-4))
    assert shape_area(SPHERE, np.array([0.1, 0, 0])) == pytest.approx(4 * math.pi * 0.01)
    r, h = 0.01, 0.05                                   # radius, half height
    assert shape_area(CYLINDER, np.array([r, h, 0])) == pytest.approx(2 * math.pi * r * 2 * h + 2 * math.pi * r * r)
    assert shape_area(CAPSULE, np.array([r, h, 0])) == pytest.approx(2 * math.pi * r * 2 * h + 4 * math.pi * r * r)


def test_config_is_validated():
    for bad in ({"bodies": ("a", "b", "c"), "initial": (300.0, 300.0)}, {"convection": -1.0}, {"ambient": 0.0},
                {"plates": ("p",)}, {"motors": ("m",), "t_max": 300.0, "t_derate": 310.0},
                {"motors": ("m",), "torque_constant": 0.0}):
        with pytest.raises(ValueError):
            Thermal.Config(**bad)


def test_registered_as_a_builtin():
    assert "thermal" in [p.name for p in ap.available()]


def test_reads_contacts_on_newtons_mujoco_c_path():
    """Newton 1.8's MuJoCo C step (use_mujoco_cpu) exports CONTACT_F too (WP0's Newton bump, 2026-11-09):
    the plug-in builds there and its conduction_energy law holds, on one world. Every law of the plug-in passed on that
    path at the bump ([internal note]). One world only: with more,
    Newton's MuJoCo C refuses separate_worlds=False, and with separate_worlds=True steps world 0 alone (DC6),
    so the other worlds read nothing."""
    from alloy.plugins.harness.check import fresh  # noqa: PLC0415
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    from alloy.plugins.thermal.laws import thermal_laws  # noqa: PLC0415

    law = next(x for x in thermal_laws() if x.name == "conduction_energy")
    scene = dataclasses.replace(law.scene or Thermal().test_scene(), worlds=1, solver_kwargs={"use_mujoco_cpu": True})
    run = run_scene(scene, [fresh(Thermal(), law.config)], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, (res.value, res.bound, res.detail)


def test_refuses_a_solver_without_contact_forces():
    """XPBD exports no contacts with forces: the plug-in refuses it at build, naming the route to use."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.plugins.thermal.laws import thermal_laws  # noqa: PLC0415

    scene = dataclasses.replace(thermal_laws()[1].scene, solver="xpbd")
    with pytest.raises(PluginError, match="need contact forces"):
        build_simulation(scene, [Thermal(bodies=("a", "b"))], "cpu", worlds=1)
