# SPDX-License-Identifier: Apache-2.0
"""Tactile v2 physics (GelSensor): geometry, the Green's tensor, Taxim's axes and validation (CPU). The laws run in the
harness quick check (tests/plugins/test_plugins_library.py runs every built-in, gel_tactile included)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.tactile import GEL_PRESETS, GelSensor
from alloy.plugins.tactile import taxim
from alloy.plugins.tactile.gel import marker_grid
from alloy.plugins.tactile.gel_laws import _green_np


def test_marker_grid_is_centred_inside_the_margin():
    xy = marker_grid((5, 7), (0.02, 0.014), 0.1)
    assert xy.shape == (35, 2)
    np.testing.assert_allclose(xy.mean(0), 0.0, atol=1e-12)
    assert np.abs(xy[:, 0]).max() == pytest.approx(0.008) and np.abs(xy[:, 1]).max() == pytest.approx(0.0056)
    assert xy[0, 1] > xy[-1, 1]                    # row 0 at +y, like the depth image
    assert marker_grid((0, 0), (0.02, 0.014), 0.1).shape == (0, 2)


@pytest.mark.parametrize("name", sorted(GEL_PRESETS))
def test_presets_divide_into_cells_and_load_blocks(name):
    h, w = GEL_PRESETS[name]["image"]
    c = GelSensor.Config(preset=name)
    assert h // c.stride % c.block == 0 and w // c.stride % c.block == 0


def test_config_is_validated():
    for bad in ({"preset": "biotac"}, {"kernel": "fem"}, {"poisson": 0.5}, {"max_depth": 0.0},
                {"bodies": ("a", "b", "c"), "origins": ((0, 0, 0), (0, 0, 0))}):
        with pytest.raises(ValueError):
            GelSensor.Config(**bad)


def test_analytic_green_tensor_structure():
    dx, dy = np.array([0.001, 0.0, 0.002]), np.array([0.0, 0.001, -0.0015])
    g = _green_np(dx, dy, 1e-4, 0.48, None, 1.0)
    np.testing.assert_allclose(g[..., 0, 1], g[..., 1, 0])           # shear block symmetric
    np.testing.assert_allclose(g[..., 0, 2], -g[..., 2, 0])          # normal/shear coupling antisymmetric
    assert (g[..., 2, 2] > 0).all() and (g[..., 0, 0] > 0).all()
    far = _green_np(np.array([0.004]), np.array([0.0]), 1e-4, 0.48, None, 1.0)
    assert far[0, 2, 2] < g[0, 2, 2] / 3.0                            # decays ~1/r


def test_taxim_rejects_unknown_files():
    with pytest.raises(ValueError):
        taxim.fetch("calibs/nothing.npz")


@pytest.mark.skipif(not (taxim.cache_dir() / "calibs/femCalib.npz").exists(), reason="Taxim's femCalib.npz not cached")
def test_taxim_tensor_in_sensor_axes():
    raw, ours = taxim.fem_tensor(), taxim.fem_tensor_sensor_axes()
    s = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(ours[450, 450], s @ raw[450, 450] @ s, rtol=1e-6)
    assert ours[450, 450, 2, 2] > 0 and ours[450, 450, 0, 0] > 0


def test_registered_as_a_builtin():
    assert "gel_tactile" in [p.name for p in ap.available()]


def test_reads_contacts_on_newtons_mujoco_c_path():
    """Newton 1.8's MuJoCo C step (use_mujoco_cpu) exports CONTACT_F too (WP0's Newton bump, 2026-11-09):
    the plug-in builds there and its force_weight law holds, on one world. Every law of the plug-in passed on that
    path at the bump ([internal note]). One world only: with more,
    Newton's MuJoCo C refuses separate_worlds=False, and with separate_worlds=True steps world 0 alone (DC6),
    so the other worlds read nothing."""
    from alloy.plugins.harness.check import fresh  # noqa: PLC0415
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    from alloy.plugins.tactile.gel_laws import gel_laws, press_scene  # noqa: PLC0415

    law = next(x for x in gel_laws() if x.name == "force_weight")
    scene = dataclasses.replace(law.scene or press_scene(), worlds=1, solver_kwargs={"use_mujoco_cpu": True})
    run = run_scene(scene, [fresh(GelSensor(), law.config)], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, (res.value, res.bound, res.detail)


def test_refuses_a_solver_without_contact_forces():
    """XPBD exports no contacts with forces: the plug-in refuses it at build, naming the route to use."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.plugins.tactile.gel_laws import press_scene  # noqa: PLC0415

    scene = dataclasses.replace(press_scene(), solver="xpbd")
    with pytest.raises(PluginError, match="need contact forces"):
        build_simulation(scene, [GelSensor()], "cpu", worlds=1)
