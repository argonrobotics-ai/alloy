# SPDX-License-Identifier: Apache-2.0
"""Tactile v1: layouts, presets and the contact export (CPU). The laws run in the harness quick check
(tests/plugins/test_plugins_library.py runs it for every built-in, tactile included); the full check is slow."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.tactile import PRESETS, Tactile, preset
from alloy.plugins.tactile.layout import build_layout, grid_layout


def test_grid_layout_is_centred_and_row_major():
    xy, reach = grid_layout((4, 2), (0.004, 0.003))
    assert xy.shape == (8, 2)
    np.testing.assert_allclose(xy.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(xy[1] - xy[0], (0.004, 0.0))
    np.testing.assert_allclose(xy[4] - xy[0], (0.0, 0.003))
    assert reach == pytest.approx(0.5 * np.hypot(0.004, 0.003))


def test_foveation_packs_taxels_toward_the_centre():
    flat, _ = grid_layout((8, 8), (0.004, 0.004))
    fov, _ = grid_layout((8, 8), (0.004, 0.004), foveation=1.0)
    inner = lambda xy: np.abs(np.diff(np.unique(xy[:, 0]))).min()  # noqa: E731
    assert inner(fov) < inner(flat)
    assert np.abs(fov).max() <= np.abs(flat).max() + 1e-12


def test_presets_resolve_with_their_sources():
    for name, p in PRESETS.items():
        assert p.source, name
        layout = build_layout(Tactile.Config(preset=name))
        assert layout.reach > 0 and len(layout.xy) == (len(p.points) or p.grid[0] * p.grid[1])
    assert len(build_layout(Tactile.Config(preset="reskin")).xy) == 5
    with pytest.raises(ValueError, match="BioTac and Inspire"):
        preset("biotac")


def test_reskin_reach_covers_its_sensing_area():
    layout = build_layout(Tactile.Config(preset="reskin"))
    corner = np.array([0.01, 0.01])
    assert np.linalg.norm(layout.xy - corner, axis=1).min() <= layout.reach


def test_config_is_validated():
    with pytest.raises(ValueError):
        Tactile.Config(bodies=("a", "b", "c"), origins=((0, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError):
        Tactile.Config(noise=-0.1)
    with pytest.raises(ValueError):
        build_layout(Tactile.Config(points=((0.0, 0.0),)))


def test_registered_as_a_builtin():
    assert "tactile" in [p.name for p in ap.available()]


def test_reads_contacts_on_newtons_mujoco_c_path():
    """Newton 1.8's MuJoCo C step (use_mujoco_cpu) exports CONTACT_F too (WP0's Newton bump, 2026-11-09):
    the plug-in builds there and its force_weight law holds, on one world. Every law of the plug-in passed on that
    path at the bump ([internal note]). One world only: with more,
    Newton's MuJoCo C refuses separate_worlds=False, and with separate_worlds=True steps world 0 alone (DC6),
    so the other worlds read nothing."""
    from alloy.plugins.harness.check import fresh  # noqa: PLC0415
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    from alloy.plugins.tactile.laws import rest_scene, tactile_laws  # noqa: PLC0415

    law = next(x for x in tactile_laws() if x.name == "force_weight")
    scene = dataclasses.replace(law.scene or rest_scene(), worlds=1, solver_kwargs={"use_mujoco_cpu": True})
    run = run_scene(scene, [fresh(Tactile(), law.config)], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, (res.value, res.bound, res.detail)


def test_refuses_a_solver_without_contact_forces():
    """XPBD exports no contacts with forces: the plug-in refuses it at build, naming the route to use."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.plugins.tactile.laws import rest_scene  # noqa: PLC0415

    scene = dataclasses.replace(rest_scene(), solver="xpbd")
    with pytest.raises(PluginError, match="need contact forces"):
        build_simulation(scene, [Tactile()], "cpu", worlds=1)


def test_refuses_newtons_mujoco_c_step_on_several_worlds():
    """Newton 1.8's own MuJoCo C step simulates world 0 only (separate worlds), so on 2 worlds a contact plug-in
    would read nothing in world 1, with no error: the host refuses it at build (WP0's Newton bump, 2026-11-09)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.plugins.tactile.laws import rest_scene  # noqa: PLC0415

    scene = dataclasses.replace(rest_scene(), solver_kwargs={"use_mujoco_cpu": True, "separate_worlds": True})
    with pytest.raises(PluginError, match="simulates world 0 only of 2"):
        build_simulation(scene, [Tactile()], "cpu", worlds=2)


def test_alloys_cpu_route_reads_contacts_in_every_world():
    """Alloy's CPU route (MjcCoreSolver: use_mujoco_cpu too, one MjData per world) is not refused: on 2 worlds the
    box's weight reaches the pad in world 1 as in world 0."""
    from alloy.plugins.harness.run import run_scene  # noqa: PLC0415
    from alloy.plugins.tactile.laws import rest_scene  # noqa: PLC0415
    from alloy.route.mjc_core_config import CPU_ROUTE_KEY, MJC  # noqa: PLC0415

    scene = dataclasses.replace(rest_scene(), worlds=2, solver_kwargs={CPU_ROUTE_KEY: MJC})
    run = run_scene(scene, [Tactile()], "cpu")
    assert run.error is None, run.error
    assert type(run.sim.stepper.solver).__name__ == "MjcCoreSolver"
    m = float(run.model.body_mass.numpy()[run.bodies("box")[0]])
    pad = np.asarray(run.trace.plugin(run.plugin.label, "pad_force"))[-1, :, 0, 0]       # [world], normal (N)
    assert pad == pytest.approx([m * 9.81, m * 9.81], rel=5e-3)
