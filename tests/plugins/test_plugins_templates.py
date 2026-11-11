# SPDX-License-Identifier: Apache-2.0
"""Every template kind scaffolds into a package that passes the harness unchanged."""

from __future__ import annotations

import pytest

import alloy.plugins as ap
from alloy.plugins.scaffold import new_plugin


def _scaffold(tmp_path, kind):
    files = new_plugin(f"my_{kind}", kind=kind, dest=tmp_path)
    text = files[0].read_text()
    assert "plugin_name" not in text and "PluginName" not in text
    return files[0]


def test_force_template_passes(tmp_path):
    report = ap.check(str(_scaffold(tmp_path, "force")), quick=True)
    assert report.passed, report.summary()


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["sensor", "object", "solver"])
def test_other_templates_pass(tmp_path, kind):
    report = ap.check(str(_scaffold(tmp_path, kind)))
    assert report.passed, report.summary()


def test_scaffold_errors(tmp_path):
    with pytest.raises(ap.PluginError, match="snake_case"):
        new_plugin("My-Plugin", dest=tmp_path)
    with pytest.raises(ap.PluginError, match="kind"):
        new_plugin("x", kind="wizard", dest=tmp_path)
    new_plugin("dup", dest=tmp_path)
    with pytest.raises(ap.PluginError, match="already exists"):
        new_plugin("dup", dest=tmp_path)


@pytest.mark.parametrize("kind", ["force", "sensor", "object", "solver"])
def test_templates_are_package_resources(kind):
    """F2: the templates ship inside the package (an installed wheel has no repository folder beside it)."""
    from alloy.plugins.scaffold import template_source

    assert "class PluginName(" in template_source(kind) and 'name = "plugin_name"' in template_source(kind)


def test_graph_replay_scene_leaves_the_mujoco_c_route(monkeypatch):
    """F3: on a CPU the default MuJoCo C route reads the host each step; graph replay runs MuJoCo Warp instead."""
    from alloy.plugins.harness.check import capturable_scene
    from alloy.route.mjc_core_config import CPU_ROUTE_KEY

    scene = ap.TestScene(build=lambda b: None)
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    assert capturable_scene(scene, "cpu").solver_kwargs[CPU_ROUTE_KEY] == "warp"
    monkeypatch.delenv("ALLOY_CPU_ROUTE")
    assert capturable_scene(scene, "cpu").solver_kwargs[CPU_ROUTE_KEY] == "warp"     # unset = mjc for users
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    assert capturable_scene(scene, "cpu") is None
    assert capturable_scene(ap.TestScene(build=lambda b: None, solver="xpbd"), "cpu") is None
