# SPDX-License-Identifier: Apache-2.0
"""Alloy plug-in kit: new forces, sensors, objects and solvers, tested automatically and sandboxed.

Example::

    import alloy.plugins as ap

    print([p.name for p in ap.available()])               # built-ins + installed "alloy.plugins" entry points
    report = ap.check("magnet")                           # laws, crash, determinism, world independence, speed
    print(report.to_json())

Write one: subclass ``ForcePlugin`` / ``SensorPlugin`` / ``ObjectPlugin`` / ``SolverPlugin`` (see GUIDE.md), or
start from the template: ``python -m alloy.plugins new my_force --kind force``.
"""

from alloy.plugins.base import ForcePlugin, Forces, ObjectPlugin, Plugin, SensorPlugin, SolverPlugin
from alloy.plugins.context import PluginContext
from alloy.plugins.errors import PluginError
from alloy.plugins.harness import CheckReport, CheckResult, Law, LawResult, Run, TestScene, body_xform, check
from alloy.plugins.host import PluginHost, simulation
from alloy.plugins.registry import available, create, describe, get, load_entry_points, load_target, register

__all__ = ["CheckReport", "CheckResult", "ForcePlugin", "Forces", "Law", "LawResult", "ObjectPlugin", "Plugin",
           "PluginContext", "PluginError", "PluginHost", "Run", "SensorPlugin", "SolverPlugin", "TestScene",
           "available", "body_xform", "check", "create", "describe", "get", "load_entry_points", "load_target",
           "register", "simulation"]
