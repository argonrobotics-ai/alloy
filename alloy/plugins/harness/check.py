# SPDX-License-Identifier: Apache-2.0
"""``check(plugin)``: run every harness check on one plug-in and return a ``CheckReport``.

Force and sensor plug-ins: contract, float-atomic lint, crash (hooks may not sync, read back or allocate),
determinism (two runs, bit-identical digests), world independence (identical worlds agree; perturbing worlds 1..
leaves world 0 alone), graph replay (captured periods = eager), action-reaction for
``internal`` forces, the plug-in's own laws, and speed (step time with / without it). Objects: contract, lint, a drop
test (crash, determinism, independence, mass and inertia, comes to rest). Solvers: the same run checks on the
solver's own test scene, inside a scratch solver registry.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import traceback
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.base import ForcePlugin, ObjectPlugin, Plugin, SolverPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.harness import checks
from alloy.plugins.harness.hooks import hook_checks
from alloy.plugins.harness.report import CheckReport, CheckResult
from alloy.plugins.harness.run import Run, run_scene, time_scene
from alloy.plugins.harness.scene import Law, TestScene

log = logging.getLogger("alloy.plugins")

DEFAULT_SPEED_BUDGET = 2.0


def _versions() -> dict[str, str]:
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415

    import alloy  # noqa: PLC0415

    return {"alloy": alloy.__version__, "warp": wp.config.version, "newton": newton.__version__}


def instance(target: Any) -> Plugin:
    """A plug-in instance from an instance, a class, a registry name, ``"module:Class"`` or a ``.py`` path."""
    if isinstance(target, Plugin):
        return target
    if isinstance(target, type) and issubclass(target, Plugin):
        return target()
    if isinstance(target, str):
        from alloy.plugins.registry import load_target  # noqa: PLC0415

        classes = load_target(target)
        if len(classes) > 1:
            raise PluginError(f"{target} defines {len(classes)} plug-ins {[c.name for c in classes]}: check one, "
                              f"e.g. '{target}:{classes[0].__name__}' or its registry name")
        return classes[0]()
    raise PluginError(f"cannot check {target!r}: pass a plug-in, its class, its name or 'module:Class'")


def fresh(plugin: Plugin, overrides: Mapping[str, Any] | None = None) -> Plugin:
    """A new instance with the same config (plus ``overrides``) and label (setup allocates per simulation)."""
    return type(plugin)(plugin.config, label=plugin.label, **dict(overrides or {}))


def check(target: Any, *, device: str = "cpu", quick: bool = False, laws: bool = True,
          speed: bool = True) -> CheckReport:
    """Run the harness on ``target``; ``quick`` skips the repeat runs (determinism, independence) and speed.

    Example::

        report = check("magnet")                 # or check(MyForce(k=2.0)), check("my_pkg.mod:MyForce")
        assert report.passed, report.summary()
    """
    t0 = time.perf_counter()
    plugin = instance(target)
    cls = type(plugin)
    report = CheckReport(cls.name, cls.kind, f"{cls.__module__}:{cls.__qualname__}", device, versions=_versions())
    report.add(checks.contract(cls))
    report.add(checks.lint(cls))
    try:
        if isinstance(plugin, ObjectPlugin):
            _check_object(plugin, report, device, quick)
        elif isinstance(plugin, SolverPlugin):
            _check_solver(plugin, report, device, quick, laws)
        else:
            _check_stepping(plugin, report, device, quick, laws, speed)
    except PluginError as err:
        report.add(CheckResult("setup", "contract", False, detail=str(err), fix=str(err)))
    report.seconds = time.perf_counter() - t0
    log.info("plug-in check %s: %s in %.1f s", cls.name, "PASS" if report.passed else "FAIL", report.seconds)
    return report


def _scene(plugin: Plugin, report: CheckReport) -> TestScene | None:
    try:
        scene = plugin.test_scene()
    except PluginError as err:
        report.add(CheckResult("test_scene", "contract", False, detail=str(err), fix=str(err)))
        return None
    if not isinstance(scene, TestScene):
        report.add(CheckResult("test_scene", "contract", False, detail=f"test_scene() returned {type(scene)}",
                               fix="return alloy.plugins.TestScene(build=...)"))
        return None
    return scene


def capturable_scene(scene: TestScene, device: str) -> TestScene | None:
    """``scene`` on a route whose steps can be captured, or None when it already is one. On a CPU, a MuJoCo entry
    on Alloy's MuJoCo C route (the default there) reads the host every step, so the graph replay runs it on MuJoCo
    Warp (the entry's ``alloy:cpu_route``) in both its eager and its captured arm."""
    from alloy.route.mjc_core_config import CPU_ROUTE_KEY, MJC, WARP, requested_route  # noqa: PLC0415

    if not wp.get_device(device).is_cpu or scene.solver != "mujoco":
        return None
    if requested_route(dict(scene.solver_kwargs)) != MJC:
        return None
    return dataclasses.replace(scene, solver_kwargs={**dict(scene.solver_kwargs), CPU_ROUTE_KEY: WARP})


def _graph_replay(a: Run, scene: TestScene, plugins: list[Plugin], device: str) -> CheckResult:
    warp_scene = capturable_scene(scene, device)
    if warp_scene is None:
        return checks.graph_replay(a, run_scene(scene, [fresh(x) for x in plugins], device, graph=True))
    eager = run_scene(warp_scene, [fresh(x) for x in plugins], device)
    res = checks.graph_replay(eager, run_scene(warp_scene, [fresh(x) for x in plugins], device, graph=True))
    res.detail += " (CPU: both arms on MuJoCo Warp; the default MuJoCo C route reads the host each step)"
    return res


def _repeat_checks(a: Run, scene: TestScene, plugins: list[Plugin], device: str) -> list[CheckResult]:
    """Determinism, world independence and graph replay against run ``a`` (three or four more runs)."""
    b = run_scene(scene, [fresh(p) for p in plugins], device)
    p = run_scene(scene, [fresh(x) for x in plugins], device, perturb=True)
    return [checks.determinism(a, b), *checks.independence(a, p), _graph_replay(a, scene, plugins, device)]


def _run_checks(scene: TestScene, plugins: list[Plugin], report: CheckReport, device: str, quick: bool,
                engine_may_differ: bool = True) -> Run:
    """Crash on one run; hook-level checks; whole-run determinism, world independence and graph replay (unless
    ``quick``). On a GPU, a whole-run repeat check that fails while every hook-level check passed is the engine's
    (MuJoCo contacts are not run-to-run deterministic there by default): it is skipped with that reason, unless
    ``engine_may_differ`` is False (solver plug-ins: the solver is the plug-in)."""
    a = run_scene(scene, plugins, device)
    report.add(checks.crash(a))
    hook = hook_checks(a) if plugins else []
    report.add(hook)
    if quick:
        report.add(CheckResult.skip("determinism", "determinism", "quick"))
        return a
    results = _repeat_checks(a, scene, plugins, device)
    if engine_may_differ and wp.get_device(device).is_cuda and all(h.passed for h in hook):
        results = [_engine_verdict(r, device, bool(hook)) for r in results]
    report.add(results)
    return a


def _engine_verdict(res: CheckResult, device: str, hooks_ran: bool) -> CheckResult:
    if res.passed or res.skipped:
        return res
    why = "the plug-in's own hooks passed the same property on this device" if hooks_ran else "objects are geometry"
    return CheckResult.skip(res.name, res.group, f"engine on {device}: {res.detail}; MuJoCo contacts on a GPU are "
                            f"not run-to-run deterministic by default and {why} (the CPU run tests it end to end)")


def _check_stepping(plugin: Plugin, report: CheckReport, device: str, quick: bool, laws: bool, speed: bool) -> None:
    scene = _scene(plugin, report)
    if scene is None:
        return
    a = _run_checks(scene, [fresh(plugin)], report, device, quick)
    if isinstance(plugin, ForcePlugin) and plugin.internal:
        report.add(checks.action_reaction(a))
    if laws:
        for law in plugin.laws():
            report.add(_law(law, scene, [fresh(plugin, law.config)], device))
    if speed and not quick and a.error is None:
        report.add(_speed(plugin, scene, device))


def _law(law: Law, default: TestScene, plugins: list[Plugin], device: str,
         subject: Plugin | None = None) -> CheckResult:
    if not isinstance(law, Law):
        return CheckResult(str(law), "law", False, detail="laws() must return alloy.plugins.Law objects",
                           fix="return [Law(name, check=fn, scene=TestScene(...))]")
    run = run_scene(law.scene or default, plugins, device)
    run.plugin = run.plugin or subject            # objects and solvers: the plug-in itself
    if run.error is not None:
        return CheckResult(law.name, "law", False, detail=f"law scene crashed: {run.error.strip().splitlines()[-1]}",
                           fix=run.error[-1500:])
    try:
        res = law.check(run)
    except Exception:  # noqa: BLE001 - a broken law is a failing law, with its traceback
        return CheckResult(law.name, "law", False, detail="law check raised", fix=traceback.format_exc()[-1500:])
    return CheckResult(law.name, "law", bool(res.passed), res.value, res.bound, res.detail or law.doc, res.fix)


def _speed(plugin: Plugin, scene: TestScene, device: str) -> CheckResult:
    periods = max(2, min(scene.periods, 10))
    with_p = time_scene(scene, [fresh(plugin)], device, periods)
    without = time_scene(scene, [], device, periods)
    return checks.speed(with_p, without, float(getattr(plugin, "speed_budget", DEFAULT_SPEED_BUDGET)))


def _check_object(plugin: ObjectPlugin, report: CheckReport, device: str, quick: bool) -> None:
    scene = _scene(plugin, report)
    if scene is None:
        return
    a = _run_checks(scene, [], report, device, quick)
    if a.error is None:
        rows = np.nonzero(a.model.body_world.numpy() == 0)[0]
        report.add(checks.object_mass(a, rows))
        report.add(checks.settles(a, rows))
    for law in plugin.laws():
        report.add(_law(law, scene, [], device, plugin))


def _check_solver(plugin: SolverPlugin, report: CheckReport, device: str, quick: bool, laws: bool) -> None:
    from alloy.core.registry import isolated_registry, register_solver  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

    with isolated_registry():
        register_builtin_solvers()
        cap = plugin.capability()
        register_solver(cap, replace=True)
        scene = _scene(plugin, report)
        if scene is None:
            return
        if scene.solver != cap.name:
            report.add(CheckResult("test_scene", "contract", False, detail=f"scene solver {scene.solver!r}",
                                   fix=f"set TestScene(solver={cap.name!r}) so the scene runs your solver"))
            return
        _run_checks(scene, [], report, device, quick, engine_may_differ=False)   # the solver is the plug-in
        if laws:
            for law in plugin.laws():
                report.add(_law(law, scene, [], device, plugin))
        report.add(CheckResult.skip("speed", "speed", "solver speed is reported by the bench, not gated here"))


__all__ = ["check", "fresh", "instance"]
