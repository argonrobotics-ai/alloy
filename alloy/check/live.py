# SPDX-License-Identifier: Apache-2.0
"""``check_sim(sim)``: check a simulation that is already running, at its current state (no new run).

Covers overlap at load while the simulation has not stepped yet (t = 0: the start poses, every world), the health
record so far (INVALID worlds and what tripped them) and the deepest contact now. Once it has stepped, contact depth
is resting depth, not a start overlap: it goes to the penetration check (warn past 1 mm, error past 10 mm) and the
overlap check is skipped. Rest drift and energy need a timed run from the start: use ``alloy.check(scene)``.
``sim`` is an ``alloy.api`` ``Sim`` (its ``engine``) or an ``IntegratedSimulation``.

Example::

    >>> sim = alloy.build(alloy.load("demo_peg_d12"), worlds=4)      # doctest: +SKIP
    >>> sim.step(10)
    >>> out = check_sim(sim)
    >>> out["ok"], out["problems"][:1]
    (True, [])
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from alloy.check.contacts import ContactProbe
from alloy.check.objects import ObjectIndex
from alloy.check.run import RunResult
from alloy.check.report import CHECKS, CheckReport, Finding

log = logging.getLogger("alloy.check")

SOURCE = "alloy.check"


def check_sim(sim: Any) -> dict[str, Any]:
    """JSON-ready check of ``sim`` now: ``ok`` (no errors), ``problems`` (errors and warnings as
    {"world", "what", "fix", "check", "level", "objects"}), ``source`` and the full ``report`` (schema
    alloy.check/1).

    Example::

        >>> result = check_sim(sim)  # doctest: +SKIP
        >>> result["ok"], result["source"]
        (True, 'alloy.check')
    """
    report = check_sim_report(sim)
    problems = [_problem(f) for f in report.findings if f.level != "info"]
    return {"ok": report.ok, "problems": problems, "source": SOURCE, "report": report.to_dict()}


def check_sim_report(sim: Any) -> CheckReport:
    """The ``CheckReport`` behind ``check_sim``.

    Example::

        >>> check_sim_report(sim).checks["overlap"]  # doctest: +SKIP
        'pass'
    """
    from alloy.check import events  # noqa: PLC0415
    from alloy.check.overlap import overlap_findings, probe_findings  # noqa: PLC0415

    t0 = time.perf_counter()
    engine = getattr(sim, "engine", sim)
    built = engine.built
    index = ObjectIndex.build(built)
    t = float(engine.clock.t)
    report = CheckReport(scene=_label(built), settings={"worlds": int(built.world_count), "t": t, "live": True})
    report.skip("cards", "checked when the scene was built")
    probe = ContactProbe(built)
    if int(engine.periods) == 0:
        found, metrics = overlap_findings(built, index, engine.state, probe, when="at start")
        report.metrics["overlap"] = metrics
        report.add(found, "overlap")
    else:
        report.skip("overlap", f"start poses only (t = 0); at t={t:.3g} s contact depth is the penetration check")
    res = _result(engine, probe, index, t)
    for name, fn in (("health", events.health_findings), ("penetration", events.penetration_findings)):
        found, metrics = fn(res, index)
        report.metrics[name] = metrics
        report.add(found, name)
    report.add(probe_findings(probe, "penetration"), "penetration")
    for c in CHECKS:
        report.checks.setdefault(c, "skipped: needs a timed run from the start (alloy.check(scene))")
    report.timings_s["total"] = time.perf_counter() - t0
    return report


def _result(engine: Any, probe: ContactProbe, index: ObjectIndex, t: float) -> RunResult:
    report = engine.report()
    h = report.get("health")
    res = RunResult(health=h, control_dt=float(engine.built.control_dt), periods=int(engine.periods),
                    steps=int(engine.steps))
    if h is None:
        return res
    measured = np.asarray(h.pen_measured, bool) if h.pen_measured is not None else np.ones(len(h.severity), bool)
    res.pen_max = np.where(measured, np.asarray(h.max_pen_m, np.float64), np.nan)
    res.pen_t = np.full(len(h.severity), t)
    res.pair_pen_max = np.nan_to_num(np.asarray(h.max_pair_pen_m, np.float64))
    rows = probe.depths(engine.state, index)
    if rows:
        res.attributions.append((t, rows[0]))
    return res


def _problem(f: Finding) -> dict[str, Any]:
    return {"world": f.worlds[0] if f.worlds else None, "what": f.message, "fix": f.fix, "check": f.check,
            "level": f.level, "objects": list(f.objects)}


def _label(built: Any) -> str:
    plan = getattr(built, "_plan", None)
    return str(getattr(getattr(plan, "scene", None), "name", None) or "simulation")


__all__ = ["check_sim", "check_sim_report"]
