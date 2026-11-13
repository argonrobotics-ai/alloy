# SPDX-License-Identifier: Apache-2.0
"""``check(scene, worlds, seconds)``: every physics check of a scene in one call, as a ``CheckReport``.

Stages: card validation, assembly, overlap at load (every world, so randomized poses are covered), then a timed run
with no control input for rest drift, maximum penetration, health events and energy drift. ``run=False`` stops after
the load checks (about the assembly time: what importers and the scene builder call after every placement).

Defaults: on a CUDA device 64 worlds for 2 s; on CPU (no GPU, or ``device="cpu"``) 2 worlds for 0.5 s, for small
scenes. ``seconds`` and ``worlds`` override both. On a GPU the run uses at most the worlds that fit in free device
memory (``alloy.check.memory``: an 80 mm MPM dough ball needs about 0.7 GB per world); ``settings["worlds"]`` is the count that
ran, ``settings["worlds_why"]`` says why, and a smaller count than asked is an info finding.

Example::

    >>> from alloy.check import check
    >>> report = check("demo_peg_d12", worlds=2, seconds=0.2, device="cpu")
    >>> report.ok, sorted(report.checks)[:2]
    (True, ['cards', 'energy'])
    >>> print(report.to_json())                                      # doctest: +SKIP
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from alloy.check.cards import card_error_finding, card_findings
from alloy.check.contacts import ContactProbe
from alloy.check.memory import WorldFit, fit_worlds, free_bytes
from alloy.check.objects import ObjectIndex
from alloy.check.run import RunOptions
from alloy.check.source import Source, load_source
from alloy.check.report import CHECKS, CheckReport, Finding
from alloy.core.errors import AlloyError, CardError

log = logging.getLogger("alloy.check")

GPU_WORLDS, GPU_SECONDS = 64, 2.0
CPU_WORLDS, CPU_SECONDS = 2, 0.5
RUN_CHECKS = ("rest_drift", "penetration", "health", "energy")
ExtraCheck = Callable[[Any, Any, Any], tuple[list[Finding], dict[str, Any]]]   # fn(res, built, index)


def check(scene: Any, worlds: int | None = None, seconds: float | None = None, *, device: str | None = None,
          run: bool = True, expect_rest: bool = True, profile: str = "standard", seed: int = 0,
          library: Any = None, budget_s: float | None = 25.0, extra: Sequence[ExtraCheck] = (),
          contacts: str | None = None) -> CheckReport:
    """Check ``scene`` (YAML path, library scene name, YAML text or ``SceneCard``) and return a ``CheckReport``.

    Never raises for a broken scene: problems are findings, each with the change that fixes it. Raises
    ``ValueError`` only for a bad argument (the message says what to pass). ``budget_s`` caps the run's wall time:
    a slow scene runs fewer seconds (``metrics["run"]["truncated"]``; None runs them all). ``extra``: more checks on
    the same run, each ``fn(res, built, index) -> (findings, metrics)`` (``alloy.check.run.RunResult``,
    ``BuiltScene``, ``alloy.check.objects.ObjectIndex``), named by ``fn.check_name`` (else ``fn.__name__``); their
    findings carry that name and their verdicts and metrics join the report under it. ``contacts``: the run's
    contact pipeline, as ``alloy.build(contacts=)`` ("auto", "mujoco", "newton"; None: the profile's own).

    Example::

        >>> r = check("library/scenes/demo_peg_d12.yaml", run=False)
        >>> r.checks["overlap"]
        'pass'
    """
    t0 = time.perf_counter()
    dev, worlds, seconds = resolve_settings(device, worlds, seconds)
    if contacts not in (None, "auto", "mujoco", "newton"):
        from alloy.api.errors import SceneError  # noqa: PLC0415

        raise SceneError(f"check(contacts=...) got {contacts!r}", "pass contacts='auto', 'mujoco', 'newton' or None "
                         "(None: the profile's own pipeline)", field="contacts")
    try:
        src = load_source(scene, library)
    except CardError as exc:
        report = CheckReport(scene=str(scene)[:80])
        report.add([card_error_finding(exc)], "cards")
        return _skip_rest(report, "the scene card does not load", t0)
    report = CheckReport(scene=src.label, settings={"worlds": worlds, "seconds": seconds if run else 0.0,
                                                    "device": dev, "profile": profile, "seed": seed,
                                                    "expect_rest": expect_rest, "contacts": contacts})
    _stage(report, "cards", lambda: report.add(_fetch(src) + card_findings(src), "cards"))
    if not report.ok:
        return _skip_rest(report, "card errors (fix them first)", t0)
    built = _assemble_fitting(report, src, dev, worlds) if run else _assemble(report, src, dev, worlds)
    if built is None:
        return _skip_rest(report, "the scene does not assemble", t0)
    index = ObjectIndex.build(built, src.card, src.lib)
    probe = ContactProbe(built)
    _stage(report, "overlap", lambda: _overlap(report, built, index, probe))
    if run:
        opts = RunOptions(seconds, profile, seed, budget_s, contacts)
        _stage(report, "run", lambda: _dynamic(report, built, index, probe, opts, expect_rest, extra))
    else:
        for c in (*RUN_CHECKS, *(extra_name(fn) for fn in extra)):
            report.skip(c, "run=False (load checks only)")
    report.timings_s["total"] = time.perf_counter() - t0
    log.info("check %s: %s in %.1f s", src.label, "OK" if report.ok else "FAIL", report.timings_s["total"])
    return report


def resolve_settings(device: str | None, worlds: int | None, seconds: float | None) -> tuple[str, int, float]:
    """(device, worlds, seconds) with the GPU or CPU defaults filled in; validates the values.

    Example::

        >>> resolve_settings("cpu", None, None)
        ('cpu', 2, 0.5)
    """
    import warp as wp  # noqa: PLC0415

    dev = device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    cpu = not wp.get_device(dev).is_cuda
    worlds = int(worlds) if worlds is not None else (CPU_WORLDS if cpu else GPU_WORLDS)
    seconds = float(seconds) if seconds is not None else (CPU_SECONDS if cpu else GPU_SECONDS)
    if worlds < 1:
        raise ValueError(f"worlds={worlds}: pass worlds >= 1 (64 is the GPU default)")
    if not seconds > 0.0:
        raise ValueError(f"seconds={seconds}: pass seconds > 0 (2.0 is the GPU default), or run=False")
    return dev, worlds, seconds


def _fetch(src: Source) -> list[Finding]:
    """Files of imported objects the scene places, fetched from the asset store first (``alloy.check.fetch``)."""
    from alloy.check.fetch import fetch_imported  # noqa: PLC0415

    return fetch_imported(src.lib, [pl.object for pl in src.card.objects])


def _stage(report: CheckReport, name: str, fn) -> None:
    t = time.perf_counter()
    fn()
    report.timings_s[name] = time.perf_counter() - t


def _assemble(report: CheckReport, src: Source, device: str, worlds: int) -> Any:
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    t = time.perf_counter()
    try:
        built = SceneAssembler(src.lib).assemble(src.card, device, world_count=worlds)
    except AlloyError as exc:
        report.add([Finding("cards", "error", f"the scene does not assemble: {type(exc).__name__}: {exc}",
                            "change what the message names (a route, capacity or card value); card checks passed, "
                            "so this is a combination the engine cannot build")], "cards")
        return None
    report.timings_s["assemble"] = report.timings_s.get("assemble", 0.0) + time.perf_counter() - t
    report.metrics["scene"] = {"bodies": int(built.model.body_count), "shapes": int(built.model.shape_count),
                               "particles": int(built.model.particle_count), "worlds": int(built.world_count),
                               "control_dt": float(built.control_dt),
                               "entries": [e.capability for e in built.entries if e.name]}
    return built


def _assemble_fitting(report: CheckReport, src: Source, device: str, worlds: int) -> Any:
    """Assemble ``worlds`` worlds; when fewer fit the device memory, assemble that many instead (``_fit``)."""
    free = free_bytes(device)                        # before assembly: what the run may use
    built = _assemble(report, src, device, worlds)
    if built is None or not _fit(report, built, device, free).capped:
        return built
    built = None
    gc.collect()                                     # release the first assembly before the smaller one
    return _assemble(report, src, device, report.settings["worlds"])


def _fit(report: CheckReport, built: Any, device: str, free: float | None) -> WorldFit:
    """Size the run's world count to the device memory (``alloy.check.memory``); the report says how many and why."""
    fit = fit_worlds(built, int(built.world_count), device, free)
    report.settings.update(worlds=fit.worlds, worlds_asked=fit.asked, worlds_why=fit.why)
    if fit.free_bytes is not None:
        report.metrics["memory"] = fit.to_dict()
    if fit.capped:
        report.add([Finding("health", "info", f"ran {fit.worlds} of the {fit.asked} worlds asked, as many as fit "
                            f"in free GPU memory ({fit.basis})",
                            "nothing to change in the scene; for more worlds free GPU memory or use a larger GPU "
                            "(worlds= / CLI --worlds sets the count asked)", value=fit.worlds, limit=fit.asked,
                            unit="worlds")])
    return fit


def _overlap(report: CheckReport, built: Any, index: ObjectIndex, probe: Any) -> None:
    from alloy.check.overlap import overlap_findings  # noqa: PLC0415

    found, metrics = overlap_findings(built, index, probe=probe)
    report.metrics["overlap"] = metrics
    report.add(found, "overlap")


def _dynamic(report: CheckReport, built: Any, index: ObjectIndex, probe: Any, opts: Any, expect_rest: bool,
             extra: Sequence[ExtraCheck] = ()) -> None:
    from alloy.check import events, motion  # noqa: PLC0415
    from alloy.check.run import run_scene  # noqa: PLC0415

    res = run_scene(built, index, opts, probe)
    report.metrics["run"] = {"periods": res.periods, "steps": res.steps, "native_contacts": res.native,
                             "seconds_run": res.seconds_run, "truncated": res.truncated,
                             "graph_error": res.graph_error}
    if res.truncated:
        report.add([Finding("rest_drift", "info", f"ran {res.seconds_run:.3g} s of the {opts.seconds:.3g} s asked, "
                            f"to stay within the {opts.budget_s:.0f} s wall budget ({res.steps} solver steps)",
                            "pass budget_s=None (CLI --budget 0) for the full run, or a smaller worlds= to go faster",
                            value=res.seconds_run, limit=opts.seconds, unit="s")])
    stages = (("rest_drift", lambda: motion.motion_findings(res, built, index, expect_rest)),
              ("health", lambda: events.health_findings(res, index)),
              ("penetration", lambda: events.penetration_findings(res, index)),
              ("energy", lambda: events.energy_findings(res, index)))
    for name, fn in stages:
        if res.periods == 0 and name != "health":
            report.skip(name, "the simulation did not run (the health finding says why)")
            continue
        found, metrics = fn()
        report.metrics[name] = metrics
        if name == "penetration" and metrics.get("max_pen_m") is None:
            report.skip(name, "no contact source measured penetration")
        elif name == "energy" and not metrics:
            report.skip(name, "energy monitors unavailable for this scene")
        else:
            report.add(found, name)
    _probe_lost(report, probe)
    _extra_checks(report, res, built, index, extra)


def _probe_lost(report: CheckReport, probe: Any) -> None:
    """The probe lost contacts during the run (not at the start): the penetration pairs are incomplete; say so."""
    from alloy.check.overlap import probe_findings  # noqa: PLC0415

    ran = not str(report.checks.get("penetration", "skipped")).startswith("skipped")
    late = probe_findings(probe, "penetration")
    if late:
        report.add(late, "penetration" if ran else None)


def extra_name(fn: ExtraCheck) -> str:
    """The check name of an extra run check: ``fn.check_name``, else ``fn.__name__``.

    Example::

        >>> def soft(res, built, index): return [], {}
        >>> extra_name(soft)
        'soft'
    """
    return str(getattr(fn, "check_name", None) or getattr(fn, "__name__", "extra"))


def _extra_checks(report: CheckReport, res: Any, built: Any, index: ObjectIndex, extra: Sequence[ExtraCheck]) -> None:
    """Run each extra check on the finished run; a crash is a warning under its name, not a crash of the check."""
    for fn in extra:
        name = extra_name(fn)
        try:
            found, metrics = fn(res, built, index)
        except Exception as exc:  # noqa: BLE001 - another package's check must not break the report
            log.warning("check: extra check %s raised %s: %s", name, type(exc).__name__, exc)
            found, metrics = [Finding(name, "warn", f"the extra check {name!r} raised {type(exc).__name__}: {exc}",
                                      f"fix the {name!r} check function (its findings are missing from this "
                                      f"report)")], {}
        report.metrics[name] = metrics
        report.add(list(found), name)


def _skip_rest(report: CheckReport, reason: str, t0: float) -> CheckReport:
    for c in CHECKS:
        if c not in report.checks:
            report.skip(c, reason)
    report.timings_s["total"] = time.perf_counter() - t0
    return report


__all__ = ["ExtraCheck", "check", "extra_name", "resolve_settings"]
