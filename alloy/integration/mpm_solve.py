# SPDX-License-Identifier: Apache-2.0
"""The implicit-MPM rheology solve's tolerance (SCR-WP12-3 (3); coordinator ruling 2026-11-08 10:05Z).

The rheology solve (Newton's projected Gauss-Seidel) is two thirds of an eager MPM period at 256 dough worlds (L4: 6.3
of 9.8 s). It stops when the stress residual is under the tolerance (Newton's default 1e-4) or at 250 iterations.
``ALLOY_MPM_TOLERANCE`` sets it for every MPM entry (an entry's own solver kwargs win); unset, it is
``DEFAULT_TOLERANCE``, 3e-4 since every MPM law and the dough, paste and soil tasks re-passed at it within their own
bands (2026-11-08; old against new in [internal note] notes/wp12-results/rheo-default/regression.md).
``ALLOY_MPM_TOLERANCE=1e-4`` restores Newton's default. 3e-4 cut a 256-world dough period 22 % (7.60 against 9.78 s;
sand unchanged, 6.69 against 6.73 s) and moved a pressed dough ball's particles by mean 0.08 mm, max 0.65 mm, 2.3x the
solver's own floor (CPU, 0.2 s). Values outside ``TOLERANCE_RANGE`` are refused: looser ones leave solves unconverged
(the ruling rejected 1e-3 and a 100-iteration cap).

Newton's older Q1 collider basis (38 % faster at 256 dough worlds) was measured for the same ruling and is not offered:
it failed its contact laws (docs/known_gaps/mpm-solve-speed.md).

The friction cone. Newton's Drucker-Prager cone is sqrt(J2) <= mu p, and a granular card's friction angle phi
becomes mu = tan(phi) (``CONE_DEFAULT``, "tan"). That is the repose match: a flow that keeps its volume lines the
stress up with the strain rate, so a slope rests at tan(beta) = mu. Laws (tests/integration/test_mpm_cone.py, CPU,
sand of 34.5 deg): a layer between end walls rests at 33 deg and flows at 36, and a free end relaxes to a wedge at
33 deg. ``ALLOY_MPM_CONE=mc`` gives the Mohr-Coulomb compression-cone match instead,
mu = 6 sin(phi) / (sqrt(3) (3 - sin(phi))) (0.806 for 34.5 deg, the cone of a 38.9 deg sand): the same layer then
rests up to 40 deg, and the dry-collapse runout (CARD-9, Lube et al. 2004) reads 31 % short against 2 % with "tan".
It is an opt-in for comparing with codes that use that match, not a fix: the first reading, that "tan" yields below
the card's angle, came from a layer with a free end (docs/known_gaps/mpm-grain-slope-resolution.md).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

from alloy.core.errors import SolverOptionError

__all__ = ["CONES", "CONE_DEFAULT", "CONE_ENV", "DEFAULT_TOLERANCE", "PARTIAL_DEFAULT_TAU", "PARTIAL_ENV",
           "PARTIAL_RANGE", "MPM_SWEEPS", "TOLERANCE_ENV", "TOLERANCE_RANGE", "attach_notes", "check", "cone_friction",
           "describe", "effective_tau", "partial_tau", "route_note", "solve_kwargs", "sweeps_tau"]

TOLERANCE_ENV = "ALLOY_MPM_TOLERANCE"
NEWTON_TOLERANCE = 1e-4                    # SolverImplicitMPM.Config.tolerance
DEFAULT_TOLERANCE: float | None = 3e-4      # the flip, its own commit: a revert restores None (Newton's 1e-4)
TOLERANCE_RANGE = (1e-5, 3e-4)             # the ruled range: no looser than 3e-4
CONE_ENV = "ALLOY_MPM_CONE"
CONES = ("tan", "mc")                      # mu = tan(phi); Mohr-Coulomb's compression-cone match (module doc)
CONE_DEFAULT = "tan"
PARTIAL_ENV = "ALLOY_MPM_PARTIAL"           # SPEED-NOW S5 (opt-in): partial Gauss-Seidel sweeps (mpm_worlds)
PARTIAL_DEFAULT_TAU = 0.3                  # "on": L4, 256 dough worlds 7,795 -> 6,021 ms per period
PARTIAL_RANGE = (0.05, 1.0)                # tau x tolerance: a block stays active while a row moves more
MPM_SWEEPS = ("full", "partial")           # alloy.build(mpm_sweeps=...): every sweep full (default), or partial
MODEL_TAU = "alloy_mpm_partial_tau"        # the built model's attribute that carries it (mpm_worlds reads it)


def check(tolerance: float, source: str) -> float:
    """``tolerance`` if it is in ``TOLERANCE_RANGE``; else ``SolverOptionError`` naming ``source`` (the environment
    variable, or an MPM entry's own solver kwargs: WP4's review, no path past the ruled range)."""
    tol = float(tolerance)
    lo, hi = TOLERANCE_RANGE
    if not lo <= tol <= hi:
        raise SolverOptionError(f"{source}={tolerance!r}: the MPM rheology tolerance must be a number from {lo:g} "
                                f"to {hi:g} (looser leaves solves unconverged). Fix: unset it, or set it within that "
                                "range")
    return tol


def _tolerance() -> float | None:
    raw = os.environ.get(TOLERANCE_ENV, "").strip()
    if not raw:
        return DEFAULT_TOLERANCE
    try:
        tol = float(raw)
    except ValueError:
        tol = float("nan")
    check(tol, TOLERANCE_ENV)
    return tol


def solve_kwargs() -> dict[str, Any]:
    """Newton ``SolverImplicitMPM.Config`` kwargs for the solve (empty: Newton's defaults)."""
    tol = _tolerance()
    return {} if tol is None else {"tolerance": tol}


def describe() -> str:
    """The effective tolerance and where it comes from, for an MPM object's route reasons (``sim.routes()``): the
    environment is not in Scene JSON, so a row names what it ran with (WP2's review)."""
    tol = _tolerance()
    where = f"{TOLERANCE_ENV}" if os.environ.get(TOLERANCE_ENV, "").strip() else "default"
    cone = f"; friction cone {_cone()} ({CONE_ENV})" if os.environ.get(CONE_ENV, "").strip() else ""
    tau = partial_tau()
    partial = f"; partial sweeps tau {tau:g} ({PARTIAL_ENV})" if tau is not None else ""
    return f"rheology solve tolerance {NEWTON_TOLERANCE if tol is None else tol:g} ({where}){cone}{partial}"


def sweeps_tau(sweeps: str, error: type) -> float | None:
    """``alloy.build(mpm_sweeps=...)``: None for "full", ``PARTIAL_DEFAULT_TAU`` for "partial"; else ``error`` (the
    API's BuildError: message, fix, field)."""
    if sweeps not in MPM_SWEEPS:
        raise error(f"unknown mpm_sweeps {sweeps!r}", f"pass mpm_sweeps={MPM_SWEEPS[0]!r} (default) or "
                    f"{MPM_SWEEPS[1]!r}", field="mpm_sweeps")
    return PARTIAL_DEFAULT_TAU if sweeps == "partial" else None


def effective_tau(model: Any) -> float | None:
    """The partial-sweep threshold for a solve on ``model``: ``ALLOY_MPM_PARTIAL`` when it is set (a run-wide
    override, e.g. for law campaigns), else the build's ``mpm_sweeps`` (``MODEL_TAU`` on the model)."""
    if os.environ.get(PARTIAL_ENV, "").strip():
        return partial_tau()
    return getattr(model, MODEL_TAU, None)


def route_note(model: Any) -> Any:
    """An MPM entry solver's ``alloy_route_notes`` for a build with ``mpm_sweeps="partial"`` (``sim.routes()``; WP12's
    review): what its solves run now, read when asked (``ALLOY_MPM_PARTIAL`` set later still wins)."""
    def notes() -> dict[str, str]:
        tau = effective_tau(model)
        how = f"partial, tau {tau:g}" if tau is not None else "full"
        return {"mpm_sweeps": f"{how} (alloy.build mpm_sweeps; {PARTIAL_ENV} overrides it when set)"}
    return notes


def partial_tau() -> float | None:
    """The partial-sweep threshold under ``ALLOY_MPM_PARTIAL`` (SPEED-NOW S5; ``mpm_worlds``): None when unset or
    "off" (every sweep full, the default); ``PARTIAL_DEFAULT_TAU`` for "on"; else a number in ``PARTIAL_RANGE``. In each
    group of sweeps the last is full and the stop test reads only full sweeps, so the solve stops on the same rule."""
    raw = os.environ.get(PARTIAL_ENV, "").strip().lower()
    if raw in ("", "off"):
        return None
    if raw == "on":
        return PARTIAL_DEFAULT_TAU
    try:
        tau = float(raw)
    except ValueError:
        tau = float("nan")
    lo, hi = PARTIAL_RANGE
    if not lo <= tau <= hi:
        raise SolverOptionError(f"{PARTIAL_ENV}={raw!r}: partial sweeps take on, off, or a threshold from {lo:g} to "
                                f"{hi:g} (x the tolerance). Fix: unset it, or set {PARTIAL_ENV}=on")
    return tau


def _cone() -> str:
    raw = os.environ.get(CONE_ENV, "").strip().lower() or CONE_DEFAULT
    if raw not in CONES:
        raise SolverOptionError(f"{CONE_ENV}={raw!r}: the MPM friction cone must be one of {', '.join(CONES)}. Fix: "
                                f"unset it (tan(phi), the default) or set {CONE_ENV}=mc")
    return raw


def cone_friction(phi: float) -> float:
    """Newton's cone coefficient mu for a card friction angle ``phi`` [rad] under ``ALLOY_MPM_CONE`` (module doc)."""
    if _cone() == "tan":
        return math.tan(phi)
    s = math.sin(phi)
    return 6.0 * s / (math.sqrt(3.0) * (3.0 - s))


def attach_notes(solver: Any, tag: str, notes: Callable[[], dict[str, str]]) -> None:
    """Add ``notes`` to ``solver.alloy_route_notes`` (``sim.routes()``) once per ``tag``, keeping the others."""
    done = solver.__dict__.setdefault("_alloy_note_tags", set())
    if tag in done:
        return
    done.add(tag)
    before = getattr(solver, "alloy_route_notes", None)
    solver.alloy_route_notes = (lambda: {**(before() if callable(before) else {}), **notes()}) if before else notes
