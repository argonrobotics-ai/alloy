# SPDX-License-Identifier: Apache-2.0
"""The exchange clock of body pairs into MPM (speed-now-1109 S5): "standard", "fast" or "auto".

A coupled MPM entry is a contact-free destination: its collider response is an implicit grid projection, not a
penalty spring, so its pairs set no stiffness bound (``proxy_contacts.CONTACT_FREE_DESTINATIONS``) and the clock runs
at the ``dt_x_max`` cap (2 ms: 1.25 ms on a 10 ms control grid), every exchange in a lagged pair's two passes, so each
MPM solve is done twice. The MPM step is implicit and stable at a larger step; its partner, a robot or a free body,
substeps inside the exchange on its own stable step.

``fast``: a scene whose every two-way coupled pair has an MPM destination takes one lagged pass per exchange
(``SchedulerPolicy.lagged_iterations`` 1: the MPM force reaches its partner one exchange late, the explicit lagged
scheme) and a 2.5 ms exchange cap (``ClockPolicy.dt_x_max``: 4 exchanges per 10 ms). A caller's own values (anything
but the defaults) and a precision pin's cap win; other scenes are unchanged.
``auto``: one lagged pass per exchange at today's exchange step (``AUTO_PASSES``, ``AUTO_DT_X_MAX``) when, in addition,
the scene has a servo-driven robot and every body that starts near the MPM material is welded or that robot's link
(``mpm_robots.robot_only``); else standard (coordinator ruling 2026-11-09, option B). Decided once, at build: a free
body that reaches the material later runs on auto's clock too; a 2 cm steel block dropped 4 cm onto sand beside a
Panda rests 0.46 mm higher than on the standard clock and moves under 6 mm/s from 0.3 s (the ruled bound: 0.5 mm,
1 cm/s; tests/integration/test_mpm_clock.py). fast's 2.5 ms step would rest it 3.3 mm deeper, hence opt-in only.
``standard``: the clock before 2026-11-09. The default is auto.

Where it is set: ``ALLOY_MPM_COUPLING`` (a run-wide override, e.g. for law campaigns) wins over
``alloy.build(..., mpm_coupling=...)`` (``requested``), which wins over ``DEFAULT``. ``sim.routes()`` names the
clock and why on the MPM objects (``route_notes``).

Robot + sand rows (times per 10 ms period, same-build comparisons): docs/known_gaps/mpm-solve-speed.md. A free
body resting on MPM is less exact: on tests/integration/test_mpm_contact_free.py's cup, one pass makes it bob by
about 15 um (up to 7 mm/s) for about 0.3 s, and a 2.5 ms step with two passes rests it 0.36 mm deeper.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import logging
import os
from collections.abc import Iterator
from typing import Any

from alloy.core.defaults import DT_X_MAX
from alloy.core.errors import SolverOptionError
from alloy.coupling.types import SchedulerPolicy

log = logging.getLogger("alloy.integration")

__all__ = ["CLOCKS", "DEFAULT", "ENV", "FAST_DT_X_MAX", "FAST_PASSES", "MODEL_ATTR", "check", "clock", "decide",
           "pair_policy", "requested", "route_notes"]

ENV = "ALLOY_MPM_COUPLING"
CLOCKS = ("standard", "fast", "auto")
DEFAULT = "auto"               # coordinator ruling 2026-11-09 (option B), after its laws
FAST_DT_X_MAX = 2.5e-3          # s: 4 exchanges per 10 ms control period
FAST_PASSES = 1                 # lagged passes per exchange
AUTO_DT_X_MAX: float | None = None      # auto's cap (None: today's); see the module doc for the dropped-body rows
AUTO_PASSES = 1
DEFAULT_PASSES = SchedulerPolicy().lagged_iterations
MODEL_ATTR = "alloy_mpm_coupling"     # the build's decision on its model: (clock, why) for sim.routes()
_REQUESTED: contextvars.ContextVar[str | None] = contextvars.ContextVar("alloy_mpm_coupling", default=None)


def check(value: str, source: str) -> str:
    """``value`` when it is one of ``CLOCKS``; else ``SolverOptionError`` naming ``source``."""
    raw = str(value).strip().lower()
    if raw not in CLOCKS:
        raise SolverOptionError(f"{source}={value!r}: the MPM pair clock must be one of {', '.join(CLOCKS)}. Fix: "
                                f"use standard (today's), fast or auto")
    return raw


@contextlib.contextmanager
def requested(value: str | None) -> Iterator[None]:
    """``alloy.build(mpm_coupling=...)``: the pair clock for the scenes wired inside (None: ``DEFAULT``)."""
    token = _REQUESTED.set(None if value is None else check(value, "mpm_coupling"))
    try:
        yield
    finally:
        _REQUESTED.reset(token)


def clock() -> str:
    """The pair clock asked for: ``ALLOY_MPM_COUPLING``, else the build's ``mpm_coupling``, else ``DEFAULT``."""
    raw = os.environ.get(ENV, "").strip()
    if raw:
        return check(raw, ENV)
    return _REQUESTED.get() or DEFAULT


def _source() -> str:
    if os.environ.get(ENV, "").strip():
        return ENV
    return "alloy.build mpm_coupling" if _REQUESTED.get() else "default"


def _mpm_pairs_only(built: Any) -> bool:
    cap = {e.name: e.capability for e in built.entries}
    pairs = [p for p in getattr(built.coupler, "pairs", ()) or () if not p.one_way]
    return bool(pairs) and all(cap.get(p.destination) == "mpm" for p in pairs)


def decide(built: Any) -> tuple[str, str] | None:
    """("fast", "auto" (auto's settings) or "standard", why) for a scene whose two-way pairs all end in MPM; None for
    any other scene."""
    if not _mpm_pairs_only(built):
        return None
    asked, where = clock(), _source()
    if asked != "auto":
        return asked, f"{asked} ({where})"
    from alloy.integration.mpm_robots import robot_only  # noqa: PLC0415

    ok, why = robot_only(built)
    return ("auto" if ok else "standard"), f"auto ({where}): {why}; decided at build"


def pair_policy(policy: SchedulerPolicy, built: Any) -> SchedulerPolicy:
    """``policy`` with one lagged pass and the 2.5 ms exchange cap when ``decide`` says fast (module doc); a value the
    caller changed from its default is kept. The decision goes on ``built.model`` for ``route_notes``."""
    got = decide(built)
    if got is None:
        return policy
    setattr(built.model, MODEL_ATTR, got)
    if got[0] == "standard":
        return policy
    want_passes, want_cap = (FAST_PASSES, FAST_DT_X_MAX) if got[0] == "fast" else (AUTO_PASSES, AUTO_DT_X_MAX)
    passes = want_passes if policy.lagged_iterations == DEFAULT_PASSES else policy.lagged_iterations
    cap = want_cap if want_cap is not None and policy.clock.dt_x_max == DT_X_MAX else policy.clock.dt_x_max
    log.info("MPM pair clock %s, %s: %d lagged pass(es) per exchange, dt_x_max %.4g s", got[0], got[1], passes, cap)
    return dataclasses.replace(policy, lagged_iterations=passes, clock=dataclasses.replace(policy.clock, dt_x_max=cap))


def _settings(name: str) -> str:
    if name == "standard":
        return "today's clock"
    passes, cap = (FAST_PASSES, FAST_DT_X_MAX) if name == "fast" else (AUTO_PASSES, AUTO_DT_X_MAX)
    exch = f"exchanges up to {cap * 1e3:g} ms" if cap is not None else "today's exchange step"
    return f"{passes} lagged pass{'es' if passes != 1 else ''} per exchange, {exch}"


def route_notes(model: Any) -> Any:
    """An MPM entry solver's ``alloy_route_notes`` (``sim.routes()``): the pair clock and why, plus the partial-sweep
    row ``mpm_solve.route_note`` gives when a build asked for it (one notes function per solver)."""
    from alloy.integration import mpm_solve  # noqa: PLC0415

    def notes() -> dict[str, str]:
        out = {}
        got = getattr(model, MODEL_ATTR, None)
        if got is not None:
            out["mpm_coupling"] = f"{got[0]}: {_settings(got[0])}; {got[1]}"
        if mpm_solve.effective_tau(model) is not None:
            out.update(mpm_solve.route_note(model)())
        return out
    return notes
