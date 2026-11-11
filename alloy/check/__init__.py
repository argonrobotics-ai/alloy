# SPDX-License-Identifier: Apache-2.0
"""Physics check of a scene: ``alloy.check(scene, worlds=64, seconds=2) -> CheckReport`` (JSON with
``report.to_json()``).

Checks: card validation, overlap at load, rest drift (stability), maximum penetration, health events (NaN, overflow,
blow-up) and energy drift. Every finding says what to change, e.g. "peg_d12 and table_top_plywood overlap by 10 mm
at start -> lift peg_d12 by 10.5 mm". Command line: ``python -m alloy.check scene.yaml --json``.

The package itself is callable, so ``alloy.check(scene)`` and ``alloy.check.check(scene)`` are the same call.

Example::

    >>> import alloy.check
    >>> report = alloy.check("demo_peg_d12", run=False)       # load checks only (cards, overlap)
    >>> report.ok
    True
    >>> report = alloy.check("demo_peg_d12", worlds=4, seconds=0.5)
    >>> [f.text() for f in report.findings if f.level == "error"]
    []
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from alloy.check.report import CHECKS, CheckReport, Finding

__all__ = ["CHECKS", "CheckReport", "Finding", "apply_fixes", "check", "check_load", "check_object", "check_sim"]


def check(scene: Any, worlds: int | None = None, seconds: float | None = None, **kwargs: Any) -> CheckReport:
    """Run every check on ``scene``; see ``alloy.check.api.check`` for the keyword arguments.

    Example::

        >>> check("demo_peg_d12", worlds=2, seconds=0.2, device="cpu").ok
        True
    """
    from alloy.check.api import check as _check  # noqa: PLC0415 - keeps `import alloy.check` free of warp

    return _check(scene, worlds, seconds, **kwargs)


def check_load(scene: Any, worlds: int | None = None, **kwargs: Any) -> CheckReport:
    """Load checks only (cards, assembly, overlap at load in every world): what to call after each placement.

    Example::

        >>> check_load("demo_peg_d12", worlds=4).checks["overlap"]
        'pass'
    """
    return check(scene, worlds, None, run=False, **kwargs)


def apply_fixes(card: Any, report: CheckReport, levels: tuple[str, ...] = ("error", "warn")) -> tuple[Any, list[str]]:
    """Apply the report's suggested poses and object names to ``card``; see ``alloy.check.fix.apply_fixes``.

    Example::

        >>> fixed, applied = apply_fixes(card, check_load(card))     # doctest: +SKIP
    """
    from alloy.check.fix import apply_fixes as _apply  # noqa: PLC0415

    return _apply(card, report, levels)


def check_object(obj: Any, worlds: int | None = None, seconds: float | None = None, **kwargs: Any) -> CheckReport:
    """Drop test of one object card on a generated floor (cards, overlap, rest drift, penetration, health, energy):
    what an importer calls after writing each object; see ``alloy.check.drop``.

    Example::

        >>> check_object("cube_beech_30", worlds=2, seconds=0.25, device="cpu").ok
        True
    """
    from alloy.check.drop import check_object as _drop  # noqa: PLC0415

    return _drop(obj, worlds, seconds, **kwargs)


def check_sim(sim: Any) -> dict[str, Any]:
    """Check a running simulation at its current state (overlaps, health so far, deepest contact now) as JSON-ready
    data: {"ok", "problems": [{"world", "what", "fix", ...}], "source", "report"}; see ``alloy.check.live``.

    Example::

        >>> sim.step(10); alloy.check.check_sim(sim)["ok"]             # doctest: +SKIP
        True
    """
    from alloy.check.live import check_sim as _check_sim  # noqa: PLC0415

    return _check_sim(sim)


class _CallableModule(ModuleType):
    """``alloy.check(...)`` works whether ``alloy.check`` resolves to this package or to the function."""

    def __call__(self, scene: Any, worlds: int | None = None, seconds: float | None = None,
                 **kwargs: Any) -> CheckReport:
        return check(scene, worlds, seconds, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
