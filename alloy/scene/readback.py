# SPDX-License-Identifier: Apache-2.0
"""Readback (SPEC §3.4, CARD-3): every card value vs the compiled Model array, in every world (randomized values
included). Leaves that cannot reach the Model are listed in `report.unsupported` with the reason; any other
mismatch raises ReadbackError(message, mismatches)."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from alloy.core.errors import ReadbackError
from alloy.core.quantity import Q
from alloy.route.types import Route
from alloy.scene.readback_checks import CHECKERS, Check, routing_inputs
from alloy.scene.types import BuiltScene

if TYPE_CHECKING:
    from alloy.scene.assembler import Plan

log = logging.getLogger("alloy.scene")
Mismatch = tuple[str, float, float]


def leaves(obj, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Dotted paths and values of every Q leaf (and float field) in a card dataclass tree."""
    if isinstance(obj, Q):
        yield prefix, obj.value
        return
    if not dataclasses.is_dataclass(obj):
        return
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        path = f"{prefix}.{f.name}" if prefix else f.name
        if isinstance(val, Q) or dataclasses.is_dataclass(val):
            yield from leaves(val, path)
        elif isinstance(val, tuple) and val and all(dataclasses.is_dataclass(v) for v in val):
            for i, v in enumerate(val):
                yield from leaves(v, f"{path}.{i}")   # SCR-WP1-1 grammar


def _compare(checks: list[Check], rtol: float) -> list[Mismatch]:
    bad = []
    for c in checks:
        exp, got = np.asarray(c.expected, np.float64), np.asarray(c.actual, np.float64)
        tol = rtol * np.maximum(np.abs(exp), c.atol / max(rtol, 1e-30))
        err = np.abs(got - exp)
        if exp.shape != got.shape or np.any(err > tol + c.atol):
            i = int(np.argmax(err)) if err.size and exp.shape == got.shape else 0
            bad.append((c.path, float(exp.ravel()[i]) if exp.size else np.nan,
                        float(got.ravel()[i]) if got.size else np.nan))
    return bad


def readback(built: BuiltScene, lib=None, rtol: float = 1e-6, *, plan: Plan | None = None) -> list[Mismatch]:
    """Check every card value; unreachable ones go to `report.unsupported`. Returns the (empty) mismatch list."""
    plan = plan if plan is not None else getattr(built, "_plan", None)
    if plan is None:
        raise ReadbackError("readback needs the assembly plan (built by SceneAssembler)", [])
    built._plan = plan
    mismatches: list[Mismatch] = []
    covered = 0
    for t in plan.objects:
        if t.obj is None:
            continue
        checks, unsupported = CHECKERS.get(t.decision.route, CHECKERS[Route.RIGID])(built, plan, t)
        mismatches += [(f"{t.name}.{p}", e, a) for p, e, a in _compare(checks, rtol)]
        verified = {c.path for c in checks} | set(unsupported) | routing_inputs()
        for path, reason in unsupported.items():
            _note(built, f"{t.name}.{path}", reason)
        for path, _ in leaves(t.obj.material, "material"):
            if path not in verified:
                _note(built, f"{t.name}.{path}", f"{t.decision.route}: no Model array for this value")
        for path, _ in leaves(t.obj.card, "card"):
            if path not in verified:
                _note(built, f"{t.name}.{path}", "card field consumed by routing/assembly, not a Model value")
        covered += len(checks)
    if mismatches:
        raise ReadbackError(f"{len(mismatches)} card values differ from the compiled Model: {mismatches[:5]}",
                            mismatches)
    log.info("readback: %d checks passed, %d values listed unsupported", covered, len(built.report.unsupported))
    return mismatches


def _note(built: BuiltScene, field: str, reason: str) -> None:
    entry = (field, reason)
    if entry not in built.report.unsupported:
        built.report.unsupported.append(entry)
