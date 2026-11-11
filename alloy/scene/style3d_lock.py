# SPDX-License-Identifier: Apache-2.0
"""Style3D's projective-dynamics lock (DC5, 2026-11-08; [internal note] notes/dc5-style3d,
docs/known_gaps/cloth-style3d-stretch-lock.md).

Style3D solves each step with a fixed global matrix built from the cloth's stretch (and bending) stiffness. When that
matrix's diagonal outweighs the inertia term m/dt² by more than about a thousand per vertex, its iterations release
out-of-plane motion only a little per step: the sheet creeps (a KES woven card on the Cusick rig: about 2 mm/s) and
stays nearly flat while every health flag reads OK. DC5 measured the KES fabrics at 9,700 (F1) and 2,900 (F9): locked;
the same card with its stretch ÷100 (about 100) drapes. ``warn_if_locked`` says so at build, with the ratio.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("alloy.scene")

LOCK_RATIO = 1.0e3          # median PD diagonal / (m / dt²) above which Style3D's solve creeps (DC5: 970 still locked)


def lock_ratio(solver: Any, model: Any, dt: float) -> float:
    """Median over moving vertices of Style3D's PD diagonal over m/dt² (0 when there is nothing to judge)."""
    diag = getattr(solver, "pd_diags", None)
    if diag is None or not dt > 0.0:
        return 0.0
    d, m = np.asarray(diag.numpy(), np.float64), np.asarray(model.particle_mass.numpy(), np.float64)
    moving = m > 0.0
    return float(np.median(d[moving] / (m[moving] / dt ** 2))) if moving.any() else 0.0


def warn_if_locked(solver: Any, model: Any, dt: float) -> float:
    """The lock ratio; a user warning when it exceeds LOCK_RATIO."""
    ratio = lock_ratio(solver, model, dt)
    if ratio > LOCK_RATIO:
        from alloy.api._say import warn  # noqa: PLC0415 - the API's user-warning helper

        warn("a Style3D cloth (an elastic.cloth card) is stiff in stretch for its step: its solve matrix outweighs "
             "the inertia term %.0f times per vertex (over %.0f), so the sheet drapes slowly or stays too flat while "
             "health reads OK (docs/known_gaps/cloth-style3d-stretch-lock.md). For a woven fabric's drape, give the "
             "card no elastic.cloth (it routes to the VBD cloth), or check one drape against a measurement first",
             ratio, LOCK_RATIO)
    return ratio


__all__ = ["LOCK_RATIO", "lock_ratio", "warn_if_locked"]
