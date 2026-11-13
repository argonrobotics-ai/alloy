# SPDX-License-Identifier: Apache-2.0
"""MPM entries push their particles back out of colliders after every step (the default from 0.1.3).

Newton's implicit MPM holds material out of a collider through the collider's grid nodes. A lone grain resting on the
ground has one cell, and on the sparse grid the ground lies midway between two node planes (docs/known_gaps/mpm-sparse-
floor-offset.md): such a grain could sink into the ground, and once under it nothing pushed it back, so it fell for
good. Newton's own MPM examples call ``SolverImplicitMPM.project_outside`` after every step for this (it moves a
particle deeper than the collider's projection threshold back to the surface, with the collider's friction on its
velocity), and so does Alloy's granular twin (``alloy/scene/twins/granular.py``, its own solver, which this module never
wraps). The engine's MPM entries did not, through 0.1.2: in soil_dig on one L4 the grains that spilled onto the ground
went under it in every tree and mode (the shipped 0.1.2 tree at 16 eager worlds: 10 of 16 worlds), with ``sim.check()``
ok until a falling grain passed 20 m/s.

``install(solver)`` wraps an MPM entry's ``step`` to end with ``solver.project_outside(state_out, state_out, dt)``,
after any body carry (``mpm_bodies``) so the colliders are at the step's end pose. ``ALLOY_MPM_PROJECT=off`` leaves the
step as Newton's (A/B runs and the before rows); any other value, or none, projects.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

log = logging.getLogger("alloy.integration")

__all__ = ["ENV", "enabled", "install", "install_for_entries"]

ENV = "ALLOY_MPM_PROJECT"


def enabled() -> bool:
    """Whether MPM entries project particles out of colliders after each step (default yes; ``ALLOY_MPM_PROJECT=off``
    turns it off). Example: ``enabled()`` -> True."""
    return os.environ.get(ENV, "on").strip().lower() != "off"


def install(solver: Any) -> bool:
    """Wrap ``solver.step`` to project its particles out of colliders at the step's end; False when ``solver`` has no
    ``project_outside`` or is already wrapped. Example: ``install(solvers["mpm"])`` -> True."""
    if getattr(solver, "_alloy_project_outside", False) or not callable(getattr(solver, "project_outside", None)):
        return False
    prev = solver.step

    def step(state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, **kw: Any) -> None:
        prev(state_in, state_out, control, contacts, dt, **kw)
        solver.project_outside(state_out, state_out, dt)

    solver.step = step
    solver._alloy_project_outside = True
    return True


def install_for_entries(built: Any, solvers: Mapping[str, Any]) -> list[str]:
    """``install`` on every MPM entry's solver unless ``enabled()`` is False; one note each. Example:
    ``notes += install_for_entries(built, solvers)``."""
    if not enabled():
        return []
    notes = []
    for entry in built.entries:
        solver = solvers.get(entry.name)
        if entry.capability != "mpm" or solver is None or not install(solver):
            continue
        line = f"mpm entry {entry.name}: particles projected out of colliders after every step ({ENV})"
        log.info("%s", line)
        notes.append(line)
    return notes
