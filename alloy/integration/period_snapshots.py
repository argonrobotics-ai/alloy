# SPDX-License-Identifier: Apache-2.0
"""When ``IntegratedSimulation`` saves its period-start snapshots (SPEED-NOW S1, WP7; SCR-WP0-RIGIDDT-2's snapshots).

Every period the simulation saves three copies: ``_good`` (what a rollback restores, and the health latch's source when
every array has a per-world axis), the per-world latch copy when it has not, and health's own period-start record
(``_health_snap``, which only a rollback or an MPM grid regrow restores). On XB4-9's resting stacks at 3,072 worlds
those saves take about 0.7 ms of a 15-20 ms period on an L4.

A simulation is **sparse** when nothing can restore a period start but the latch, and its periods end on check steps:
- its stepper never rolls back (``steppers._NoRollback``'s ``rollback_requested``, a single entry's or the stock
  proxy's);
- it has no MPM grid that can regrow (``grid_growth.regrow`` restores ``_good``);
- health checks every ``every`` > 1 steps (the fast preset's cheap checks: every 10 steps);
- ``every`` and the steps per period share no factor (one 10 ms step per period under the rigid fast mode), so a
  period start right after a check step comes around (WP5: at 8 steps per period and every 10 none ever does).
It is decided every period (three getattrs), so a stepper, grid or cadence that changes later never meets a stale
decision (WP0, WP5).

A sparse simulation saves the copies only at the start of a period whose previous step was a check step, so the latch
source is a checked state (``HealthMonitor.is_check_step``'s contract), and keeps no health copy. Saves then land
every lcm(``every``, steps per period) steps (WP5), so a world latched INVALID freezes at its last saved checked
state, up to that many steps before the failing check: up to ``every`` - 1 steps before the failure at one step per
period (the condition the coordinator set: FAST_COST says so), instead of at the start of its failing period, which
with checks every 10 steps was not checked either. Every other simulation saves all three every period, as before.
Example: ``save_period_start(sim)`` at the top of ``IntegratedSimulation.step_period``.

``make_period_snapshots`` makes the copies themselves, from the arrays the simulation has at that moment:
``IntegratedSimulation.__init__`` calls it, and so does a pour that adds solver state after the build
(``alloy.api._pour.start``), so one place says how they are made (WP4's review, 2026-11-11).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from alloy.integration.row_growth import row_guard

log = logging.getLogger("alloy.integration")

__all__ = ["make_period_snapshots", "save_period_start", "sparse"]


def sparse(sim: Any) -> bool:
    """Whether ``sim``'s period-start saves can follow health's checks this period (module docstring)."""
    from alloy.scene.steppers import _NoRollback  # noqa: PLC0415

    health = getattr(sim, "health", None)
    cfg = getattr(health, "cfg", None)
    if cfg is None or cfg.level == "off" or cfg.every <= 1 or not callable(getattr(health, "is_check_step", None)):
        return False
    if getattr(type(sim.stepper), "rollback_requested", None) is not _NoRollback.rollback_requested:
        return False
    if getattr(getattr(sim, "parts", None), "mpm_grid", None) is not None or row_guard(sim) is not None:
        return False
    out = math.gcd(int(cfg.every), int(sim.clock.steps_per_period)) == 1
    if out and not getattr(sim, "_alloy_sparse_said", False):
        sim._alloy_sparse_said = True
        period = math.lcm(int(cfg.every), int(sim.clock.steps_per_period))
        log.info("period snapshots after check steps only (every %d steps; no rollback, no MPM regrow): a world "
                 "latched INVALID freezes at its last saved checked state, up to %d steps before the failing check",
                 cfg.every, period)
    return out


def save_period_start(sim: Any) -> None:
    """Save ``sim``'s period-start snapshots when they are due (every period unless ``sparse``)."""
    if sparse(sim):
        if getattr(sim, "_alloy_saved_at", None) is None or sim.health.is_check_step(sim.steps - 1):
            sim._good.save(sim.state)
            if sim._latch_snap is not sim._good:
                sim._latch_snap.save(sim.state)
            sim._alloy_saved_at = sim.steps
        return
    rows = row_guard(sim)
    if rows is not None:                     # health's dropped-pairs counts as this period starts (after any reset)
        rows.keep_dropped()
    sim._good.save(sim.state)
    if sim._latch_snap is not sim._good:
        sim._latch_snap.save(sim.state)
    if sim._health_snap is not None:
        sim._health_snap.save(sim.state)


def make_period_snapshots(sim: Any, *, good: bool = False) -> None:
    """Make ``sim``'s period snapshots from the state arrays it has now. ``_latch_snap`` is ``_good`` itself when
    every array has a per-world axis, else a per-world copy. ``_end_snap`` stays empty until a per-world rollback
    needs it. ``_health_snap`` holds health's arrays. A WP2 base's ``_latch`` / ``_end`` point at these copies.

    ``good=True`` first remakes ``_good`` itself, for solver state added after construction. A snapshot keeps only
    the arrays it was made with, so a pour's emitter cursor added later was never restored by a rollback (GAP-LC,
    2026-11-11: 244 of 294 particles poured). Remake before the first step only: a captured graph holds the
    snapshots' arrays.
    Example: ``make_period_snapshots(sim)`` (``IntegratedSimulation.__init__``)."""
    if good:
        sim._good = sim._snapshot()
        sim._good.save(sim.state)
    sim._latch_snap = sim._good if not sim._good.whole_arrays else sim._per_world_snapshot()
    sim._latch_snap.save(sim.state)
    sim._end_snap = None
    sim._health_snap = sim._health_snapshot()
    for name in ("_latch", "_end"):     # newer WP2 bases keep their own copies; this class uses its own
        if hasattr(sim, name):
            setattr(sim, name, sim._latch_snap if name == "_latch" else None)
