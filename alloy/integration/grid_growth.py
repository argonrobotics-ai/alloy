# SPDX-License-Identifier: Apache-2.0
"""Re-run a period whose MPM sparse grid outgrew its reservation, on a bigger grid (SCR-WP12-3 a2).

The rebuildable grid is reserved from the cells the material fills at the start (``mpm_grid.graph_cell_capacity``), so
material that spreads can outgrow it. Newton then drops the cells past the reservation and reports it in a status
word, read once per period (``GridGuard``). ``regrow`` turns that into a re-run instead of an error:

1. ``GridGuard.grow`` doubles each overflowed reservation (up to the particle count, where the grid cannot overflow)
   and gives that solver a fresh grid state (``mpm_grid._reset_grid_state``): the next step allocates the grid at the
   new size. Particle-backed warm starts (the rebuildable grid's) survive; grid-side ones start cold.
2. Every world goes back to the period start (``_good``; health's period-start copy) and the clock is wound back. The
   stepper replays the attempt unchanged (``replay_period``: same dt_x and substeps, nothing counted or escalated;
   WP4's review); a stepper without it gets ``on_rollback(None)``, which counts as an escalation rollback.
3. Captured graphs hold the old buffers: they are dropped, and each key's next period runs eagerly before its capture.

The snapshots are kept, not made again: one made mid-run would also copy State arrays that appear after the first
step and have no per-world axis (sand_heap's ``collider_ids``), which breaks the health latch's masked restore. So
they go on copying the old grid's warm-start arrays (``mpm.last_step.*``), which the solver no longer reads: after a
growth, a rollback restores the particles and their particle-backed warm start, not the grid-side one, and a re-run
differs at the solver's tolerance. The old arrays stay alive with the snapshots (at most the new grid's size, since
reservations double). So the kept-worlds-bitwise rule of a per-world rollback (``simulation`` item 3) stops covering
the grid-side warm start after a growth, said once in the log. The re-run also differs from a run that had the bigger
grid from the start; each growth is logged, ``GridGuard.grown`` lists every one and ``sim.grid_regrows`` counts them.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("alloy.integration")

__all__ = ["drop_graphs", "regrow", "restore_period_start"]


def regrow(sim: Any, start: tuple[int, int]) -> bool:
    """When an MPM grid of ``sim`` (an IntegratedSimulation) outgrew its reservation this period: grow it, restore the
    period start (``start``: clock ticks and steps) and return True so the caller re-runs the period."""
    guard = getattr(getattr(sim, "parts", None), "mpm_grid", None)
    if guard is None or not guard.grow():
        return False
    restore_period_start(sim, start)
    drop_graphs(sim)
    sim.grid_regrows = getattr(sim, "grid_regrows", 0) + 1
    if sim.grid_regrows == 1:
        log.warning("MPM grid grown: from now on a per-world rollback restores the particle-backed warm start but not "
                    "the grid-side one, so kept worlds are no longer bitwise in it (snapshots copy the old grid)")
    log.info("period %d re-runs on the grown MPM grid: %s", sim.periods, guard.grown[-1])
    return True


def restore_period_start(sim: Any, start: tuple[int, int]) -> None:
    """Every world back to the period start, the stepper ready to replay the attempt, the clock wound back (also
    ``row_growth``'s re-run)."""
    sim._good.restore(sim.state)
    if sim._health_snap is not None:
        sim._health_snap.restore(sim.state)
    replay = getattr(sim.stepper, "replay_period", None)
    if callable(replay):
        replay()
    else:
        sim.stepper.on_rollback(None)
    sim.clock.ticks, sim.steps = start
    rows = getattr(sim, "_alloy_row_guard", None)
    if rows is not None:                     # the abandoned attempt's dropped self-contact pairs are no period's
        rows.restore_dropped()


def drop_graphs(sim: Any) -> None:
    """Drop the captured graphs (they hold replaced buffers): each key's next period runs eagerly, then captures."""
    if sim._graphs is not None:
        sim._graphs.invalidate()
        sim.exchange_nodes.clear()
        sim._warm.clear()
