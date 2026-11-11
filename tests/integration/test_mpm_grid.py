# SPDX-License-Identifier: Apache-2.0
"""MPM sparse grid under graph capture (alloy.integration.mpm_grid): reserved from the cells in use at the start, grown
on overflow (SCR-WP12-3 a2), never past one cell per particle.

WP5's sand_heap at 16 worlds ran out of GPU memory when its first period's graph was instantiated: the grid reserved
8 cells per particle. An active cell holds at least one particle, so the particle count bounds it. On CPU, sand_heap
(1 world, 6,912 particles) used 1,183 to 1,397 cells over 0.6 s (wp12-results "WP12-MPM-CELLS").
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from alloy.core.registry import isolated_registry
from alloy.integration import mpm_grid as G
from alloy.integration.mpm_grid import GridGuard, MPMGridOverflowError, graph_cell_capacity
from alloy.testing import integration as I
from alloy.testing import scene as S

PERIODS = 12                 # 0.24 s: sand_heap's grid grows past 1,250 cells by 0.12 s


def _sand_heap(worlds: int, cells: int | None = None):
    """sand_heap built with graph capture's grid kwargs (``wiring._graph_entries``), or a forced ``cells``."""
    from alloy.integration.wiring import _graph_entries

    _, card, lib = next(x for x in S.mixed_scenes() if x[0] == "sand_heap")
    built = _graph_entries(I.assemble(dataclasses.replace(card, worlds=worlds), lib, "cpu", worlds), [])
    if cells is None:
        return built
    entries = [dataclasses.replace(e, solver_kwargs={**dict(e.solver_kwargs), "max_active_cell_count": cells})
               if e.capability == "mpm" else e for e in built.entries]
    return dataclasses.replace(built, entries=entries)


class _Grid:
    def __init__(self, rebuildable: bool, fail: bool = False) -> None:
        self._sparse_rebuildable, self.fail, self.max_active_cell_count = rebuildable, fail, 7

    def check_sparse_grid_rebuild_status(self) -> None:
        if self.fail:
            raise RuntimeError("capacity was exceeded for active voxels (status 8)")


def test_guard_checks_only_rebuildable_grids() -> None:
    assert GridGuard.for_solvers({"a": _Grid(False), "b": object()}) is None
    GridGuard.for_solvers({"a": _Grid(True)}).check()
    with pytest.raises(MPMGridOverflowError, match="mpm entry sand.*max_active_cell_count=7.*status 8"):
        GridGuard.for_solvers({"sand": _Grid(True, fail=True), "a": _Grid(False, fail=True)}).check()


def test_graph_capacity_follows_the_cells_in_use_at_the_start(monkeypatch) -> None:
    with isolated_registry(empty=True):
        built = _sand_heap(2)
    (e,) = [e for e in built.entries if e.capability == "mpm"]
    cells = G.start_cells(built.model, e.particles, e.solver_kwargs["voxel_size"])
    assert cells == 2 * 864                                       # per world at the start (CPU); 1,393 by 0.12 s
    reserved = int(e.solver_kwargs["max_active_cell_count"])
    assert reserved == graph_cell_capacity(len(e.particles), cells) == max(G.MIN_CELLS, 2 * cells)
    assert reserved < len(e.particles) == 2 * 6912
    monkeypatch.setenv(G.CELLS_ENV, "particles")                  # before SCR-WP12-3 a2: one cell per particle
    assert graph_cell_capacity(len(e.particles), cells) == len(e.particles)


@pytest.mark.integration
@pytest.mark.timeout(900)
@pytest.mark.parametrize("cells", [None, 1250])
def test_sand_heap_grid_holds_or_grows(cells: int | None) -> None:
    """The reservation holds sand_heap as it spreads; 1,250 cells (under the 1,393 it reaches) grows to 2,500 once and
    the run goes on (it stopped before SCR-WP12-3 a2)."""
    import numpy as np

    from alloy.integration import build_simulation

    with isolated_registry(empty=True):
        sim = build_simulation(_sand_heap(1, cells), "standard", graph="off")
        assert sim.parts.mpm_grid is not None
        for _ in range(PERIODS):
            sim.step_period()
        grown = sim.parts.mpm_grid.grown
        assert grown == ([] if cells is None else [(grown[0][0], 1250, 2500)])
        assert getattr(sim, "grid_regrows", 0) == len(grown) and sim.rollbacks == 0
        assert np.isfinite(sim.state.particle_q.numpy()).all()

class _Solver:
    """A rebuildable MPM solver stand-in whose status reports an overflow until it is reset."""

    def __init__(self, cap: int, overflow: bool) -> None:
        self._sparse_rebuildable, self.max_active_cell_count, self.overflow = True, cap, overflow
        self._scratchpad, self._last_step_data = ["old grid"], ["old warm start"]

    def check_sparse_grid_rebuild_status(self) -> None:
        if self.overflow:
            raise RuntimeError("capacity was exceeded for active voxels (status 8)")

    def _clear_sparse_grid_rebuild_status(self) -> None:
        self.overflow = False


def test_grow_doubles_an_overflowed_reservation_up_to_the_particle_bound() -> None:
    a, b = _Solver(1000, True), _Solver(500, False)
    guard = GridGuard.for_solvers({"sand": a, "dough": b}, {"sand": 1500, "dough": 4000})
    assert guard.grow()
    assert a.max_active_cell_count == 1500 and guard.grown == [("sand", 1000, 1500)]   # 2 x 1000, capped at 1500
    assert a._scratchpad == [] and a._last_step_data == [] and not a.overflow     # fresh grid state, status cleared
    assert b.max_active_cell_count == 500 and not guard.grow()                     # nothing overflowed: no growth
    a.overflow = True
    with pytest.raises(MPMGridOverflowError, match="particle bound 1500"):         # at the bound it cannot grow
        guard.grow()


def test_start_cells_counts_per_world_and_gives_up_on_inactive_particles() -> None:
    import numpy as np
    import warp as wp
    from alloy._vendor.newton import ParticleFlags

    q = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.012, 0, 0], [0.001, 0, 0]], np.float32)
    model = SimpleNamespace(particle_q=wp.array(q, dtype=wp.vec3, device="cpu"),
                            particle_world=wp.array([0, 0, 0, 1], dtype=wp.int32, device="cpu"),
                            particle_flags=wp.array([int(ParticleFlags.ACTIVE)] * 4, dtype=wp.int32, device="cpu"))
    assert G.start_cells(model, [0, 1, 2, 3], 0.01) == 3                           # 2 cells in world 0, 1 in world 1
    assert G.start_cells(model, [0, 1], 0.01) == 1 and G.start_cells(model, [0, 1], None) is None
    model.particle_flags = wp.array([int(ParticleFlags.ACTIVE), 0, 0, 0], dtype=wp.int32, device="cpu")
    assert G.start_cells(model, [0, 1, 2, 3], 0.01) is None                        # a pour's spares: one per particle




def test_regrow_replays_the_attempt_when_the_stepper_can() -> None:
    """WP4's review: a grid growth replays the period unchanged (``replay_period``), not an escalation rollback."""
    from alloy.integration.grid_growth import regrow

    calls = []
    snap = SimpleNamespace(restore=lambda state: calls.append("restore"))
    guard = SimpleNamespace(grow=lambda: True, grown=[("sand", 1, 2)])
    stepper = SimpleNamespace(replay_period=lambda: calls.append("replay"),
                              on_rollback=lambda mask: calls.append("rollback"))
    sim = SimpleNamespace(parts=SimpleNamespace(mpm_grid=guard), _good=snap, _health_snap=None, stepper=stepper,
                          clock=SimpleNamespace(ticks=9, steps=9), _graphs=None, state="s", periods=3, rollbacks=0)
    assert regrow(sim, (4, 5)) and calls == ["restore", "replay"]
    assert (sim.clock.ticks, sim.steps, sim.grid_regrows, sim.rollbacks) == (4, 5, 1, 0)
    del stepper.replay_period                                       # a stepper without it: the old rollback
    assert regrow(sim, (4, 5)) and calls[-1] == "rollback" and sim.grid_regrows == 2
