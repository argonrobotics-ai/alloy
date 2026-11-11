# SPDX-License-Identifier: Apache-2.0
"""SCR-SOFT-2-5: the MPM grid choice (alloy.integration.mpm_grid). Sparse by default; dense on request, with a memory
guard; the persistent sparse grid under graph capture; one warning for sparse material resting on the ground.

Newton's sparse grid puts its node planes at half voxels, so the ground (z = 0) lies between two of them and a
20 Pa s film on a 10 deg slope ran at 0.82 of the analytic speed (dense: 0.97). The dense grid has no plane inside a
sub-voxel floor on the ground, and honey leaked through liquid_box_56's 4 mm floor; sparse holds, so it stays default.
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace

import pytest

from alloy.core.registry import isolated_registry
from alloy.integration import mpm_grid as G
from alloy.integration.mpm_grid import (GRID_ENV, MPMGridOverflowError, default_grid_type, dense_cell_budget,
                                        grid_override, guard_dense_grid, warn_sparse_ground)
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.integration]


def _sand(worlds: int = 1):
    _, card, lib = next(x for x in S.mixed_scenes() if x[0] == "sand_heap")
    return I.assemble(dataclasses.replace(card, worlds=worlds), lib, "cpu", worlds)


def _mpm(sim):
    from alloy._vendor import newton  # noqa: PLC0415

    (solver,) = [s for s in sim.solvers if isinstance(s, newton.solvers.SolverImplicitMPM)]
    return solver


def test_the_default_grid_is_sparse_and_dense_is_guarded(monkeypatch) -> None:
    from alloy.integration import build_simulation  # noqa: PLC0415

    monkeypatch.delenv(GRID_ENV, raising=False)
    with isolated_registry(empty=True):
        assert _mpm(build_simulation(_sand(), "standard", graph="off")).grid_type == "sparse"
        monkeypatch.setenv(GRID_ENV, "dense")
        solver = _mpm(build_simulation(_sand(), "standard", graph="off"))
    assert solver.grid_type == "dense" and solver.alloy_grid["grid_type"] == "dense"
    assert solver.alloy_grid["budget_cells"] == dense_cell_budget(solver.model.particle_count)


@pytest.mark.parametrize("worlds, expect", [(1, "dense"), (2, "dense")])
def test_the_override_asks_for_dense_and_several_cpu_worlds_are_dense_anyway(monkeypatch, worlds: int,
                                                                              expect: str) -> None:
    from alloy.integration import build_simulation  # noqa: PLC0415

    monkeypatch.setenv(GRID_ENV, "sparse")
    with isolated_registry(empty=True), grid_override("dense"):
        solver = _mpm(build_simulation(_sand(worlds), "standard", graph="off"))
    assert solver.grid_type == expect and default_grid_type() == "sparse"     # the override ends with the block


def test_a_bad_grid_name_raises(monkeypatch) -> None:
    monkeypatch.setenv(GRID_ENV, "dence")
    with pytest.raises(ValueError, match=GRID_ENV):
        default_grid_type()
    with pytest.raises(ValueError, match="grid_override"), grid_override("coarse"):
        pass


def _view(min_z: float, plane: bool):
    from alloy._vendor import newton  # noqa: PLC0415

    arr = lambda a: SimpleNamespace(numpy=lambda: a)                         # noqa: E731
    types = [int(newton.GeoType.PLANE)] if plane else [int(newton.GeoType.BOX)]
    return SimpleNamespace(shape_type=arr(types), particle_q=arr([[0.0, 0.0, min_z], [0.0, 0.0, 0.05]]))


def test_sparse_material_on_the_ground_warns_once(monkeypatch, caplog) -> None:
    monkeypatch.setattr(G, "_WARNED_GROUND", [])
    solver = SimpleNamespace(grid_type="sparse", _mpm_model=SimpleNamespace(voxel_size=0.005))
    assert not warn_sparse_ground(solver, _view(0.012, plane=True))          # 12 mm up: a fill on a floor
    assert not warn_sparse_ground(solver, _view(0.001, plane=False))         # no ground plane
    assert not warn_sparse_ground(SimpleNamespace(grid_type="dense"), _view(0.001, plane=True))
    with caplog.at_level(logging.WARNING, logger="alloy.integration"):
        assert warn_sparse_ground(solver, _view(0.00125, plane=True))
        assert not warn_sparse_ground(solver, _view(0.00125, plane=True))   # once per process
    assert sum("ALLOY_MPM_GRID=dense" in r.getMessage() for r in caplog.records) == 1


def test_graph_capture_keeps_the_persistent_sparse_grid_unless_the_entry_chose() -> None:
    from alloy.integration.wiring import _graph_entries  # noqa: PLC0415

    with isolated_registry(empty=True):
        built = _sand()
    notes: list[str] = []
    (entry,) = [e for e in _graph_entries(built, notes).entries if e.capability == "mpm"]
    assert entry.solver_kwargs["grid_type"] == "sparse" and entry.solver_kwargs["max_active_cell_count"] > 0
    own = dataclasses.replace(built, entries=[dataclasses.replace(e, solver_kwargs={"grid_type": "dense"})
                                              if e.capability == "mpm" else e for e in built.entries])
    (entry,) = [e for e in _graph_entries(own, []).entries if e.capability == "mpm"]
    assert entry.solver_kwargs["grid_type"] == "dense"


class _Grid:
    def __init__(self, res: tuple[int, int, int], envs: int = 1) -> None:
        self.res, self._envs = res, envs

    def environment_count(self) -> int:
        return self._envs


def _fake_solver(res: tuple[int, int, int]):
    calls: list[str] = []
    solver = SimpleNamespace(grid_type="dense", _scratchpad=SimpleNamespace(), _last_step_data=SimpleNamespace())

    def allocate(*_a, **_k):
        calls.append(solver.grid_type)
        return _Grid(res) if solver.grid_type == "dense" else "sparse grid"

    solver._allocate_grid = allocate
    return solver, calls


def test_a_dense_grid_over_its_budget_falls_back_to_sparse_once(caplog) -> None:
    budget = dense_cell_budget(1000)
    solver, calls = _fake_solver((100, 100, budget // 10_000 + 1))           # just over the budget
    scratch = solver._scratchpad
    guard_dense_grid(solver, 1000, "sand", can_fall_back=True)
    with caplog.at_level(logging.WARNING, logger="alloy.integration"):
        assert solver._allocate_grid() == "sparse grid"
        assert solver._allocate_grid() == "sparse grid"
    assert calls == ["dense", "sparse", "sparse"] and solver.grid_type == "sparse"
    assert solver._scratchpad is not scratch                                   # fields rebuilt for the sparse space
    assert solver.alloy_grid["fallback"]["cells"] > budget
    assert sum("switching to the sparse grid" in r.getMessage() for r in caplog.records) == 1


def test_within_budget_stays_dense_and_without_a_fallback_raises() -> None:
    solver, calls = _fake_solver((40, 40, 40))
    guard_dense_grid(solver, 1000, "sand", can_fall_back=True)
    assert isinstance(solver._allocate_grid(), _Grid) and solver.alloy_grid["fallback"] is None
    big, _ = _fake_solver((1000, 1000, 21))                                    # over the 20M cap
    guard_dense_grid(big, 1000, "sand", can_fall_back=False)
    with pytest.raises(MPMGridOverflowError, match="FAR_FROM_ORIGIN"):
        big._allocate_grid()


def test_a_real_solver_switches_to_sparse_mid_run_and_keeps_stepping() -> None:
    """The switch needs fresh Newton fields: before, the next step raised "Incompatible space and/or space partition"
    (the soil dig's carry on a dense grid went over the old 100k-cell budget)."""
    from alloy.integration import build_simulation  # noqa: PLC0415

    with isolated_registry(empty=True), grid_override("dense"):
        sim = build_simulation(_sand(), "standard", graph="off")
    solver = _mpm(sim)
    assert solver.grid_type == "dense" and solver.alloy_grid["fallback"] is None
    solver.alloy_grid["budget_cells"] = 10                                   # the next step is over its budget
    for _ in range(3):
        sim.step_period()
    fb = solver.alloy_grid["fallback"]
    assert solver.grid_type == "sparse" and fb["budget_cells"] == 10 and fb["cells"] > 10 and "at" not in fb


def test_a_dense_entry_over_its_budget_at_the_start_is_built_sparse(monkeypatch, caplog) -> None:
    """Newton's constructor builds the sparse solver: no switch inside a step (with separate worlds on CUDA that one
    failed on the next step, WP13's L4 log at 64 worlds)."""
    from alloy.integration import build_simulation  # noqa: PLC0415

    monkeypatch.setattr(G, "DENSE_MIN_CELLS", 10)
    monkeypatch.setattr(G, "DENSE_CELLS_PER_PARTICLE", 0)                    # budget 10 cells
    with isolated_registry(empty=True), grid_override("dense"), caplog.at_level(logging.WARNING, "alloy.integration"):
        sim = build_simulation(_sand(), "standard", graph="off")
        solver = _mpm(sim)
        sim.step_period()
    fb = solver.alloy_grid["fallback"]
    assert solver.grid_type == "sparse" and solver.alloy_grid["grid_type"] == "sparse"
    assert fb["at"] == "build" and fb["budget_cells"] == 10 and fb["cells"] > 10
    assert sum("building it on the sparse grid" in r.getMessage() for r in caplog.records) == 1
    assert not any("switching to the sparse grid" in r.getMessage() for r in caplog.records)


def _points_view(points, flags, device: str = "cuda:0", count: int | None = None):
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    q = wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device="cpu")
    f = wp.array(np.asarray(flags, np.int32), dtype=wp.int32, device="cpu")
    return SimpleNamespace(particle_q=q, particle_flags=f, device=device, particle_count=count or len(points))


def test_dense_start_cells_is_newtons_box_by_newtons_particle_rule() -> None:
    """CUDA: the active particles (Newton's bounds kernel reads the flags); CPU: all of them. Float32 arithmetic as
    Newton's: 0.31 / 0.005 rounds to 62 in float32 (63 in float64, WP4's review)."""
    pts, flags = [[0.001, 0.001, 0.001], [0.019, 0.009, 0.004], [5.0, 5.0, 5.0]], [1, 1, 0]   # the far one inactive
    assert G.dense_start_cells(_points_view(pts, flags), 0.005) == 4 * 2 * 1          # [0, 20] x [0, 10] x [0, 5] mm
    assert G.dense_start_cells(_points_view(pts, flags), 0.005, padding=1, worlds=3) == 6 * 4 * 3 * 3
    assert G.dense_start_cells(_points_view(pts, flags, "cpu"), 0.005) == 1000 ** 3
    assert G.dense_start_cells(_points_view([[0.0, 0.0, 0.0], [0.31, 0.31, 0.31]], [1, 1]), 0.005) == 62 ** 3


def test_dense_start_fallback_counts_separate_worlds_only(monkeypatch) -> None:
    monkeypatch.setattr(G, "DENSE_MIN_CELLS", 10)
    monkeypatch.setattr(G, "DENSE_CELLS_PER_PARTICLE", 0)                    # budget 10; one world's box is 8 cells
    view = _points_view([[0.001, 0.001, 0.001], [0.019, 0.009, 0.004]], [1, 1])
    cfg = SimpleNamespace(grid_type="dense", voxel_size=0.005, grid_padding=0, separate_worlds=True)
    assert G.dense_start_fallback(view, cfg, 1, "sand") is None
    assert G.dense_start_fallback(view, cfg, 3, "sand") == {"at": "build", "cells": 24, "budget_cells": 10}
    shared = SimpleNamespace(**{**vars(cfg), "separate_worlds": False})
    assert G.dense_start_fallback(view, shared, 3, "sand") is None                   # one shared box: 8 cells
    assert G.dense_start_fallback(view, SimpleNamespace(**{**vars(cfg), "grid_type": "sparse"}), 3, "sand") is None


def test_chosen_says_whether_the_user_picked_the_grid(monkeypatch) -> None:
    """``mpm_grid.chosen``: the environment variable or an active ``grid_override``; a route with its own default
    (a j2 solid's dense grid on the CPU) yields to it."""
    from alloy.integration import mpm_grid as G

    monkeypatch.delenv(G.GRID_ENV, raising=False)
    assert not G.chosen()
    with G.grid_override("dense"):
        assert G.chosen()
    assert not G.chosen()
    monkeypatch.setenv(G.GRID_ENV, "sparse")
    assert G.chosen()
