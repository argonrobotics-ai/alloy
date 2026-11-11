# SPDX-License-Identifier: Apache-2.0
"""Implicit MPM's grids: sparse by default, dense on request (with a memory guard), the persistent grid under graphs.

**Which grid** (SCR-SOFT-2-5, coordinator 2026-11-07: option A). Neither of Newton's grids handles a wall that lies
between two node planes; they differ in where their planes are. Sparse: planes at (k + 1/2) voxel (``grid_offset =
0.5``, newton@ac7f7830 solver_implicit_mpm.py:3118), so the ground (z = 0) lies midway between two, and the bottom
particle layers of a flow on it shear too little (a 20 Pa s film on a 10 deg slope: 0.82 of the analytic speed at any
voxel or dt). Dense: planes on whole voxels (grid min = floor(bbox_min / voxel)), so flows on the ground are right
(0.970, 200 Pa s 0.995), but a floor thinner than a voxel resting on the ground has no node plane inside it and
leaks particles (honey in liquid_box_56's 4 mm floor: min z -0.24 mm in 0.3 s; sparse holds). Fills in containers are
the common case, so sparse stays the default; ``ALLOY_MPM_GRID=dense``, an entry's ``grid_type`` or
``grid_override("dense")`` (law rigs: the film laws on the ground) ask for dense. ``warn_sparse_ground`` warns once
when a sparse entry starts with material within a voxel of a ground plane. Known gap: mpm-sparse-floor-offset.md.

A dense grid covers the particles' bounding box, every step, so grains that fly off (an ejected grain falling for
seconds) would grow it without bound: ``guard_dense_grid`` checks each allocation against
``dense_cell_budget`` (4 cells per particle, within 2M-20M) and switches that solver to the sparse grid for good, with
one warning (``solver.alloy_grid`` records the grid in use; the switching step starts without a warm start). Where
Newton has no sparse grid to fall back to (several separate worlds on the CPU, which are always dense) it raises
``MPMGridOverflowError`` instead of running out of memory.

Graph capture needs SolverImplicitMPM's rebuildable sparse grid: a positive ``max_active_cell_count``, since the default
grid reads the particle bounds on the host every step (``wiring._graph_entries``). Everything built on that grid (the
velocity, strain and collider spaces, their matrices and the solve's temporaries) is sized by the capacity, not by
the cells in use (newton@ac7f7830 solver_implicit_mpm.py, "Build active partition").

An active cell holds at least one particle. The rebuildable grid needs ``grid_padding == 0``, and the guard cells
between worlds are empty index spacing, not active cells (warp.fem ``Nanogrid.from_environment_voxels``). So the
particle count bounds the active cells (``CELLS_PER_PARTICLE = 1`` cannot overflow). The old 8 per particle reserved 8x
that bound: 885k cells for WP5's sand_heap at 16 worlds, whose first captured period ran out of memory on an L4
(wp_cuda_graph_create_exec).

**Cells in use** (SCR-WP12-3 a2). Material fills one cell per ~5 particles (sand_heap, wp12-results "WP12-MPM-CELLS")
to ~9 (a dough ball, 2 per axis), and everything on the rebuildable grid is sized by its reservation: one cell per
particle cost dough 398 MiB per world as graphs, 4x fewer cells 184 MiB, and ran 2.3x faster ([internal note]
notes/wp12-results/mpm-memory). ``start_cells`` counts the cells the particles fill at the start; the reservation is
``CELL_HEADROOM`` times that, at least ``MIN_CELLS``, at most the particle count. A scene with inactive particles at
the start (a pour's spares) keeps one cell per particle. ``GridGuard.grow`` doubles a reservation that overflowed
(up to the particle count) and the simulation re-runs the period on the new grid (``grid_growth``).
``ALLOY_MPM_CELLS=particles`` keeps one cell per particle. Eager scenes keep Newton's per-step grid (on the reserved
grid, 64 dough worlds ran 1.48x slower in 3.4x the memory: the step works over the spare cells).

Leaf nodes keep Newton's default, one per reserved cell. Lower and upper nodes are Newton's estimate from the first
grid with 16x headroom, and those can overflow. ``GridGuard.check`` reads each rebuildable grid's status once per
period (a 4-byte readback) and raises ``MPMGridOverflowError``: a replayed graph does not report it otherwise.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

log = logging.getLogger("alloy.integration")

__all__ = ["CELLS_ENV", "CELLS_PER_PARTICLE", "CELL_HEADROOM", "DENSE_CELLS_PER_PARTICLE", "GRID_ENV", "GridGuard",
           "MPMGridOverflowError", "by_particles", "chosen", "default_grid_type", "dense_box", "dense_cell_budget",
           "dense_start_cells", "dense_start_fallback", "entry_cells", "graph_cell_capacity", "grid_override",
           "guard_dense_grid", "start_cells", "warn_sparse_ground"]

CELLS_PER_PARTICLE = 1              # the bound: an active cell holds at least one particle
CELL_HEADROOM = 2.0                 # reserved cells per cell in use at the start (SCR-WP12-3 a2)
MIN_CELLS = 4096                    # per entry, all worlds
GROWTH = 2                          # an overflowed reservation grows by this factor, up to the particle count
CELLS_ENV = "ALLOY_MPM_CELLS"       # "particles": one cell per particle (before SCR-WP12-3 a2)
DENSE_CELLS_PER_PARTICLE = 4        # a resting fill takes ~1 cell per 8 particles: 32x headroom for its bounding box
DENSE_MIN_CELLS = 2_000_000        # a small scene may spread (a scoop carries soil 16 cm: 100k cells for 15k particles)
DENSE_MAX_CELLS = 20_000_000        # any scene: the reconcile row measures bytes per cell ([internal note] xb4 queue)
GRID_ENV = "ALLOY_MPM_GRID"         # "dense" asks for the dense grid (module docstring)
_OVERRIDE: list[str] = []
_WARNED_GROUND: list[bool] = []


class MPMGridOverflowError(RuntimeError):
    """A rebuildable MPM sparse grid went over its reserved capacity (cells or NanoVDB nodes)."""


def graph_cell_capacity(particles: int, cells: int | None = None) -> int:
    """``max_active_cell_count`` for an MPM entry of ``particles`` particles that fill ``cells`` cells at the start
    (None: unknown, one cell per particle; module docstring)."""
    bound = max(1, CELLS_PER_PARTICLE * int(particles))
    if cells is None or by_particles():
        return bound
    return int(min(bound, max(MIN_CELLS, math.ceil(CELL_HEADROOM * int(cells)))))


def entry_cells(model: Any, entry: Any) -> int | None:
    """``start_cells`` of an MPM entry (its particles at its voxel size)."""
    return start_cells(model, entry.particles, entry.solver_kwargs.get("voxel_size"))


def by_particles() -> bool:
    """``ALLOY_MPM_CELLS=particles``: the reservation and eager grid as before SCR-WP12-3 a2."""
    return os.environ.get(CELLS_ENV, "").strip().lower() == "particles"


def start_cells(model: Any, particles: Sequence[int], voxel: float | None) -> int | None:
    """Cells the ``particles`` of ``model`` fill now, counted per world at ``voxel``; None when unknown (no voxel)
    or when any of them is inactive (a pour's spares sit together until they are poured)."""
    from alloy._vendor.newton import ParticleFlags  # noqa: PLC0415

    if not voxel or not len(particles):
        return None
    idx = np.asarray(particles, dtype=np.int64)
    flags = getattr(model, "particle_flags", None)
    if flags is not None and not np.all(flags.numpy()[idx] & int(ParticleFlags.ACTIVE)):
        return None
    keys = np.floor(model.particle_q.numpy()[idx] / float(voxel)).astype(np.int64)
    world = getattr(model, "particle_world", None)
    world = world.numpy()[idx].astype(np.int64)[:, None] if world is not None else np.zeros((len(idx), 1), np.int64)
    return int(len(np.unique(np.hstack([world, keys]), axis=0)))


class GridGuard:
    """Checks the rebuildable sparse grids of a scene's MPM solvers after each period. Duck-typed: any solver with
    ``_sparse_rebuildable`` set and a ``check_sparse_grid_rebuild_status`` method."""

    def __init__(self, solvers: Mapping[str, Any], limits: Mapping[str, int] | None = None) -> None:
        self._solvers = {n: s for n, s in solvers.items() if getattr(s, "_sparse_rebuildable", False)
                         and callable(getattr(s, "check_sparse_grid_rebuild_status", None))}
        self._limits = dict(limits or {})           # per entry: the particle count, where growth stops
        self.grown: list[tuple[str, int, int]] = []  # (entry, cells before, cells after)

    @classmethod
    def for_solvers(cls, solvers: Mapping[str, Any], limits: Mapping[str, int] | None = None) -> GridGuard | None:
        guard = cls(solvers, limits)
        if not guard._solvers:
            return None
        log.info("MPM sparse-grid capacity checked once per period: %s", sorted(guard._solvers))
        return guard

    def check(self) -> None:
        for name, solver in self._solvers.items():
            try:
                solver.check_sparse_grid_rebuild_status()
            except RuntimeError as exc:
                cap = getattr(solver, "max_active_cell_count", "?")
                raise MPMGridOverflowError(f"mpm entry {name}: the sparse grid went over its reserved capacity "
                                           f"(max_active_cell_count={cap}): {exc}") from exc

    def grow(self) -> bool:
        """Grow every reservation that overflowed this period (``GROWTH`` x, up to the entry's particle count) and
        reset that solver's grid state; True when one grew. One at the particle count raises as ``check`` does."""
        grew = False
        for name, solver in self._solvers.items():
            try:
                solver.check_sparse_grid_rebuild_status()
                continue
            except RuntimeError as exc:
                cap, limit = int(solver.max_active_cell_count), self._limits.get(name, 0)
                if cap >= limit:
                    raise MPMGridOverflowError(f"mpm entry {name}: the sparse grid went over its reserved capacity "
                                               f"(max_active_cell_count={cap}, particle bound {limit}): {exc}") from exc
            new = min(limit, GROWTH * cap)
            log.warning("mpm entry %s: the sparse grid outgrew its %d reserved cells; %d from now on, and this period "
                        "re-runs without a warm start (SCR-WP12-3 a2)", name, cap, new)
            solver.max_active_cell_count = new
            _reset_grid_state(solver)
            solver._clear_sparse_grid_rebuild_status()
            self.grown.append((name, cap, new))
            grew = True
        return grew


def chosen() -> bool:
    """True when the user picked the grid: ``ALLOY_MPM_GRID`` is set or a ``grid_override`` is active. A caller with
    its own default (a route that needs the dense grid) yields to that choice. Example: ``chosen()`` is False in a
    plain build."""
    return bool(_OVERRIDE) or bool(os.environ.get(GRID_ENV, "").strip())


def default_grid_type() -> str:
    """The MPM grid an entry gets unless it sets ``grid_type``: ``grid_override``'s inside one, else ALLOY_MPM_GRID,
    else ``"sparse"``."""
    if _OVERRIDE:
        return _OVERRIDE[-1]
    value = os.environ.get(GRID_ENV, "sparse").strip().lower()
    if value not in ("dense", "sparse"):
        raise ValueError(f"{GRID_ENV}={value!r}: use 'sparse' (default) or 'dense'")
    return value


@contextlib.contextmanager
def grid_override(grid_type: str):
    """MPM entries built inside the block default to ``grid_type`` (process-wide, not for builds in parallel threads).
    Example: ``with grid_override("dense"): sim = alloy.build(scene)`` (a film on the ground, SOFT-2's L4/L5)."""
    if grid_type not in ("dense", "sparse"):
        raise ValueError(f"grid_override({grid_type!r}): use 'sparse' or 'dense'")
    _OVERRIDE.append(grid_type)
    try:
        yield
    finally:
        _OVERRIDE.pop()


def warn_sparse_ground(solver: Any, view: Any) -> bool:
    """One warning per process when ``solver`` runs the sparse grid and its material starts within a voxel of a ground
    plane: flows along the ground then run slow (module docstring). One host read of the particles at build."""
    if _WARNED_GROUND or getattr(solver, "grid_type", None) != "sparse":
        return False
    try:
        from alloy._vendor import newton  # noqa: PLC0415

        planes = np.asarray(view.shape_type.numpy()) == int(newton.GeoType.PLANE)
        z = np.asarray(view.particle_q.numpy())[:, 2]
        voxel = float(solver._mpm_model.voxel_size)
    except (AttributeError, TypeError, ValueError):            # a view without shapes or particles: nothing to say
        return False
    if not planes.any() or not len(z) or float(z.min()) > voxel:
        return False
    _WARNED_GROUND.append(True)
    log.warning("MPM material within a voxel (%.3g m) of the ground on the sparse grid: flows along the ground run "
                "slow (a 20 Pa s film at 0.82 of its speed). For flows on the ground without floors thinner than a "
                "voxel, set ALLOY_MPM_GRID=dense (docs/known_gaps/mpm-sparse-floor-offset.md)", voxel)
    return True


def dense_cell_budget(particles: int) -> int:
    """Most dense-grid cells (all worlds) an MPM entry of ``particles`` particles may allocate in one step."""
    return min(DENSE_MAX_CELLS, max(DENSE_MIN_CELLS, DENSE_CELLS_PER_PARTICLE * int(particles)))


def dense_box(q: np.ndarray, voxel: float, padding: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Integer voxel bounds (lo, hi) of Newton's dense grid over the points ``q`` (N, 3): floor(min / voxel) - padding
    to ceil(max / voxel) + padding, in the points' own float32 arithmetic as Newton does it (a Python float divisor
    keeps float32 under NumPy 2; newton@ac7f7830 solver_implicit_mpm.py:2798-2804). A float64 copy rounds differently
    next to voxel planes, one layer either way (WP4's review: 6 of 4M random points at a 2 mm voxel)."""
    lo = np.floor(q.min(axis=0) / voxel) - padding
    hi = np.ceil(q.max(axis=0) / voxel) + padding
    return lo, hi


def dense_start_cells(view: Any, voxel: float, padding: int = 0, worlds: int = 1) -> int:
    """Cells of the dense grid Newton would build over ``view``'s particles at their start, ``worlds`` copies of one
    box (Newton's Grid3D with one environment per separate world). Newton's rule for which particles count: on CUDA
    the active ones (its bounds kernel reads the flags), on the CPU all of them (``positions.numpy()``)."""
    q = view.particle_q.numpy()
    flags = getattr(view, "particle_flags", None)
    if flags is not None and str(getattr(view, "device", "cpu")).startswith("cuda"):
        q = q[(flags.numpy() & 1) != 0]
    if len(q) == 0:
        return 0
    lo, hi = dense_box(q, voxel, padding)
    return int(np.prod(hi - lo)) * max(1, int(worlds))


def dense_start_fallback(view: Any, config: Any, worlds: int, name: str) -> dict | None:
    """A dense ``config`` whose grid is over ``dense_cell_budget`` already at the start: one warning and the fallback
    record (the entry is then built sparse, as Newton builds any sparse solver); else None."""
    if config.grid_type != "dense":
        return None
    budget = dense_cell_budget(int(view.particle_count))
    cells = dense_start_cells(view, float(config.voxel_size), int(config.grid_padding),
                              worlds if config.separate_worlds else 1)
    if cells <= budget:
        return None
    log.warning("mpm entry %s: the dense grid over the particles' start box needs %d cells, over the budget %d "
                "(%d per particle); building it on the sparse grid (docs/known_gaps/mpm-sparse-floor-offset.md)",
                name, cells, budget, DENSE_CELLS_PER_PARTICLE)
    return {"at": "build", "cells": cells, "budget_cells": budget}


def _reset_grid_state(solver: Any) -> None:
    """Fresh scratchpad and warm-start fields: Newton binds them to the grid's space, and a dense (Grid3D) space does
    not rebind to a sparse (NanoVDB) one ("Incompatible space", newton@ac7f7830 solver_implicit_mpm.py:577). The step
    reads both after allocating the grid (:2146-2148), so the switching step starts cold (no warm start); a rollback
    across the switch restores the old, unused warm-start arrays."""
    solver._scratchpad = type(solver._scratchpad)()
    solver._last_step_data = type(solver._last_step_data)()


def _grid_cells(grid: Any) -> int:
    envs = getattr(grid, "environment_count", 1)
    envs = envs() if callable(envs) else envs
    return int(np.prod([int(r) for r in grid.res])) * max(1, int(envs))


def guard_dense_grid(solver: Any, particles: int, name: str, can_fall_back: bool, *, fallback: dict | None = None,
                     no_fallback_why: str = "no sparse grid to fall back to (separate worlds on the CPU)") -> None:
    """Check every dense allocation of ``solver`` (a SolverImplicitMPM) against ``dense_cell_budget``; over it, switch
    to the sparse grid for the rest of the run with one warning, or raise ``MPMGridOverflowError`` when
    ``can_fall_back`` is False (``no_fallback_why`` says why). ``solver.alloy_grid`` records the grid in use and any
    fallback (``fallback``: one already taken when the solver was built, ``dense_start_cells``)."""
    budget = dense_cell_budget(particles)
    inner = solver._allocate_grid
    solver.alloy_grid = {"grid_type": solver.grid_type, "budget_cells": budget, "fallback": fallback}

    def allocate(*args: Any, **kwargs: Any) -> Any:
        grid = inner(*args, **kwargs)
        if solver.grid_type != "dense" or not hasattr(grid, "res"):
            return grid
        cells, budget = _grid_cells(grid), solver.alloy_grid["budget_cells"]
        if cells <= budget:
            return grid
        why = (f"mpm entry {name}: the dense grid over the particles' bounding box needs {cells} cells, over the "
               f"budget {budget} ({DENSE_CELLS_PER_PARTICLE} per particle): particles have spread far apart")
        if not can_fall_back:
            raise MPMGridOverflowError(f"{why}; {no_fallback_why}. Look for particles thrown out of the scene (health "
                                       "FAR_FROM_ORIGIN), or use the sparse grid (the default)")
        log.warning("%s; switching to the sparse grid for the rest of the run "
                    "(docs/known_gaps/mpm-sparse-floor-offset.md)", why)
        solver.grid_type = "sparse"
        solver.alloy_grid.update(grid_type="sparse", fallback={"cells": cells, "budget_cells": budget})
        _reset_grid_state(solver)
        return inner(*args, **kwargs)

    solver._allocate_grid = allocate
