# SPDX-License-Identifier: Apache-2.0
"""Implicit MPM: one world's solve does not depend on the other worlds (SCR-WP13-MPMISO-1).

Newton's rheology loop (``solve_rheology._run_solver_loop``, newton@ac7f7830) keeps one iteration count for every world
and stops only when all of them are under tolerance (``update_batched_condition``). A world that converged early goes
on iterating while another one has not, so melting world 1 moved world 0 by 1.66e-7 m (melt_chocolate's scene, CPU, 2
worlds). This module wraps the loop: each world's iterate (strain stress and its last change, velocity nodes, collider
impulses and velocities, by the per-world node offsets) is saved at its first converged check and restored after the
loop, so each world ends where it would have stopped alone. The shared loop still runs until the last world converges
(or ``max_iterations``); only the result is per world.

- One world, or a model without per-world environments: Newton's loop runs untouched (bit for bit).
- Graph path (CUDA): the same launches inside ``wp.capture_while``, iteration count and stop rule as Newton's.
- The dense grid: Newton takes its bounds from every world's particles each step, and its cells' size and origin
  (``Grid3D``: ``(hi - lo) / res``, ``(x - lo) / cell_size``) then round differently when another world spreads: one
  float32 ulp in world 0 (1.5-3e-8 m). Here a multi-world dense grid keeps its integer voxel bounds (the first step's
  plus up to MARGIN voxels, within mpm_grid's dense budget) while every particle stays inside, so each world's
  arithmetic stays put. A particle leaving them grows the box for every world (logged), to the particles' box and
  the held one before its margin, plus one margin, and the worlds couple by an ulp at that step
  (docs/known_gaps/mpm-dense-grid-world-coupling.md). Per-world sparse grids (CUDA) index voxels from the origin.
- Newton also sets some material switches from every world's particles when the solver is built: the compliant path
  (``min_young_modulus < inf``) and hardening (``max_hardening > 0``). Alloy's cards always give a finite E, so the
  compliant path is on in every Alloy MPM scene; worlds that differ in whether they harden would still couple.
- The Newton internals relied on are pinned by tests/integration/test_mpm_world_independence.py, so a Newton bump that
  changes them fails there. The real fix is per-world masks in Newton's solver kernels (WP0's package process).

- Converged worlds are skipped (GAPS-NOW #5a): the loop hands its done flags to Newton's Gauss-Seidel sweep
  (``skip_converged``, vendored), which leaves a done world's strain rows alone. They are frozen until the restore
  anyway, so the result is bit for bit the same; a solve then costs the mean world's sweeps, not the slowest
  world's times every world. L4, 256 dough worlds whose presses start at spread heights: 7.87 against 10.10 s per
  period. Identical worlds stop together and gain nothing. ``SKIP_CONVERGED = False`` sweeps them as before.
- Partial sweeps (SPEED-NOW S5, opt-in ``alloy.build(..., mpm_sweeps="partial")``, or ``ALLOY_MPM_PARTIAL=on`` for a
  whole run; mpm_solve.effective_tau): after the scaled L2 residual
  is met (~23 sweeps), the max-row test holds a solve (~103) while a few rows still move. In each group of
  PARTIAL_GROUP sweeps the others visit only the colour blocks with a row that moved more than tau x tolerance in
  the last full sweep (vendored ``skip_blocks`` / ``mark_active_blocks``); the group's last sweep is full and the
  stop test reads only full sweeps, so a solve stops on the same rule. L4, 256 dough worlds: 7,795 -> 6,021 ms per
  period, the dough within 0.006 mm mean of plain. One world runs this loop too when partial sweeps are on (``[0, n]``
  tables; off, one world keeps Newton's loop bit for bit). A block
  skipped by a partial sweep also skips its coloured cell contacts; a block whose rows span worlds is always swept
  (WP8's review), and the threshold is in the max-row test's units (unscaled, as ``_converged``).

``install()`` is idempotent and chains onto whatever is in place.
"""

from __future__ import annotations

import logging
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
import warp.fem as fem
from alloy._vendor.newton._src.solvers.implicit_mpm import solve_rheology as SR
from alloy._vendor.newton._src.solvers.implicit_mpm import solver_implicit_mpm as SIM

from alloy.integration.mpm_grid import dense_box, dense_cell_budget

log = logging.getLogger("alloy.integration")

LANES = wp.constant(64)              # threads per world in the copy kernels
GRAPH_GRANULARITY = 5                # Newton's graph-path check interval (solve_rheology.py:1739)
SKIP_CONVERGED = True                # GAPS-NOW #5a: a converged world's strain rows are skipped (tests: False)
PARTIAL_TAU: float | None = None     # SPEED-NOW S5: a test override of the build setting (mpm_solve.effective_tau)
PARTIAL_GROUP = 5                    # S5: every PARTIAL_GROUP-th sweep is full, and so is every checked one
MARGIN = 8                           # voxels around the first dense box (the box then holds until a particle leaves)
_CTX: dict[str, Any] = {}            # the current solve's velocity and collider node offsets (set by _solve_rheology)


@wp.func
def _converged(residual: wp.array2d[float], scale: wp.array[float], tol2: float, e: int) -> bool:
    """Newton's per-world test (update_batched_condition)."""
    s = scale[e]
    return residual[0, e] < tol2 * s * s and residual[1, e] < tol2


@wp.func
def _copy(src: wp.array(dtype=Any), dst: wp.array(dtype=Any), offsets: wp.array[int], e: int, lane: int):
    for i in range(offsets[e] + lane, offsets[e + 1], LANES):
        dst[i] = src[i]


@wp.kernel
def _slices_kernel(residual: wp.array2d[float], scale: wp.array[float], tol2: float, done: wp.array[wp.int32],
                   restore: int, so: wp.array[int], vo: wp.array[int], co: wp.array[int],
                   stress: wp.array(dtype=Any), dstress: wp.array(dtype=Any), vel: wp.array(dtype=Any),
                   imp: wp.array(dtype=Any), cvel: wp.array(dtype=Any), s_stress: wp.array(dtype=Any),
                   s_dstress: wp.array(dtype=Any), s_vel: wp.array(dtype=Any), s_imp: wp.array(dtype=Any),
                   s_cvel: wp.array(dtype=Any)):
    """Save (restore 0): a world converged for the first time copies its slices of the iterate. Restore (1): every
    converged world copies them back. LANES threads per world."""
    e, lane = wp.tid()
    if restore == 0:
        if done[e] != 0 or not _converged(residual, scale, tol2, e):
            return
        _copy(stress, s_stress, so, e, lane)
        _copy(dstress, s_dstress, so, e, lane)
        _copy(vel, s_vel, vo, e, lane)
        _copy(imp, s_imp, co, e, lane)
        _copy(cvel, s_cvel, co, e, lane)
    elif done[e] != 0:
        _copy(s_stress, stress, so, e, lane)
        _copy(s_dstress, dstress, so, e, lane)
        _copy(s_vel, vel, vo, e, lane)
        _copy(s_imp, imp, co, e, lane)
        _copy(s_cvel, cvel, co, e, lane)


@wp.kernel
def _condition_kernel(residual: wp.array2d[float], scale: wp.array[float], tol2: float, done: wp.array[wp.int32],
                      granularity: int, max_iterations: int, iteration: wp.array[int], condition: wp.array[int]):
    """Marks the worlds that converged, then Newton's stop rule on the flags: every world converged, or past
    ``max_iterations``."""
    it = iteration[0] + granularity
    every = bool(True)
    for e in range(done.shape[0]):
        if done[e] == 0 and _converged(residual, scale, tol2, e):
            done[e] = 1
        every = every and done[e] == 1
    iteration[0] = it
    condition[0] = wp.where(every or it > max_iterations, 0, 1)


def _iterate(rs: Any, cs: Any) -> list[tuple[wp.array, wp.array | None]] | None:
    """(array, per-world node offsets) of everything an iteration writes and the post-processing reads: strain stress
    and its last change, velocity nodes, collider impulses and velocities (offsets None: no colliders). None when an
    offset table is missing (Newton's loop then runs untouched)."""
    strain, vel, col = rs.rheology.strain_environment_offsets, _CTX.get("velocity"), _CTX.get("collider")
    if strain is None or vel is None:
        return None
    out = [(rs.rheology.stress, strain), (rs.delta_stress, strain), (rs.momentum.velocity, vel)]
    if cs is not None and cs.collision.has_colliders:
        if col is None:
            return None
        return out + [(cs.collision.collider_impulse, col), (cs.collision.collider_velocities, col)]
    return out + [(rs.momentum.velocity, None), (rs.momentum.velocity, None)]     # empty ranges: nothing to copy


class _Freeze:
    """The per-world flags and saved iterates for one solve (two launches per check: the slices, the condition)."""

    def __init__(self, arrays: list, worlds: int, store: Any, device: Any) -> None:
        self.worlds, self.device = worlds, device
        self.ints = fem.borrow_temporary(store, shape=(worlds + 1 + 2,), dtype=int, device=device)
        self.ints.zero_()
        empty, self.counter = self.ints[: worlds + 1], self.ints[worlds + 1:]
        self.offsets = [arrays[0][1], arrays[2][1], arrays[3][1] if arrays[3][1] is not None else empty]
        self.arrays = [a for a, _ in arrays]
        self.saved = [fem.borrow_temporary_like(a, store) for a in self.arrays]
        self.done = fem.borrow_temporary(store, shape=(worlds,), dtype=wp.int32, device=device)
        self.done.zero_()

    def _slices(self, residual: Any, scale: Any, tol2: float, restore: int) -> None:
        wp.launch(_slices_kernel, dim=(self.worlds, LANES), inputs=[residual, scale, tol2, self.done, restore,
                  *self.offsets, *self.arrays, *self.saved], device=self.device)

    def check(self, residual: wp.array, scale: wp.array, tolerance: float, granularity: int, max_iterations: int,
              iteration: wp.array, condition: wp.array) -> None:
        tol2 = tolerance * tolerance
        self._slices(residual, scale, tol2, 0)
        wp.launch(_condition_kernel, dim=1, inputs=[residual, scale, tol2, self.done, granularity, max_iterations,
                  iteration, condition], device=self.device)

    def restore(self, residual: wp.array, scale: wp.array) -> None:
        self._slices(residual, scale, 0.0, 1)

    def release(self) -> None:
        for t in (*self.saved, self.done, self.ints):
            t.release()


class _Partial:
    """SPEED-NOW S5 partial sweeps: per colour block, still moving after the last full sweep (1) or not (0); every
    solve starts with all blocks active."""

    def __init__(self, rs: Any, tolerance: float, tau: float, store: Any) -> None:
        self.held = fem.borrow_temporary(store, shape=(rs.rheology.color_blocks.shape[1],), dtype=wp.int32)
        self.active = self.held
        self.active.fill_(1)
        self.threshold = tau * tolerance

    def mark(self, rs: Any, evaluate: bool = True) -> None:
        rs.mark_active_blocks(self.active, self.threshold, evaluate)


@wp.kernel
def _table_kernel(table: wp.array[int], n: int):
    table[0] = 0
    table[1] = n


@wp.kernel
def _scale_kernel(out: wp.array[float], value: float):
    out[0] = value


class _OneWorld:
    """SPEED-NOW S5: the [0, n] tables (and the scale as an array) of a one-world solve, which Newton keeps none for,
    so partial sweeps run through this loop too. Filled by kernels: a graph capture may be recording."""

    def __init__(self, rs: Any, cs: Any, scale: float, store: Any) -> None:
        self.held: list[Any] = []
        strain, vel = self._table(rs.size, store), self._table(rs.momentum.velocity.shape[0], store)
        self.arrays = [(rs.rheology.stress, strain), (rs.delta_stress, strain), (rs.momentum.velocity, vel)]
        if cs is not None and cs.collision.has_colliders:
            col = self._table(cs.collision.collider_impulse.shape[0], store)
            self.arrays += [(cs.collision.collider_impulse, col), (cs.collision.collider_velocities, col)]
        else:
            self.arrays += [(rs.momentum.velocity, None), (rs.momentum.velocity, None)]   # nothing to copy
        self.scale = fem.borrow_temporary(store, shape=(1,), dtype=float)
        self.held.append(self.scale)
        wp.launch(_scale_kernel, dim=1, inputs=[self.scale, float(scale)])

    def _table(self, n: int, store: Any) -> Any:
        t = fem.borrow_temporary(store, shape=(2,), dtype=int)
        self.held.append(t)
        wp.launch(_table_kernel, dim=1, inputs=[t, int(n)])
        return t

    def release(self) -> None:
        for t in self.held:
            t.release()


def _wants_partial(rs: Any) -> bool:
    tau = PARTIAL_TAU if PARTIAL_TAU is not None else _CTX.get("partial")
    return bool(tau) and getattr(rs, "skips_converged", False) and hasattr(rs, "skip_blocks")


def _loop_tables(rs: Any, cs: Any, scale: Any, store: Any) -> tuple[list, int, Any, _OneWorld | None] | None:
    """(arrays, worlds, scale, one-world tables) for Alloy's loop, or None for Newton's loop as it is: one world
    without partial sweeps (bit for bit Newton's), or a solve whose per-world tables are missing."""
    offsets = rs.rheology.strain_environment_offsets
    worlds = 1 if offsets is None else offsets.shape[0] - 1
    arrays = _iterate(rs, cs)
    if worlds > 1:
        return None if arrays is None else (arrays, worlds, scale, None)
    if not _wants_partial(rs):
        return None
    if offsets is not None:
        return None if arrays is None else (arrays, 1, scale, None)
    one = _OneWorld(rs, cs, scale, store)
    return one.arrays, 1, one.scale, one


def _partial(rs: Any, tolerance: float, store: Any) -> _Partial | None:
    """Partial sweeps when the build (``mpm_sweeps``), ``ALLOY_MPM_PARTIAL`` or ``PARTIAL_TAU`` asks for them and the
    solver can skip blocks."""
    tau = PARTIAL_TAU if PARTIAL_TAU is not None else _CTX.get("partial")
    blocks = getattr(rs, "skip_blocks", None) if tau else None
    return _Partial(rs, tolerance, tau, store) if blocks and blocks(None) else None


def _iterate_once(rs: Any, cs: Any, granularity: int, part: _Partial | None = None) -> None:
    """``granularity`` sweeps; with ``part``, every PARTIAL_GROUP-th is full and the others visit only the active
    blocks; the last sweep is always full (the stop test reads it)."""
    for k in range(granularity):
        full = part is None or k % PARTIAL_GROUP == PARTIAL_GROUP - 1 or k == granularity - 1
        if part is not None:
            rs.skip_blocks(None if full else part.active)
        cs.solve()
        rs.solve()
        if part is not None and full and k != granularity - 1:   # the batch's last: marked after its check
            part.mark(rs)


def _graph_loop(rs: Any, cs: Any, fz: _Freeze, max_iterations: int, tolerance: float, scale: wp.array,
                store: Any, part: _Partial | None = None) -> Any:
    """Newton's graph path (solve_rheology.py:1737-1803) with the per-world checks and the restore."""
    ic = fem.borrow_temporary(store, shape=(2,), dtype=int)
    ic.fill_(1)
    iteration, condition = ic[:1], ic[1:]
    holder = {}

    def body() -> None:
        _iterate_once(rs, cs, GRAPH_GRANULARITY, part)
        holder["residual"] = rs.eval_residual()
        fz.check(holder["residual"], scale, tolerance, GRAPH_GRANULARITY, max_iterations, iteration, condition)
        if part is not None:
            part.mark(rs, evaluate=False)

    graph = None
    with SR._ScopedDisableGC():
        if fz.device.is_capturing:
            if part is not None:
                part.active.fill_(1)
            wp.capture_while(condition, body)
            fz.restore(holder["residual"], scale)
        else:
            with wp.ScopedCapture(force_module_load=False) as capture:
                if part is not None:
                    part.active.fill_(1)
                wp.capture_while(condition, body)
                fz.restore(holder["residual"], scale)
            graph = capture.graph
            wp.capture_launch(graph)
    ic.release()
    return graph


def _host_loop(rs: Any, cs: Any, fz: _Freeze, max_iterations: int, tolerance: float, scale: wp.array,
               part: _Partial | None = None) -> None:
    """Newton's host path (solve_rheology.py:1804-1835): blocks of the solver's granularity, the per-world checks."""
    granularity = rs.solve_granularity
    iteration, condition = fz.counter[:1], fz.counter[1:]
    residual = None
    for _batch in range((max_iterations + granularity - 1) // granularity):
        _iterate_once(rs, cs, granularity, part)
        residual = rs.eval_residual()
        fz.check(residual, scale, tolerance, granularity, max_iterations, iteration, condition)
        if part is not None:
            part.mark(rs, evaluate=False)
        if bool(fz.done.numpy().all()):
            break
    if residual is not None:
        fz.restore(residual, scale)


def _wrap_loop(prev: Any) -> Any:
    def run_solver_loop(rheology_solver: Any, contact_solver: Any, max_iterations: int, tolerance: float,
                        l2_tolerance_scale: Any, use_graph: bool, verbose: bool, temporary_store: Any) -> Any:
        args = (rheology_solver, contact_solver, max_iterations, tolerance, l2_tolerance_scale, use_graph, verbose,
                temporary_store)
        tables = _loop_tables(rheology_solver, contact_solver, l2_tolerance_scale, temporary_store)
        if tables is None:
            return prev(*args)                      # one world (no partial sweeps), or no per-world tables
        arrays, worlds, scale, one = tables
        fz = _Freeze(arrays, worlds, temporary_store, rheology_solver.device)
        skip = getattr(rheology_solver, "skip_converged", None) if SKIP_CONVERGED else None
        skipping = bool(skip and skip(fz.done))     # GAPS-NOW #5a: converged worlds cost no strain sweeps
        part = _partial(rheology_solver, tolerance, temporary_store)  # SPEED-NOW S5 (opt-in)
        try:
            if use_graph:
                return _graph_loop(rheology_solver, contact_solver, fz, max_iterations, tolerance, scale,
                                   temporary_store, part)
            _host_loop(rheology_solver, contact_solver, fz, max_iterations, tolerance, scale, part)
            return None
        finally:
            if skipping:
                skip(None)
            if part is not None:
                rheology_solver.skip_blocks(None)
                part.held.release()
            fz.release()
            if one is not None:
                one.release()

    run_solver_loop._alloy_world_freeze = True      # type: ignore[attr-defined]
    return run_solver_loop


def _wrap_solve(prev: Any) -> Any:
    def solve_rheology(self: Any, scratch: Any, rigidity_operator: Any) -> Any:
        _CTX["velocity"], _CTX["collider"] = scratch.velocity_environment_offsets, scratch.collider_environment_offsets
        from alloy.integration.mpm_solve import effective_tau  # noqa: PLC0415

        _CTX["partial"] = effective_tau(self.model)  # SPEED-NOW S5: build(mpm_sweeps="partial") / ALLOY_MPM_PARTIAL
        try:
            return prev(self, scratch, rigidity_operator)
        finally:
            _CTX.clear()

    solve_rheology._alloy_world_freeze = True       # type: ignore[attr-defined]
    return solve_rheology


def _with_margin(lo: np.ndarray, hi: np.ndarray, worlds: int, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """The box plus the widest margin up to MARGIN voxels whose cells (all worlds) fit the dense budget (mpm_grid)."""
    for m in range(MARGIN, 0, -1):
        if int(np.prod(hi - lo + 2 * m)) * worlds <= budget:
            return lo - m, hi + m
    return lo, hi


def _dense_bounds(self: Any, positions: wp.array, flags: wp.array, voxel: float, pad: int) -> tuple[Any, Any]:
    """Integer voxel bounds of the multi-world dense grid: Newton's box (solver_implicit_mpm.py:2768-2802; on CUDA its
    active particles only, as compute_bounds), held while it covers them, else grown to them plus a margin (up to
    MARGIN voxels, within the dense budget). Growth unions the particles' box with the held core (no margin) and puts
    the margin on once: growing from the margined box stacked a margin per growth (WP12's 100k red <gpu-vm>:
    2.1M cells over the dense guard for 332k cells of particles)."""
    p = positions.numpy()
    if positions.device.is_cuda:
        p = p[(flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0]
    lo, hi = dense_box(p, voxel, pad)                  # mpm_grid's copy of Newton's rule (float32, as Newton)
    box = getattr(self, "_alloy_dense_box", None)
    if box is None or (lo < box[0]).any() or (hi > box[1]).any():
        core = getattr(self, "_alloy_dense_core", None)
        if box is not None:
            log.warning("implicit MPM dense grid: a particle left the held box, grown for every world (the worlds "
                        "couple by about one float32 ulp this step; mpm_worlds)")
            lo, hi = np.minimum(lo, core[0]), np.maximum(hi, core[1])
        _hold(self, lo, hi, self._environment_count, dense_cell_budget(int(self.model.particle_count)))
    return self._alloy_dense_box


def _hold(solver: Any, lo: np.ndarray, hi: np.ndarray, worlds: int, budget: int) -> None:
    """Hold the core ``lo``, ``hi`` (no margin) and the box: the core plus ``_with_margin``'s margin."""
    solver._alloy_dense_core = (lo, hi)
    solver._alloy_dense_box = _with_margin(lo, hi, worlds, budget)


def _wrap_grid(prev: Any) -> Any:
    def allocate_grid(self: Any, positions: wp.array, particle_flags: wp.array, voxel_size: float, temporary_store: Any,
                      padding_voxels: int = 0, **kw: Any) -> Any:
        held = getattr(self, "_alloy_dense_box", None) is not None   # one world: only CPU mode A holds a box
        if self.grid_type != "dense" or (self._environment_count <= 1 and not held):
            return prev(self, positions, particle_flags, voxel_size, temporary_store, padding_voxels, **kw)
        lo, hi = _dense_bounds(self, positions, particle_flags, voxel_size, padding_voxels)
        return fem.Grid3D(bounds_lo=wp.vec3(lo * voxel_size), bounds_hi=wp.vec3(hi * voxel_size),
                          res=wp.vec3i((hi - lo).astype(int)), env_count=self._environment_count)

    allocate_grid._alloy_world_freeze = True        # type: ignore[attr-defined]
    return allocate_grid


def start_box(solver: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """The integer voxel box (no margin) a dense-grid MPM solver's first allocation takes over its particles' start
    positions (``_dense_bounds``'s rule); None on a sparse grid. CPU mode A (``alloy.parallel.cpu_worlds``) takes it
    from every worker and holds the union in all (``hold_box``), the box one process of all the worlds holds."""
    if getattr(solver, "grid_type", None) != "dense":
        return None
    points, flags, _, _ = solver._grid_points(solver.model.particle_q)
    p = points.numpy()
    if points.device.is_cuda:
        p = p[(flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0]
    return dense_box(p, solver._mpm_model.voxel_size, solver.grid_padding)


def hold_box(solver: Any, lo: np.ndarray, hi: np.ndarray, worlds: int, particles: int) -> None:
    """Hold ``solver``'s dense grid at the box one process of ``worlds`` worlds and ``particles`` particles holds:
    their start box (``lo``, ``hi``, union of every worker's ``start_box``) plus the same margin. Also for a worker
    of one world, where Newton's own grid would follow the particles."""
    _hold(solver, np.asarray(lo), np.asarray(hi), int(worlds),   # dtype kept (float32, as dense_box gives it)
          dense_cell_budget(int(particles)))


_PREV: dict[str, Any] = {}           # Newton's functions as found by install() (uninstall() puts them back)
ENABLED = True                       # tests: False keeps install() from wrapping (Newton's own loop and grid)


def install() -> None:
    """Wraps Newton's rheology loop, the solver method that calls it and the grid allocation (idempotent; chains onto
    what is in place)."""
    if not ENABLED or _PREV:                       # off, or already in place (even under someone else's wrapper)
        return
    _PREV.update(loop=SR._run_solver_loop, solve=SIM.SolverImplicitMPM._solve_rheology,
                 grid=SIM.SolverImplicitMPM._allocate_grid)
    SR._run_solver_loop = _wrap_loop(SR._run_solver_loop)
    SIM.SolverImplicitMPM._solve_rheology = _wrap_solve(SIM.SolverImplicitMPM._solve_rheology)
    SIM.SolverImplicitMPM._allocate_grid = _wrap_grid(SIM.SolverImplicitMPM._allocate_grid)
    log.info("implicit MPM: each world's rheology solve stops at its own convergence (mpm_worlds)")


def uninstall() -> None:
    """Puts back what install() found (tests: Newton's own loop for comparison)."""
    if _PREV:
        SR._run_solver_loop = _PREV["loop"]
        SIM.SolverImplicitMPM._solve_rheology = _PREV["solve"]
        SIM.SolverImplicitMPM._allocate_grid = _PREV["grid"]
        _PREV.clear()


__all__ = ["hold_box", "install", "start_box", "uninstall"]
