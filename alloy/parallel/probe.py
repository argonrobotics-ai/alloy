# SPDX-License-Identifier: Apache-2.0
"""Probe: measure T(N), iterations, capacities and memory of a pack, and fit a WorldProfile (parallel §4.1, §4.9).

Protocol (PRF-05/PRF-08): build the pack, warm up, capture the period graph (CUDA, or a CPU whose Warp records CPU
graphs, ``graphs.cpu_graphs_supported``; graph-safe targets only), then time ``windows`` windows of
about ``window_s`` wall seconds each (at least ``min_window_sim_s`` simulated seconds, for gates that need settling;
the step count per window is sized from one timed warm-up step). Each window runs in ``chunks`` chunks with a device
sync at the end of each;
counters are read between chunks, outside the timed region. The step time is the median over windows (min and
spread are kept). Memory is the live bytes after warm-up and capture minus before build: mempool used plus the
CUDA graph memory pool, where MJWarp's in-step scratch lives under a graph (``counters.live_bytes``; CUDA only, CPU
reports None). ``probe_targets`` sweeps N ascending and fits ``t_floor, c_w`` (``fit_cost_model``) and
``bytes_fixed, bytes_per_world`` (linear LSQ). Timing numbers count only from an exclusive run on an idle GPU
(SPEC §6.3); CPU numbers are for plumbing tests.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from alloy.core.ids import digest_of
from alloy.core.newton_compat import NEWTON_PIN, installed_newton_commit
from alloy.parallel.costmodel import fit_cost_model
from alloy.parallel.counters import (Counters, ProbeSample, TraceReplayer, live_bytes, mujoco_counters,
                                     trim_graph_pool)
from alloy.parallel.graphs import PeriodGraph, graphs_by_default
from alloy.parallel.types import Trace, WorldProfile

if TYPE_CHECKING:
    from alloy.cards.types import SceneCard
    from alloy.core.protocols import SceneAssemblerLike, SimulationLike
    from alloy.scene.types import BuiltScene

log = logging.getLogger("alloy.parallel")


@dataclass
class StepTarget:
    """What the probe times: each ``step()`` advances every world by ``sim_dt`` simulated seconds."""

    step: Callable[[], None]
    nworld: int
    sim_dt: float
    device: str
    counters: Counters | None = None
    graph_safe: bool = True  # False: host control inside step(); graphs off
    buffers: Callable[[], Sequence[wp.array]] | None = None  # identity check across a captured step
    model: object | None = None  # newton.Model, for structure and particle counts
    close: Callable[[], None] | None = None


@dataclass(frozen=True)
class TimingOptions:
    window_s: float = 2.0  # wall seconds per window (PRF-08: >= 5 windows of 2 s)
    min_window_sim_s: float = 0.0  # floor on simulated seconds per window
    windows: int = 5
    chunks: int = 10  # syncs (and counter reads) per window
    warmup_steps: int = 4
    use_graph: bool | None = None  # None: graphs when the target is graph-safe, on CUDA and on a CPU-graph Warp
    budget_s: float = 120.0  # wall-clock cap per N; windows past it are dropped (recorded)


@dataclass
class ProbeRun:
    n: int
    sim_dt: float  # simulated seconds per step() call
    step_s: float  # median over windows [s per step() call]
    step_s_min: float
    step_s_spread: float  # (max - min) / median over windows
    steps: int
    windows: int
    graph: bool
    mem_bytes: int | None
    samples: list[ProbeSample] = field(default_factory=list)


def time_target(target: StepTarget, opts: TimingOptions, mem_before: int | None = None) -> ProbeRun:
    """Warm up, capture, time and sample one pack."""
    use_graph = _graph_on(target, opts)
    graph = PeriodGraph(target.step, target.device, use_graph=use_graph, buffers=target.buffers)
    for _ in range(opts.warmup_steps):
        target.step()
    if use_graph:
        graph.launch()  # capture + first replay, untimed
    wp.synchronize_device(target.device)
    mem = None if mem_before is None else _mem_current(target.device) - mem_before
    per_window = _steps_per_window(graph, target, opts)
    times, steps, samples = _timed_windows(graph, target, opts, per_window)
    med = float(np.median(times))
    return ProbeRun(n=target.nworld, sim_dt=target.sim_dt, step_s=med, step_s_min=float(np.min(times)),
                    step_s_spread=float((np.max(times) - np.min(times)) / med), steps=steps, windows=len(times),
                    graph=graph.mode == "graph", mem_bytes=mem, samples=samples)


def _steps_per_window(graph: PeriodGraph, target: StepTarget, opts: TimingOptions) -> int:
    t0 = time.perf_counter()
    graph.launch()
    wp.synchronize_device(target.device)
    est = max(time.perf_counter() - t0, 1e-6)
    by_wall = round(opts.window_s / est)
    by_sim = math.ceil(opts.min_window_sim_s / target.sim_dt - 1e-9) if opts.min_window_sim_s > 0 else 0
    return max(opts.chunks, by_wall, by_sim)


def _timed_windows(graph: PeriodGraph, target: StepTarget, opts: TimingOptions,
                   per_window: int) -> tuple[list[float], int, list[ProbeSample]]:
    chunk = max(1, per_window // opts.chunks)
    times, samples, steps, t_start = [], [], 0, time.perf_counter()
    for _ in range(opts.windows):
        elapsed, done = 0.0, 0
        while done < per_window:
            n = min(chunk, per_window - done)
            t0 = time.perf_counter()
            for _ in range(n):
                graph.launch()
            wp.synchronize_device(target.device)
            elapsed += time.perf_counter() - t0
            done += n
            if target.counters is not None:
                samples.append(target.counters())
        times.append(elapsed / done)
        steps += done
        if time.perf_counter() - t_start > opts.budget_s:
            log.warning("probe n=%d hit the %.0f s budget after %d windows", target.nworld, opts.budget_s, len(times))
            break
    return times, steps, samples


def _graph_on(target: StepTarget, opts: TimingOptions) -> bool:
    """Graphs only for a graph-safe target: on CPU a host read inside the captured step would not fail as it does on
    CUDA but read stale data (PeriodGraph notes the common ones), so the CPU default rides on the same flag."""
    if not target.graph_safe:
        return False
    if opts.use_graph is not None:
        return opts.use_graph
    return graphs_by_default(target.device)


def _mem_current(device: str) -> int | None:
    return live_bytes(device)


def run_ladder_point(make_target: Callable[[int], StepTarget], n: int, device: str, opts: TimingOptions) -> ProbeRun:
    """Build the pack of ``n`` worlds on ``device``, time it, release it (memory = live bytes it added)."""
    gc.collect()
    trim_graph_pool(device)  # earlier packs' graph memory would otherwise stay counted as used
    mem0 = _mem_current(device)
    target = make_target(n)
    try:
        return time_target(target, opts, mem0)
    finally:
        if target.close is not None:
            target.close()
        del target
        gc.collect()


@dataclass(frozen=True)
class ProfileMeta:
    scene_hash: str
    solver_config_hash: str
    newton_commit: str
    gpu_model: str


def stat_p(samples: Sequence[ProbeSample], name: str, q: float) -> float:
    vals = [getattr(s, name) for s in samples if getattr(s, name) is not None]
    if not vals:
        return float("nan")
    return float(np.percentile(np.concatenate([np.ravel(v) for v in vals]), q))


def fit_memory(points: Sequence[tuple[int, int | None]]) -> tuple[int, int]:
    """(bytes_fixed, bytes_per_world) from (N, bytes) points, bytes = fixed + N * per_world; (0, 0) if unmeasured."""
    pts = [(n, m) for n, m in points if m is not None]
    if len({n for n, _ in pts}) < 2:
        return 0, 0
    n, m = np.array(pts, dtype=np.float64).T
    per_world, fixed = np.polyfit(n, m, 1)
    return int(max(0.0, fixed)), int(max(0.0, per_world))


def profile_from_runs(runs: Sequence[ProbeRun], meta: ProfileMeta, sim_dt: float,
                      structure: tuple[int, int, int, int] = (0, 0, 0, 0)) -> WorldProfile:
    """WorldProfile from an N sweep; ``structure`` = (nv, n_particles, island_nv_max, chain_len_max) per world."""
    t_floor, c_w = fit_cost_model([r.n for r in runs], [r.step_s for r in runs])
    fixed, per_world = fit_memory([(r.n, r.mem_bytes) for r in runs])
    samples = [s for r in runs for s in r.samples]
    nv, n_particles, island_nv, chain = structure
    return WorldProfile(
        scene_hash=meta.scene_hash, solver_config_hash=meta.solver_config_hash, newton_commit=meta.newton_commit,
        gpu_model=meta.gpu_model, dt=sim_dt, nv=nv, n_particles=n_particles, island_nv_max=island_nv,
        chain_len_max=chain, rows_p99=_int_p(samples, "rows", 99), contacts_p99=_int_p(samples, "contacts", 99),
        pairs_peak=_int_p(samples, "pairs", 100), t_floor_s=t_floor, c_w_s=c_w,
        iters_p50=stat_p(samples, "iters", 50), iters_p99=stat_p(samples, "iters", 99),
        bytes_fixed=fixed, bytes_per_world=per_world)


def _int_p(samples: Sequence[ProbeSample], name: str, q: float) -> int:
    v = stat_p(samples, name, q)
    return 0 if np.isnan(v) else int(np.ceil(v))


def probe_targets(make_target: Callable[[int], StepTarget], n_values: Sequence[int], device: str,
                  meta: ProfileMeta, opts: TimingOptions = TimingOptions(),
                  structure: tuple[int, int, int, int] = (0, 0, 0, 0)) -> tuple[WorldProfile, list[ProbeRun]]:
    """Sweep N ascending (so the mempool high-water mark follows the largest pack) and fit the profile."""
    runs = []
    for n in sorted(set(n_values)):
        run = run_ladder_point(make_target, n, device, opts)
        runs.append(run)
        log.info("probe %s n=%d: %.4g ms/step (min %.4g, spread %.1f%%), graph=%s, mem=%s B", meta.scene_hash[:12],
                 n, run.step_s * 1e3, run.step_s_min * 1e3, 100 * run.step_s_spread, run.graph, run.mem_bytes)
    return profile_from_runs(runs, meta, runs[0].sim_dt, structure), runs


# ---- SceneCard path ---------------------------------------------------------------------------------------------
_CAPACITY_KEYS = frozenset({"njmax", "nconmax", "njmax_nnz", "rigid_contact_max", "shape_pairs_max"})


def probe_scene(scene: SceneCard, replay: Trace | None, device: str, n_values: Sequence[int], *,
                assembler: SceneAssemblerLike, sim_factory: Callable[[BuiltScene], SimulationLike],
                opts: TimingOptions = TimingOptions(), cache: ProfileCache | None = None) -> WorldProfile:
    """Profile a SceneCard pack on ``device``: assemble + Simulation per N, one ``step_period`` per timed call.

    ``Simulation.step_period`` decides its own graph use (it calls PeriodGraph), so the probe does not capture.
    """
    first: dict[str, Any] = {}

    def make(n: int) -> StepTarget:
        built = assembler.assemble(scene, device, world_count=n)
        sim = sim_factory(built)
        first.setdefault("built", built)
        replayer = _replayer(replay, sim, n, device)

        def step() -> None:
            if replayer is not None:
                replayer.apply()
            sim.step_period()

        return StepTarget(step=step, nworld=n, sim_dt=built.control_dt, device=device, graph_safe=False,
                          counters=sim_counters(sim, built.model), model=built.model)

    meta_key = digest_of(scene)
    if cache is not None and (hit := cache.get_by_scene(meta_key, device)) is not None:
        return hit
    runs = [run_ladder_point(make, n, device, opts) for n in sorted(set(n_values))]
    built = first["built"]
    meta = ProfileMeta(scene_hash=meta_key, solver_config_hash=solver_config_hash(built),
                       newton_commit=installed_newton_commit() or NEWTON_PIN, gpu_model=wp.get_device(device).name)
    prof = profile_from_runs(runs, meta, runs[0].sim_dt, structure_of(built.model))
    if cache is not None:
        cache.put(prof)
    return prof


def solver_config_hash(built: BuiltScene) -> str:
    """Digest of each entry's capability and solver options, capacities excluded (they scale with N)."""
    rows = [(e.capability, e.substeps, e.max_substeps, {k: v for k, v in dict(e.solver_kwargs).items()
                                                        if k not in _CAPACITY_KEYS}) for e in built.entries]
    return digest_of(rows)


def structure_of(model: Any) -> tuple[int, int, int, int]:
    """(nv, n_particles, island_nv_max, chain_len_max) per world, from ``classify_structure``."""
    from alloy.parallel.structure import classify_structure

    plan = classify_structure(model)
    worlds = max(1, int(model.world_count))
    nv = sum(i.nv for i in plan.islands)
    return (nv, int(model.particle_count) // worlds, max((i.nv for i in plan.islands), default=0),
            max((i.chain_len for i in plan.islands), default=0))


def sim_counters(sim: Any, model: Any) -> Counters | None:
    """Counters for a Simulation whose stepper wraps one SolverMuJoCo (else None: not exposed)."""
    solver = getattr(getattr(sim, "stepper", None), "solver", None)
    if solver is None or getattr(solver, "mjw_data", None) is None:
        return None
    return mujoco_counters(solver, model, getattr(sim, "contacts", None))


def _replayer(replay: Trace | None, sim: Any, n: int, device: str) -> TraceReplayer | None:
    if replay is None:
        return None
    targets = {name: getattr(sim.control, name) for name in replay.inputs}
    return TraceReplayer(dict(replay.inputs), targets, n, device)


class ProfileCache:
    """JSON file of WorldProfiles keyed by (scene_hash, solver_config_hash, newton_commit, gpu_model)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._rows: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(p: WorldProfile) -> str:
        return "|".join((p.scene_hash, p.solver_config_hash, p.newton_commit, p.gpu_model))

    def put(self, p: WorldProfile) -> None:
        self._rows[self.key(p)] = asdict(p)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._rows, indent=1, sort_keys=True))

    def get(self, scene_hash: str, solver_config_hash: str, newton_commit: str, gpu_model: str) -> WorldProfile | None:
        row = self._rows.get("|".join((scene_hash, solver_config_hash, newton_commit, gpu_model)))
        return None if row is None else WorldProfile(**row)

    def get_by_scene(self, scene_hash: str, device: str) -> WorldProfile | None:
        """The cached profile for this scene on this device model and Newton commit, if exactly one exists."""
        gpu, commit = wp.get_device(device).name, installed_newton_commit() or NEWTON_PIN
        hits = [r for r in self._rows.values()
                if r["scene_hash"] == scene_hash and r["gpu_model"] == gpu and r["newton_commit"] == commit]
        return WorldProfile(**hits[0]) if len(hits) == 1 else None
