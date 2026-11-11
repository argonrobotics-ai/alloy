# SPDX-License-Identifier: Apache-2.0
"""Bitwise-repeat protocol R1-R8 (SPEC §3.7 ``repeat_test``, §6.5; research precision §4.6).

``make_sim(world_count)`` builds a ``SimulationLike``; ``steps`` counts control periods (``step_period`` is the only
stepping call a ``SimulationLike`` exposes). Every period the world digests (state + ``sim.state_arrays()`` extras)
are recorded, so a mismatch reports its first divergent period.

- R1 same process: snapshot (with extras) -> run -> restore -> run again.
- R2 fresh process: the same run in a spawned interpreter (``make_sim`` must be importable: a module-level function).
- R3 eager vs CUDA graph: needs ``make_sim(n, use_graph=...)``; CPU and sims without the keyword are ``not_run``.
- R4 batch invariance: world digests at every N in ``world_counts`` against world 0 of the smallest N (identical
  worlds); world-order permutation needs global world ids in ``make_sim`` and is reported ``not_run``.
- R5 save -> load into a fresh Simulation -> continue, against the uninterrupted run.
- R6 another GPU of the same model: ``not_run`` (one GPU per VM).
- R8 negative control: a 1-ulp change of one world's state must change that world's digest and no other.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.core.protocols import SimulationLike
from alloy.core.snapshot import WorldSnapshot
from alloy.precision.config import gpu_tier, preset
from alloy.precision.determinism import effective_determinism, native_share
from alloy.precision.digest import world_digests
from alloy.precision.types import Determinism, PrecisionConfig, RepeatResult

log = logging.getLogger("alloy.precision")

MakeSim = Callable[[int], SimulationLike]
CASES = ("R1", "R2", "R3", "R4", "R5", "R6", "R8")


@dataclass(frozen=True)
class SimParts:
    model: Any
    extra: Mapping[str, wp.array]
    world_maps: Mapping[str, wp.array]
    solvers: tuple[Any, ...]
    config: PrecisionConfig

    def snapshot(self, state) -> WorldSnapshot:
        return WorldSnapshot(self.model, state, self.extra, extra_world=self.world_maps)

    def digests(self, state, whole_extras: bool = True) -> np.ndarray:
        return world_digests(self.model, state, self.extra, extra_world=self.world_maps, whole_extras=whole_extras)


def sim_parts(sim: SimulationLike) -> SimParts:
    """Model, extra state arrays (+ item->world maps), solvers and precision config of a ``SimulationLike``."""
    model = getattr(sim, "model", None) or getattr(getattr(sim, "built", None), "model", None)
    if model is None:
        raise TypeError(f"{type(sim).__name__} exposes no model (SimulationLike.model)")
    extra = dict(sim.state_arrays()) if hasattr(sim, "state_arrays") else {}
    maps = dict(sim.state_world_maps()) if hasattr(sim, "state_world_maps") else {}
    solvers = tuple(getattr(sim, "solvers", ()) or ())
    stepper = getattr(sim, "stepper", None)
    if not solvers and stepper is not None:
        inner = getattr(stepper, "solver", None)
        solvers = (inner,) if inner is not None else (stepper,)
        if not extra and hasattr(stepper, "state_arrays"):
            extra = dict(stepper.state_arrays())
    cfg = getattr(sim, "precision", None) or preset("standard", gpu_tier(model.device))
    return SimParts(model, extra, maps, solvers, cfg)


def run_digests(sim: SimulationLike, periods: int, whole_extras: bool = True) -> list[np.ndarray]:
    """World digests after each of ``periods`` control periods (index 0 = before the first)."""
    parts = sim_parts(sim)
    out = [parts.digests(sim.state, whole_extras)]
    for _ in range(periods):
        sim.step_period()
        out.append(parts.digests(sim.state, whole_extras))
    return out


def first_divergence(a: Sequence[np.ndarray], b: Sequence[np.ndarray], world: int | None = None) -> int | None:
    """First period index where the two digest series differ (all worlds, or one), None if equal."""
    for k, (x, y) in enumerate(zip(a, b)):
        xs, ys = (x, y) if world is None else (x[world], y[world])
        if not np.array_equal(xs, ys):
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def _snapshot(sim: SimulationLike) -> WorldSnapshot:
    snap = sim_parts(sim).snapshot(sim.state)
    snap.save(sim.state)
    return snap


def _case_r1(make_sim: MakeSim, n: int, periods: int) -> tuple[bool, int | None, str]:
    sim = make_sim(n)
    snap, ticks = _snapshot(sim), sim.clock.ticks
    a = run_digests(sim, periods)
    snap.restore(sim.state)
    sim.clock.ticks = ticks
    b = run_digests(sim, periods)
    k = first_divergence(a, b)
    return k is None, k, ""


def _spawned_run(make_sim: MakeSim, n: int, periods: int, queue) -> None:  # pragma: no cover - child process
    queue.put([d.tolist() for d in run_digests(make_sim(n), periods)])


def _case_r2(make_sim: MakeSim, n: int, periods: int) -> tuple[bool, int | None, str]:
    here = run_digests(make_sim(n), periods)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_spawned_run, args=(make_sim, n, periods, queue))
    proc.start()
    there = [np.asarray(d, dtype=np.uint64) for d in queue.get(timeout=3600)]
    proc.join(timeout=60)
    k = first_divergence(here, there)
    return k is None, k, f"fresh process exit {proc.exitcode}"


def _case_r3(make_sim: MakeSim, n: int, periods: int) -> tuple[bool | None, int | None, str]:
    probe = make_sim(n)
    if not wp.get_device(sim_parts(probe).model.device).is_cuda:
        return None, None, "not_run: CUDA graphs need a GPU"
    try:
        eager, graph = make_sim(n, use_graph=False), make_sim(n, use_graph=True)  # type: ignore[call-arg]
    except TypeError:
        return None, None, "not_run: make_sim takes no use_graph keyword"
    k = first_divergence(run_digests(eager, periods), run_digests(graph, periods))
    return k is None, k, ""


def _case_r5(make_sim: MakeSim, n: int, periods: int) -> tuple[bool, int | None, str]:
    split = max(periods // 2, 1)
    ref = run_digests(make_sim(n), periods)
    sim_a = make_sim(n)
    run_digests(sim_a, split)
    host = _snapshot(sim_a).to_host()  # what a save file holds
    sim_b = make_sim(n)
    load = sim_parts(sim_b).snapshot(sim_b.state)
    load.from_host(host)
    load.restore(sim_b.state)
    sim_b.clock.ticks = sim_a.clock.ticks
    cont = run_digests(sim_b, periods - split)
    k = first_divergence(ref[split:], cont)
    return k is None, None if k is None else k + split, f"saved after {split} periods"


def _case_r8(make_sim: MakeSim, n: int, periods: int) -> tuple[bool, int | None, str]:
    sim = make_sim(n)
    parts = sim_parts(sim)
    base = parts.digests(sim.state)
    arr = sim.state.body_q if sim.state.body_q is not None and sim.state.body_q.size else sim.state.particle_q
    host = arr.numpy()
    flat = host.reshape(host.shape[0], -1)
    flat[0, 0] = np.nextafter(flat[0, 0], np.float32(np.inf))
    arr.assign(host)
    nudged = parts.digests(sim.state)
    moved = np.flatnonzero((nudged != base).any(axis=1)).tolist()
    world0 = int(parts.model.body_world.numpy()[0]) if arr is sim.state.body_q else 0
    return moved == [world0], None, f"1-ulp nudge moved worlds {moved}"


def _case_r4(make_sim: MakeSim, counts: Sequence[int], periods: int
             ) -> tuple[dict[int, bool], dict[int, int | None], str]:
    ref_n = min(counts)
    ref = [d[0] for d in run_digests(make_sim(ref_n), periods, whole_extras=False)]
    equal, first = {}, {}
    for n in counts:
        series = run_digests(make_sim(n), periods, whole_extras=False)
        k = next((i for i, d in enumerate(series) if not (d == ref[i]).all()), None)
        equal[n], first[n] = k is None, k
    return equal, first, ("not_run part: permuted world order needs global world ids in make_sim; extras without a "
                          "per-world axis are left out (they depend on the other worlds by construction)")


def repeat_test(make_sim: MakeSim, case: str, steps: int = 2000, world_counts: Sequence[int] = (1, 64, 1024, 4096)
                ) -> RepeatResult:
    """Run one repeat case at every world count; ``steps`` = control periods."""
    if case not in CASES:
        raise ValueError(f"unknown repeat case {case!r}; expected one of {CASES}")
    counts = tuple(int(n) for n in world_counts)
    probe = make_sim(min(counts))
    parts = sim_parts(probe)
    mode, reasons = effective_determinism(parts.config, parts.solvers, max(counts))
    share = native_share(parts.solvers)
    notes: dict[str, str] = {"periods": str(steps)}
    if case == "R4":
        equal, first, note = _case_r4(make_sim, counts, steps)
        notes["R4"] = note
    elif case == "R6":
        equal, first = {}, {}
        notes["R6"] = "not_run: needs a second GPU of the same model"
    else:
        equal, first = {}, {}
        fn = {"R1": _case_r1, "R2": _case_r2, "R3": _case_r3, "R5": _case_r5, "R8": _case_r8}[case]
        for n in counts:
            ok, k, note = fn(make_sim, n, steps)
            if ok is None:
                notes[f"{case}@{n}"] = note
                continue
            equal[n], first[n] = bool(ok), k
            if note:
                notes[f"{case}@{n}"] = note
    log.info("repeat %s at %s: equal %s (det %s, native share %.2f)", case, counts, equal, mode.name, share)
    return RepeatResult(case, counts, equal, first, share, Determinism(mode), tuple(reasons), notes)
