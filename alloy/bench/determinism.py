# SPDX-License-Identifier: Apache-2.0
"""Determinism suite D1-D3 (bench §4.5, SPEC §6.5, D9), on the A3 contact scene (drop, impact, sliding friction).

- D1 rerun: two fresh builds, same seed; every recorded series and the final body state must match bitwise.
  Metric `max_ulp`: largest float32 ULP distance between the runs (0 required on CPU, 1,000 steps in full mode).
- D2 batch invariance: world 0 alone vs world 0 among 15 neighbours with other conditions; world 0's trajectory
  must not change (GPUSimBench Eq. 8). Metric `batch_dev_m`: max |x_alone - x_batch| over time [m] (0 on CPU).
- D3 rerun spread: K fresh runs; assignment W1 between the final positions of each run and run 0, max over runs
  (GPUSimBench Eq. 11). Metric `rerun_w1_m` [m] (0 on CPU; on GPU outside RUN_TO_RUN it is a noise floor).
Population per D9: on CPU every solver is expected to be bitwise; the tolerance is 0.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alloy.bench import compare as C
from alloy.bench.analytic.energy import ContactEnergyTwin
from alloy.bench.law import register_law
from alloy.bench.runner import RunContext, SteppedRun, _World, step_twin
from alloy.bench.solvers import RIGID_SOLVERS
from alloy.bench.types import Condition, LawSpec, MetricVerdict, Tolerance

log = logging.getLogger("alloy.bench")
_SRC = "bench §4.5 D1-D3; SPEC §6.5: 0 ulp / 0 mm on CPU"
RERUNS_QUICK, RERUNS_FULL = 3, 10


def _spec(law_id: str, metric: str) -> LawSpec:
    return LawSpec(id=law_id, suite="determinism", metrics={metric: Tolerance(abs=0.0, source=_SRC)},
                   stat={metric: "distribution" if metric == "rerun_w1_m" else "mean"}, dt=1.25e-3, substeps=1,
                   duration_s=1.25, worlds_per_condition=1, solvers=RIGID_SOLVERS)


LAW_D1 = _spec("D1.rerun", "max_ulp")
LAW_D2 = _spec("D2.batch", "batch_dev_m")
LAW_D3 = _spec("D3.spread", "rerun_w1_m")


def ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Max ULP distance between two float32 arrays (NaN payloads compared bitwise; shape mismatch -> huge)."""
    x = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    y = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    if x.shape != y.shape:
        return 2**31
    # map sign-magnitude to a monotonic integer line so -0.0 and +0.0 are 0 apart
    x = np.where(x < 0, -(x & 0x7FFFFFFF), x)
    y = np.where(y < 0, -(y & 0x7FFFFFFF), y)
    return int(np.max(np.abs(x - y))) if x.size else 0


def _positions(run: SteppedRun) -> np.ndarray:
    """[W, T, 3] recorded positions of each world's box."""
    return np.stack([run.log.series(a) for a in "xyz"], -1)


class _DetTwin:
    """Direct law: runs the inner scene several times and compares the runs."""

    def __init__(self) -> None:
        self.monitors: dict[str, Any] = {}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return ContactEnergyTwin().conditions(quick)


class _Recording(ContactEnergyTwin):
    """The A3 contact scene, also recording box positions for the comparisons."""

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: Any) -> None:
        super().record(step, state, contacts, obs, log)
        self.record_first_body(step, state, log)


def _run_recording(spec: LawSpec, ctx: RunContext, conds: list[Condition]) -> SteppedRun:
    return step_twin(spec, _Recording(), [_World(c, {}) for c in conds], ctx)


class RerunTwin(_DetTwin):
    law_id = LAW_D1.id

    def evaluate(self, ctx: RunContext) -> tuple[MetricVerdict, ...]:
        conds = list(self.conditions(ctx.quick))
        a, b = (_run_recording(LAW_D1, ctx, conds) for _ in range(2))
        ulps = [ulp_distance(a.log.series(n), b.log.series(n)) for n in ("x", "y", "z")]
        for name in ("body_q", "body_qd"):
            ulps.append(ulp_distance(getattr(a.state, name).numpy(), getattr(b.state, name).numpy()))
        self.monitors = {"ulp_per_channel": ulps, "steps": a.monitors["steps"]}
        return (C.compare_errors("max_ulp", "ulp", np.array([float(max(ulps))]), LAW_D1.metrics["max_ulp"]),)


class BatchTwin(_DetTwin):
    law_id = LAW_D2.id
    neighbours = 15

    def evaluate(self, ctx: RunContext) -> tuple[MetricVerdict, ...]:
        conds = list(self.conditions(ctx.quick))
        target = conds[-1]
        others = [conds[i % (len(conds) - 1)] for i in range(self.neighbours)]
        alone = _positions(_run_recording(LAW_D2, ctx, [target]))[0]
        batch = _positions(_run_recording(LAW_D2, ctx, [target] + others))[0]
        dev = float(np.max(np.abs(alone.astype(np.float64) - batch))) if alone.shape == batch.shape else np.inf
        self.monitors = {"target": target.id, "neighbours": [c.id for c in others]}
        return (C.compare_errors("batch_dev_m", "m", np.array([dev]), LAW_D2.metrics["batch_dev_m"]),)


class SpreadTwin(_DetTwin):
    law_id = LAW_D3.id

    def evaluate(self, ctx: RunContext) -> tuple[MetricVerdict, ...]:
        conds = list(self.conditions(ctx.quick))
        k = RERUNS_QUICK if ctx.quick else RERUNS_FULL
        finals = [_positions(_run_recording(LAW_D3, ctx, conds))[:, -1, :].astype(np.float64) for _ in range(k)]
        w1 = [C.assignment_w1(finals[0], f)[0] for f in finals[1:]]
        self.monitors = {"reruns": k, "w1_m": w1}
        return (C.compare_errors("rerun_w1_m", "m", np.array([max(w1)]), LAW_D3.metrics["rerun_w1_m"],
                                 stat="distribution"),)


LAWS = ((LAW_D1, RerunTwin), (LAW_D2, BatchTwin), (LAW_D3, SpreadTwin))
for _spec_, _factory in LAWS:
    register_law(_spec_, _factory)

__all__ = ["LAW_D1", "LAW_D2", "LAW_D3", "RerunTwin", "BatchTwin", "SpreadTwin", "ulp_distance"]
