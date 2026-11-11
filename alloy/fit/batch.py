# SPDX-License-Identifier: Apache-2.0
"""Batched candidate evaluation: one candidate x condition per world group, one Newton model per chunk (SPEC §3.12).

Candidate k on condition c runs as the *virtual condition* ``"<c>~<k>"`` with c's parameters, after
``twin.bind_values("<c>~<k>", {field: value})``. The twin's own per-condition expansion (e.g. the R3 tilt grid) and
its per-condition scalars therefore work unchanged, and candidates never share a world. Worlds are laid out
condition-major per candidate inside a chunk; a chunk holds at most ``cfg.worlds`` candidate-conditions and is
built and stepped by WP6's `plan_worlds` / `step_twin` on a fresh deep copy of the pristine twin. Perturbation
samples (`LawSpec.perturb`) are common random numbers: every candidate sees the same draws per condition.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from alloy.bench.runner import RunContext, plan_worlds, step_twin
from alloy.bench.types import Condition
from alloy.fit.errors import FitError
from alloy.fit.objective import Score, SimOutcome, score
from alloy.fit.problem import FitSetup

log = logging.getLogger("alloy.fit")
SEP = "~"


@dataclass(frozen=True)
class BatchResult:
    x: np.ndarray                       # [P, n] SI candidates as evaluated
    objective: np.ndarray               # [P]
    scores: tuple[Score, ...]
    worlds: int                         # simulated worlds over all chunks
    wall_s: float


def virtual_id(cond_id: str, k: int) -> str:
    return f"{cond_id}{SEP}{k}"


def split_virtual(vid: str) -> tuple[str, int]:
    base, _, k = vid.rpartition(SEP)
    return base, int(k)


def bind_candidates(twin: object, conds: Sequence[Condition], fields: Sequence[str], x: np.ndarray,
                    first: int) -> list[list[Condition]]:
    """Per candidate row (numbered from `first`), its virtual conditions, each bound to its values on `twin`."""
    out = []
    for i, row in enumerate(np.atleast_2d(x)):
        values = {f: float(v) for f, v in zip(fields, row)}
        vconds = [Condition(virtual_id(c.id, first + i), dict(c.params)) for c in conds]
        for vc in vconds:
            twin.bind_values(vc.id, values)  # type: ignore[attr-defined]
        out.append(vconds)
    return out


def _group(cond_ids: Sequence[str], values: Mapping[str, np.ndarray], series: Mapping[str, np.ndarray],
           times: np.ndarray) -> dict[int, dict[str, SimOutcome]]:
    """Per candidate, per base condition: the outcome of that candidate's worlds."""
    ids = np.asarray(cond_ids)
    out: dict[int, dict[str, SimOutcome]] = {}
    for vid in dict.fromkeys(cond_ids):
        base, k = split_virtual(vid)
        mask = ids == vid
        sc = {m: np.asarray(v, np.float64)[mask] for m, v in values.items()}
        se = {m: np.asarray(s)[mask] for m, s in series.items()}
        out.setdefault(k, {})[base] = SimOutcome(sc, times, se or None)
    return out


class TwinBatch:
    """Evaluates candidate rows on the fit twin; `evaluate` is the search's only contact with simulation."""

    def __init__(self, setup: FitSetup, device: str, worlds: int, seed: int = 0) -> None:
        if worlds <= 0:
            raise FitError(f"FitConfig.worlds must be > 0, got {worlds}")
        self.setup = setup
        self.device = device
        self.seed = seed
        self.chunk = max(1, worlds // len(setup.targets.conditions))
        self.curve_metrics = tuple(sorted({t.metric for t in setup.targets.targets if t.stat == "curve"}))
        self.evaluations = 0
        self.worlds_run = 0

    def _run_chunk(self, x: np.ndarray, first: int) -> tuple[dict[int, dict[str, SimOutcome]], int]:
        s = self.setup
        twin = copy.deepcopy(s.twin)
        ctx = RunContext(s.spec, self.device, self.seed, s.solver, False)
        expand = getattr(twin, "expand", None)
        worlds = []
        for vconds in bind_candidates(twin, s.targets.conditions, s.fields, x, first):
            # planned per candidate: condition i draws the same perturbation samples for every candidate (common
            # random numbers) and the same ones run_law draws in the final validation
            worlds += plan_worlds(s.spec, vconds, ctx, expand)
        try:
            run = step_twin(s.spec, twin, worlds, ctx)
            values = twin.scalars(run.log)
        except Exception as exc:  # noqa: BLE001 - re-raised with the fit context
            raise FitError(f"{s.spec.id} on {s.solver}/{self.device}: twin run failed for candidates "
                           f"{first}..{first + len(x) - 1}: {type(exc).__name__}: {exc}") from exc
        series = {m: run.log.series(m) for m in self.curve_metrics}
        return _group([w.condition.id for w in worlds], values, series, run.log.times()), len(worlds)

    def evaluate(self, x: np.ndarray) -> BatchResult:
        """Objective per candidate row of `x` ([P, n], SI)."""
        t0 = time.perf_counter()
        x = np.atleast_2d(np.asarray(x, np.float64))
        prior = self.setup.space.prior_penalty(x)
        outcomes: dict[int, dict[str, SimOutcome]] = {}
        n_worlds = 0
        for first in range(0, len(x), self.chunk):
            got, w = self._run_chunk(x[first:first + self.chunk], first)
            outcomes.update(got)
            n_worlds += w
        targets = self.setup.targets.targets
        scores = tuple(score(targets, outcomes.get(k, {}), self.setup.objective, float(prior[k]))
                       for k in range(len(x)))
        self.evaluations += len(x)
        self.worlds_run += n_worlds
        wall = time.perf_counter() - t0
        log.debug("batch of %d candidates, %d worlds, %.2f s", len(x), n_worlds, wall)
        return BatchResult(x, np.array([sc.objective for sc in scores]), scores, n_worlds, wall)


__all__ = ["TwinBatch", "BatchResult", "bind_candidates", "virtual_id", "split_virtual"]
