# SPDX-License-Identifier: Apache-2.0
"""Bench API types: Law, Rig, SimTwin, Score (SPEC §3.10; comparators, run_law, reports and scoreboard are WP6)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, Control, State
from alloy._vendor.newton.solvers import SolverBase

REPORT_SCHEMA: Final[str] = "alloy.bench.report/1"
RUN_SCHEMA: Final[str] = "alloy.bench.run/1"


@dataclass(frozen=True)
class Tolerance:
    abs: float = 0.0
    rel: float = 0.0
    source: str = ""                        # required (plan | rig noise | paper | derived)

    def bound(self, ref: float) -> float:
        return self.abs + self.rel * abs(ref)


@dataclass(frozen=True)
class Quantity:
    name: str
    unit: str
    mean: float
    std: float
    n: int
    u95: float


@dataclass(frozen=True)
class Condition:
    id: str
    params: Mapping[str, float]


@dataclass(frozen=True, eq=False)
class RigTrial:
    condition_id: str
    index: int
    split: Literal["calib", "valid"]
    series: Mapping[str, np.ndarray]        # SI; "t" first
    scalars: Mapping[str, float]


@dataclass(frozen=True)
class RigCard:                              # bench/rigs/<law>/<id>/rig_card.json
    law_id: str
    dataset_id: str
    protocol_version: str
    objects: Mapping[str, Mapping[str, float]]          # measured dims/masses (SI) with *_u95
    instruments: Mapping[str, Mapping[str, Any]]        # range, resolution, calibration date, u95
    holder: Mapping[str, float]                         # k_x [N/m], k_theta [N·m/rad], stage profile params
    environment: Mapping[str, float]                    # temperature, humidity


@dataclass(frozen=True, eq=False)
class RigDataset:
    card: RigCard
    conditions: tuple[Condition, ...]
    trials: tuple[RigTrial, ...]
    sha256: str

    def split(self, which: Literal["calib", "valid"]) -> tuple[RigTrial, ...]:
        """Trials of one split, in stored order. Fitting reads ``calib`` only (§3.12)."""
        if which not in ("calib", "valid"):
            raise ValueError(f"split must be 'calib' or 'valid', got {which!r}")
        return tuple(t for t in self.trials if t.split == which)


@runtime_checkable
class TwinLog(Protocol):
    """Device-side recording surface (no per-step host sync, §3.10; SCR-WP6-2). WP6: ``alloy.bench.twin.DeviceTwinLog``."""

    n_worlds: int
    dt: float

    def column(self, step: int) -> int:
        """Time column of ``step`` in the [W, T] buffers, -1 = not sampled."""
        ...

    def times(self) -> np.ndarray:
        """[T] sample times [s]."""
        ...

    def channel(self, name: str, dtype: Any = wp.float32) -> wp.array:
        """[W, T] device buffer for ``name``, allocated on first use."""
        ...

    def accumulator(self, name: str, dtype: Any = wp.float64) -> wp.array:
        """[W] device running reduction for ``name``, allocated on first use."""
        ...

    def series(self, name: str) -> np.ndarray:
        """[W, T] host view after the run."""
        ...


@runtime_checkable
class SimTwin(Protocol):
    """One law's simulated twin (§3.10, SCR-WP6-1).

    ``run_law`` also calls these optional hooks when a twin defines them (not protocol members):
    ``make_pipeline(model, solver) -> CollisionPipeline | None`` (contact source; default Newton's pipeline unless the
    solver collides internally), ``init_state(model, state)``, ``world_gravity(cond) -> vec3 | None`` (for
    ``begin_world(gravity=)``), ``prepare_builder(builder, solver)``, ``units: Mapping[str, str]``,
    ``references(log)`` (per-world reference values for rel tolerances), ``settings_record(solver)`` (solver kwargs
    for the report), ``evaluate(ctx)`` (laws that are not one stepped run: A16 self-test, D1-D3).
    """

    law_id: str

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        """Conditions to run when there is no RigDataset (analytic and determinism laws)."""
        ...

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None: ...

    def make_solver(self, model: newton.Model, solver: str) -> SolverBase:
        """Build the solver for capability ``solver`` (chosen by ``run_law`` from ``LawSpec.solvers``)."""
        ...

    def drive(self, step: int, t: float, state: State, control: Control) -> None:
        """Device-array driven (no host sync per step)."""
        ...

    def record(self, step: int, state: State, contacts: Contacts | None,
               obs: SolverBase.Observables | None, log: TwinLog) -> None: ...

    def scalars(self, log: TwinLog) -> dict[str, np.ndarray]:
        """Per world, SI."""
        ...


Stat = Literal["mean", "curve", "distribution", "rate"]
Verdict = Literal["pass", "fail", "inconclusive", "not_run", "error"]
VERDICTS: Final[tuple[str, ...]] = ("pass", "fail", "inconclusive", "not_run", "error")


def _quick_default() -> dict[str, float]:
    return {"worlds": 16, "duration_s": 0.5}


@dataclass(frozen=True)
class LawSpec:
    id: str                                   # "A5.coulomb", "R3.incline_friction", "LIQ-4.hydrostatic"
    suite: Literal["analytic", "determinism", "rig"]
    metrics: Mapping[str, Tolerance]
    stat: Mapping[str, Stat]
    dt: float                                 # free (not bound to the D4 grid) unless run via Simulation
    substeps: int
    duration_s: float
    perturb: Mapping[str, float] = field(default_factory=dict)     # 1-sigma per parameter
    worlds_per_condition: int = 16
    min_tur: float = 3.0
    solvers: tuple[str, ...] = ()             # registered capabilities this law runs on; () = all applicable
    production_dt: bool = True                # False -> results reported fit_dt_only
    producer: str = "WP6"                     # package that registers it; not merged -> not_run
    quick: Mapping[str, float] = field(default_factory=_quick_default)
    sleeping: bool = False                    # creep/drift laws must keep False (§3.5)


@dataclass(frozen=True)
class MetricVerdict:
    name: str
    unit: str
    real: Quantity | None
    sim: Quantity
    error: float
    bound: float
    tur: float | None
    stat: Stat
    verdict: Verdict


@dataclass(frozen=True)
class LawResult:
    law: str
    solver: str
    device: str
    verdicts: tuple[MetricVerdict, ...]
    monitors: Mapping[str, Any]
    provenance: Mapping[str, str]
    wall_s: float
    fit_dt_only: bool
    command: str = ""                         # CLI that produced it (bench §4 report/1)
    config: Mapping[str, Any] = field(default_factory=dict)   # solver, kwargs sha, dt, substeps, worlds
    data: Mapping[str, str] = field(default_factory=dict)     # dataset path, sha256 (rig laws)

    @property
    def verdict(self) -> Verdict:
        """Worst metric verdict: error > fail > inconclusive > not_run > pass (no metrics -> not_run)."""
        order = ("error", "fail", "inconclusive", "not_run", "pass")
        present = {v.verdict for v in self.verdicts}
        return next((v for v in order if v in present), "not_run")  # type: ignore[return-value]


@dataclass(frozen=True)
class ScoreRow:                               # one line of bench/SCOREBOARD.json
    metric_id: str                            # §5 id, e.g. "SB-3", "CT-1"
    value: float | None
    unit: str
    target: str
    verdict: Verdict
    source_rows: tuple[str, ...]              # report file paths
    command: str
    commit: str
    date: str
    status: Literal["active", "deferred"] = "active"   # deferred rows never enter the release scoreboard (§5)
    reason: str = ""                          # why a row is not_run/error/deferred (WP6, defaulted field per D17)
