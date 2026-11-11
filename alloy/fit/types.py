# SPDX-License-Identifier: Apache-2.0
"""Material-fitting types (SPEC §3.12; the fitter and write-back are WP11)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from alloy.bench.types import LawSpec, MetricVerdict, RigDataset, SimTwin
from alloy.core.quantity import Q


@dataclass(frozen=True)
class FitParam:
    leaf: str                                 # "material:pp.surface.mu_static" | "pair:pp__steel.mu_dynamic"
    lo: float
    hi: float                                 # SI
    scale: Literal["linear", "log"] = "linear"
    prior: tuple[float, float] | None = None  # (mean, sigma) in scale space


@dataclass(frozen=True, eq=False)
class FitProblem:
    law: LawSpec
    twin: SimTwin
    data: RigDataset
    params: tuple[FitParam, ...]
    objective: Literal["law_error", "nrmse", "w1"] = "law_error"   # on calib trials only
    solver: str | None = None                 # WP11: registered solver to fit on; None = law.solvers[0]
    conditions: tuple[str, ...] = ()          # WP11: dataset conditions sharing these leaves; () = all


@dataclass(frozen=True)
class FitConfig:
    method: Literal["cma", "grad", "hybrid"] = "cma"
    worlds: int = 1024                        # candidates x conditions per batch
    generations: int = 30
    restarts: int = 1
    seed: int = 0
    wall_budget_s: float = 3600.0             # FIT-2
    grad_steps: int = 50                      # Newton differentiable contacts where supported
    quick: bool = False                       # WP11: run each candidate for the law's quick.duration_s (CPU)
    per_condition: int | None = None          # WP11: worlds per (candidate, condition); None = law's value
    substeps: int | None = None               # WP11: solver substeps per law dt; None = law's (else fit_dt_only)
    tol_x: float = 1e-4                       # WP11: stop a CMA-ES run when every unit-box axis step is below this


@dataclass(frozen=True)
class FitResult:
    params: Mapping[str, Q]                   # source="fit", unc from the final population spread, note = fit id
    calib_error: Mapping[str, float]
    valid_error: Mapping[str, float]
    heldout_gap: Mapping[str, float]
    verdicts: tuple[MetricVerdict, ...]       # validation laws on valid trials
    wall_s: float
    evaluations: int
    fit_id: str
    report: Mapping[str, Any] = field(default_factory=dict)   # WP11: history, stop reasons, method, provenance
