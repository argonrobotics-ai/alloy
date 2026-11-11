# SPDX-License-Identifier: Apache-2.0
"""``apply_precision``: the kwargs a precision config implies for pipelines and solvers (SPEC §3.7, D9, D10).

Returned mapping (consumed by WP2's assembler/Simulation and WP3's contact layer):

- ``"bulk_pipeline"``: ``{"deterministic": bool}`` for the Newton bulk ``CollisionPipeline`` (sorted contacts).
- ``"precision_pipeline"``: ``{"deterministic": True, "gjk_tol_m", "mpr_tol_m"}`` (always ordered, F3 tolerances).
- ``"solvers"``: entry name -> ``{"deterministic": wp.DeterministicMode}`` for entries whose capability is NATIVE
  or WARP_LOWERING. An explicit ``deterministic`` in ``EntrySpec.solver_kwargs`` is never overridden. SolverMuJoCo
  gets Warp's lowering only when explicitly requested there (research §3.3: ~1,100x cost, OOM at 1024 worlds).
  Under a RUN_TO_RUN preset it gets Alloy's ordered route on the built solver instead (P3-1 round 5,
  ``alloy.precision.mjw_route``, enabled by ``enable_solver_determinism`` / the HealthMonitor) when its path is
  covered; an uncovered path is reported as a downgrade.
- ``"use_graph"``: False when an entry runs Warp's deterministic lowering (its scratch does not fit a captured
  10-substep graph, research §3.3).
- ``"native_contacts"``: the D10 path; ``"reasons"``: why something was not set.

Also enforces F1: raises ``PrecisionError`` when worlds are not at the origin.
"""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from alloy.core import registry
from alloy.core.errors import RegistryError
from alloy.core.protocols import DetCapability
from alloy.precision.frames import check_worlds_at_origin
from alloy.precision.types import Determinism, PrecisionConfig

log = logging.getLogger("alloy.precision")

MUJOCO_NAMES = frozenset({"mujoco", "SolverMuJoCo"})


def _warp_mode(det: Determinism) -> wp.DeterministicMode:
    return wp.DeterministicMode.RUN_TO_RUN if det != Determinism.NONE else wp.DeterministicMode.NOT_GUARANTEED


def _entry_capability(entry: Any) -> DetCapability | None:
    try:
        return registry.solver(entry.capability).determinism
    except RegistryError:
        return None


def _entry_kwargs(entry: Any, cfg: PrecisionConfig, world_count: int, reasons: list[str]) -> tuple[dict, bool]:
    """(kwargs for this entry, whether it runs Warp's deterministic lowering)."""
    if "deterministic" in entry.solver_kwargs:
        mode = entry.solver_kwargs["deterministic"]
        lowering = _entry_capability(entry) == DetCapability.WARP_LOWERING and mode != wp.DeterministicMode.NOT_GUARANTEED
        return {}, lowering
    cap = _entry_capability(entry)
    if cap is None:
        reasons.append(f"{entry.name}: capability {entry.capability!r} not registered; determinism left at default")
        return {}, False
    want = Determinism(min(cfg.determinism, Determinism.RUN_TO_RUN))
    if cap == DetCapability.NONE or want == Determinism.NONE:
        return {}, False
    if entry.capability in MUJOCO_NAMES:
        reasons.append(f"{entry.name}: SolverMuJoCo Warp lowering only on explicit request "
                       f"(solver_kwargs['deterministic']); built NOT_GUARANTEED, then Alloy's ordered route is enabled "
                       f"on the instance when its path is covered (alloy.precision.mjw_route)")
        return {"deterministic": wp.DeterministicMode.NOT_GUARANTEED}, False
    return {"deterministic": _warp_mode(want)}, cap == DetCapability.WARP_LOWERING


def apply_precision(built: Any, cfg: PrecisionConfig) -> dict[str, Any]:
    """Pipeline and solver kwargs for ``built`` (a ``BuiltScene``) under ``cfg``; raises if F1 is violated."""
    check_worlds_at_origin(built.model, cfg.frames)
    reasons: list[str] = []
    solvers: dict[str, dict] = {}
    lowering = False
    for entry in built.entries:
        kwargs, low = _entry_kwargs(entry, cfg, int(built.world_count), reasons)
        solvers[entry.name] = kwargs
        lowering |= low
    if lowering:
        reasons.append("Warp deterministic lowering in use: CUDA graph capture off")
    out = {
        "bulk_pipeline": {"deterministic": cfg.determinism != Determinism.NONE},
        "precision_pipeline": {"deterministic": True, "gjk_tol_m": cfg.pair_gjk_tol_m,
                               "mpr_tol_m": cfg.pair_mpr_tol_m},
        "solvers": solvers,
        "use_graph": not lowering,
        "native_contacts": cfg.native_contacts,
        "reasons": reasons,
    }
    log.info("apply_precision: %d entries, graph %s; %s", len(solvers), out["use_graph"], "; ".join(reasons) or "-")
    return out
