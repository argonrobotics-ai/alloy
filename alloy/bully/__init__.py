# SPDX-License-Identifier: Apache-2.0
"""Bully agent (WP7, SPEC §3.11): find legal inputs that make a simulation do something impossible.

- B0 ``stress.stress_phases``: VAL-115's four legal phases plus drops and pushes, in action space.
- B1 ``search.search``: batched CMA-ES over action knots, one candidate per world, objective = badness
  ``max_m v_m/t_m + 0.1·Σ_rest``; a candidate becomes a ``Finding`` only after replay (10/10 CPU, >= 9/10 GPU).
- B2 ``brief.audit_prompt``: the brief for an auditing agent (never edits product code).
- ``monitors.BenchMonitors``: bench-only checks over ``StepHealth`` (passive-subsystem energy ledger, non-target
  displacement, teleport, impossible motion, joint separation, contact-buffer penetration); ``severity.classify``
  applies the §4.3 table. ``corpus.Corpus``: ``bench/bully/corpus/BUL-nnnn/`` with replay and regressions.
- Targets: ``targets.register_target`` / ``get_target``; ``direct.DirectSim`` runs raw-Newton targets
  (``xpbd_box_stack``); SceneCard targets use ``alloy.sim.Simulation``.

- ``audit(scene, episodes, seed, device)``: N random-legal episodes on any scene card, as findings (S1/S2 class,
  cause, first tick, body, replay handle); ``replay_finding(handle)`` reruns one alone. See ``scene_audit``.

CLI: ``python -m alloy.bully {run,b0,replay,try,sb7,targets,audit}`` (``--help`` per command). Types:
``alloy.bully.types``.
"""

from __future__ import annotations

from typing import Any

_LAZY = {  # imported on first use: ``import alloy.bully.types`` stays light and free of import cycles
    "audit": "alloy.bully.scene_audit", "replay_finding": "alloy.bully.scene_audit",
    "AuditReport": "alloy.bully.scene_audit", "AuditFinding": "alloy.bully.scene_audit",
    "ReplayHandle": "alloy.bully.scene_audit", "AuditOptions": "alloy.bully.scene_target",
}
__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module 'alloy.bully' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value
