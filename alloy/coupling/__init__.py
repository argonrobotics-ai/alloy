# SPDX-License-Identifier: Apache-2.0
"""Coupling and clocks (WP4): Scheduler(ADMM), grid clocks, watchdog, sleeping, estimators and state adapters.

Types live in ``alloy.coupling.types`` (WP0). Entry points:
- ``alloy.coupling.scheduler.Scheduler`` / ``SchedulerADMM``: ``StepperLike`` couplers on Newton's
  ``SolverCoupledProxy`` / ``SolverCoupledADMM`` (one exchange per ``step``; ``choose_clock`` once per period;
  ``rollback_requested``/``on_rollback`` for the Simulation loop, SCR-WP4-1);
- ``alloy.coupling.scheduler.strong_pair`` (D5); ``alloy.coupling.clocks`` (D4 grid clock, rung hysteresis);
- ``alloy.coupling.sleep.mujoco_sleep_kwargs`` (SolverMuJoCo sleeping kwargs, off for ``native_contacts="newton"``);
- ``register_builtins()``: stability estimators and state adapters for Newton's solvers, into the registry in use
  (idempotent per registry; run when ``alloy.coupling.scheduler`` is imported and when a Scheduler is built, so a
  Scheduler built inside ``isolated_registry(empty=True)`` gets them too).

This ``__init__`` stays import-light: importing ``alloy.coupling.types`` must not import solver modules.
"""

from __future__ import annotations

_REGISTERED_IN: object | None = None      # the registry object the builtins were last checked into


def register_builtins() -> None:
    """Register WP4's estimators and state adapters for Newton solver classes into the current registry unless it
    already has them. Idempotent per registry: ``isolated_registry`` swaps WP0's registry object, and an empty one
    gets the builtins on the next call; a copy keeps the ones it was copied with (and any test overrides)."""
    global _REGISTERED_IN
    from alloy.core import registry

    current = registry._REG                    # WP0's live registry (replaced by isolated_registry)
    if current is _REGISTERED_IN:
        return
    import alloy._vendor.newton.solvers as ns

    if ns.SolverMuJoCo not in current.estimators:
        from alloy.coupling.adapters import register_builtin_adapters
        from alloy.coupling.estimators import register_builtin_estimators

        register_builtin_estimators()
        register_builtin_adapters()
    _REGISTERED_IN = current
