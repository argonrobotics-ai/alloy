# SPDX-License-Identifier: Apache-2.0
"""Integration (WP12): ``Simulation`` with every real part injected, graph mode, first scoreboard (SPEC §7 WP12).

Use::

    from alloy.integration import build_simulation
    sim = build_simulation(built, "standard")            # or "precision", "fast", "debug"; profile fields as kwargs
    for _ in range(periods):
        sim.step_period()
    sim.report()                                          # health, coupling, gap, topology, graph stats

Modules:

- ``profiles``: ``IntegrationProfile`` (precision preset + graph mode + which parts to build), named ``PROFILES``.
- ``stepper``: BuiltScene -> stepper (WP2 single entry, WP4 ``Scheduler`` / ``SchedulerADMM``), ``entry_solvers``.
- ``wiring``: ``wire(built, profile) -> Parts`` (WP3 contact layer + MuJoCo append + gap monitor, WP5 health,
  WP10 topology) and ``build_simulation``.
- ``simulation``: ``IntegratedSimulation(alloy.sim.Simulation)``: graph-safe periods through WP8's ``PeriodGraph``
  (PAR-14 counter), separate latch and rollback snapshots, per-world rollback that leaves other worlds bitwise
  unchanged, topology events re-baselining the health energy reference.

Design note: ``[internal note]``; results: ``docs/notes/wp12-results.md``.
"""

from __future__ import annotations

from alloy.integration.profiles import PROFILES, IntegrationProfile, profile
from alloy.integration.simulation import IntegratedSimulation
from alloy.integration.wiring import Parts, build_simulation, wire

__all__ = ["PROFILES", "IntegratedSimulation", "IntegrationProfile", "Parts", "build_simulation", "profile", "wire"]
