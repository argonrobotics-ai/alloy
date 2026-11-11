# SPDX-License-Identifier: Apache-2.0
"""Graph-budget rungs (speed-now S2, WP10 2026-11-09; opt-in: ``ClockPolicy.graph_budget_rungs``).

``IntegratedSimulation`` sends a graph key to eager periods when one exchange is over its node budget
(``alloy.integration.graph_budget.GRAPH_NODE_BUDGET``, 2^16; WP12). With the contact gate a released pair lets the
exchange coarsen to 1.25 ms, and an entry then takes many substeps per exchange: a towel falling beside water
(robot_cloth_liquid, 64 worlds, L4) took 16 VBD substeps, 74,344 nodes per exchange, so its periods ran eagerly at 31 s
against 7 s captured. The physics did not need that exchange: 0.625 ms with 8 substeps steps every entry at the same h.

With the flag on, the Simulation's report of an over-budget key caps the entry with the most substeps one rung lower
(``ClockController.cap_rung``); the clock then holds that entry within its cap by taking a finer dt_x (a bound like the
interface bound), and the next key is captured. Exchanges get finer, never coarser, and every entry keeps its own step.

Rows (towel + water x64, L4, periods 3-18): base 17.8 s per period, contact gate 15.0, contact gate + this 11.1 (from
period 14: 13.0 s instead of 30.2 s eager); 0 INVALID.

Settled drape with the contact gate and this on (a Panda at rest with a draped towel, 8 nudged worlds, 2 s, one L4
run per row): it FAILS the rate rule (below). The drape does not repeat on the GPU, so one run decides nothing:
- seeds 0-7, 100-107: the towel stays on the arm in 8 of 8, at rest;
- seeds 200-207: 7 of 8, 7 still moving at 2 s;
- seeds 300-307: 8 of 8 at rest in one run, 0 of 8 with check() False in a repeat of the same run (a contact-band
  trip took VBD to 4.9 us steps, where its Chebyshev rho 0.9 over-relaxes; every towel was thrown).
This lever stayed inert in every one of those runs (no cap, no eager key; the armed exchange is 5.7k nodes).
A verdict is a pass rate over >= 3 runs per seed set against the defaults' on the same seeds (coordinator
2026-11-09 22:3xZ). Measured 2026-11-10 (9 runs of 8 worlds per arm, seeds 100 / 200 / 300, tree main 2c76e7b6e
before the Newton 1.8 bump; towels off the arm / at rest, of 72): defaults 3 / 64 (one run lost 3 towels: they slid
off with no band trip and no flag); contact gate + this 2 / 48 (in 3 runs every towel was still moving at 2 s). The
rule asks for no more off the arm than the defaults (met) and an at-rest rate of 0.816 or more (0.667: failed).
Failures come by run, so 9 runs per arm separate the arms weakly. The sentence above about 4.9 us steps is one
run's reading: keeping the rung's own step through the trip (``ClockPolicy.rungs_follow_dt_x``) lost more towels,
not fewer (11 of 72). Max strain 1.72 % against the defaults' 0.71 % on seeds 0-7 is within that spread too (the
defaults reach 1.81 % on seeds 100-107). With it on, dt_x depends on graph node counts, so CPU, graph=False and GPU
runs take different clocks ([internal note] scr/WP10.md, speed-now S2 and "S2 drape repeats").
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

log = logging.getLogger("alloy.coupling")

__all__ = ["GraphBudgetRungsMixin"]


class GraphBudgetRungsMixin:
    """For the Scheduler: ``graph_over_budget``, called by ``IntegratedSimulation`` when a key runs eagerly for its
    node count."""

    _alloy_policy: Any
    _clock: Any

    def graph_over_budget(self, substeps: Mapping[str, int], nodes: int) -> str | None:
        """One exchange at ``substeps`` is ``nodes`` graph nodes, over the budget: with the flag on, cap the entry with
        the most substeps one rung lower (``ClockController.cap_rung``). Returns that entry, or None."""
        if not self._alloy_policy.clock.graph_budget_rungs:
            return None
        log.debug("coupling: one exchange at substeps %s is %d graph nodes", dict(substeps), nodes)
        return self._clock.cap_rung(substeps)
