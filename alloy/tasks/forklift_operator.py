# SPDX-License-Identifier: Apache-2.0
"""Scripted forklift operator: one generic phase-machine engine over the five GA task phase lists.

``Operator(plan, truck).step(obs) -> Controls`` once per control period; ``phase`` names the active phase and
``done`` is True after the last. Interface types live in ``forklift_control``, paths in ``forklift_path``, phase
types in ``forklift_phases`` and the task phase lists in ``forklift_plans``. Host side only: no physics.

Park is commanded only at rest (below 0.02 m/s). A phase that follows a drive starts only after the drive's stop,
so direction changes happen at rest.
"""

from __future__ import annotations

import logging

from alloy.tasks.forklift_control import Controls, Observation, TruckSpec
from alloy.tasks.forklift_phases import DriveSegment, Run
from alloy.tasks.forklift_plans import Phase, Plan, phases_for

__all__ = ["Controls", "Observation", "Operator", "Plan", "TruckSpec"]

log = logging.getLogger("alloy.tasks.forklift")


class Operator:
    """Sequences a task's phases; stateful, one instance per episode."""

    def __init__(self, plan: Plan, truck: TruckSpec | None = None) -> None:
        self.plan, self.truck = plan, truck or TruckSpec()
        self._phases: list[Phase] = phases_for(plan, self.truck)
        self._i = 0
        self._run: Run | None = None
        self._last = Controls()
        self.history: list[tuple[str, Observation]] = []     # (phase, observation when it finished)

    @property
    def phase_list(self) -> list[str]:
        return [p.name for p in self._phases]

    @property
    def done(self) -> bool:
        return self._i >= len(self._phases)

    @property
    def phase(self) -> str:
        return "done" if self.done else self._phases[self._i].name

    @property
    def speed_cap(self) -> float | None:
        """Speed cap (m/s) of the active drive phase, None in a stationary one."""
        if self.done or not isinstance(self._phases[self._i], DriveSegment):
            return None
        return self._phases[self._i].cap

    def step(self, obs: Observation) -> Controls:
        if self.done:
            return self._last
        phase = self._phases[self._i]
        if self._run is None:
            self._run = phase.start(obs)
            log.info("forklift: phase %s at t=%.2f s", phase.name, obs.t)
        self._last, finished = phase.act(self._run, obs, self.truck)
        if finished:
            self.history.append((phase.name, obs))
            self._i += 1
            self._run = None
        return self._last
