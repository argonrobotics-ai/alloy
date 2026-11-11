# SPDX-License-Identifier: Apache-2.0
"""Bully episodes on the product loop: ``SceneSim`` wraps the ``IntegratedSimulation`` that
``alloy.integration.wiring.build_simulation`` builds for a scene card (every route, couplings, precision pairs,
topology and WP5 health included, as ``alloy.check`` runs it) and adds what ``EpisodeRunner`` needs on top:

- ``pre_substep`` hooks for force actions (``body_f``, ``particle_f``): run on the period's start state, then after
  every step on the step's output state, which is the next step's input. A force command is constant within a
  control tick, so this is the same force before every step as ``DirectSim`` applies. Kinematic ``body_pose``
  actions are not supported here (they count substeps); use ``DirectSim`` targets for those.
- MuJoCo-native contacts exported for the bench monitors when the product loop has no contact buffer (one
  ``SolverMuJoCo`` entry running its own collision), as ``DirectSim(export_contacts=True)`` does.
- Push work booked with health: WP5's energy check (on for passive scenes) is told the work of the force actions
  every step (``HealthMonitor.add_external_work``), as the bench energy ledger books it; without it a legal push
  reads as energy created (ENERGY_JUMP).
- Per-episode health: ``reset`` also restores the health records (and the MuJoCo Warp guard counters) of the reset
  worlds, which the product reset leaves running, so an episode never inherits an earlier episode's flags.

Graphs stay off (``graph="off"``): the bully's observers run eagerly, and a culprit probe reads the host.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bully import kernels as K
from alloy.bully.direct import health_arrays, restore_worlds
from alloy.core.protocols import StepObserver

log = logging.getLogger("alloy.bully")


class SceneSim:
    """``SimulationLike`` over a product ``IntegratedSimulation`` (module doc)."""

    def __init__(self, sim: Any, *, book_work: bool = True) -> None:
        self.sim = sim
        self.built = sim.built
        self.model: newton.Model = sim.model
        self.clock = sim.clock
        self.control_dt = float(sim.clock.control_dt)
        solvers = _entry_solvers(sim)
        self.entry_solvers = solvers
        self.solver = solvers[0] if len(solvers) == 1 else None
        self.ordered = False  # WP5's ordered MuJoCo Warp route on (scene_target.ordered_route)
        self.observers: list[StepObserver] = []
        self.pre_substep: list[Callable[[newton.State], None]] = []
        self.contacts, self._export = _contact_export(sim, self.solver)
        self.periods = 0
        health = getattr(sim, "health", None)
        self._ledger = WorkLedger(self.model) if book_work and hasattr(health, "add_external_work") else None
        self._health_initial = {k: wp.clone(a) for k, a in health_arrays(sim.health, self.solver).items()}
        sim.add_step_observer(self._after_step)
        log.info("SceneSim: %d worlds, solvers %s, contacts %s", self.model.world_count,
                 [type(s).__name__ for s in solvers], "exported" if self._export else "product")

    @property
    def state(self) -> newton.State:
        return self.sim.state

    @property
    def control(self) -> newton.Control:
        return self.sim.control

    def add_step_observer(self, fn: StepObserver) -> None:
        self.observers.append(fn)

    def _after_step(self, s0: newton.State, s1: newton.State, control: newton.Control, contacts: Any,
                    dt: float) -> None:
        if self._export is not None:
            self._export(self.contacts, s1)
            contacts = self.contacts
        for fn in self.observers:
            fn(s0, s1, control, contacts, dt)
        if self.pre_substep and self._ledger is not None:  # before the product's health.update of this step
            self.sim.health.add_external_work(self._ledger.work(s0, s1, dt))
        for hook in self.pre_substep:  # s1 is the next step's input state
            hook(s1)

    def step_period(self) -> None:
        for hook in self.pre_substep:
            hook(self.sim.state)
        self.sim.step_period()
        self.periods += 1

    def reset(self, world_mask: wp.array | None = None) -> None:
        """The product reset, plus the health records and guard counters of the reset worlds (module doc)."""
        self.sim.reset(world_mask)
        mask = None if world_mask is None else np.asarray(world_mask.numpy()).astype(bool)
        live = health_arrays(self.sim.health, self.solver)
        for name, init in self._health_initial.items():
            restore_worlds(live[name], init, mask)
        if mask is None:
            self.periods = 0

    def report(self) -> Mapping[str, Any]:
        return self.sim.report()

    def state_arrays(self) -> Mapping[str, wp.array]:
        return self.sim.state_arrays()

    def state_world_maps(self) -> Mapping[str, wp.array]:
        return self.sim.state_world_maps()


class WorkLedger:
    """Per-world work [J] of the applied body wrenches over one step (``kernels.body_force_work_kernel``)."""

    def __init__(self, model: newton.Model) -> None:
        start, index = K.world_csr(model.body_world.numpy(), model.world_count)
        dev = model.device
        self.start = wp.array(np.asarray(start, np.int32), dtype=wp.int32, device=dev)
        self.index = wp.array(np.asarray(index, np.int32), dtype=wp.int32, device=dev)
        self.out = wp.zeros(model.world_count, dtype=wp.float64, device=dev)
        self.worlds, self.bodies = model.world_count, model.body_count

    def work(self, s0: newton.State, s1: newton.State, dt: float) -> wp.array:
        if self.bodies and s0.body_f is not None:
            wp.launch(K.body_force_work_kernel, dim=self.worlds, device=self.out.device,
                      inputs=[s0.body_f, s0.body_qd, s1.body_qd, float(dt), self.start, self.index],
                      outputs=[self.out])
        return self.out


def _entry_solvers(sim: Any) -> list[Any]:
    """The live solver of every entry (``wire``'s ``Parts.solvers``; else WP12's ``entry_solvers`` on the stepper)."""
    parts = getattr(sim, "parts", None)
    if parts is not None and getattr(parts, "solvers", None):
        return list(parts.solvers.values())
    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415

    return list(entry_solvers(sim.stepper).values())


def _contact_export(sim: Any, solver: Any) -> tuple[Any, Callable | None]:
    """(contacts the monitors read, exporter or None): the product's buffer when it has one, else MuJoCo's own
    contacts copied out after every step."""
    if sim.contacts is not None or solver is None:
        return sim.contacts, None
    naconmax = int(getattr(getattr(solver, "mjw_data", None), "naconmax", 0))
    if naconmax <= 0 or not hasattr(solver, "update_contacts"):
        return None, None
    return newton.Contacts(naconmax, 0, device=sim.model.device), solver.update_contacts
