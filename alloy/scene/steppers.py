# SPDX-License-Identifier: Apache-2.0
"""StepperLike implementations Simulation builds when none is injected (SPEC §2.3, D18), and the fallback contact
path used when no AlloyContactLayer is injected.

- `SingleEntryStepper`: one dynamic entry; `step` = one solver.step at dt_x; clock = the entry's D4 target.
- `StockProxyStepper`: two entries through Newton's stock `SolverCoupledProxy` (lagged, 2 iterations, ω = 1,
  coupling-time §3.2), rigid entry as the proxy source (solver_coupled_proxy.py:245-326 allows ≤ 2 entries).
- `NewtonContacts`: a plain `CollisionPipeline` buffer with the three per-contact arrays attached (D13), so the pair
  table can write friction scale / stiffness / damping; Newton zeroes them in `clear()` and the pipeline writes
  only hydroelastic values (collide.py:177-180), and `PairTable.apply` overwrites every slot after collide. An
  optional per-world `ContactBudget` (triangle pairs, rigid and soft contacts) bounds its buffers through WP4's
  sized pipeline, and an always-on device counter records lost contacts for the health monitor (SCR-WP4-6).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, Control, State
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.errors import CouplingError
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.core.registry import solver as capability
from alloy.core.timebase import snap_to_grid
from alloy.coupling.types import CouplingReport
from alloy.scene.types import BuiltScene, EntrySpec

log = logging.getLogger("alloy.sim")
PROXY_ITERATIONS = 2


def solver_kwargs(entry: EntrySpec, native: str) -> dict[str, Any]:
    kw = {**capability(entry.capability).default_kwargs, **dict(entry.solver_kwargs)}
    if entry.capability == "mujoco":
        kw["use_mujoco_contacts"] = native == "mujoco"
    return kw


def needs_newton_contacts(entries: Sequence[EntrySpec], native: str) -> bool:
    """True unless the only dynamic entry is SolverMuJoCo running its own contacts."""
    return not (len(entries) == 1 and entries[0].capability == "mujoco" and native == "mujoco")


def fallback_needs_rigid(entries: Sequence[EntrySpec], native: str, shape_body: np.ndarray) -> bool:
    """Whether the fallback buffer must collide rigid shapes (SCR-WP4-7). False only when one SolverMuJoCo entry runs
    its own contacts and owns the body of every shape on a body (static shapes are MJWarp's world geoms): MJWarp then
    collides every rigid shape and the health monitor reads its penetration from MJWarp's contacts. Any other case,
    including a shape on a body no entry owns, keeps rigid contacts (fails closed)."""
    mujoco = [e for e in entries if e.capability == "mujoco"]
    if native != "mujoco" or len(mujoco) != 1:
        return True
    on_body = np.unique(np.asarray(shape_body)[np.asarray(shape_body) >= 0])
    return not np.isin(on_body, np.asarray(list(mujoco[0].bodies), dtype=np.int64)).all()


@dataclass(frozen=True)
class ContactBudget:
    """Per-world capacities of the fallback contact buffer (SCR-WP4-6); None keeps Newton's automatic size. Any field
    set also builds soft candidates only for shapes with COLLIDE_PARTICLES (WP4's filtered builder): Newton's list is
    every particle x every shape per world."""

    triangle_pairs: int | None = None
    rigid_contacts: int | None = None
    soft_contacts: int | None = None

    @property
    def is_set(self) -> bool:
        return any(v is not None for v in (self.triangle_pairs, self.rigid_contacts, self.soft_contacts))


class NewtonContacts:
    """Fallback `ContactLayerLike` without the WP3 sidecar: CollisionPipeline + per-contact arrays.

    ``budget`` (SCR-WP4-6): per-world caps for triangle pairs, rigid and soft contacts. Without one, Newton's automatic
    sizes stand (12.6 GB of triangle and reducer buffers, 7.3 GB of rigid slots and every particle x shape at 1,024 DC5
    worlds). ``overflow`` is an int32[1] device count of contacts lost (rigid, soft or triangle pairs over capacity,
    the reducer's loss flag, a shape excluded from the candidates whose COLLIDE_PARTICLES is on now), sticky until
    ``reset_overflow`` and always on. As a HealthMonitor capacity flag it turns every world INVALID (SC0-1; the buffer
    is shared by all worlds, so a loss is not attributed to one). ``rigid=False`` (SCR-WP4-7, see
    ``fallback_needs_rigid``): soft contacts only, with no rigid pairs, so no broad/narrow phase or reducer buffers.
    ``particles`` (SCR-JOB2-4): the only particles given soft candidates (None: all). The wiring leaves out
    contact-free (SPH) particles: their solver ignores contacts and has its own wall and leak checks, so their soft
    contacts would only feed the health monitor's penetration test (a PEN_FAIL latch froze JOB2's robot).
    """

    def __init__(self, model: newton.Model, *, budget: ContactBudget | None = None, rigid: bool = True,
                 particles: Sequence[int] | None = None) -> None:
        from alloy.coupling.proxy_contacts import ProxyContactPipeline  # noqa: PLC0415 - WP4's sized pipeline

        self.budget, self.rigid = budget or ContactBudget(), bool(rigid)
        b = self.budget
        self._factory = ProxyContactPipeline(b.soft_contacts, b.rigid_contacts, rigid_shapes=None if rigid else (),
                                             triangle_pairs_per_world=b.triangle_pairs,
                                             filter_candidates=b.is_set or not rigid or particles is not None,
                                             label="fallback", particles=particles,
                                             pass_through=False,   # its particles include SPH (own leak monitor)
                                             publish_capacity=rigid)   # soft-only: no model-wide rigid capacity
        self.pipeline = self._factory(model)
        self._contacts = self.pipeline.contacts()
        n = self._contacts.rigid_contact_max
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            if getattr(self._contacts, name, None) is None:
                setattr(self._contacts, name, wp.zeros(n, dtype=wp.float32, device=model.device))
        self.overflow: wp.array = self._factory.overflow

    def contacts(self) -> Contacts:
        return self._contacts

    def sidecar(self) -> None:
        raise CouplingError("the fallback contact path has no AlloyContactSidecar (inject WP3's layer)")

    def collide(self, state: State, contacts: Contacts, *, dt: float) -> None:
        self.pipeline.collide(state, contacts)
        self._factory.update_overflow()

    def reset_overflow(self) -> None:
        self._factory.reset_overflow()


def solver_state(name: str, solver: SolverBase) -> tuple[dict[str, wp.array], dict[str, wp.array]]:
    """An entry solver's registered state-adapter arrays and item->world maps, keys prefixed `<entry>/`."""
    from alloy.core.registry import state_arrays_for, state_world_maps_for

    arrays = {f"{name}/{k}": v for k, v in state_arrays_for(solver).items()}
    maps = {f"{name}/{k}": v for k, v in state_world_maps_for(solver).items()}
    return arrays, maps


class _NoRollback:
    """StepperLike rollback hooks for steppers without a watchdog (SCR-WP4-1)."""

    def rollback_requested(self, report: CouplingReport) -> bool:
        return False

    def on_rollback(self, world_mask=None) -> None:
        return None


def _quiet(dt_x: float, substeps: Mapping[str, int], worlds: int, notes: Mapping[str, Any]) -> CouplingReport:
    rep = CouplingReport.quiet(dt_x, dict(substeps), worlds)
    rep.notes.update(notes)
    return rep


class SingleEntryStepper(_NoRollback):
    """`StepperLike` for single-entry scenes (no Scheduler): one solver.step per exchange."""

    def __init__(self, built: BuiltScene, entry: EntrySpec, native: str = "mujoco") -> None:
        cap = capability(entry.capability)
        self.entry, self.name = entry, entry.name
        self.solver: SolverBase = cap.factory(built.model, solver_kwargs(entry, native))
        self.worlds = built.world_count
        self.dt_x = entry.target_dt
        if entry.capability == "style3d":       # its solve creeps when stretch outweighs inertia (alloy.scene.style3d_lock)
            from alloy.scene.style3d_lock import warn_if_locked

            warn_if_locked(self.solver, built.model, self.dt_x)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None,
             dt: float) -> None:
        self.solver.step(state_in, state_out, control, contacts, dt)

    def choose_clock(self, control_dt: float, last: CouplingReport | None) -> tuple[float, dict[str, int]]:
        return snap_to_grid(self.dt_x, control_dt), {self.name: 1}

    def report(self) -> CouplingReport:
        return _quiet(self.dt_x, {self.name: 1}, self.worlds, {"stepper": "single_entry"})

    def state_arrays(self) -> Mapping[str, wp.array]:
        return solver_state(self.name, self.solver)[0]

    def state_world_maps(self) -> Mapping[str, wp.array]:
        return solver_state(self.name, self.solver)[1]

    def solvers(self) -> list[SolverBase]:
        return [self.solver]


class StockProxyStepper(_NoRollback):
    """`StepperLike` around Newton's stock SolverCoupledProxy for two-entry scenes (WP4's Scheduler replaces it)."""

    def __init__(self, built: BuiltScene, native: str = "newton") -> None:
        entries = [e for e in built.entries if e.name]
        if len(entries) != 2:
            raise CouplingError(f"the stock proxy couples exactly 2 entries, got {len(entries)} (3+ need WP4's "
                                "SchedulerADMM, D5)")
        self.strong_unrouted = tuple(built.coupler.strong_unrouted)
        if self.strong_unrouted:
            log.warning("strong pairs %s run lagged on the stock proxy (no ADMM before WP4)",
                        built.coupler.strong_unrouted)
        pol = built.coupler.pairs[0] if built.coupler.pairs else None
        src = pol.source if pol is not None else entries[0].name
        dst = pol.destination if pol is not None else entries[1].name
        by = {e.name: e for e in entries}
        if not by[src].bodies and by[src].particles and by[dst].capability == "sph":
            raise CouplingError(f"{src} -> {dst}: a soft source into SPH needs particle proxies, staggered (SPH-VBD P1); "
                                "the stock proxy maps bodies only and runs lagged, so the liquid would pass through. "
                                "Use WP4's Scheduler (make_stepper without stock=True)")
        self.entries = (by[src], by[dst])
        self.worlds = built.world_count
        self.dt_x = min(e.target_dt for e in entries)
        self.solver = SolverCoupledProxy(model=built.model, entries=[self._entry(e, native) for e in self.entries],
                                         coupling=SolverCoupledProxy.Config(proxies=[self._proxy(by[src], dst)],
                                                                            iterations=PROXY_ITERATIONS))

    @staticmethod
    def _entry(e: EntrySpec, native: str):
        cap, kw = capability(e.capability), solver_kwargs(e, native)
        return SolverCoupledProxy.Entry(name=e.name, solver=lambda v, cap=cap, kw=kw: cap.factory(v, kw),
                                        bodies=list(e.bodies), particles=list(e.particles), joints=list(e.joints),
                                        substeps=int(e.substeps))

    @staticmethod
    def _proxy(src: EntrySpec, dst: str):
        return SolverCoupledProxy.Proxy(source=src.name, destination=dst, bodies=list(src.bodies), mode="lagged",
                                        collision_pipeline=lambda v: newton.CollisionPipeline(v), collide_interval=1)

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None,
             dt: float) -> None:
        self.solver.step(state_in, state_out, control, contacts, dt)

    def choose_clock(self, control_dt: float, last: CouplingReport | None) -> tuple[float, dict[str, int]]:
        return snap_to_grid(self.dt_x, control_dt), {e.name: int(e.substeps) for e in self.entries}

    def report(self) -> CouplingReport:
        notes = {"stepper": "stock_proxy", "iterations": PROXY_ITERATIONS}
        if self.strong_unrouted:
            notes["strong_unrouted_run_lagged"] = self.strong_unrouted
        return _quiet(self.dt_x, {e.name: int(e.substeps) for e in self.entries}, self.worlds, notes)

    def _entry_solvers(self) -> dict[str, SolverBase]:
        """Entry solvers inside the stock coupler (private `SolverCoupled._entries`, a PRIVATE_HOOKS entry)."""
        entries = getattr(self.solver, "_entries", {}) or {}
        return {n: e.solver for n, e in entries.items() if getattr(e, "solver", None) is not None}

    def state_arrays(self) -> Mapping[str, wp.array]:
        out: dict[str, wp.array] = {}
        for name, s in self._entry_solvers().items():
            out.update(solver_state(name, s)[0])
        return out

    def state_world_maps(self) -> Mapping[str, wp.array]:
        out: dict[str, wp.array] = {}
        for name, s in self._entry_solvers().items():
            out.update(solver_state(name, s)[1])
        return out

    def solvers(self) -> list[SolverBase]:
        return [self.solver]


def default_stepper(built: BuiltScene, native: str):
    dyn = [e for e in built.entries if e.name]
    if not dyn:
        raise CouplingError("scene has no dynamic entry to step")
    if len(dyn) == 1:
        return SingleEntryStepper(built, dyn[0], native)
    return StockProxyStepper(built, native)


__all__ = ["NewtonContacts", "SingleEntryStepper", "StockProxyStepper", "default_stepper", "needs_newton_contacts"]
