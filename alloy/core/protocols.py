# SPDX-License-Identifier: Apache-2.0
"""Protocols other packages are typed against (SPEC §3.1, D17, D18).

Every class that one package calls in another has a ``*Like`` protocol here and a fake in ``alloy.testing.fakes``,
so builders can work in parallel. All protocols are ``runtime_checkable``: ``isinstance`` checks member presence only.

The five §3.1 protocols (``ContactLayerLike``, ``GapMonitorLike``, ``HealthLike``, ``StepperLike``,
``TopologyLike``) are what ``Simulation`` is typed against. WP0 adds ``CardLibraryLike``, ``RouterLike``,
``SceneAssemblerLike``, ``SimulationLike`` and ``PeriodGraphLike`` for the other cross-package calls (WP8 packs
through the assembler, WP5/WP7 drive simulations, WP2/WP12 capture graphs).
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, Control, Model, State
from alloy._vendor.newton.solvers import SolverBase

if TYPE_CHECKING:
    from alloy.cards.types import MaterialCard, ObjectCard, PairCard, Placement, ResolvedObject, SceneCard
    from alloy.contact.types import AlloyContactSidecar, GapReport
    from alloy.core.snapshot import WorldSnapshot
    from alloy.core.timebase import SimClock
    from alloy.coupling.types import CouplingReport
    from alloy.parallel.types import PackSpec
    from alloy.precision.types import HealthReport
    from alloy.route.types import RouteDecision, SceneContext
    from alloy.scene.types import BuiltScene
    from alloy.topology.types import TopologyLedger


@runtime_checkable
class StabilityEstimator(Protocol):
    """Per-world stable step and energy of one solver (registered per solver class, or implemented by a plug-in)."""

    def alloy_stable_dt(self, state: State, contacts: Contacts | None, out: wp.array) -> None:
        """Write the largest stable dt [s] per world into ``out`` ([W] float32)."""
        ...

    def alloy_energy(self, state: State, z_ref: wp.array, out: wp.array) -> None:
        """KE + Σ m·g·(z − z_ref[world]) (+ elastic) [J] per world into ``out`` ([W] float64); z_ref [W] float64 is
        the lowest height at reset (D20)."""
        ...


class DetCapability(enum.IntEnum):
    """Determinism a solver path can deliver (D9). NATIVE = bitwise run-to-run and batch-invariant, no float atomics."""

    NONE = 0
    WARP_LOWERING = 1
    NATIVE = 2


@runtime_checkable
class AlloySolverPlugin(Protocol):
    """Contract of every Alloy solver plug-in (§3.8), next to ``SolverBase`` and ``CouplingInterface``."""

    plugin_name: ClassVar[str]
    routes: ClassVar[frozenset[str]]
    determinism: ClassVar[DetCapability]

    @staticmethod
    def register_custom_attributes(builder: newton.ModelBuilder) -> None: ...

    @staticmethod
    def emit(builder: newton.ModelBuilder, obj: ResolvedObject, decision: RouteDecision) -> None: ...

    def alloy_state_arrays(self, state: State) -> Mapping[str, wp.array]:
        """Every array the next step reads (digest, snapshot, save/load)."""
        ...

    def alloy_capacity_report(self) -> Mapping[str, int]:
        """Overflow counters of fixed-capacity pools."""
        ...

    def alloy_state_worlds(self) -> Mapping[str, wp.array]:
        """Item->world int32 map for each per-item array of ``alloy_state_arrays`` (keys absent = world-major or
        whole); passed to ``WorldSnapshot(extra_world=)`` (SCR-WP9-4)."""
        ...


@runtime_checkable
class ShapeWorkSource(Protocol):
    """Optional plug-in capability (§3.8, SCR-WP9-6), separate from ``AlloySolverPlugin`` so plug-ins without it
    still satisfy the contract. Energy ledgers count a robot shape's work on the solver's particles with it."""

    def alloy_shape_work(self, out: wp.array) -> None:
        """Write the work [J] each shape's boundary did on the solver's particles during the last step into ``out``
        (float64 ``[model.shape_count]``, model shape indices; static shapes 0)."""
        ...


@runtime_checkable
class Emitter(Protocol):
    """Writes one routed object into the world-template builder (public ModelBuilder methods only, §2.2)."""

    def emit(self, b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None: ...


# --- Protocols that Simulation (WP2/WP12) is typed against (D17, D18) -------------------------------------------


@runtime_checkable
class ContactLayerLike(Protocol):
    """One contact buffer for every solver plus the Alloy sidecar (§3.5)."""

    def contacts(self) -> Contacts: ...

    def sidecar(self) -> AlloyContactSidecar: ...

    def collide(self, state: State, contacts: Contacts, *, dt: float) -> None: ...


@runtime_checkable
class GapMonitorLike(Protocol):
    """Exact-geometry precision-pair penetration monitor (D11); graph-safe ``update``."""

    def update(self, state_prev: State, state: State, health: wp.array | None = None) -> None: ...

    def report(self) -> GapReport: ...


@runtime_checkable
class HealthLike(Protocol):
    """Per-world ``StepHealth`` producer with latch-on-INVALID (§3.7)."""

    health: wp.array  # [W] StepHealth

    def update(self, step: int, state_in: State, state_out: State, contacts: Contacts | None,
               obs: SolverBase.Observables | None = None) -> None: ...

    def latch(self, state_out: State, snapshot: WorldSnapshot) -> None: ...

    def poll(self) -> HealthReport: ...


@runtime_checkable
class StepperLike(Protocol):
    """One exchange of length dt (Scheduler, SchedulerADMM, SingleEntryStepper); never owns the loop (D18)."""

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None,
             dt: float) -> None: ...

    def choose_clock(self, control_dt: float, last: CouplingReport | None) -> tuple[float, dict[str, int]]:
        """Host, once per period: (dt_x on the D4 grid, substeps per entry name)."""
        ...

    def report(self) -> CouplingReport: ...

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Lagged feedback, Aitken omega, ... (for snapshots). Coupled steppers include their entry solvers'
        registered state adapters (prefixed by entry name); Simulation does not add them again."""
        ...

    def state_world_maps(self) -> Mapping[str, wp.array]:
        """Item->world int32 maps for the per-item arrays of ``state_arrays`` (``WorldSnapshot(extra_world=)``;
        absent keys are world-major or whole) (SCR-WP4-3)."""
        ...

    def rollback_requested(self, report: CouplingReport) -> bool:
        """After a period: True if a world escalated (``WatchState.ESCALATED``) and the stepper still has rollbacks
        left this period (``WatchdogPolicy.max_rollbacks_per_period``) (SCR-WP4-1)."""
        ...

    def on_rollback(self, world_mask: np.ndarray | None = None) -> None:
        """Called after Simulation restored the period snapshot (``world_mask`` = rolled-back worlds, None = all);
        the stepper counts it, marks ROLLED_BACK, and its next ``choose_clock`` re-plans the re-run (SCR-WP4-1)."""
        ...


@runtime_checkable
class TopologyLike(Protocol):
    """Bond breaking / tear / cut on fixed topology (D19). Optional: ``ContactAdjuster.adjust_contacts`` (SCR-SC5-1)."""

    def pre_step(self, state: State, contacts: Contacts, dt: float) -> None: ...

    def apply(self, state: State) -> None: ...

    def ledger(self) -> TopologyLedger: ...

    step_events: wp.array    # [W] int32, topology events applied by the last ``apply`` (SCR-WP10-2)
    body_active: wp.array    # [body_count] int8, 0 = body retired by crumble; energy checks skip it (SCR-WP10-2)


@runtime_checkable
class ContactAdjuster(Protocol):
    """Optional ``TopologyLike`` capability (SCR-SC5-1): edit the step's contacts after collide and before the solver
    step (a blade's side-aware contact on a cut seam). Separate from ``TopologyLike`` so a topology without it still
    meets the contract; ``Simulation`` calls it through :func:`adjust_contacts`, a no-op for such a topology."""

    def adjust_contacts(self, state: State, contacts: Contacts) -> None:
        """Edit ``contacts`` in place for ``state``; launches only (capture-safe, no host sync)."""
        ...


def adjust_contacts(topology: TopologyLike | ContactAdjuster | None, state: State, contacts: Contacts) -> None:
    """``topology.adjust_contacts(state, contacts)`` when the object has it (a topology, or a standalone adjuster such
    as a blade's contact module); otherwise nothing (the default)."""
    if isinstance(topology, ContactAdjuster):
        topology.adjust_contacts(state, contacts)


# --- Further cross-package classes (WP0 additions under D17) ---------------------------------------------------


@runtime_checkable
class CardLibraryLike(Protocol):
    """Card library (WP1 ``CardLibrary``); used by the assembler (WP2), topology (WP10) and fitting (WP11)."""

    def material(self, name: str) -> MaterialCard: ...

    def pair(self, a: str, b: str) -> PairCard | None: ...

    def object(self, name: str) -> ObjectCard: ...

    def resolve(self, card: ObjectCard, placement: Placement | None = None) -> ResolvedObject: ...

    def put(self, card: MaterialCard | PairCard | ObjectCard, *, reason: str) -> Path: ...


@runtime_checkable
class RouterLike(Protocol):
    """Physics-check router (WP2 ``Router``, §3.3)."""

    def route(self, obj: ResolvedObject, ctx: SceneContext) -> RouteDecision: ...


@runtime_checkable
class SceneAssemblerLike(Protocol):
    """Scene assembler (WP2 ``SceneAssembler``, §3.4); WP8 builds packs through it, WP13 assembles task scenes."""

    def assemble(self, scene: SceneCard, device: str, *, world_count: int | None = None, world_id_offset: int = 0,
                 capacities: PackSpec | None = None, world_ids: Sequence[int] | None = None) -> BuiltScene:
        """``world_ids`` (len == world_count) gives the global id of every local world and overrides
        ``world_id_offset`` (repacked packs hold non-consecutive ids, SCR-WP8-2)."""
        ...


StepObserver = Callable[[State, State, Control, Contacts | None, float], None]


@runtime_checkable
class SimulationLike(Protocol):
    """The control-period loop owner (WP2 ``Simulation``, D18); driven by WP5 repeat tests, WP7 bully, WP8 packs."""

    model: Model
    state: State
    control: Control
    clock: SimClock

    def step_period(self) -> None: ...

    def reset(self, world_mask: wp.array | None = None) -> None: ...

    def report(self) -> Mapping[str, object]: ...

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Every array outside ``State`` the next step reads: registered state adapters of each entry solver,
        ``stepper.state_arrays()``, plug-in ``alloy_state_arrays`` (the ``WorldSnapshot`` extras) (SCR-WP5-3)."""
        ...

    def state_world_maps(self) -> Mapping[str, wp.array]:
        """Item->world maps for the per-item arrays of ``state_arrays`` (``WorldSnapshot(extra_world=)``)."""
        ...

    def add_step_observer(self, fn: StepObserver) -> None:
        """Call ``fn(state_in, state_out, control, contacts, dt)`` once per step after ``stepper.step`` (next to the
        gap monitor and health). Observers only launch kernels, so the call is graph-capturable (SCR-WP7-2)."""
        ...


@runtime_checkable
class PeriodGraphLike(Protocol):
    """Graph capture utility (WP8 ``PeriodGraph``); owns no stepping logic (D18).

    Replay contract for the captured ``step_fn`` (SCR-WP8-3): after one control period every buffer the next period
    reads through a Python reference must be the same array object as before (an even number of state/state_next
    swaps, or a copy back; note j = 0 gives one step per period), and it must not read device data on the host
    (no ``.numpy()``, ``wp.synchronize*``, host-side ``choose_clock``): those are the §2.3 "host:" lines, outside the
    captured period. A step_fn that breaks this raises ``GraphCaptureError``.
    """

    host_syncs: int

    def launch(self) -> None: ...


__all__ = [
    "AlloySolverPlugin", "CardLibraryLike", "ContactAdjuster", "ContactLayerLike", "DetCapability", "Emitter",
    "GapMonitorLike", "HealthLike", "PeriodGraphLike", "RouterLike", "SceneAssemblerLike", "ShapeWorkSource",
    "SimulationLike", "StabilityEstimator", "StepObserver", "StepperLike", "TopologyLike", "adjust_contacts",
]
