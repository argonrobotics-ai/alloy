# SPDX-License-Identifier: Apache-2.0
"""Fakes for every ``*Like`` protocol in ``alloy.core.protocols``, plus ``FakeSolver`` and ``synthetic_rig`` (SPEC §6.1).

Each fake satisfies its protocol (``isinstance`` with ``runtime_checkable``) and does the least real work that keeps
downstream tests meaningful: ``FakeHealth`` really flags non-finite and over-speed worlds and latches them from a
``WorldSnapshot``; ``FakeSimulation`` really runs the §2.3 loop eagerly with whatever parts are injected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, Control, Model, State
from alloy._vendor.newton.solvers import SolverBase

from alloy.cards.types import SceneCard
from alloy.contact.types import AlloyContactSidecar, GapReport
from alloy.core.ids import digest_of
from alloy.core.protocols import StepObserver
from alloy.core.snapshot import WorldSnapshot
from alloy.core.timebase import SimClock, grid_index, snap_to_grid
from alloy.coupling.types import CouplingReport, WatchState
from alloy.precision.types import Determinism, HealthFlags, HealthReport, Severity, StepHealth
from alloy.scene.types import BuiltScene, CouplerChoice, EntrySpec, PairTable, RouteReport
from alloy.testing.core import (  # noqa: F401  (re-exported)
    FakeCardLibrary, FakeRouter, FakeSolver, default_leaves, synthetic_rig, world_starts, worst_end,
)
from alloy.topology.types import TopologyLedger

__all__ = [
    "FakeAssembler", "FakeCardLibrary", "FakeContactLayer", "FakeGapMonitor", "FakeHealth", "FakePeriodGraph",
    "FakeRouter", "FakeSimulation", "FakeSolver", "FakeStepper", "FakeTopology", "synthetic_rig",
]


class FakeContactLayer:
    """``ContactLayerLike``: a plain ``Contacts`` + zeroed sidecar; ``collide`` runs a real Newton pipeline only if
    ``real_pipeline``. Records the dts it was called with."""

    def __init__(self, model: Model, *, real_pipeline: bool = False, rigid_contact_max: int = 64) -> None:
        self.pipeline = newton.CollisionPipeline(model) if real_pipeline else None
        if self.pipeline is not None:
            self._contacts = self.pipeline.contacts()
        else:
            self._contacts = Contacts(rigid_contact_max, 0, device=model.device, per_contact_shape_properties=True)
        self._sidecar = AlloyContactSidecar.allocate(self._contacts.rigid_contact_max, device=model.device)
        self.calls: list[float] = []

    def contacts(self) -> Contacts:
        return self._contacts

    def sidecar(self) -> AlloyContactSidecar:
        return self._sidecar

    def collide(self, state: State, contacts: Contacts, *, dt: float) -> None:
        self.calls.append(dt)
        if self.pipeline is not None:
            self.pipeline.collide(state, contacts)


class FakeGapMonitor:
    """``GapMonitorLike``: counts updates; reports ``GapReport.empty`` or a scripted per-world max."""

    def __init__(self, pairs: int = 0, worlds: int = 1, world_max_m: np.ndarray | None = None) -> None:
        self.pairs, self.worlds, self.updates = pairs, worlds, 0
        self.world_max_m = world_max_m

    def update(self, state_prev: State, state: State, health: wp.array | None = None) -> None:
        self.updates += 1

    def report(self) -> GapReport:
        rep = GapReport.empty(self.pairs, self.worlds)
        if self.world_max_m is not None:
            rep.world_max_m[:] = self.world_max_m
        return rep


@wp.kernel
def _health_check(q: wp.array(dtype=wp.transform), qd: wp.array(dtype=wp.spatial_vector),
                  start: wp.array(dtype=wp.int32), step: wp.int64, speed_fail: wp.float32,
                  health: wp.array(dtype=StepHealth)):
    w = wp.tid()
    h = health[w]
    flags = h.flags
    vmax = float(0.0)
    for i in range(start[w], start[w + 1]):
        p = wp.transform_get_translation(q[i])
        v = wp.spatial_top(qd[i])
        if not (wp.isfinite(p) and wp.isfinite(wp.transform_get_rotation(q[i]))):
            flags = flags | wp.uint32(1)            # NONFINITE_Q
        if not (wp.isfinite(v) and wp.isfinite(wp.spatial_bottom(qd[i]))):
            flags = flags | wp.uint32(2)            # NONFINITE_QD
        elif wp.length(v) > vmax:
            vmax = wp.length(v)
    if vmax > speed_fail:
        flags = flags | wp.uint32(2048)             # SPEED_FAIL
    h.max_speed_mps = vmax
    if (flags & wp.uint32(2051)) != wp.uint32(0) and h.severity < 2:
        h.severity = 2
        if h.first_bad_step < wp.int64(0):
            h.first_bad_step = step
    h.flags = flags
    health[w] = h


@wp.kernel
def _invalid_mask(health: wp.array(dtype=StepHealth), mask: wp.array(dtype=wp.int32)):
    w = wp.tid()
    mask[w] = 0
    if health[w].severity >= 2:
        mask[w] = 1
        h = health[w]
        h.flags = h.flags | wp.uint32(32768)        # LATCHED
        health[w] = h


class FakeHealth:
    """``HealthLike``: flags NONFINITE_Q/QD and SPEED_FAIL per world (per-world loops, no atomics); ``inject``
    scripts flags from the host; ``latch`` restores INVALID worlds from a snapshot and sets LATCHED."""

    def __init__(self, model: Model, *, speed_fail_mps: float = 20.0) -> None:
        self.model, self.world_count = model, int(model.world_count)
        self.speed_fail = speed_fail_mps
        init = wp.zeros(self.world_count, dtype=StepHealth, device="cpu").numpy()
        init["first_bad_step"] = -1
        init["residual"] = np.nan
        self.health = wp.array(init, dtype=StepHealth, device=model.device)
        bw = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
        self._start = wp.array(world_starts(bw, self.world_count), dtype=wp.int32, device=model.device)
        self._mask = wp.zeros(self.world_count, dtype=wp.int32, device=model.device)

    def update(self, step: int, state_in: State, state_out: State, contacts: Contacts | None,
               obs: SolverBase.Observables | None = None) -> None:
        if self.model.body_count:
            wp.launch(_health_check, dim=self.world_count, device=self.model.device,
                      inputs=[state_out.body_q, state_out.body_qd, self._start, step, self.speed_fail, self.health])

    def inject(self, world: int, flags: HealthFlags, severity: Severity, step: int = 0) -> None:
        """Host-side scripting: OR ``flags`` into ``world`` and raise its severity."""
        h = self.health.numpy()
        h["flags"][world] |= int(flags)
        h["severity"][world] = max(int(h["severity"][world]), int(severity))
        if h["first_bad_step"][world] < 0 and severity > Severity.OK:
            h["first_bad_step"][world] = step
        self.health.assign(wp.array(h, dtype=StepHealth, device=self.model.device))

    def latch(self, state_out: State, snapshot: WorldSnapshot) -> None:
        wp.launch(_invalid_mask, dim=self.world_count, inputs=[self.health, self._mask], device=self.model.device)
        snapshot.restore(state_out, self._mask)

    def poll(self) -> HealthReport:
        h = self.health.numpy()
        sev = h["severity"].astype(np.int32)
        return HealthReport(sev, h["flags"].copy(), h["first_bad_step"].copy(), h["max_pen_m"].astype(float),
                            h["max_pair_pen_m"].astype(float), h["max_speed_mps"].astype(float),
                            tuple(int(w) for w in np.flatnonzero(sev >= Severity.INVALID)), Determinism.RUN_TO_RUN, ())


class FakeStepper:
    """``StepperLike`` around one solver: ``step`` = one ``solver.step``; fixed clock; quiet reports.
    ``request_rollbacks = n`` makes the next n periods ask for a rollback of every world (loop tests)."""

    def __init__(self, solver: SolverBase, *, dt_x: float = 1.25e-3, name: str = "fake") -> None:
        self.solver, self.dt_x, self.name = solver, dt_x, name
        self.world_count = int(solver.model.world_count)
        self.exchanges = 0
        self.request_rollbacks = 0
        self.rollbacks = 0

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None,
             dt: float) -> None:
        self.solver.step(state_in, state_out, control, contacts, dt)
        self.exchanges += 1

    def choose_clock(self, control_dt: float, last: CouplingReport | None) -> tuple[float, dict[str, int]]:
        return snap_to_grid(self.dt_x, control_dt), {self.name: 1}

    def report(self) -> CouplingReport:
        rep = CouplingReport.quiet(self.dt_x, {self.name: 1}, self.world_count)
        if self.request_rollbacks > 0:
            rep.watch_state[:] = int(WatchState.ESCALATED)
        return rep

    def state_arrays(self) -> Mapping[str, wp.array]:
        return {}

    def state_world_maps(self) -> Mapping[str, wp.array]:
        return {}

    def rollback_requested(self, report: CouplingReport) -> bool:
        return self.request_rollbacks > 0 and bool(np.any(report.watch_state == int(WatchState.ESCALATED)))

    def on_rollback(self, world_mask: np.ndarray | None = None) -> None:
        self.request_rollbacks -= 1
        self.rollbacks += 1


class FakeTopology:
    """``TopologyLike`` and ``ContactAdjuster``: counts calls; reports an empty ledger (or ``events`` set by the
    test); ``step_events`` are zeros and ``body_active`` ones (SCR-WP10-2); ``adjust_contacts`` changes nothing."""

    def __init__(self, world_count: int, *, body_count: int = 0, device: str | None = None) -> None:
        self.world_count, self.pre_steps, self.applies, self.adjusts = world_count, 0, 0, 0
        self.events = np.zeros(world_count, np.int64)
        self.step_events = wp.zeros(world_count, dtype=wp.int32, device=device)
        self.body_active = wp.ones(body_count, dtype=wp.int8, device=device)

    def pre_step(self, state: State, contacts: Contacts, dt: float) -> None:
        self.pre_steps += 1

    def apply(self, state: State) -> None:
        self.applies += 1

    def adjust_contacts(self, state: State, contacts: Contacts) -> None:
        self.adjusts += 1

    def ledger(self) -> TopologyLedger:
        led = TopologyLedger.empty(self.world_count)
        led.events[:] = self.events
        return led


class FakeAssembler:
    """``SceneAssemblerLike``: ignores the card contents and builds ``scene_factory(world_count)`` (default
    ``alloy.testing.scenes.box_on_plane``) into a ``BuiltScene`` with one entry, no coupler and a 1-material table."""

    def __init__(self, scene_factory: Callable[..., Any] | None = None, solver: str = "xpbd") -> None:
        from alloy.testing.scenes import box_on_plane

        self.scene_factory = scene_factory or box_on_plane
        self.solver = solver
        self.calls: list[tuple[int, int]] = []

    def assemble(self, scene: SceneCard, device: str, *, world_count: int | None = None, world_id_offset: int = 0,
                 capacities: Any | None = None, world_ids: Sequence[int] | None = None) -> BuiltScene:
        n = int(world_count or (len(world_ids) if world_ids is not None else scene.worlds))
        if world_ids is not None and len(world_ids) != n:
            raise ValueError(f"world_ids has {len(world_ids)} ids for {n} worlds")
        ref = self.scene_factory(n, device=device)
        model = ref.model
        self.calls.append((n, world_id_offset))
        entry = EntrySpec(self.solver, self.solver, tuple(range(model.body_count)), tuple(range(model.particle_count)),
                          tuple(range(model.joint_count)), tuple(range(model.shape_count)), 1, ref.dt)
        return BuiltScene(model, [entry], CouplerChoice("none", ()), {}, _one_material_table(model), (),
                          scene.control_dt, n, world_id_offset, wp.zeros(n, dtype=wp.vec3d, device=device), None, {},
                          scene.emitters, RouteReport.empty(), digest_of({"fake": ref.name, "worlds": n}),
                          None if world_ids is None else wp.array(list(world_ids), dtype=wp.int32, device=device))


def _one_material_table(model: Model) -> PairTable:
    dev = model.device
    return PairTable(("default",), wp.array([[wp.vec4(0.5, 0.5, 0.0, 0.0)]], dtype=wp.vec4, device=dev),
                     wp.array([[1.0e6]], dtype=wp.float32, device=dev),
                     wp.zeros(model.shape_count, dtype=wp.int32, device=dev),
                     wp.zeros(model.particle_count, dtype=wp.int32, device=dev),
                     wp.full(model.shape_count, 0.5, dtype=wp.float32, device=dev), 0,
                     wp.zeros(1, dtype=wp.float32, device=dev))


class FakePeriodGraph:
    """``PeriodGraphLike``: ``launch`` calls ``step_fn`` eagerly; ``host_syncs`` stays 0."""

    def __init__(self, step_fn: Callable[[], None], device: str = "cpu") -> None:
        self.step_fn, self.device = step_fn, device
        self.host_syncs = 0
        self.launches = 0

    def launch(self) -> None:
        self.step_fn()
        self.launches += 1


class FakeSimulation:
    """``SimulationLike``: the §2.3 loop run eagerly with the injected parts (WP2's ``Simulation`` replaces it).

    Per step: save a good snapshot, topology.pre_step, contact_layer.collide, stepper.step, step observers,
    gap_monitor.update (with the health record, D11), health.update + latch, topology.apply, swap, clock advance.
    Per period: if ``stepper.rollback_requested(report)``, restore the period snapshot, ``stepper.on_rollback()`` and
    re-run (SCR-WP4-1). ``reset(world_mask)`` restores the initial state. Not graph-safe (one swap per step).
    """

    def __init__(self, model: Model, stepper: Any, *, control_dt: float = 0.02, contact_layer: Any | None = None,
                 gap_monitor: Any | None = None, health: Any | None = None, topology: Any | None = None) -> None:
        self.model, self.stepper = model, stepper
        self.contact_layer, self.gap_monitor, self.health, self.topology = contact_layer, gap_monitor, health, topology
        self.state, self._next, self.control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        self.contacts = contact_layer.contacts() if contact_layer is not None else None
        dt_x, _ = stepper.choose_clock(control_dt, None)
        self.clock = SimClock(control_dt, grid_index(dt_x, control_dt))
        self.observers: list[StepObserver] = []
        extra, maps = self.state_arrays(), self.state_world_maps()
        self._initial = WorldSnapshot(model, self.state, extra, extra_world=maps)
        self._initial.save(self.state)
        self._good = WorldSnapshot(model, self.state, extra, extra_world=maps)
        self._period = WorldSnapshot(model, self.state, extra, extra_world=maps)
        self.steps = 0
        self.rollbacks = 0
        self._last: CouplingReport | None = None

    def state_arrays(self) -> Mapping[str, wp.array]:
        return dict(self.stepper.state_arrays())

    def state_world_maps(self) -> Mapping[str, wp.array]:
        return dict(self.stepper.state_world_maps())

    def add_step_observer(self, fn: StepObserver) -> None:
        self.observers.append(fn)

    def step_period(self) -> None:
        ticks, steps = self.clock.ticks, self.steps
        self._period.save(self.state)
        while True:
            dt_x, _ = self.stepper.choose_clock(self.clock.control_dt, self._last)
            self.clock.set_dt(dt_x)
            for _ in range(self.clock.steps_per_period):
                self._step_once(dt_x)
            self._last = self.stepper.report()
            if not self.stepper.rollback_requested(self._last):
                return
            self._period.restore(self.state)
            self.clock.ticks, self.steps = ticks, steps
            self.stepper.on_rollback(None)
            self.rollbacks += 1

    def _step_once(self, dt: float) -> None:
        self._good.save(self.state)
        if self.topology is not None:
            self.topology.pre_step(self.state, self.contacts, dt)
        if self.contact_layer is not None:
            self.contact_layer.collide(self.state, self.contacts, dt=dt)
        self.stepper.step(self.state, self._next, self.control, self.contacts, dt)
        for fn in self.observers:
            fn(self.state, self._next, self.control, self.contacts, dt)
        if self.gap_monitor is not None:
            self.gap_monitor.update(self.state, self._next, self.health.health if self.health is not None else None)
        if self.health is not None:
            self.health.update(self.steps, self.state, self._next, self.contacts)
            self.health.latch(self._next, self._good)
        if self.topology is not None:
            self.topology.apply(self._next)
        self.state, self._next = self._next, self.state
        self.clock.advance(1)
        self.steps += 1

    def reset(self, world_mask: wp.array | None = None) -> None:
        self._initial.restore(self.state, world_mask)

    def report(self) -> Mapping[str, Any]:
        return {
            "steps": self.steps, "t": self.clock.t, "rollbacks": self.rollbacks, "coupling": self.stepper.report(),
            "health": self.health.poll() if self.health is not None else None,
            "gap": self.gap_monitor.report() if self.gap_monitor is not None else None,
            "topology": self.topology.ledger() if self.topology is not None else None,
        }
