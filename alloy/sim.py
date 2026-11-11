# SPDX-License-Identifier: Apache-2.0
"""Simulation: the only owner of the control-period loop (SPEC §2.3, D18), typed against core protocols.

Per exchange of length dt_x: topology.pre_step → contacts.collide → pair_table.apply(contacts, state) →
stepper.step → gap_monitor.update → health.update + latch → topology.apply → swap → clock.advance (int64 ticks, D4).
Once per period on the host: stepper.choose_clock (dt_x, substeps), a period-start snapshot for latching, and the
reports. Absent parts are skipped. A rollback restores only the escalated worlds and keeps every other world's
first end-of-period state. Without an injected stepper: one entry → SingleEntryStepper, two → Newton's stock
SolverCoupledProxy (StockProxyStepper); three or more need WP4's SchedulerADMM. Without an injected contact layer:
SolverMuJoCo alone runs its own contacts (native "mujoco"); every other scene gets the fallback Newton pipeline
with per-contact arrays so the pair table applies. This loop runs eagerly; CUDA graph capture of the period (odd swap
counts, host counters, per-key graph cache) is WP12's `alloy.integration.simulation.IntegratedSimulation`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts

from alloy.core.errors import CouplingError
from alloy.core.protocols import ContactLayerLike, GapMonitorLike, HealthLike, StepperLike, TopologyLike
from alloy.core.snapshot import WorldSnapshot
from alloy.core.timebase import SimClock, grid_index
from alloy.precision.types import PrecisionConfig
from alloy.scene.pairs import apply_pair_table
from alloy.scene.steppers import NewtonContacts, default_stepper, needs_newton_contacts
from alloy.scene.types import BuiltScene

log = logging.getLogger("alloy.sim")


class Simulation:
    """Runs §2.3 for one BuiltScene (D18). Optional parts are injected through core protocols."""

    def __init__(self, built: BuiltScene, *, precision: PrecisionConfig | None = None,
                 contact_layer: ContactLayerLike | None = None, gap_monitor: GapMonitorLike | None = None,
                 health: HealthLike | None = None, topology: TopologyLike | None = None,
                 stepper: StepperLike | None = None, use_graph: bool | None = None, seed: int = 0) -> None:
        self.built, self.precision, self.seed = built, precision, seed
        self.model = model = built.model
        self._observers: list = []
        dyn = [e for e in built.entries if e.name]
        self.native = self._native(precision, dyn, contact_layer)
        self.stepper = stepper if stepper is not None else default_stepper(built, self.native)
        self.contact_layer = contact_layer
        if contact_layer is None and needs_newton_contacts(dyn, self.native):
            self.contact_layer = NewtonContacts(model)
        self.contacts: Contacts | None = self.contact_layer.contacts() if self.contact_layer is not None else None
        self.gap_monitor, self.health, self.topology = gap_monitor, health, topology
        self.state, self._next, self.control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        dt_x, self.substeps = self.stepper.choose_clock(built.control_dt, None)
        self.clock = SimClock(built.control_dt, grid_index(dt_x, built.control_dt))
        self._initial = self._snapshot()
        self._initial.save(self.state)
        self._good = self._snapshot()       # period start: what a rollback restores
        self._latch = self._snapshot()      # health's last good step (WP5 refreshes it for every valid world)
        self._end = self._snapshot()        # accepted end-of-period state of worlds a rollback keeps
        self.steps, self.periods = 0, 0
        self.reset_hooks: list[Any] = []    # fn(world_mask) after each reset's restore and redraw; empty: no-op
        if use_graph:
            log.warning("Simulation runs eagerly; graph capture is IntegratedSimulation (WP12)")
        self.use_graph = False
        self._last_coupling = None
        log.info("Simulation: %d worlds, entries %s, native=%s, dt_x %.4g s, graph %s", built.world_count,
                 [e.name for e in dyn], self.native, dt_x, self.use_graph)

    @staticmethod
    def _native(precision: PrecisionConfig | None, dyn, layer) -> str:
        if precision is not None:
            return precision.native_contacts
        only_mujoco = len(dyn) == 1 and dyn[0].capability == "mujoco"
        return "mujoco" if only_mujoco and layer is None else "newton"

    def _snapshot(self) -> WorldSnapshot:
        extra, maps = dict(self.state_arrays()), dict(self.state_world_maps())
        return WorldSnapshot(self.model, self.state, extra or None, extra_world=maps or None)

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Arrays outside State the next step reads: the stepper's (incl. its entry solvers' state adapters)."""
        return dict(self.stepper.state_arrays())

    def state_world_maps(self) -> Mapping[str, wp.array]:
        maps = getattr(self.stepper, "state_world_maps", None)
        return dict(maps()) if maps is not None else {}

    def add_step_observer(self, fn) -> None:
        """`fn(state_in, state_out, control, contacts, dt)` once per step after stepper.step (SCR-WP7-2)."""
        self._observers.append(fn)

    # --- loop ------------------------------------------------------------------------------------------------

    def step_period(self) -> None:
        """One control period; re-run from the period snapshot while the stepper requests a rollback (SCR-WP4-1).
        Only escalated worlds are rolled back: the others keep their first end-of-period state (they are re-run with
        the rest, then put back), and the step counter and clock restart from the period start on every re-run."""
        self._good.save(self.state)
        start_ticks, start_steps = self.clock.ticks, self.steps
        accepted = np.zeros(self.built.world_count, np.int32)
        while True:
            dt_x, self.substeps = self.stepper.choose_clock(self.clock.control_dt, self._last_coupling)
            self.clock.set_dt(dt_x)
            self._run_period(dt_x)
            self._last_coupling = self.stepper.report()
            wants = getattr(self.stepper, "rollback_requested", None)
            if wants is None or not wants(self._last_coupling):
                break
            mask = _escalated(self._last_coupling)
            if mask is not None:
                newly = ((mask == 0) & (accepted == 0)).astype(np.int32)
                self._end.save(self.state, self._device_mask(newly))
                accepted |= newly
            self._good.restore(self.state, None if mask is None else self._device_mask(mask))
            getattr(self.stepper, "on_rollback", lambda m: None)(mask)
            self.clock.ticks, self.steps = start_ticks, start_steps
            log.info("period %d rolled back (worlds %s)", self.periods, "all" if mask is None else int(mask.sum()))
        if accepted.any():
            self._end.restore(self.state, self._device_mask(accepted))
        self.periods += 1

    def _device_mask(self, mask: np.ndarray) -> wp.array:
        return wp.array(np.asarray(mask, np.int32), dtype=wp.int32, device=self.model.device)

    def _run_period(self, dt_x: float) -> None:
        for _ in range(self.clock.steps_per_period):
            self._exchange(dt_x)

    def _exchange(self, dt: float) -> None:
        s0, s1, c = self.state, self._next, self.contacts
        if self.topology is not None:
            self.topology.pre_step(s0, c, dt)
        if self.contact_layer is not None:
            self.contact_layer.collide(s0, c, dt=dt)
            if c is not None and c.rigid_contact_friction is not None:
                apply_pair_table(self.built.pair_table, c, s0,   # = PairTable.apply(c, state), SCR-WP2-1
                                 precision_external=not isinstance(self.contact_layer, NewtonContacts))
        adjust = getattr(self.topology, "adjust_contacts", None)   # SCR-SC5-1: after collide, before the solve
        if adjust is not None:
            adjust(s0, c)
        self.stepper.step(s0, s1, self.control, c, dt)
        for fn in self._observers:
            fn(s0, s1, self.control, c, dt)
        if self.gap_monitor is not None:
            self.gap_monitor.update(s0, s1, getattr(self.health, "health", None))
        if self.health is not None:
            self.health.update(self.steps, s0, s1, c)
            self.health.latch(s1, self._latch)
        if self.topology is not None:
            self.topology.apply(s1)
        self.state, self._next = s1, s0
        self.clock.advance(1)
        self.steps += 1

    # --- reset / report --------------------------------------------------------------------------------------

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Restore the initial state (masked worlds), redraw their randomized values at the scene's draw seed
        (`randomize.draw_seed`: the assembly draw unless the caller redrew all worlds), refresh solvers."""
        self._initial.restore(self.state, world_mask)
        plan = getattr(self.built, "_plan", None)
        if plan is not None and any(p.randomize for p in plan.scene.objects):
            self._redraw(plan, world_mask)
        for hook in self.reset_hooks:
            hook(world_mask)

    def _redraw(self, plan: Any, world_mask: wp.array | None) -> None:
        """Redraw the reset worlds' object rows (TRAIN-FIX 3a, TA-8 F1). Only worlds whose draw differs from the value
        the model holds are written, and only then are the solvers told: on a plain reset the draw seed gives every
        world its values again, so nothing is written. A world whose start draw changed (a caller redrew it at another
        seed) gets its objects' root coordinates from the model and forward kinematics on their joints alone. The
        state of every other world is left as it is: the restore above already put the reset worlds at their start.

        Cost (TRAIN-FIX 3b, TA-8 F3): every row's draws at the seed are made once for all worlds (``_draws``); a reset
        compares them with the held values and reads the mask back from the device only when one differs."""
        from alloy.scene.randomize import POSE_AXES, apply_randomization, draw_seed

        seed, count = draw_seed(self.built), self.built.world_count
        differ = _differing(self.built, self._draws(plan, seed))
        if not any(d.any() for d in differ.values()):
            return
        mask = np.ones(count, bool) if world_mask is None else world_mask.numpy()[:count] != 0
        stale, moved = np.zeros(count, bool), {}
        for (name, field), d in differ.items():
            d = d & mask
            stale |= d
            if field in POSE_AXES and d.any():
                moved.setdefault(name, set()).update(int(w) for w in np.flatnonzero(d))
        if not stale.any():
            return
        ids = np.flatnonzero(stale).astype(np.int32)
        apply_randomization(self.built, seed, wp.array(ids, dtype=wp.int32, device="cpu"))
        self._notify(newton.ModelFlags.ALL)
        if moved:
            self._restart(moved)

    def _draws(self, plan: Any, seed: int) -> dict[tuple[str, str], np.ndarray]:
        """{(object, field): (worlds,) draws at ``seed``}, made once per seed and kept (the draws never change)."""
        cached = getattr(self, "_draw_cache", None)
        if cached is None or cached[0] != seed:
            cached = self._draw_cache = (seed, _all_draws(self.built, plan, seed))
        return cached[1]

    def _restart(self, moved: Mapping[str, set[int]]) -> None:
        """Put each object of ``moved`` ({name: worlds}) at its start in those worlds: its root joint's coordinates
        from the model (where the start draws are written), then FK on that object's joints only."""
        model, handles = self.model, self.built.handles
        qs, mq = model.joint_q_start.numpy(), model.joint_q.numpy()
        jq, joints = self.state.joint_q.numpy().copy(), np.zeros(model.joint_count, bool)
        for name, worlds in moved.items():
            h = handles[name]
            for w in worlds:
                first = h.template.joint.start + w * len(h.world_stride.joint)
                jq[qs[first]:qs[first + 1]] = mq[qs[first]:qs[first + 1]]
                joints[first:first + len(h.template.joint)] = True
        self.state.joint_q.assign(jq)
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state,
                       joint_mask=wp.array(joints, dtype=wp.bool, device=model.device))

    def _notify(self, flags) -> None:
        for s in getattr(self.stepper, "solvers", lambda: [])():
            s.notify_model_changed(flags)

    def report(self) -> Mapping[str, Any]:
        return {
            "steps": self.steps, "periods": self.periods, "t": self.clock.t, "ticks": self.clock.ticks,
            "coupling": self.stepper.report(), "native": self.native,
            "health": self.health.poll() if self.health is not None else None,
            "gap": self.gap_monitor.report() if self.gap_monitor is not None else None,
            "topology": self.topology.ledger() if self.topology is not None else None,
        }


def _escalated(report) -> np.ndarray | None:
    """int32 mask of escalated worlds (WatchState 2) from a CouplingReport, None = roll back every world."""
    ws = getattr(report, "watch_state", None)
    if ws is None or len(ws) == 0:
        return None
    return (np.asarray(ws) == 2).astype(np.int32)


def _all_draws(built: BuiltScene, plan: Any, seed: int) -> dict[tuple[str, str], np.ndarray]:
    """{(object, field): (worlds,) draw at ``seed`` in every world}. Rows are paired with objects as
    ``apply_randomization`` pairs them, and a later row on the same field wins there too."""
    from alloy.scene.randomize import draw

    gids = [built.global_world_id(w) for w in range(built.world_count)]
    names = [t.name for t in plan.objects if t.obj is not None]
    return {(name, r.field): np.asarray([draw(seed, g, f"{name}.{r.field}", r.lo, r.hi) for g in gids])
            for name, placement in zip(names, plan.scene.objects) for r in placement.randomize}


def _differing(built: BuiltScene, draws: Mapping[tuple[str, str], np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    """{(object, field): (worlds,) bool, the draw differs from the value ``built.randomized`` holds}. The draws are
    exact, so a value drawn at the same seed compares equal; a world never drawn (NaN, or no array) differs."""
    held = getattr(built, "randomized", {})
    return {k: ~(np.asarray(held[k]) == v) if k in held else np.ones(len(v), bool) for k, v in draws.items()}


__all__ = ["CouplingError", "Simulation"]
