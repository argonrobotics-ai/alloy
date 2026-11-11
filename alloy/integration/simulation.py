# SPDX-License-Identifier: Apache-2.0
"""IntegratedSimulation: WP2's ``Simulation`` with the integration fixes WP12 owns (SPEC §2.3, D18, §7 WP12).

``alloy.sim.Simulation`` stays the loop (D18); this subclass replaces the parts that break once every real part is
injected, until WP2 folds them into ``alloy/sim.py``:

1. **Graph period.** One ``alloy.parallel.graphs.PeriodGraph`` per ``(dt_x, substeps)`` key and chunk
   (``GraphCache``). The captured body runs a power-of-two chunk of exchanges, at most ``MAX_GRAPH_STEPS`` and, on
   CUDA, within ``GRAPH_NODE_BUDGET`` graph nodes (each key's first replayed period runs one-exchange graphs, counted
   before their first launch; WP5's 1.19 M-node ADMM graph crashed the launch; a key whose one exchange is over the
   budget runs eagerly instead; a chunk whose graph cannot be created or launched for lack of device memory halves,
   and a one-exchange graph that does not fit runs eagerly: F10), and is launched ``2**j / chunk`` times per period (no
   host sync between launches); an odd count (j = 0) ends with one ``State.assign`` copy-back so the
   Python state references are the same objects after every period (SCR-WP8-3), and the cache checks the state
   pointers across the capture. The first period of each key runs eagerly (solvers allocate lazily and read the
   host once on their first step; those belong outside the captured period). Counters (steps, ticks) advance on the
   host after each launch, because the captured Python body runs only once. ``host_syncs`` is PAR-14. CPU graphs
   (Warp records them too) are opt-in.
2. **Latch source.** ``HealthMonitor.latch`` restores INVALID worlds from the snapshot it is given and freezes them;
   the caller owns saving it (WP5). That snapshot is the period start (no per-step save, which would cost a full
   masked state copy every step); a scene with arrays that have no per-world axis gets a per-world-only copy.
   ``latch_log`` warns once per latched world with its flags (``latch_log.py``, SCR-JOB2-4).
3. **Per-world rollback.** The escalated worlds go back to the period start and the period is re-run; every other
   world is then put back from an end-of-period snapshot (state, solver/stepper/topology arrays and its health
   record), so it keeps its first-run result bitwise. Reports are merged per world. Falls back to an all-world
   rollback when a state array has no per-world axis (masked restore impossible). The re-run worlds' health arrays
   (record, latch, clock, energy reference and work books: ``HealthMonitor.state_arrays``) go back to the period
   start too, from a health-only snapshot saved with it (WP4: the clock had counted every discarded attempt). They
   cannot live in the period-start snapshot, which is also the latch source and is restored every step.
4. **Topology → health.** Worlds with ``topology.step_events > 0`` re-baseline health's energy (SCR-WP10-2).
5. Energy references (health, coupling watchdog) are set at construction, before any capture.
6. Mixed-speed levers ([internal note] notes/mixed-speed-1107.md). #1: the graph key carries what the stepper bakes into a
   capture (``graph_key``: its captured coupling passes), and a re-run also takes back the worlds a capped capture left
   short of passes (``capture_passes``). #3: ``graph_choice`` times each key's eager and graph periods once and sends a
   key whose graphs are slower to eager periods.

Known limits: health ``first_bad_step`` inside a replayed graph is the step index seen at capture (WP5 keeps the step
on the host); a re-run adds its samples to the gap-monitor accumulators and its topology events to the ledger.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.protocols import (ContactAdjuster, ContactLayerLike, GapMonitorLike, HealthLike, StepperLike,
                                  TopologyLike, adjust_contacts)
from alloy.core.snapshot import WorldSnapshot
from alloy.coupling.capture_passes import merge_reports, short_worlds  # merge_reports: re-exported
from alloy.coupling.capture_passes import rerun_mask as _rerun_mask
from alloy.coupling.onset import escalated_beyond_onset
from alloy.coupling.types import CouplingReport
from alloy.integration.graph_memory_note import note_graph_memory
from alloy.integration.built_copy import draw_seed
from alloy.integration.graph_budget import (GRAPH_NODE_BUDGET, chunk_for, countable, graph_nodes, out_of_memory,
                                            pow2_floor, release)
from alloy.integration.graph_choice import GraphChoice
from alloy.integration.graph_checks import chunk_graph, send_eager, timed_what
from alloy.integration.graph_checks import launch as launch_period
from alloy.integration.grid_growth import regrow
from alloy.integration.mpm_memory import room_to_step
from alloy.integration.row_growth import regrow_rows
from alloy.integration.latch_log import LatchLog
from alloy.integration.period_snapshots import save_period_start
from alloy.integration.reset_graphs import after_reset, watch
from alloy.integration.stepper import entry_solvers
from alloy.precision.types import PrecisionConfig
from alloy.scene.pairs import apply_pair_table
from alloy.scene.steppers import NewtonContacts
from alloy.scene.types import BuiltScene
from alloy.sim import Simulation

log = logging.getLogger("alloy.integration")

__all__ = ["IntegratedSimulation", "merge_reports"]

MAX_GRAPH_STEPS = 64   # exchanges per captured graph at most: a 1,024-step period with unrolled coupling passes as
                       # one graph crashed capture_launch on the L4 (box/cloth/cable, Scheduler at dt_x = 19.5 us);
                       # on CUDA the node budget (graph_budget.GRAPH_NODE_BUDGET) can make a key's chunk smaller


class IntegratedSimulation(Simulation):
    """``Simulation`` with graph-safe periods, separate latch/rollback snapshots and per-world rollback."""

    def __init__(self, built: BuiltScene, *, precision: PrecisionConfig | None = None,
                 contact_layer: ContactLayerLike | None = None, gap_monitor: GapMonitorLike | None = None,
                 health: HealthLike | None = None, topology: TopologyLike | None = None,
                 stepper: StepperLike | None = None, use_graph: bool | None = None, cpu_graph: bool = False,
                 seed: int = 0, contact_adjusters: Sequence[ContactAdjuster] = ()) -> None:
        super().__init__(built, precision=precision, contact_layer=contact_layer, gap_monitor=gap_monitor,
                         health=health, topology=topology, stepper=stepper, use_graph=None, seed=seed)
        device = wp.get_device(self.model.device)
        self.use_graph = bool(use_graph) and (device.is_cuda or cpu_graph)
        self.solvers: tuple[SolverBase, ...] = tuple(entry_solvers(self.stepper).values())
        self.contact_adjusters = tuple(contact_adjusters)     # beside the topology, e.g. WP10's BladeContact
        self.max_graph_steps = MAX_GRAPH_STEPS
        self.graph_node_budget = GRAPH_NODE_BUDGET
        self.exchange_nodes: dict[Any, int | None] = {}   # per graph key: nodes of one captured exchange (CUDA)
        self.graph_chunk: dict[Any, int] = {}              # per graph key: exchanges per graph in its last period
        self.eager_keys: dict[Any, int] = {}    # keys whose one exchange is over the budget: nodes (kept on rollback)
        self.memory_chunk: dict[Any, int] = {}  # per key: the largest chunk left after a graph ran out of device memory
        self.memory_eager: set = set()          # keys whose one-exchange graph did not fit in device memory
        self.memory_notes: list[str] = []       # one plain line per step down (alloy.core.log's NullHandler hides logs)
        self.rollbacks = self.short_reruns = 0      # short re-runs (lever #1) apart: bench rows read rollbacks
        watch(self)                                 # reset keeps the period graphs when safe (reset_graphs)
        self._latch_snap = self._good if not self._good.whole_arrays else self._per_world_snapshot()
        self._latch_snap.save(self.state)
        self._end_snap: WorldSnapshot | None = None
        self._health_snap = self._health_snapshot()
        for name in ("_latch", "_end"):     # newer WP2 bases keep their own copies; this class uses its own
            if hasattr(self, name):
                setattr(self, name, self._latch_snap if name == "_latch" else None)
        self._state_names = tuple(sorted(k for k, v in vars(self.state).items() if isinstance(v, wp.array)))
        self._warm: set = set()
        self._graphs = None
        self.graph_choice: GraphChoice | None = GraphChoice() if self.use_graph else None      # lever #3
        if callable(getattr(self.stepper, "allow_capped_captures", None)):     # lever #1: this loop re-runs short
            self.stepper.allow_capped_captures()                                # worlds (_rerun_mask)
        if self.use_graph:
            from alloy.parallel.graphs import GraphCache  # noqa: PLC0415 - WP8

            self._graphs = GraphCache(str(device), use_graph=True, buffers=self._graph_buffers)
        self.latch_log = LatchLog.for_health(self.health)      # SCR-JOB2-4: a latched world says so, once
        self._health_dt = _takes_dt(self.health)
        self._set_energy_references()
        log.info("IntegratedSimulation: graph %s (%s), solvers %s", self.use_graph, device,
                 [type(s).__name__ for s in self.solvers])

    # --- state providers -------------------------------------------------------------------------------------

    def state_arrays(self) -> Mapping[str, wp.array]:
        """Stepper arrays (incl. entry solvers' state adapters) plus the topology controller's bond/phase arrays."""
        out = dict(self.stepper.state_arrays())
        topo = getattr(self, "topology", None)
        if topo is not None and hasattr(topo, "state_arrays"):
            out.update({f"topology/{k}": v for k, v in topo.state_arrays().items()})
        return out

    def state_world_maps(self) -> Mapping[str, wp.array]:
        out = dict(super().state_world_maps())
        topo = getattr(self, "topology", None)
        if topo is not None and hasattr(topo, "state_worlds"):
            out.update({f"topology/{k}": v for k, v in topo.state_worlds().items()})
        return out

    def _per_world_snapshot(self) -> WorldSnapshot:
        """Snapshot for the health latch (masked save/restore every step): arrays without a per-world axis are
        left out, because a latched world is restored every step and never reads them back."""
        snap = self._snapshot()
        whole = {name.removeprefix("extra:") for name in snap.whole_arrays}
        if not whole:
            return snap
        log.warning("health latch snapshot leaves out arrays without a per-world axis: %s", sorted(whole))
        extra = {k: v for k, v in self.state_arrays().items() if k not in whole}
        maps = {k: v for k, v in self.state_world_maps().items() if k in extra}
        return WorldSnapshot(self.model, self.state, extra or None, extra_world=maps or None)

    def _health_snapshot(self) -> WorldSnapshot | None:
        """Period-start copy of the health arrays a re-run world restores (module docstring, item 3); None when the
        stepper never asks for a rollback or health has no per-world arrays."""
        arrays = _health_arrays(self.health)
        if getattr(self.stepper, "rollback_requested", None) is None or not arrays:
            return None
        return WorldSnapshot(self.model, newton.State(), {f"health/{k}": v for k, v in arrays.items()})

    def _set_energy_references(self) -> None:
        """D20 datum/E0 for health and the coupling watchdog, before any graph capture."""
        if self.health is not None and hasattr(self.health, "reset_reference"):
            self.health.reset_reference(self.state)
        if hasattr(self.stepper, "reset_energy_reference"):
            self.stepper.reset_energy_reference(self.state)

    def _notify(self, flags) -> None:
        for s in self.solvers:
            s.notify_model_changed(flags)

    # --- loop ------------------------------------------------------------------------------------------------

    def step_period(self) -> None:
        """One control period; per-world re-run while the stepper asks for a rollback (SCR-WP4-1)."""
        save_period_start(self)       # every period, or after check steps when nothing else restores them (S1)
        start = (self.clock.ticks, self.steps)
        self._observer_clocks("save_period")
        keep: np.ndarray | None = None
        first: CouplingReport | None = None
        while True:
            dt_x, self.substeps = self.stepper.choose_clock(self.clock.control_dt, self._last_coupling)
            self.clock.set_dt(dt_x)
            self._run_period(dt_x)
            # an MPM grid outgrew its reservation (grid_growth), or self-contact rows dropped pairs (row_growth)
            if regrow(self, start) or regrow_rows(self, start):
                self._observer_clocks("restore_period")
                keep = first = None
                continue
            report = self.stepper.report()
            if keep is not None:
                self._restore_kept(keep)
                report = merge_reports(first, report, keep)
            self._last_coupling = report
            wants = getattr(self.stepper, "rollback_requested", None)
            if wants is None or not wants(report):
                break
            fine = getattr(self, "fine_rerun", None) if keep is None else None     # #10b P1: E alone, fine_rerun.py
            if fine is not None and (merged := fine(self, report, start)) is not None:
                self._last_coupling = merged
                break
            keep, first = self._roll_back(report, start), report
        self.periods += 1
        pin = getattr(getattr(self, "parts", None), "rung_pin", None)      # SCR-JOB2-1: precision rung pinned
        if pin is not None:
            pin.update(self._last_coupling)
        if self.latch_log is not None:
            self.latch_log.poll()

    def _observer_clocks(self, what: str) -> None:
        """Observers with a clock of their own (``sim.drive``) keep it at the period start and go back with a re-run."""
        for fn in self._observers:
            getattr(fn, what, lambda: None)()

    def _roll_back(self, report: CouplingReport, start: tuple[int, int]) -> np.ndarray | None:
        """Restore escalated worlds (and worlds a capped capture left short) to the period start; returns the int32
        keep mask (None = all rolled back)."""
        escalated = _rerun_mask(report)
        per_world = escalated is not None and not escalated.all() and not self._good.whole_arrays
        mask = self._mask(escalated) if per_world else None
        if per_world:
            self._end_snapshot().save(self.state)
        elif escalated is not None and self._good.whole_arrays:
            log.warning("rollback of all worlds: arrays without a per-world axis %s", self._good.whole_arrays)
        self._good.restore(self.state, mask)
        if self._health_snap is not None:
            self._health_snap.restore(self.state, mask)
        self.stepper.on_rollback(escalated if per_world else None)
        self.clock.ticks, self.steps = start
        self._observer_clocks("restore_period")
        whole = bool((report.notes or {}).get("rerun_all"))     # lever #2's gate: every world, its escalations dropped
        short = bool(short_worlds(report).any()) and not whole
        self.short_reruns, self.rollbacks = self.short_reruns + short, self.rollbacks + (not short)
        if self._graphs is not None and escalated_beyond_onset(report).any() and not (short or whole):
            self._graphs.invalidate()   # the stepper may have switched mode/ρ (Python constants baked into graphs)
            self.exchange_nodes.clear()   # and with it the graph's size: count each key again
        # a force-jump re-run (SCR-WP11-4) only refines dt_x for this period: the graphs stay valid
        log.info("period %d rolled back (%s worlds)", self.periods, int(escalated.sum()) if per_world else "all")
        return (1 - escalated).astype(np.int32) if per_world else None

    def _end_snapshot(self) -> WorldSnapshot:
        if self._end_snap is None:
            extra = dict(self.state_arrays())
            extra.update({f"health/{k}": v for k, v in _health_arrays(self.health).items()})
            maps = dict(self.state_world_maps())
            self._end_snap = WorldSnapshot(self.model, self.state, extra or None, extra_world=maps or None)
        return self._end_snap

    def _restore_kept(self, keep: np.ndarray) -> None:
        """Worlds not rolled back take their first-run end state (and health record) back."""
        self._end_snap.restore(self.state, self._mask(keep))

    def _mask(self, mask: np.ndarray) -> wp.array:
        return wp.array(np.asarray(mask, np.int32), dtype=wp.int32, device=self.model.device)

    def _run_period(self, dt_x: float) -> None:
        n, key = self.clock.steps_per_period, self._key(dt_x)
        done = self._warm_up(key, dt_x, n)
        graph, chunk = self._period_graph(dt_x, 1 if done else n) if done < n else (None, 1)
        choice = self.graph_choice
        what = timed_what(self, key, graph, chunk, n, done)       # lever #3: never a period that captures
        with choice.timed(key, what, self.model.device, n) if what else contextlib.nullcontext():
            if graph is None or not self._launch(graph, key, n - done, chunk, dt_x):
                for _ in range(n - done):
                    self._exchange(dt_x)
        self.clock.advance(n)

    def _warm_up(self, key: Any, dt_x: float, n: int) -> int:
        """A graph key's first period (SCR-WP10-3): its first two exchanges run eagerly (lazy solver allocations and
        one-time host reads stay outside any capture; an even count leaves the State objects in place), the second
        timed for lever #3; the rest replays a one-exchange graph. Today's whole first period ran eagerly: a force-jump
        cascade's fresh keys made 768 eager exchanges, 243 of 348 s (robot_cloth_liquid, L4). Returns the exchanges
        run: 0 when graphs are off, the key is not new or runs eagerly; a one-exchange period runs it (timed)."""
        if self._graphs is None or key in self._warm or key in self.eager_keys or key in self.memory_eager:
            return 0
        self._warm.add(key)
        if n > 1:
            self._exchange(dt_x)
        choice = self.graph_choice
        with choice.timed(key, "eager", self.model.device, 1) if choice is not None else contextlib.nullcontext():
            self._exchange(dt_x)
        return min(n, 2)

    def _launch(self, graph: Any, key: Any, n: int, chunk: int, dt_x: float) -> bool:
        """Launch the period as ``n // chunk`` graphs (a power of two <= n = 2**j), each with its steps' checks; False
        if the first cannot be made or launched for lack of memory (``graph_checks.launch``)."""
        return launch_period(self, graph, key, n, chunk, dt_x)

    def _out_of_memory(self, key: Any, chunk: int, graph: Any, exc: BaseException) -> None:
        """A ``chunk``-exchange graph of ``key`` did not fit in device memory (combination sweep F10: MPM coupled to
        another entry at 64 worlds on an L4): later periods try half the chunk; a one-exchange graph that does not fit
        sends the key to eager periods (slower, same results). One warning per step down. The period it was for
        steps eagerly at once: an MPM scene with no room for that is refused here (``mpm_memory.room_to_step``)."""
        free = release(self.model.device, graph, self._graphs, exc)   # the graphs, what the process holds unused
        why = str(exc).strip().splitlines()[-1][:160] if str(exc).strip() else type(exc).__name__
        note_graph_memory(self.memory_notes, chunk, why)
        if chunk > 1:
            self.memory_chunk[key] = chunk // 2
            log.warning("graph key %s: a %d-exchange period graph does not fit in device memory (%s); trying %d per "
                        "graph", key, chunk, why, chunk // 2)
        else:
            self.memory_eager.add(key)
            log.warning("graph key %s: a one-exchange graph does not fit in device memory (%s); this key runs eagerly "
                        "(no graphs: slower, same results). Fewer worlds bring graphs back", key, why)
        room_to_step(self.built, self.model.device, free)   # this period steps eagerly now: MPM refuses, cleanly

    def _key(self, dt_x: float) -> tuple:
        baked = getattr(self.stepper, "graph_key", None)       # lever #1: e.g. the captured coupling passes
        return (dt_x, tuple(sorted(self.substeps.items())), *(baked() if callable(baked) else ()))

    def _period_graph(self, dt_x: float, n: int) -> tuple[Any, int]:
        """This period's graph and chunk, or ``(None, 1)`` to run it eagerly: graphs off, a key whose one exchange is
        over ``graph_node_budget`` (``eager_keys``) or that lever #3 timed slower as graphs. A key's first exchanges
        run eagerly before this (``_warm_up``)."""
        if self._graphs is None:
            return None, 1
        key = self._key(dt_x)
        if key in self.eager_keys or key in self.memory_eager:
            return None, 1
        if self.graph_choice is not None and self.graph_choice.eager(key):    # lever #3: timed slower as graphs
            return None, 1
        chunk = self._chunk(key, n)
        graph = chunk_graph(self, key, chunk, dt_x)
        uncounted = chunk == 1 and key not in self.exchange_nodes and countable(self.model.device)
        if uncounted and not self._fits(key, graph):
            return None, 1
        return graph, chunk

    def _chunk(self, key: Any, n: int) -> int:
        """Exchanges per captured graph for ``key``: a power of two, at most ``n`` and ``max_graph_steps``, and on
        CUDA within ``graph_node_budget`` nodes. A key not yet counted gets a one-exchange graph first (WP5's MuJoCo +
        VBD segfault: 16 ADMM exchanges were 1.19 M nodes), counted by ``_fits`` before its first launch."""
        cap = pow2_floor(min(n, self.max_graph_steps, self.memory_chunk.get(key, n)))
        if cap > 1 and key not in self.exchange_nodes and countable(self.model.device):
            return 1
        self.graph_chunk[key] = chunk_for(self.exchange_nodes.get(key), cap, self.graph_node_budget)
        return self.graph_chunk[key]

    def _fits(self, key: Any, graph: Any) -> bool:
        """Capture ``key``'s one-exchange graph and count its nodes before the first launch, which sizes the key's
        later chunks. One exchange over the budget is never launched (WP5: the ordered-VBD exchange segfaulted in
        capture_launch): the graph is dropped and the key runs eagerly from then on."""
        if graph._graph is None:
            steps = self.steps                 # the capture runs the Python body (and its counter) once
            try:
                graph._capture()               # WP8's PeriodGraph: no public capture-only call; launch() would replay
            except Exception as exc:           # noqa: BLE001 - only an allocation failure is handled
                self.steps = steps
                if not out_of_memory(exc):
                    raise
                self._out_of_memory(key, 1, graph, exc)
                return False
            self.steps = steps
        nodes = self.exchange_nodes[key] = graph_nodes(graph._graph)
        if nodes is None or nodes <= self.graph_node_budget:
            log.info("graph key %s: %s nodes per exchange", key, nodes)
            return True
        graph.invalidate()
        log.warning("graph key %s: one exchange is %d graph nodes (budget %d); this key runs eagerly", key, nodes,
                    self.graph_node_budget)
        send_eager(self, key, nodes)                     # S2 (opt-in): the clock may take a finer dt_x next
        return False

    def _period_body(self, dt: float, n: int) -> None:
        """One captured chunk: ``n`` exchanges, then a copy-back if ``n`` is odd (references unchanged)."""
        a, b = self.state, self._next
        for _ in range(n):
            self._exchange(dt)
        if self.state is not a:
            a.assign(self.state)
            self.state, self._next = a, b

    def _graph_buffers(self) -> Sequence[wp.array]:
        """The State arrays present at construction, for the capture's pointer-identity check (arrays a solver
        adds to State lazily are not compared)."""
        return [getattr(s, name, None) for s in (self.state, self._next) for name in self._state_names]

    def _exchange(self, dt: float) -> None:
        """One exchange (§2.3). The clock advances in ``_run_period``."""
        s0, s1, c = self.state, self._next, self.contacts
        if self.topology is not None:
            self.topology.pre_step(s0, c, dt)
        if self.contact_layer is not None:
            self.contact_layer.collide(s0, c, dt=dt)
            if c is not None and c.rigid_contact_friction is not None:
                apply_pair_table(self.built.pair_table, c, s0, precision_external=getattr(
                    self.contact_layer, "precision_external", not isinstance(self.contact_layer, NewtonContacts)))
            self._adjust_contacts(s0, c)
        self.stepper.step(s0, s1, self.control, c, dt)
        for fn in self._observers:
            fn(s0, s1, self.control, c, dt)
        if self.gap_monitor is not None:
            self.gap_monitor.update(s0, s1, getattr(self.health, "health", None))
        if self.health is not None:
            if self._health_dt:                       # WP5 9e19d854: sim time of the first-failure snapshot
                self.health.update(self.steps, s0, s1, c, dt=dt)
            else:
                self.health.update(self.steps, s0, s1, c)
            self.health.latch(s1, self._latch_snap)
        if self.topology is not None:
            self.topology.apply(s1)
            self._topology_rebase(s1)
        self.state, self._next = s1, s0
        self.steps += 1

    def _adjust_contacts(self, state, contacts) -> None:
        """SCR-SC5-1: the topology (WP0's optional ``ContactAdjuster``), then each adjuster beside it, edit the collided
        contacts before the solver step. Without them (or without contacts) the step is unchanged."""
        if contacts is None:
            return
        adjust_contacts(self.topology, state, contacts)
        for adjuster in self.contact_adjusters:
            adjuster.adjust_contacts(state, contacts)

    def _topology_rebase(self, state) -> None:
        events = getattr(self.topology, "step_events", None)
        if events is not None and self.health is not None and hasattr(self.health, "reset_reference"):
            self.health.reset_reference(state, events)

    # --- reset / report --------------------------------------------------------------------------------------

    def reset(self, world_mask: wp.array | None = None) -> None:
        """WP2's reset, with the redraw at the scene's draw seed (``built_copy.draw_seed``: the seed the assembled
        values were drawn at), not at this simulation's ``seed`` (SCR-DC3-8: one seed for assembly and reset)."""
        seed, self.seed = self.seed, draw_seed(self.built)    # WP2's Simulation.reset redraws at self.seed
        try:
            super().reset(world_mask)
        finally:
            self.seed = seed
        if hasattr(self.stepper, "reset_worlds"):  # per-world history the snapshot does not hold (coupling.match_reset)
            self.stepper.reset_worlds(world_mask)
        self._latch_snap.save(self.state, world_mask)
        if self.health is not None and hasattr(self.health, "reset_worlds"):   # SCR-WP5-19: unlatch, records as built
            self.health.reset_worlds(world_mask)
        if self.health is not None and hasattr(self.health, "reset_reference"):
            self.health.reset_reference(self.state, world_mask)
        if self.latch_log is not None:
            self.latch_log.reset(world_mask)
        after_reset(self)                                     # the period graphs: kept when safe (reset_graphs)

    @property
    def host_syncs(self) -> int:
        """Host syncs that broke a period capture (PAR-14); 0 when graphs are off."""
        return 0 if self._graphs is None else self._graphs.host_syncs

    def graph_stats(self) -> dict[str, Any]:
        if self._graphs is None:
            return {"mode": "eager", "graphs": 0, "host_syncs": 0, "captures": 0, "launches": 0, "memory_notes": []}
        graphs = list(self._graphs._graphs.values())
        return {"mode": "graph", "graphs": len(graphs), "host_syncs": self.host_syncs,
                "captures": sum(g.captures for g in graphs), "launches": sum(g.launches for g in graphs),
                "chunks": sorted({k[1] for k in self._graphs._graphs}),
                "exchange_nodes": sorted({v for v in self.exchange_nodes.values() if v is not None}),
                "sizes": sorted((self.graph_chunk.get(k, 1), v) for k, v in self.exchange_nodes.items() if v),
                "eager_keys": sorted(self.eager_keys.values()), "memory_chunks": sorted(self.memory_chunk.values()),
                "memory_eager": len(self.memory_eager), "memory_notes": list(self.memory_notes),
                "reset_kept": self.reset_graphs["kept"], "reset_dropped": self.reset_graphs["dropped"],
                **(self.graph_choice.stats() if self.graph_choice is not None else {})}

    def report(self) -> Mapping[str, Any]:
        return {**super().report(), "graph": self.graph_stats(), "rollbacks": self.rollbacks, "short_reruns":
                self.short_reruns, "fine_reruns": getattr(getattr(self, "fine_rerun", None), "reruns", 0)}


def _takes_dt(health: Any) -> bool:
    """Whether ``health.update`` takes ``dt=`` (WP5's monitor since 9e19d854; fakes and older monitors do not)."""
    try:
        return "dt" in inspect.signature(health.update).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def _health_arrays(health: Any) -> Mapping[str, wp.array]:
    """Per-world arrays a health monitor wants rolled back with its world: ``state_arrays()`` (WP5: record, inbox,
    latched, energy reference), else the ``health`` record array alone."""
    fn = getattr(health, "state_arrays", None)
    if callable(fn):
        return dict(fn())
    record = getattr(health, "health", None)
    return {"record": record} if record is not None else {}
