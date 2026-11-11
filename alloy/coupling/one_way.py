# SPDX-License-Identifier: Apache-2.0
"""One-way followers (SCR-DC5-6): an entry that follows the bodies of a coupled stepper on its own clock and never
feeds back (D5 one-way: a ≤ 30 mg drip puts no load on the bodies).

Newton's proxy coupler takes at most two entries (newton@ac7f7830:newton/_src/solvers/coupled/
solver_coupled_proxy.py:331-332) and re-steps a source once per (source, destination) group with only that group's
feedback; a one-way destination needs neither. ``OneWayFollower`` wraps any stepper (single entry, ``Scheduler``,
``SchedulerADMM``):

1. the base steps the exchange [t, t + dt_x] as usual (a follower adds no force anywhere);
2. each follower steps a scratch copy of the state ``rung`` times at dt_x/rung, every body starting at its begin pose
   and moving at its end velocity (the lagged proxy sync, solver_coupled_proxy.py:1436-1451; with MuJoCo's
   semi-implicit integrator that lands on the end position). The follower solver is built on the full model, moves
   only its own particles and keeps every body kinematic;
3. the follower's particles are copied into ``state_out``; bodies and all other particles are the base's.

The rung follows the follower's stability estimator (SPH: ``estimators.SPHEstimator``) with the Scheduler's clock
rules (raise at once, lower one ladder step after ``downshift_quiet_periods`` quiet periods; D4, D21). Rollback state
is the base's arrays plus the follower solvers' registered state adapter arrays.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.core.errors import CouplingError
from alloy.core.registry import estimator_for, state_arrays_for, state_world_maps_for
from alloy.coupling.clocks import _Hysteresis, pick_rung
from alloy.coupling.types import ClockPolicy

log = logging.getLogger("alloy.coupling")


WORK_SCALE = 1099511627776.0      # 2^40 per J: int64 fixed point, deterministic sums (9.1e-13 J resolution)


@wp.kernel(enable_backward=False)
def add_shape_work_kernel(shape_world: wp.array[wp.int32], work: wp.array[wp.float64],
                          per_world: wp.array[wp.int64]):
    s = wp.tid()
    w = shape_world[s]
    if w >= 0 and work[s] != wp.float64(0.0):
        wp.atomic_add(per_world, w, wp.int64(wp.round(work[s] * wp.float64(WORK_SCALE))))


@wp.kernel(enable_backward=False)
def scatter_particles_kernel(ids: wp.array[wp.int32], src_q: wp.array[wp.vec3], src_qd: wp.array[wp.vec3],
                             dst_q: wp.array[wp.vec3], dst_qd: wp.array[wp.vec3]):
    i = ids[wp.tid()]
    dst_q[i] = src_q[i]
    dst_qd[i] = src_qd[i]


def validate_one_way(pairs: Sequence[Any]) -> None:
    """CouplingError unless every one-way pair is lagged and its destination is a leaf used by no other pair."""
    for p in (q for q in pairs if q.one_way):
        if p.mode != "lagged":
            raise CouplingError(f"one-way pair {p.source}->{p.destination}: mode must be 'lagged' (got {p.mode!r}); "
                                "a one-way destination is stepped once per exchange and never iterated")
        others = [q for q in pairs if q is not p and p.destination in (q.source, q.destination)]
        if others:
            raise CouplingError(f"one-way pair {p.source}->{p.destination}: its destination also appears in "
                                f"{[(q.source, q.destination) for q in others]}; a one-way destination must be a "
                                "leaf (no feedback, no chains)")


@dataclass
class OneWayLeaf:
    """A follower entry: ``solver`` is built on the full model and moves only ``particles`` (global ids)."""

    name: str
    solver: Any
    particles: Sequence[int]
    rung: int = 1                   # initial substeps per exchange (on the ClockPolicy ladder)
    adaptive: bool = True           # follow the solver's stability estimator


@dataclass
class _LeafState:
    leaf: OneWayLeaf
    ids: wp.array
    level: _Hysteresis
    estimator: Any
    stable: wp.array
    wall_work: wp.array             # (W,) int64 fixed point (WORK_SCALE) J the hub's shapes did on the leaf (ledger)
    shape_work: wp.array | None     # per-shape scratch when the solver reports wall work (SPH alloy_shape_work)
    estimated: bool = False         # no estimate before the first exchange: hold the initial rung


def _check_leaf(leaf: OneWayLeaf, ladder: Sequence[int]) -> None:
    if leaf.rung not in ladder:
        raise CouplingError(f"one-way leaf {leaf.name!r}: rung {leaf.rung} is not on the ladder {tuple(ladder)}")
    cfg = getattr(leaf.solver, "config", None)
    if getattr(cfg, "integrate_bodies", None) == "dynamic":
        raise CouplingError(f"one-way leaf {leaf.name!r}: its solver integrates bodies (integrate_bodies='dynamic'); "
                            "a follower must keep the base's bodies kinematic")
    if getattr(cfg, "coupling", "one_way") != "one_way":
        log.info("one-way leaf %r: solver config coupling=%r; its wrenches are ignored", leaf.name, cfg.coupling)


class OneWayFollower:
    """StepperLike: ``base`` plus one-way leaves stepped after it, each on its own rung (module docstring)."""

    def __init__(self, base: Any, model: Any, leaves: Sequence[OneWayLeaf], clock: ClockPolicy = ClockPolicy()):
        names = [leaf.name for leaf in leaves]
        if not leaves or len(set(names)) != len(names):
            raise CouplingError(f"OneWayFollower needs uniquely named leaves, got {names}")
        self.base, self.model, self.clock = base, model, clock
        self._a, self._b = model.state(), model.state()
        self._leaves = [self._leaf_state(leaf) for leaf in leaves]
        self._rung_changes = 0
        log.info("one-way followers %s at rungs %s over %s", names, [leaf.rung for leaf in leaves],
                 type(base).__name__)

    def _leaf_state(self, leaf: OneWayLeaf) -> _LeafState:
        _check_leaf(leaf, self.clock.substep_ladder)
        dev, w = self.model.device, int(self.model.world_count)
        ids = wp.array(np.asarray(leaf.particles, dtype=np.int32), dtype=wp.int32, device=dev)
        est = estimator_for(leaf.solver) if leaf.adaptive else None
        if leaf.adaptive and est is None:
            log.warning("one-way leaf %r has no stability estimator: its rung holds at %d", leaf.name, leaf.rung)
        stable = wp.full(w, math.inf, dtype=float, device=dev)
        wall = wp.zeros(w, dtype=wp.int64, device=dev)
        has_work = hasattr(leaf.solver, "alloy_shape_work") and int(self.model.shape_count) > 0
        scratch = wp.zeros(int(self.model.shape_count), dtype=wp.float64, device=dev) if has_work else None
        return _LeafState(leaf, ids, _Hysteresis(leaf.rung), est, stable, wall, scratch)

    # ------------------------------------------------------------------ stepping
    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        if state_out is state_in:
            raise CouplingError("OneWayFollower needs distinct state_in/state_out (the begin poses drive the leaves)")
        self.base.step(state_in, state_out, control, contacts, dt)
        for ls in self._leaves:
            self._follow(ls, state_in, state_out, dt)

    def _follow(self, ls: _LeafState, state_in: Any, state_out: Any, dt: float) -> None:
        a, b = self._a, self._b
        _copy_begin(a, state_in, state_out)
        n = ls.level.value
        for _ in range(n):
            ls.leaf.solver.step(a, b, None, None, dt / n)
            a, b = b, a
            self._add_wall_work(ls)
        if ls.ids.shape[0]:
            wp.launch(scatter_particles_kernel, dim=ls.ids.shape[0],
                      inputs=[ls.ids, a.particle_q, a.particle_qd, state_out.particle_q, state_out.particle_qd],
                      device=self.model.device)
        if ls.estimator is not None:
            ls.estimator.alloy_stable_dt(a, None, ls.stable)
            ls.estimated = True

    def _add_wall_work(self, ls: _LeafState) -> None:
        if ls.shape_work is None:
            return
        ls.leaf.solver.alloy_shape_work(ls.shape_work)
        wp.launch(add_shape_work_kernel, dim=ls.shape_work.shape[0],
                  inputs=[self.model.shape_world, ls.shape_work, ls.wall_work], device=self.model.device)

    # ------------------------------------------------------------------ clocks and reports
    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        dt_x, substeps = self.base.choose_clock(control_dt, last)
        ladder = tuple(self.clock.substep_ladder)
        for ls in self._leaves:
            if ls.estimated:
                desired = pick_rung(dt_x, float(np.min(ls.stable.numpy())), self.clock.safety, ladder)
                lower = ladder[max(ladder.index(ls.level.value) - 1, 0)]
                self._rung_changes += int(ls.level.update(desired, lower, self.clock.downshift_quiet_periods))
            substeps = {**substeps, ls.leaf.name: ls.level.value}
        return dt_x, dict(substeps)

    def rungs(self) -> dict[str, int]:
        return {ls.leaf.name: ls.level.value for ls in self._leaves}

    def report(self) -> Any:
        rep = self.base.report()
        return dataclasses.replace(rep, substeps={**dict(rep.substeps), **self.rungs()},
                                   rung_changes=int(rep.rung_changes) + self._rung_changes)

    def energy_ledger(self) -> dict[str, np.ndarray]:
        """The base's D20 ledger plus, per leaf, ``<leaf>.hub_work`` = 0 (one-way: no load, so no work, on the hub's
        bodies; exact by construction) and ``<leaf>.wall_work`` (J the hub's moving shapes did on the leaf)."""
        ledger = dict(self.base.energy_ledger()) if hasattr(self.base, "energy_ledger") else {}
        w = int(self.model.world_count)
        for ls in self._leaves:
            ledger[f"{ls.leaf.name}.hub_work"] = np.zeros(w)
            ledger[f"{ls.leaf.name}.wall_work"] = ls.wall_work.numpy().astype(np.float64) / WORK_SCALE
        return ledger

    # ------------------------------------------------------------------ rollback and lookup
    def state_arrays(self) -> Mapping[str, wp.array]:
        out = dict(self.base.state_arrays())
        for ls in self._leaves:
            out.update({f"{ls.leaf.name}.{k}": v for k, v in state_arrays_for(ls.leaf.solver).items()})
        return out

    def state_world_maps(self) -> Mapping[str, wp.array]:
        out = dict(self.base.state_world_maps())
        for ls in self._leaves:
            out.update({f"{ls.leaf.name}.{k}": v for k, v in state_world_maps_for(ls.leaf.solver).items()})
        return out

    def rollback_requested(self, report: Any) -> bool:
        return self.base.rollback_requested(report)

    def on_rollback(self, world_mask: Any = None) -> None:
        self.base.on_rollback(world_mask)

    def solver(self, name: str) -> Any:
        for ls in self._leaves:
            if ls.leaf.name == name:
                return ls.leaf.solver
        return self.base.solver(name)

    def __getattr__(self, name: str) -> Any:
        if name == "base":                      # not set yet (construction failed early)
            raise AttributeError(name)
        return getattr(self.base, name)


def _copy_begin(dst: Any, state_in: Any, state_out: Any) -> None:
    """Scratch state for the leaves: begin poses with end velocities (lagged sync), particles at the begin."""
    if dst.body_q is not None and dst.body_q.shape[0]:
        wp.copy(dst.body_q, state_in.body_q)
        wp.copy(dst.body_qd, state_out.body_qd)
    if dst.particle_q is not None and dst.particle_q.shape[0]:
        wp.copy(dst.particle_q, state_in.particle_q)
        wp.copy(dst.particle_qd, state_in.particle_qd)
