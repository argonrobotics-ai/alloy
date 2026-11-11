# SPDX-License-Identifier: Apache-2.0
"""A bare Newton loop for bully targets that are not SceneCard scenes (raw-solver targets, plug-in B0 scenes).

``DirectSim`` implements ``SimulationLike`` (``model``, ``state``, ``control``, ``clock``, ``step_period``, ``reset``,
``report``, ``state_arrays``, ``state_world_maps``, ``add_step_observer``), so ``BenchMonitors`` sees every solver
step, plus ``pre_substep`` hooks for force actions. It is a bench harness like ``alloy.bench.run_law`` (which also
steps solvers directly), not a second owner of the product loop (D18): SceneCard targets use ``Simulation``.

Health (WP13, 2026-11-04): a WP5 ``HealthMonitor`` (``HEALTH_POLICY``: cheap checks) runs once per control period on
the solver's own sources, and ``report()["health"]`` carries its records. So the bully classifies DirectSim targets on
the health flags as it does SceneCard ones, and a world a solver guard rescued (FALLBACK_USED, e.g. the BUL-0016 pivot
clamp) reads DEGRADED, not OK. Energy stays with the bench monitors, which account for actuator work. ``reset`` starts
the reset worlds' records and the guards' cumulative counters afresh.

CPU graph replay (SCR-WP8-5 Ask 1). A period can replay a captured pair of substeps (``alloy.parallel.graphs.
PeriodGraph``; two substeps, so the Python state swap is back where it started) ``substeps / 2`` times instead of
stepping eagerly: T12 on Warp CPU went from 14.6-19 to 2.0-2.9 ms per step, bitwise equal. The health check stays
once per period on the host, and the clock advances on the host after each replay. ``graph=None`` (default) replays
only what is graph-safe on CUDA and was checked bitwise on CPU (``auto_graph_reason``): Warp CPU graphs, an even
substep count, ``SolverMuJoCo`` on MuJoCo Warp with its own contacts. ``graph=True`` asks for it on any device and
solver; ``graph=False`` never. Every period stays eager while any ``pre_substep`` hook or step observer is not
marked ``graph_safe`` (only device work on arrays that keep their identity, no host reads, no Python state that
changes per call: a substep counter or a host list breaks under replay), or when a subclass overrides ``substep``.
The first pair after construction runs eagerly (first-step allocations and host reads stay outside the capture). A
new binding (state or control objects, solver, hooks, observers) recaptures, and so does ``reset``.

CPU route fast path (SCR-WP8-8). With a solver that runs a whole period in one call (``step_many``: the MuJoCo C route,
``alloy.route.mjc_core.MjcCoreSolver``), a period with no ``pre_substep`` hooks, no step observers and no Newton
collision pipeline is one ``step_many(substeps)`` call (one C call per world): forces cleared once, the contact export
after it, the clock advanced by ``substeps``, the health check as for any period. Bitwise equal to the per-step loop
(T12: 0.19 against 0.71 ms per step). Hooks or observers (``BenchMonitors``) keep the per-step loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.errors import BullyError, GraphCaptureError
from alloy.core.protocols import StepObserver
from alloy.core.registry import state_arrays_for
from alloy.core.timebase import TICK_EXP, SimClock
from alloy.core.worldops import joint_coord_world, joint_dof_world
from alloy.parallel.graphs import PeriodGraph, cpu_graphs_supported
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy

log = logging.getLogger("alloy.bully")

_STATE_FIELDS = ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "joint_qd")
_ITEM_WORLD: dict[str, Callable[[newton.Model], wp.array | None]] = {
    "body_q": lambda m: m.body_world,
    "body_qd": lambda m: m.body_world,
    "particle_q": lambda m: m.particle_world,
    "particle_qd": lambda m: m.particle_world,
    "joint_q": joint_coord_world,
    "joint_qd": joint_dof_world,
}
HEALTH_POLICY = CheckPolicy(level="cheap", every=1)  # finiteness, speed, |p|, contacts, solver overflow and fallback
GRAPH_SAFE = "alloy_graph_safe"  # attribute that marks a pre_substep hook or step observer capturable
GRAPH_SOLVERS = frozenset({"SolverMuJoCo"})  # graph=None replays these (graph-safe on CUDA, bitwise on CPU)
F = TypeVar("F", bound=Callable)


def graph_safe(fn: F) -> F:
    """Mark a ``pre_substep`` hook or step observer capturable (module doc): decorate the function or method, or
    set ``alloy_graph_safe = True`` on a callable object."""
    setattr(fn, GRAPH_SAFE, True)
    return fn


def is_graph_safe(fn: Callable) -> bool:
    return bool(getattr(fn, GRAPH_SAFE, False))


def auto_graph_reason(solver: Any, device: Any, substeps: int, collide: bool) -> str | None:
    """Why ``graph=None`` keeps a DirectSim eager, or None when its periods replay a CPU graph (module doc)."""
    if not wp.get_device(device).is_cpu:
        return "not a CPU device (DirectSim graphs are opt-in there: graph=True)"
    if not cpu_graphs_supported():
        return "this Warp records no CPU graphs"
    if substeps % 2:
        return "one substep per period (a captured pair needs an even count)"
    name = type(solver).__name__
    if name not in GRAPH_SOLVERS or getattr(solver, "use_mujoco_cpu", False):
        return f"{name} is not on the graph-safe solver list {sorted(GRAPH_SOLVERS)} (MuJoCo Warp)"
    if int(getattr(solver, "update_data_interval", 1)) > 1:
        return "update_data_interval > 1 (the push runs on the solver's host step count)"
    if collide:
        return "Newton collision pipeline (not checked under CPU graphs)"
    return None


class DirectSim:
    """``substeps`` solver steps of ``control_dt / substeps`` per control period, collision before each step."""

    def __init__(
        self,
        model: newton.Model,
        solver: newton.solvers.SolverBase,
        *,
        control_dt: float = 0.02,
        substeps: int = 16,
        collide: bool = True,
        observers: Sequence[StepObserver] = (),
        export_contacts: bool = False,
        health: bool = True,
        graph: bool | None = None,
    ) -> None:
        j = int(substeps).bit_length() - 1
        if control_dt <= 0.0 or substeps < 1 or substeps != 1 << j or j > TICK_EXP:
            raise BullyError(f"substeps must be a power of two <= 2^{TICK_EXP} (D4 grid), got {substeps}")
        if graph and substeps % 2:
            raise BullyError("graph=True replays pairs of substeps; one substep per period cannot pair up")
        self.clock = SimClock(float(control_dt), j)
        self.model, self.solver = model, solver
        self.control_dt, self.substeps = float(control_dt), int(substeps)
        self.dt = self.control_dt / self.substeps
        self.state, self._next = model.state(), model.state()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)  # body poses from model.joint_q
        self.control = model.control()
        self.pipeline = newton.CollisionPipeline(model) if collide else None
        self.contacts = self.pipeline.contacts() if self.pipeline is not None else None
        self._export = None
        if export_contacts and self.pipeline is None:  # solver-native contacts (MuJoCo) copied out for observers
            naconmax = int(getattr(getattr(solver, "mjw_data", None), "naconmax", 0))
            if naconmax <= 0:
                raise BullyError(f"{type(solver).__name__} has no native contacts to export")
            self.contacts = newton.Contacts(naconmax, 0, device=model.device)
            self._export = solver.update_contacts
        self.observers: list[StepObserver] = list(observers)
        self.pre_substep: list[Callable[[newton.State], None]] = []
        self.periods = 0
        self._initial = {f: wp.clone(getattr(self.state, f)) for f in _STATE_FIELDS if _has(self.state, f)}
        self.health = HealthMonitor(model, HEALTH_POLICY, solvers=[solver], energy="off") if health else None
        self._health_initial = {k: wp.clone(a) for k, a in health_arrays(self.health, solver).items()}
        self.graph = graph  # None (auto), True or False; may be changed between periods
        self._auto_off = auto_graph_reason(solver, model.device, self.substeps, self.pipeline is not None)
        self._refused: str | None = None  # a refused capture (graph=None): eager from then on
        self._pairs: PeriodGraph | None = None  # the captured pair of substeps
        self._bound: tuple = ()  # what the capture read through Python references (``_binding``)
        self._warm = False  # an eager substep has run (first-step allocations / host reads stay outside capture)
        self._logged_reason: str | None = None

    def add_step_observer(self, fn: StepObserver) -> None:
        self.observers.append(fn)

    @property
    def graph_mode(self) -> str:
        """"graph" while periods replay the captured pair, else "eager"."""
        return "graph" if self._pairs is not None and self._pairs.mode == "graph" else "eager"

    @property
    def graph_captures(self) -> int:
        return 0 if self._pairs is None else self._pairs.captures

    def eager_reason(self) -> str | None:
        """Why the next period runs eagerly, or None when it replays the captured pair (module doc)."""
        if self.graph is False:
            return "graph=False"
        if self._refused is not None:
            return self._refused
        if self.graph is None and self._auto_off is not None:
            return self._auto_off
        if self.substeps % 2:
            return "one substep per period (a captured pair needs an even count)"
        if type(self).substep is not DirectSim.substep:
            return f"{type(self).__name__} overrides substep (the captured body is DirectSim's)"
        bad = next((fn for fn in (*self.pre_substep, *self.observers) if not is_graph_safe(fn)), None)
        return None if bad is None else f"{getattr(bad, '__qualname__', type(bad).__name__)} is not marked graph_safe"

    def route_period_reason(self) -> str | None:
        """Why the next period cannot run as one ``step_many`` call (module doc, CPU route fast path), or None."""
        if not callable(getattr(self.solver, "step_many", None)):
            return f"{type(self.solver).__name__} has no step_many"
        if self.pre_substep or self.observers:
            return "pre_substep hooks or step observers need every step"
        if self.pipeline is not None:
            return "Newton collision pipeline before each step"
        if type(self).substep is not DirectSim.substep:
            return f"{type(self).__name__} overrides substep"
        return None

    def step_period(self) -> None:
        if self.route_period_reason() is None:
            self._route_period()
        elif (why := self.eager_reason()) is None:
            self._replay_period()
        else:
            if why != self._logged_reason:
                log.info("DirectSim periods run eagerly: %s", why)
                self._logged_reason = why
            for _ in range(self.substeps):
                self.substep()
        if self.health is not None:
            self.health.update(self.periods, self._next, self.state, self.contacts, dt=self.control_dt)
        self.periods += 1

    def _route_period(self) -> None:
        """The period as one ``step_many`` call (module doc, CPU route fast path)."""
        self.state.clear_forces()
        self.solver.step_many(self.state, self._next, self.control, self.dt, self.substeps)
        if self._export is not None:
            self._export(self.contacts, self._next)
        self.state, self._next = self._next, self.state
        self.clock.advance(self.substeps)
        self._warm = True

    def substep(self) -> None:
        """One collision + solver step of ``dt`` with the observers (no period bookkeeping), eagerly."""
        self._work()
        self.clock.advance(1)
        self._warm = True

    def _work(self) -> None:
        """One substep without the clock: the captured pair runs this twice."""
        self.state.clear_forces()
        for hook in self.pre_substep:
            hook(self.state)
        if self.pipeline is not None:
            self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self._next, self.control, None if self._export else self.contacts, self.dt)
        if self._export is not None:
            self._export(self.contacts, self._next)
        for fn in self.observers:
            fn(self.state, self._next, self.control, self.contacts, self.dt)
        self.state, self._next = self._next, self.state

    def _replay_period(self) -> None:
        pairs = self.substeps // 2
        if not self._warm:  # nothing has stepped yet: one eager pair first
            self.substep()
            self.substep()
            pairs -= 1
        graph = self._pair_graph()
        for _ in range(pairs):
            if self._refused is not None:  # a refused capture: the rest of the period runs eagerly
                self.substep()
                self.substep()
                continue
            self._launch_pair(graph)
            self.clock.advance(2)

    def _pair_graph(self) -> PeriodGraph:
        """The pair graph for the current binding: a new state, control, solver, hook or observer recaptures."""
        if self._pairs is None:
            self._pairs = PeriodGraph(self._pair, str(self.model.device), use_graph=True, buffers=self._buffers)
        bound = self._binding()
        if len(bound) != len(self._bound) or any(a is not b for a, b in zip(bound, self._bound)):
            self._pairs.invalidate()
            self._bound = bound
        return self._pairs

    def _binding(self) -> tuple:
        return (self.state, self._next, self.control, self.contacts, self.solver, *self.pre_substep, *self.observers)

    def _pair(self) -> None:
        self._work()
        self._work()

    def _buffers(self) -> list[wp.array]:
        return [getattr(s, f) for s in (self.state, self._next) for f in _STATE_FIELDS if _has(s, f)]

    def _launch_pair(self, graph: PeriodGraph) -> None:
        """Replay the pair (capturing it first if needed). With ``graph=None`` a refused capture turns graphs off and
        runs the pair eagerly: a CPU capture records without executing, so nothing ran; ``graph=True`` raises."""
        s0, s1 = self.state, self._next
        try:
            graph.launch()
        except GraphCaptureError as exc:
            self.state, self._next = s0, s1
            if self.graph:
                raise
            self._refused = f"capture refused: {exc}"
            log.warning("DirectSim: %s; periods run eagerly from here on", self._refused)
            self._pairs = None
            self._pair()

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Restore the initial state (all worlds, or those with ``world_mask[w] != 0``) and reset the solver. A
        captured pair is dropped (the next period recaptures)."""
        if self._pairs is not None:
            self._pairs.invalidate()
        mask = None if world_mask is None else np.asarray(world_mask.numpy() if hasattr(world_mask, "numpy")
                                                          else world_mask).astype(bool)
        for name, init in self._initial.items():
            _restore(self.state, self.model, name, init, mask)
        live = health_arrays(self.health, self.solver)
        for name, init in self._health_initial.items():
            restore_worlds(live[name], init, mask)
        bool_mask = None
        if mask is not None:
            full = np.zeros(self.model.world_count + 1, dtype=bool)
            full[: mask.size] = mask
            bool_mask = wp.array(full, dtype=wp.bool, device=self.model.device)
        self.solver.reset(self.state, world_mask=bool_mask)
        if mask is None:
            self.periods = 0
            self.clock.ticks = 0

    def report(self) -> Mapping[str, Any]:
        return {"health": self.health.poll() if self.health is not None else None, "periods": self.periods,
                "t": self.clock.t}

    def state_arrays(self) -> Mapping[str, wp.array]:
        """The plug-in's ``alloy_state_arrays`` or the registered state adapter of a stock solver."""
        own = getattr(self.solver, "alloy_state_arrays", None)
        return dict(own(self.state)) if callable(own) else dict(state_arrays_for(self.solver))

    def state_world_maps(self) -> Mapping[str, wp.array]:
        maps = getattr(self.solver, "alloy_state_worlds", None)
        return dict(maps()) if callable(maps) else {}


def _has(state: newton.State, name: str) -> bool:
    arr = getattr(state, name, None)
    return arr is not None and arr.shape[0] > 0


def health_arrays(health: Any, solver: Any) -> dict[str, wp.array]:
    """Per-world [W] arrays a new episode starts from: the health records and the solver guards' cumulative fallback
    counters (the MuJoCo Warp pivot-clamp count, which HealthMonitor reads as FALLBACK_USED on every later check)."""
    if health is None:
        return {}
    out = dict(health.state_arrays())
    data = getattr(solver, "mjw_data", None)
    if data is not None:
        from alloy.precision import mjw_chol  # noqa: PLC0415  (needs mujoco_warp, as SolverMuJoCo does)

        clamps = mjw_chol.clamp_counts(data)
        if clamps is not None:
            out["guard.newton_cholesky_clamps"] = clamps
    return out


def restore_worlds(dst: wp.array, init: wp.array, mask: np.ndarray | None) -> None:
    """Copy ``init`` into the masked worlds of a world-major [W] array (all worlds if ``mask`` is None)."""
    if mask is None:
        wp.copy(dst, init)
        return
    host = dst.numpy()
    sel = np.zeros(host.shape[0], dtype=bool)
    sel[: mask.size] = mask[: host.shape[0]]
    host[sel] = init.numpy()[sel]
    dst.assign(host)


def _restore(state: newton.State, model: newton.Model, name: str, init: wp.array, mask: np.ndarray | None) -> None:
    dst = getattr(state, name)
    if mask is None:
        wp.copy(dst, init)
        return
    world_map = _ITEM_WORLD[name](model)
    if world_map is None:
        log.warning("DirectSim.reset: %s has no per-world index; restored for all worlds", name)
        wp.copy(dst, init)
        return
    item_world = world_map.numpy()
    sel = np.isin(item_world, np.nonzero(mask)[0])
    host = dst.numpy()
    host[sel] = init.numpy()[sel]
    dst.assign(host)
