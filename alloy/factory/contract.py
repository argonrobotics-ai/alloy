# SPDX-License-Identifier: Apache-2.0
"""Plug-in contract checks (SPEC §3.8 items 1-6) for ``evaluate_plugin``.

Static: base classes, ``AlloySolverPlugin`` and ``StabilityEstimator`` members, a registered ``SolverCapability``,
namespaced custom attributes. Dynamic, on the plug-in's own B0 scene (``BullyTarget.make_sim``) with 2 worlds:
no allocation and no host readback inside ``step`` (counted by wrapping Warp's allocators and ``array.numpy``;
on CUDA the step is also graph-captured), per-world ``reset`` (world 1 bitwise untouched), repeat (R1: the same
steps from the same state give bitwise equal ``alloy_state_arrays``), state-array sufficiency (restoring the listed
arrays reproduces the next step), estimator outputs finite and positive, capacity report typed.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.bully.types import BullyTarget
from alloy.core.newton_compat import CouplingInterface
from alloy.core.protocols import DetCapability
from alloy.core.registry import registered_solvers, solver

log = logging.getLogger("alloy.factory")

_ALLOCATORS = ("empty", "zeros", "ones", "full", "clone", "empty_like", "zeros_like", "ones_like", "full_like")
_PLUGIN_MEMBERS = ("register_custom_attributes", "emit", "alloy_state_arrays", "alloy_state_worlds",
                   "alloy_capacity_report", "alloy_stable_dt", "alloy_energy", "step", "reset")
STEPS = 8


@dataclass
class ContractReport:
    """Each check name -> ``None`` (passed) or the failure text."""

    checks: dict[str, str | None] = field(default_factory=dict)

    def record(self, name: str, failure: str | None) -> None:
        self.checks[name] = failure
        if failure:
            log.warning("contract %s: %s", name, failure)

    @property
    def ok(self) -> bool:
        return all(v is None for v in self.checks.values())

    @property
    def failures(self) -> list[str]:
        return [f"{k}: {v}" for k, v in self.checks.items() if v]


def static_checks(cls: type, rep: ContractReport) -> None:
    """Items 1-2: classes, members, registered capability, namespaced attributes."""
    rep.record("bases", None if issubclass(cls, SolverBase) and issubclass(cls, CouplingInterface)
               else f"{cls.__name__} must subclass SolverBase and CouplingInterface")
    missing = [m for m in _PLUGIN_MEMBERS if not callable(getattr(cls, m, None))]
    rep.record("members", f"missing {missing}" if missing else None)
    name = getattr(cls, "plugin_name", "")
    ok_attrs = isinstance(name, str) and name and isinstance(getattr(cls, "routes", None), frozenset) \
        and isinstance(getattr(cls, "determinism", None), DetCapability)
    rep.record("class_attributes", None if ok_attrs else "plugin_name (str), routes (frozenset), determinism")
    if not name:
        return
    cap = solver(name) if name in registered_solvers() else None
    rep.record("capability", None if cap is not None and cap.determinism == cls.determinism
               else f"no SolverCapability {name!r} registered on import (or its determinism differs)")
    if not callable(getattr(cls, "register_custom_attributes", None)):
        return  # already a "members" failure
    b = newton.ModelBuilder()
    before = set(getattr(b, "custom_attributes", {}))  # Newton pre-registers e.g. mujoco:* attributes
    cls.register_custom_attributes(b)
    added = set(getattr(b, "custom_attributes", {})) - before
    stray = sorted(str(k) for k in added if not str(k).startswith(f"{name}:"))
    rep.record("attribute_namespace", f"attributes outside namespace {name!r}: {stray}" if stray else None)


@contextlib.contextmanager
def _count_calls(counts: dict[str, int]) -> Iterator[None]:
    """Count Warp allocations and host readbacks while the block runs."""
    saved: dict[tuple[object, str], Callable] = {}

    def wrap(owner: object, attr: str, key: str) -> None:
        fn = getattr(owner, attr)
        saved[(owner, attr)] = fn

        def counted(*a, **k):
            counts[key] = counts.get(key, 0) + 1
            return fn(*a, **k)

        setattr(owner, attr, counted)

    for name in _ALLOCATORS:
        wrap(wp, name, "alloc")
    wrap(wp.array, "numpy", "readback")
    wrap(wp, "synchronize", "readback")
    try:
        yield
    finally:
        for (owner, attr), fn in saved.items():
            setattr(owner, attr, fn)


def _state_copy(arrays: dict[str, wp.array]) -> dict[str, np.ndarray]:
    return {k: v.numpy().copy() for k, v in arrays.items()}


def _step(sim, n: int) -> None:
    for _ in range(n):
        sim.substep()


def dynamic_checks(cls: type, target: BullyTarget, device: str, rep: ContractReport) -> None:
    """Items 3-6 on the plug-in's B0 scene with two worlds."""
    sim = target.make_sim(2, device)
    solver_obj = sim.solver
    if not isinstance(solver_obj, cls):
        rep.record("scene_solver", f"{target.name} does not step a {cls.__name__}")
        return
    _check_no_alloc(sim, rep, device)
    _check_estimators(sim, rep)
    _check_reset(sim, rep)
    _check_repeat_and_arrays(sim, rep)


def _check_no_alloc(sim, rep: ContractReport, device: str) -> None:
    _step(sim, 1)  # first step may build kernels and lazy buffers
    counts: dict[str, int] = {}
    with _count_calls(counts):
        _step(sim, STEPS)
    rep.record("no_alloc_in_step", f"{counts['alloc']} allocations in {STEPS} steps" if counts.get("alloc") else None)
    rep.record("no_readback_in_step", f"{counts['readback']} host readbacks" if counts.get("readback") else None)
    if wp.get_device(device).is_cuda:
        try:
            with wp.ScopedCapture(device=device):
                _step(sim, 1)
            rep.record("graph_capture", None)
        except Exception as exc:  # noqa: BLE001 - any capture failure is the finding
            rep.record("graph_capture", f"step is not capturable: {exc}")


def _check_estimators(sim, rep: ContractReport) -> None:
    m, s = sim.model, sim.state
    dt_out = wp.zeros(m.world_count, dtype=wp.float32, device=m.device)
    e_out = wp.zeros(m.world_count, dtype=wp.float64, device=m.device)
    sim.solver.alloy_stable_dt(s, sim.contacts, dt_out)
    sim.solver.alloy_energy(s, wp.zeros(m.world_count, dtype=wp.float64, device=m.device), e_out)
    dts, es = dt_out.numpy(), e_out.numpy()
    ok = np.all(np.isfinite(dts)) and np.all(dts > 0) and np.all(np.isfinite(es))
    rep.record("stability_estimator", None if ok else f"stable_dt {dts}, energy {es}")
    arrays, maps = dict(sim.solver.alloy_state_arrays(s)), dict(sim.solver.alloy_state_worlds())
    bad = [k for k, v in maps.items() if k not in arrays or v.shape[0] != arrays[k].shape[0]]
    rep.record("state_world_maps", f"world maps that match no state array: {bad}" if bad else None)
    cap = sim.solver.alloy_capacity_report()
    typed = all(isinstance(k, str) and isinstance(v, int | np.integer) for k, v in dict(cap).items())
    rep.record("capacity_report", None if typed else f"not Mapping[str, int]: {cap!r}")


_STATE = ("particle_q", "particle_qd", "body_q", "body_qd")


def _live(sim) -> dict[str, wp.array]:
    """Everything the next step reads that the contract can see: public State arrays plus ``alloy_state_arrays``."""
    out = {f"state.{n}": getattr(sim.state, n) for n in _STATE
           if getattr(sim.state, n, None) is not None and getattr(sim.state, n).shape[0]}
    out.update({f"plugin.{k}": v for k, v in dict(sim.solver.alloy_state_arrays(sim.state)).items()})
    return out


def _world_maps(sim, keys: list[str]) -> dict[str, np.ndarray]:
    """Item->world map per snapshot key (State arrays by their Model map, plug-in arrays by ``alloy_state_worlds``)."""
    m = sim.model
    maps = {f"plugin.{k}": v.numpy() for k, v in dict(sim.solver.alloy_state_worlds()).items()}
    for n in _STATE:
        src = m.particle_world if n.startswith("particle") else m.body_world
        if src is not None:
            maps[f"state.{n}"] = src.numpy()
    return {k: maps[k] for k in keys if k in maps}


def _run_from_reset(sim, steps: int) -> dict[str, np.ndarray]:
    sim.reset()
    _step(sim, steps)
    return _state_copy(_live(sim))


def _check_reset(sim, rep: ContractReport) -> None:
    """Trajectory test: after ``reset(world 0)`` mid-run, world 1 continues bitwise as if nothing happened and world
    0 retraces a fresh run bitwise (catches hidden state that reset misses or that leaks across worlds)."""
    sim.reset()
    _step(sim, STEPS)
    fresh_mid = _state_copy(_live(sim))
    _step(sim, STEPS)
    undisturbed = _state_copy(_live(sim))
    sim.reset()
    _step(sim, STEPS)
    mask = np.zeros(sim.model.world_count + 1, dtype=bool)
    mask[0] = True
    sim.solver.reset(sim.state, wp.array(mask, dtype=wp.bool, device=sim.model.device))
    _step(sim, STEPS)
    after = _state_copy(_live(sim))
    maps = _world_maps(sim, list(after))
    if not maps:
        rep.record("per_world_reset", "no per-world state to compare (alloy_state_worlds and State are empty)")
        return
    leak = [k for k, w in maps.items() if not np.array_equal(after[k][w == 1], undisturbed[k][w == 1])]
    stale = [k for k, w in maps.items() if not np.array_equal(after[k][w == 0], fresh_mid[k][w == 0])]
    fail = (f"reset of world 0 changed world 1's trajectory in {leak}; " if leak else "") + \
        (f"world 0 after reset does not retrace a fresh run in {stale}" if stale else "")
    rep.record("per_world_reset", fail or None)


def _check_repeat_and_arrays(sim, rep: ContractReport) -> None:
    a, b = _run_from_reset(sim, STEPS), _run_from_reset(sim, STEPS)
    rep.record("repeat_r1", None if a and all(np.array_equal(a[k], b[k]) for k in a)
               else "two runs from the reset state differ (not bitwise) or nothing to compare")
    snap = {k: wp.clone(v) for k, v in _live(sim).items()}
    _step(sim, 1)
    first = _state_copy(_live(sim))
    live = _live(sim)  # the solver may have swapped State buffers: restore into the current ones
    for k, v in snap.items():
        if k in live:
            wp.copy(live[k], v)
    _step(sim, 1)
    second = _state_copy(_live(sim))
    enough = bool(first) and all(np.array_equal(first[k], second[k]) for k in first)
    rep.record("state_arrays_sufficient", None if enough else
               "restoring State + alloy_state_arrays does not reproduce a step (hidden state outside them)")
