# SPDX-License-Identifier: Apache-2.0
"""Run a test scene and record what the checks and laws read: a ``Trace`` per control period."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.base import Plugin
from alloy.plugins.harness.scene import TestScene, build_simulation

log = logging.getLogger("alloy.plugins")


def _host(arr: Any, shape: tuple[int, ...]) -> np.ndarray:
    return arr.numpy().copy() if arr is not None and arr.shape[0] > 0 else np.zeros(shape, np.float32)


@dataclass
class Trace:
    """Host copies at t = 0 and after every period. Index ``[k]`` is after ``k`` periods.

    ``body_q [K, B, 7]`` (x, y, z, qx, qy, qz, qw of the body frame), ``body_qd [K, B, 6]`` (Newton's twist),
    ``particle_q/particle_qd [K, N, 3]``, ``body_f [K, B, 6]`` (plug-in forces of the period's last exchange),
    ``applied_q [K, B, 7]`` (the body_q those forces were computed from),
    ``reads[k] = {label: {name: [W, ...]}}``.
    """

    t: list[float] = field(default_factory=list)
    body_q: list[np.ndarray] = field(default_factory=list)
    body_qd: list[np.ndarray] = field(default_factory=list)
    particle_q: list[np.ndarray] = field(default_factory=list)
    particle_qd: list[np.ndarray] = field(default_factory=list)
    body_f: list[np.ndarray] = field(default_factory=list)
    applied_q: list[np.ndarray] = field(default_factory=list)
    reads: list[dict[str, dict[str, np.ndarray]]] = field(default_factory=list)

    def record(self, sim: Any) -> None:
        """Append the current state of ``sim`` (host sync)."""
        m, s = sim.model, sim.state
        self.t.append(float(sim.clock.t))
        self.body_q.append(_host(s.body_q, (0, 7)))
        self.body_qd.append(_host(s.body_qd, (0, 6)))
        self.particle_q.append(_host(s.particle_q, (0, 3)))
        self.particle_qd.append(_host(s.particle_qd, (0, 3)))
        self.body_f.append(_host(sim.plugins.forces.body_f, (m.body_count, 6)))
        applied = sim.plugins.applied_q
        self.applied_q.append(_host(applied, (0, 7)) if applied is not None else self.body_q[-1])
        self.reads.append(sim.plugins.read())

    def stacked(self) -> Trace:
        """The same trace with numpy-stacked fields (``trace.body_q[k, b, i]``)."""
        out = Trace()
        for name in ("t", "body_q", "body_qd", "particle_q", "particle_qd", "body_f", "applied_q"):
            setattr(out, name, np.asarray(getattr(self, name)))
        out.reads = self.reads
        return out

    def plugin(self, label: str, name: str) -> np.ndarray:
        """``[K, W, ...]``: plug-in ``label``'s array ``name`` after every period."""
        return np.stack([r[label][name] for r in self.reads])


@dataclass
class Run:
    """One harness run: the simulation, its trace and how it ended (``error`` holds the traceback tail)."""

    scene: TestScene
    sim: Any = None
    trace: Trace = field(default_factory=Trace)
    seconds: float = 0.0
    exchanges: int = 0
    error: str | None = None
    plugin: Any = None          # the plug-in under test (this run's instance), or None (objects, solvers)

    @property
    def model(self) -> newton.Model:
        return self.sim.model

    @property
    def host(self) -> Any:
        return self.sim.plugins

    def bodies(self, label: str) -> np.ndarray:
        """Global body index of ``label`` per world (``[W]``). Example: ``run.trace.body_q[:, run.bodies("a")]``."""
        return self.host.ctx.bodies(label).numpy()

    def probe(self, body_q: np.ndarray, dt: float = 1.0e-3) -> np.ndarray:
        """``[B, 6]`` plug-in forces (force xyz, torque xyz per body) at poses ``body_q [B, 7]`` for every body of
        every world, with zero velocities, without stepping.
        Example: ``q = run.trace.body_q[0].copy(); q[b, 2] = 0.3; f = run.probe(q); f[b, 2]`` = vertical force."""
        state = self.sim.model.state()
        state.body_q.assign(np.ascontiguousarray(body_q, dtype=np.float32))
        host = self.host
        host.before(state, None, dt)
        out = host.forces.body_f.numpy().copy()
        host.restore_forces(state)
        return out

    def digests(self) -> np.ndarray:
        """``[W, 2]`` per-world digest of State plus every plug-in and solver state array."""
        from alloy.testing.integration import digests  # noqa: PLC0415

        return digests(self.sim)


def perturb_worlds(sim: Any, scale: float = 1.0e-3, seed: int = 7) -> None:
    """Add small deterministic noise to the joint and particle velocities of worlds 1.. (world 0 untouched)."""
    model, state, rng = sim.model, sim.state, np.random.default_rng(seed)
    if model.joint_dof_count > 0:
        starts, jw = model.joint_qd_start.numpy(), model.joint_world.numpy()
        dof_world = np.repeat(jw, np.diff(starts))
        qd = state.joint_qd.numpy()
        hit = dof_world >= 1
        qd[hit] += rng.normal(0.0, scale, int(hit.sum())).astype(qd.dtype)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    if model.particle_count > 0:
        pw = model.particle_world.numpy()
        v = state.particle_qd.numpy()
        hit = pw >= 1
        v[hit] += rng.normal(0.0, scale, (int(hit.sum()), 3)).astype(v.dtype)
        state.particle_qd.assign(v)


def run_scene(scene: TestScene, plugins: Sequence[Plugin], device: str, *, perturb: bool = False,
              record: bool = True, periods: int | None = None, worlds: int | None = None,
              graph: bool = False) -> Run:
    """Build ``scene`` with ``plugins`` and run it (strict hooks; ``graph``: captured periods); exceptions end the run
    and land in ``Run.error``."""
    run = Run(scene, plugin=plugins[0] if plugins else None)
    try:
        run.sim = build_simulation(scene, plugins, device, worlds, graph=graph)
        run.sim.plugins.strict = True
        run.sim.plugins.record_applied()
        if perturb:
            perturb_worlds(run.sim)
        if record:
            run.trace.record(run.sim)
        t0 = time.perf_counter()
        for _ in range(periods or scene.periods):
            run.sim.step_period()
            if record:
                run.trace.record(run.sim)
        run.seconds = time.perf_counter() - t0
        run.exchanges = int(run.sim.steps)
    except Exception:  # noqa: BLE001 - the crash check reports it
        run.error = traceback.format_exc()[-3000:]
        log.warning("harness run failed: %s", run.error.strip().splitlines()[-1])
    run.trace = run.trace.stacked()
    return run


def time_scene(scene: TestScene, plugins: Sequence[Plugin], device: str, periods: int) -> float:
    """Seconds per exchange of ``scene`` with ``plugins`` (best of two timed runs after one warm-up period)."""
    run = Run(scene)
    run.sim = build_simulation(scene, plugins, device)
    sim = run.sim
    sim.step_period()                         # compiles kernels
    best = float("inf")
    for _ in range(2):
        steps0, t0 = sim.steps, time.perf_counter()
        for _ in range(periods):
            sim.step_period()
        _sync(device)
        best = min(best, (time.perf_counter() - t0) / max(sim.steps - steps0, 1))
    return best


def _sync(device: str) -> None:
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(device)


__all__ = ["Run", "Trace", "perturb_worlds", "run_scene", "time_scene"]
