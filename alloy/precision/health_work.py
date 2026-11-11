# SPDX-License-Identifier: Apache-2.0
"""External work of applied forces, booked in health's energy check (D20; fresh-agent friction item F4, 2026-11-07).

A force plug-in (wind, a magnet, suction) does work on the bodies it pushes. That energy comes from outside the
simulated system, like an actuator's: it is external work W_ext, not a numerical jump. Unbooked, a 10 m/s wind on a
10 g ball read as ENERGY_JUMP at t = 0.34 s (E_scale ~ 1 mJ) and froze the world. ``ForceWork`` measures it per world
for one step, exactly for a force held over the step (as a plug-in's is):

    W = sum over bodies of F . (c1 - c0) + T . (w0 + w1) / 2 * dt  +  sum over particles of f . (x1 - x0)

with c the centre of mass in world (where Newton applies ``body_f``'s force) and w the angular velocity. The start
state is copied in ``begin`` (before the solver step), so a solver that rewrites its input state does not matter.

A wrench ``Sim.apply_wrench`` holds is the same kind of outside work (EASE-TEST red, 2026-11-09: a held 2 N push on a
0.51 kg ball read as ENERGY_JUMP after about 0.12 J and froze the world). ``WrenchWork`` meters it the same way: its
own copy of the held wrenches (not ``body_f``, which other parts of a step also write), measured around each
``stepper.step`` from the first ``apply_wrench`` on.

A component that applies external forces registers its per-world work array on the model (``register_work``);
``HealthMonitor.update`` books every registered array into W_cum each step and zeroes it (``book_work``), check step
or not. Each world's items are summed in fixed chunks (``chunk_sums``, as health's energy is): never a float
atomic, so it is the same every run, and a world of <= 128 items is one serial sum. The
energy check stays on for the solver's own energy errors; only the forces' work is taken out.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy.precision.chunk_sums import Chunks, chunk_count, chunk_sum, make_chunks
from alloy.precision.health_kernels import EnergyRef

log = logging.getLogger("alloy.precision")

WORK_ATTR = "alloy_external_work"       # model attribute: list of float64 [world_count] arrays, J per step
WRENCH_ATTR = "_alloy_wrench_work"      # stepper / engine attribute: its WrenchWork (Sim.apply_wrench)


@wp.kernel
def _work_chunks(ch: Chunks, body_start: wp.array[wp.int32], bodies: wp.array[wp.int32],
                 body_f: wp.array[wp.spatial_vector], q0: wp.array[wp.transform], q1: wp.array[wp.transform],
                 qd0: wp.array[wp.spatial_vector], qd1: wp.array[wp.spatial_vector], com: wp.array[wp.vec3],
                 part_start: wp.array[wp.int32], parts: wp.array[wp.int32], particle_f: wp.array[wp.vec3],
                 x0: wp.array[wp.vec3], x1: wp.array[wp.vec3], dt: float, partial: wp.array[wp.float64]):
    """One chunk of a world's bodies-then-particles sequence, in item order (``chunk_sums``; no float atomics)."""
    c = wp.tid()
    w = ch.world[c]
    lo, hi = ch.lo[c], ch.hi[c]
    nb = body_start[w + 1] - body_start[w]
    total = wp.float64(0.0)
    for p in range(lo, wp.min(hi, nb)):
        b = bodies[body_start[w] + p]
        f = body_f[b]
        force = wp.vec3(f[0], f[1], f[2])
        torque = wp.vec3(f[3], f[4], f[5])
        if wp.length_sq(force) > 0.0 or wp.length_sq(torque) > 0.0:
            dc = wp.transform_point(q1[b], com[b]) - wp.transform_point(q0[b], com[b])
            a0, a1 = qd0[b], qd1[b]
            omega = 0.5 * (wp.vec3(a0[3], a0[4], a0[5]) + wp.vec3(a1[3], a1[4], a1[5]))
            total += wp.float64(wp.dot(force, dc)) + wp.float64(wp.dot(torque, omega) * dt)
    for p in range(wp.max(lo, nb), hi):
        i = parts[part_start[w] + p - nb]
        total += wp.float64(wp.dot(particle_f[i], x1[i] - x0[i]))
    partial[c] = total


@wp.kernel
def _work_sum(ch: Chunks, partial: wp.array[wp.float64], out: wp.array[wp.float64]):
    """A world's work: its chunks in chunk order (the same bits every run; batch-invariant)."""
    w = wp.tid()
    out[w] = chunk_sum(ch, partial, w)


@wp.kernel
def _book(work: wp.array[wp.float64], ref: EnergyRef):
    """W_cum += work, and the work is used up (a host that stops stepping cannot be booked twice)."""
    w = wp.tid()
    ref.w_cum[w] = ref.w_cum[w] + work[w]
    work[w] = wp.float64(0.0)


def _by_world(world: np.ndarray, worlds: int, dev: Any) -> tuple[wp.array, wp.array]:
    """(start [W + 1], items in index order) of the items in each world. Global items (world -1) are left out, as
    health's energy leaves them out (``health_setup.energy_model``)."""
    world = np.asarray(world, np.int64)
    keep = np.flatnonzero(world >= 0)
    items = keep[np.argsort(world[keep], kind="stable")]
    start = np.concatenate([[0], np.cumsum(np.bincount(world[keep], minlength=worlds))])
    return (wp.array(start.astype(np.int32), dtype=wp.int32, device=dev),
            wp.array(items.astype(np.int32), dtype=wp.int32, device=dev))


class ForceWork:
    """Per-world work [J] of a set of applied forces over one step. ``begin(state)`` before the solver step,
    ``end(body_f, particle_f, state_out, dt)`` after it; ``out`` holds the step's work (kernels and copies only, so
    it records in a CUDA graph; fixed-order chunk sums, so it is the same every run). Registered on the model for
    health (``register_work``), which uses it up each step (``book_work``)."""

    def __init__(self, model: Any, *, particles: bool = True):
        self.model, dev = model, model.device
        self.worlds = max(int(model.world_count), 1)
        self.out = wp.zeros(self.worlds, dtype=wp.float64, device=dev)
        nb = int(model.body_count)
        n_p = int(model.particle_count) if particles else 0   # False: body forces only (no particle copy per step)
        self._particles = n_p > 0
        bw = model.body_world.numpy() if nb else np.zeros(0, np.int64)
        pw = model.particle_world.numpy() if n_p else np.zeros(0, np.int64)
        self._bodies = _by_world(bw, self.worlds, dev)
        self._parts = _by_world(pw, self.worlds, dev)
        counts = np.diff(self._bodies[0].numpy()) + np.diff(self._parts[0].numpy())
        self._chunks = make_chunks(counts, dev)
        self._partial = wp.zeros(max(chunk_count(self._chunks), 1), dtype=wp.float64, device=dev)   # one per owner
        self._q0 = wp.zeros(nb, dtype=wp.transform, device=dev)
        self._qd0 = wp.zeros(nb, dtype=wp.spatial_vector, device=dev)
        self._x0 = wp.zeros(n_p, dtype=wp.vec3, device=dev)
        self._no_f = (wp.zeros(nb, dtype=wp.spatial_vector, device=dev), wp.zeros(n_p, dtype=wp.vec3, device=dev))
        self._com = model.body_com if nb else wp.zeros(0, dtype=wp.vec3, device=dev)
        register_work(model, self.out)

    def begin(self, state: Any) -> None:
        if self.model.body_count and state.body_q is not None:
            wp.copy(self._q0, state.body_q)
            wp.copy(self._qd0, state.body_qd)
        if self._particles and state.particle_q is not None:
            wp.copy(self._x0, state.particle_q)

    def end(self, body_f: wp.array | None, particle_f: wp.array | None, state_out: Any, dt: float) -> wp.array:
        m = self.model
        bq = state_out.body_q if m.body_count and state_out.body_q is not None else self._q0
        bqd = state_out.body_qd if m.body_count and state_out.body_qd is not None else self._qd0
        px = state_out.particle_q if self._particles and state_out.particle_q is not None else self._x0
        bf = body_f if body_f is not None and m.body_count else self._no_f[0]
        pf = particle_f if particle_f is not None and self._particles else self._no_f[1]
        n = chunk_count(self._chunks)
        if n:
            wp.launch(_work_chunks, dim=n, inputs=[self._chunks, *self._bodies, bf, self._q0, bq, self._qd0, bqd,
                                                   self._com, *self._parts, pf, self._x0, px, float(dt),
                                                   self._partial], device=m.device)
        wp.launch(_work_sum, dim=self.worlds, inputs=[self._chunks, self._partial, self.out], device=m.device)
        return self.out


class WrenchWork:
    """The work of the wrenches ``Sim.apply_wrench`` holds, booked by health as a force plug-in's is (module
    docstring). ``body_f`` [body_count] is a copy of the held wrenches only; ``install`` measures it around every
    ``stepper.step`` (kernels and copies, so a recaptured graph records it: the caller drops its graphs)."""

    def __init__(self, model: Any):
        self.meter = ForceWork(model, particles=False)       # registers its out with health (``register_work``)
        self._host = np.zeros((int(model.body_count), 6), np.float32)
        self.body_f = wp.zeros(int(model.body_count), dtype=wp.spatial_vector, device=model.device)

    def hold(self, rows: np.ndarray, force: np.ndarray, torque: np.ndarray) -> None:
        """Body ``rows`` now hold ``force`` / ``torque`` ((len(rows), 3) or (3,), world frame, about the COM)."""
        self._host[rows, :3], self._host[rows, 3:6] = force, torque
        self.body_f.assign(self._host)

    def clear(self) -> None:
        """No wrench held (``sim.reset``)."""
        self._host[:] = 0.0
        self.body_f.zero_()

    def install(self, stepper: Any) -> None:
        """Measure the held wrenches' work around each ``stepper.step`` (wraps this instance's ``step`` once)."""
        if getattr(stepper, WRENCH_ATTR, None) is not None:
            raise ValueError("this stepper already books a WrenchWork: keep one per simulation")
        inner, meter, body_f = stepper.step, self.meter, self.body_f

        def step(state_in, state_out, control, contacts, dt, *args, **kwargs):
            meter.begin(state_in)
            inner(state_in, state_out, control, contacts, dt, *args, **kwargs)
            meter.end(body_f, None, state_out, dt)

        stepper.step = step
        setattr(stepper, WRENCH_ATTR, self)
        log.info("health books held wrenches' work as external work on %s", type(stepper).__name__)


def wrench_work(engine: Any) -> tuple[WrenchWork | None, bool]:
    """(the engine's WrenchWork, whether it was made now). Made on the first call for an engine with a health
    monitor; None without one (nothing books work). A new one changes the step: the caller drops captured graphs."""
    ww = getattr(engine, WRENCH_ATTR, None)
    if ww is not None or getattr(engine, "health", None) is None:
        return ww, False
    ww = WrenchWork(engine.model)
    ww.install(engine.stepper)
    setattr(engine, WRENCH_ATTR, ww)
    return ww, True


def register_work(model: Any, work: wp.array) -> None:
    """Have health book ``work`` (float64 [world_count], J over the last step) as external work each step."""
    sources = getattr(model, WORK_ATTR, None)
    if sources is None:
        sources = []
        setattr(model, WORK_ATTR, sources)
    if not any(s is work for s in sources):
        sources.append(work)


def book_work(model: Any, ref: Any) -> None:
    """Add every registered work array to the energy reference's W_cum and zero it (``HealthMonitor.update``, every
    step: a step's work is booked once)."""
    for work in getattr(model, WORK_ATTR, ()):
        wp.launch(_book, dim=work.shape[0], inputs=[work, ref], device=work.device)
