# SPDX-License-Identifier: Apache-2.0
"""SPH ports: calls that other stages make on a ``SolverDFSPH`` between its steps (surface-flow drops, heat).

Every call is a few fixed-size kernel launches with no host work, so they run inside captured steps. Each loop runs one
thread per world, in particle index order, so the results are deterministic and batch-independent (D9). Only the
int64 ledgers use atomics.

- ``absorb(state, mask)``: the masked liquid particles leave the liquid. They are held for ``emit`` (phase
  ``PHASE_HELD``, back on the world's stack) and booked out of SPH's mass and flux-energy ledgers. A film takes in
  their volume.
- ``emit(state, pos, vel, n)``: per world, up to ``n[w]`` held particles become liquid at the given world-frame
  positions and velocities. q/qd are written into ``state``: the full state, which coupled entries copy in at their
  step start. They are booked into the ledgers; ``granted[w]`` says how many.
- ``retire(state, mask)``: as ``absorb``, but the particles are gone for good (``PHASE_RETIRED``; WP7's heat ask).
- ``neighbor_count``: per particle, the neighbours within h at the last step's start; ``particles(state)``: the
  solver's particle struct (flags, phase, neighbour rows) for a stage's own kernels.

The held particles at build are the port's reservoir: spares made with phase ``PHASE_HELD``
(``build.add_held_spares``). Emitters never take them, because their pools are spares of phase ``PHASE_SPARE``.
Absorbed particles join the reservoir, so a film can shed and take in liquid for as long as it runs.
``reset`` restores the build-time reservoir.

A stage finds the port of a model's SPH solver with ``port_for(model)``. Plug-ins are set up before the solvers are
built, so they look it up at their first step.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from .config import PHASE_HELD, PHASE_LIQUID, PHASE_RETIRED, sph_phase
from .emitters import flux_add
from .kernels import to_fixed
from .pressure import Particles, is_fluid

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))
_HELD = wp.constant(PHASE_HELD)
_LIQUID = wp.constant(PHASE_LIQUID)
_RETIRED = wp.constant(PHASE_RETIRED)

_PORTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # root model -> [SPHPort]


def root_model(model: Any) -> Any:
    """The full model under a chain of Newton ``ModelView``s (the model itself when it is one)."""
    while getattr(model, "parent", None) is not None:
        model = model.parent
    return model


def port_for(model: Any) -> SPHPort | None:
    """The SPH port of ``model``'s SPH solver (a view or the full model), or None without one.
    Example: ``port = port_for(sim.engine.model); port.particle_volume``."""
    ports = _PORTS.get(root_model(model), [])
    if len(ports) > 1:
        raise ValueError(f"{len(ports)} SPH solvers on this model: port_for cannot choose (pass the solver's .port)")
    return ports[0] if ports else None


@wp.kernel(enable_backward=False)
def _take_kernel(ws: wp.array[wp.int32], mask: wp.array[wp.int32], p: Particles, keep: int,
                 stack: wp.array2d[wp.int32], top: wp.array[wp.int32], taken: wp.array[wp.int32],
                 total: wp.array[wp.int32], mass_out: wp.array[wp.int64], gravity: wp.array[wp.vec3],
                 flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64]):
    """Per world: the masked liquid particles leave the liquid (keep: onto the stack as held; else retired)."""
    w = wp.tid()
    t = top[w]
    k = int(0)
    for i in range(ws[w], ws[w + 1]):
        if mask[i] == 0 or not is_fluid(p, i):
            continue
        x = p.q[i]
        v = p.qd[i]
        if wp.isfinite(x[0] + x[1] + x[2] + v[0] + v[1] + v[2]):
            flux_add(flux_e, flux_g, w, p.mass[i], v, x, gravity[w], -1.0)
        p.flags[i] = p.flags[i] & ~_ACTIVE
        p.qd[i] = wp.vec3(0.0, 0.0, 0.0)
        wp.atomic_add(mass_out, w, to_fixed(p.mass[i]))
        if keep != 0:
            p.phase[i] = wp.int8(_HELD)
            stack[w, t] = i
            t += 1
        else:
            p.phase[i] = wp.int8(_RETIRED)
        k += 1
    top[w] = t
    taken[w] = k
    total[w] = total[w] + k


@wp.kernel(enable_backward=False)
def _emit_kernel(pos: wp.array2d[wp.vec3], vel: wp.array2d[wp.vec3], n_req: wp.array[wp.int32], p: Particles,
                 stack: wp.array2d[wp.int32], top: wp.array[wp.int32], granted: wp.array[wp.int32],
                 total: wp.array[wp.int32], mass_in: wp.array[wp.int64], gravity: wp.array[wp.vec3],
                 flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64]):
    """Per world: pop held particles for the first n_req[w] requests (fewer when the reservoir runs out)."""
    w = wp.tid()
    t = top[w]
    k = int(0)
    n = wp.min(n_req[w], pos.shape[1])
    for r in range(n):
        if t == 0:
            break
        t -= 1
        i = stack[w, t]
        p.q[i] = pos[w, r]
        p.qd[i] = vel[w, r]
        p.flags[i] = p.flags[i] | _ACTIVE
        p.phase[i] = wp.int8(_LIQUID)
        wp.atomic_add(mass_in, w, to_fixed(p.mass[i]))
        flux_add(flux_e, flux_g, w, p.mass[i], p.qd[i], p.q[i], gravity[w], 1.0)
        k += 1
    top[w] = t
    granted[w] = k
    total[w] = total[w] + k


def _reservoir(phase0: np.ndarray, start: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """(stack [W, depth], top [W]) of the build-time held particles: popped in ascending id order."""
    W = len(start) - 1
    stack = np.zeros((max(W, 1), max(depth, 1)), np.int32)
    top = np.zeros(max(W, 1), np.int32)
    for w in range(W):
        ids = np.flatnonzero(phase0[start[w]:start[w + 1]] == PHASE_HELD) + start[w]
        stack[w, : len(ids)] = ids[::-1]
        top[w] = len(ids)
    return stack, top


class SPHPort:
    """The port of one ``SolverDFSPH`` (module docstring). ``solver.port``; ``port_for(model)`` finds it."""

    def __init__(self, solver: Any):
        lay, dev = solver.layout, solver.device
        self.solver = solver
        start = np.asarray(lay.world_start, np.int64)
        W = lay.world_count
        depth = int(np.max(np.diff(start))) if W else 1
        stack0, top0 = _reservoir(lay.phase.numpy() if lay.n else np.zeros(0, np.int8), start, depth)
        self._stack0, self._top0 = stack0, top0
        self.ws = wp.array(start.astype(np.int32), dtype=wp.int32, device=dev)
        self.stack = wp.array(stack0, dtype=wp.int32, device=dev)
        self.top = wp.array(top0, dtype=wp.int32, device=dev)
        z = lambda: wp.zeros(max(W, 1), dtype=wp.int32, device=dev)  # noqa: E731
        self.taken, self.granted = z(), z()                 # the last call's per-world counts
        self.absorbed, self.emitted, self.retired = z(), z(), z()   # cumulative since build / reset
        self.reservoir = int(top0.sum())
        rho = lay.rest_density.numpy() if lay.n else np.zeros(0, np.float32)
        mass = solver.model.particle_mass.numpy() if lay.n else np.zeros(0, np.float32)
        sph = sph_phase(lay.phase.numpy()) if lay.n else np.zeros(0, bool)
        self.particle_volume = float(np.median(mass[sph] / rho[sph])) if sph.any() else 0.0
        self.rest_density = float(np.median(rho[sph])) if sph.any() else 0.0
        _PORTS.setdefault(root_model(solver.model), []).append(self)

    @property
    def neighbor_count(self) -> wp.array:
        """[n] int32: neighbours within h at the last step's start (0 for particles outside the liquid)."""
        return self.solver._nbrs.count

    @property
    def world_count(self) -> int:
        return self.solver.layout.world_count

    def particles(self, state: Any) -> Particles:
        """The SPH particle struct on ``state``'s positions: flags, phase and world-local neighbour rows of the solver
        (``pressure.nbr_base`` / ``nbr_j`` decode a row; rows are from the last step's start)."""
        return self.solver._particles(state)

    def _take(self, state: Any, mask: wp.array, keep: bool, total: wp.array) -> None:
        s = self.solver
        st = s.stats
        wp.launch(_take_kernel, dim=self.world_count,
                  inputs=[self.ws, mask, self.particles(state), int(keep), self.stack, self.top, self.taken, total,
                          st.mass_out, s.model.gravity, st.flux_e, st.flux_g], device=s.device)
        s._mirror_flags()

    def absorb(self, state: Any, mask: wp.array) -> None:
        """Masked liquid particles (``mask [n]`` int32, non-zero) leave the liquid and are held for ``emit``;
        ``taken[w]`` counts them."""
        self._take(state, mask, True, self.absorbed)

    def retire(self, state: Any, mask: wp.array) -> None:
        """Masked liquid particles leave for good (phase retired); ``taken[w]`` counts them."""
        self._take(state, mask, False, self.retired)

    def emit(self, state: Any, pos: wp.array, vel: wp.array, n: wp.array) -> None:
        """Up to ``n[w]`` held particles per world become liquid at ``pos [W, R]`` with ``vel [W, R]`` (world frame,
        written into ``state``); ``granted[w]`` says how many (fewer when the reservoir is empty)."""
        s = self.solver
        st = s.stats
        wp.launch(_emit_kernel, dim=self.world_count,
                  inputs=[pos, vel, n, self.particles(state), self.stack, self.top, self.granted, self.emitted,
                          st.mass_in, s.model.gravity, st.flux_e, st.flux_g], device=s.device)
        s._mirror_flags()

    def state_arrays(self) -> dict[str, wp.array]:
        return {"sph.port_stack": self.stack, "sph.port_top": self.top, "sph.port_absorbed": self.absorbed,
                "sph.port_emitted": self.emitted, "sph.port_retired": self.retired}

    def reset(self, world_mask: wp.array | None) -> None:
        """The masked worlds' reservoir back to its build-time particles and counters to zero (host call)."""
        W = self.world_count
        mask = np.ones(W, bool) if world_mask is None else world_mask.numpy()[:W].astype(bool)
        stack, top = self.stack.numpy(), self.top.numpy()
        stack[mask], top[: W][mask] = self._stack0[mask], self._top0[: W][mask]
        self.stack.assign(stack)
        self.top.assign(top)
        for arr in (self.absorbed, self.emitted, self.retired, self.taken, self.granted):
            a = arr.numpy()
            a[: W][mask] = 0
            arr.assign(a)

    def counts(self) -> dict[str, np.ndarray]:
        """Host copies per world: held (reservoir now), absorbed, emitted, retired (cumulative)."""
        W = self.world_count
        return {"held": self.top.numpy()[:W].copy(), "absorbed": self.absorbed.numpy()[:W].copy(),
                "emitted": self.emitted.numpy()[:W].copy(), "retired": self.retired.numpy()[:W].copy()}
