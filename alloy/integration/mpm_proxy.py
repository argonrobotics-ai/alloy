# SPDX-License-Identifier: Apache-2.0
"""MPM as the source of a particle proxy pair: the coupler's force reaches the MPM particles (SCR-WP12-MIX).

Newton's coupler returns the destination's force on each proxied particle in the source's ``state.particle_f``
(``SolverCoupled._add_particle_force_input``, copied into every substep's input state). ``SolverImplicitMPM`` never
reads ``particle_f``: its momentum comes from ``particle_qd`` and gravity on the grid, so an SPH liquid's push on an
MPM paste was dropped. For every MPM entry that is the source of a two-way pair into SPH, ``install`` wraps the
solver's ``step``: the step reads ``particle_qd + f dt / m`` (an external force on the grid, since the particle-to-grid
transfer is linear in velocity). The sum is written to a scratch array that stands in for ``state_in.particle_qd``
during the step, so the input state is never changed: a restart pass that steps again from it adds the force once.
Other MPM entries are untouched (bitwise as before).

Newton 1.8 (#4964, release-1.8 db915e1c) applies ``state_in.particle_f`` itself, as an acceleration term in
``integrate_velocity``. The wrapper hides ``particle_f`` from that step (``None`` while it runs), so the force lands
once, where it did before the bump: the SPH-MPM rows stay bitwise (CPU floating slab, own_cfl off: draft 1.0879,
force 1.0311 x weight on both Newtons). Newton's own term moves the slab's draft to 1.1008 and its force to 1.0162 (the
law's draft bound is 1.10), a physics change left for its own law and ruling after the release (launch-1110).

Restart passes. A staggered pair re-steps MPM from the exchange's start state on every pass, and the per-world gate
freezes a converged world, whose re-steps must reproduce its converging pass bitwise. Newton's MPM keeps its
rheology warm start (``_last_step_data``: the impulse and stress fields of the last solve) across calls, so a
re-step from the same state differed by 1e-7 m/s, and world 0 of two moved by 2.4e-5 m in 5 periods when world 1's
paste sat 4 mm aside. ``install`` also wraps ``coupling_notify_input_state_update``: the exchange's first state
notification saves those arrays, a restart's restores them (an array whose shape changed, a rebuilt sparse grid, is
left as is), so every pass starts from the same warm start.

``wall_policy``: such a pair's interface force is spread over thousands of particles of a few mN each, so the default
per-endpoint force gate (the max over particles) held most exchanges at the pass cap. The L2 norm (scene-wide: one
Scheduler gate for every pair, so a robot's lagged pair beside it gates on it too) over a world's
endpoints (``InterfaceTolerance.force_norm = "l2"``, WP11's option for SPH-VBD P2) took 2.48 passes per exchange
instead of 4.12 (cap 8) and 41 % less time (a 1.6 cm paste cube on 2.4 cm of water, CPU, 0.6 s); a higher Aitken
start (omega0 0.5) changed nothing.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy._vendor.newton import ParticleFlags, StateFlags
from alloy.coupling.adapters import mpm_state_arrays
from alloy.coupling.types import SchedulerPolicy

log = logging.getLogger("alloy.integration")

__all__ = ["install", "install_for_pairs", "wall_policy"]

_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.kernel(enable_backward=False)
def _dv_kernel(qd: wp.array[wp.vec3], f: wp.array[wp.vec3], mass: wp.array[wp.float32], flags: wp.array[wp.int32],
               dt: float, out: wp.array[wp.vec3]):
    i = wp.tid()
    v = qd[i]
    m = mass[i]
    if (flags[i] & _ACTIVE) != 0 and m > 0.0:
        v = v + f[i] * (dt / m)
    out[i] = v


def install(solver: Any) -> bool:
    """Wrap ``solver.step`` (module doc); False when already wrapped or the solver has no particles."""
    if getattr(solver, "_alloy_proxy_force", False):
        return False
    model = solver.model
    n = int(model.particle_count or 0)
    if n == 0:
        return False
    scratch = wp.zeros(n, dtype=wp.vec3, device=model.device)
    prev = solver.step

    def step(state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, **kw: Any) -> None:
        f = getattr(state_in, "particle_f", None)
        if f is None:
            prev(state_in, state_out, control, contacts, dt, **kw)
            return
        wp.launch(_dv_kernel, dim=n, inputs=[state_in.particle_qd, f, model.particle_mass, model.particle_flags,
                                             float(dt)], outputs=[scratch], device=model.device)
        state_in.particle_f = None                   # Newton 1.8 would apply it a second time (module doc)
        try:
            if state_in is state_out:   # in place: the force lands in the input; the coupler re-distributes every pass
                wp.copy(state_in.particle_qd, scratch)
                prev(state_in, state_out, control, contacts, dt, **kw)
                return
            keep = state_in.particle_qd
            state_in.particle_qd = scratch
            try:
                prev(state_in, state_out, control, contacts, dt, **kw)
            finally:
                state_in.particle_qd = keep
        finally:
            state_in.particle_f = f

    solver.step = step
    solver.coupling_notify_input_state_update = _repeatable_notify(solver)
    solver._alloy_proxy_force = True
    return True


def _repeatable_notify(solver: Any) -> Any:
    """``coupling_notify_input_state_update`` that saves the MPM warm start at an exchange's first state notification
    and restores it on a restart pass (module doc)."""
    prev = solver.coupling_notify_input_state_update
    saved: dict[str, wp.array] = {}
    q_flag = int(StateFlags.PARTICLE_Q)
    warned = []

    def notify(state: Any, flags: int, *, iteration_restart: bool = False, dt: float = 0.0) -> None:
        if int(flags) & q_flag:
            for name, arr in mpm_state_arrays(solver).items():
                keep = saved.get(name)
                if iteration_restart:
                    if keep is not None and keep.shape == arr.shape:
                        wp.copy(arr, keep)
                    continue
                if keep is None or keep.shape != arr.shape:
                    if wp.get_device(arr.device).is_capturing:   # a grid grown inside a capture: no allocation
                        saved.pop(name, None)
                        if not warned:
                            warned.append(name)
                            log.info("MPM warm start %s changed shape inside a graph capture: its restart passes keep "
                                     "Newton's own warm start for that exchange", name)
                        continue
                    keep = saved[name] = wp.empty_like(arr)
                wp.copy(keep, arr)
        prev(state, flags, iteration_restart=iteration_restart, dt=dt)

    return notify


def install_for_pairs(built: Any, solvers: Mapping[str, Any]) -> list[str]:
    """``install`` on every MPM entry that is the source of a two-way pair into an SPH entry; one note each. Also the
    proxy body carry of MPM destinations (``mpm_bodies``), the pair clock's route note (``mpm_clock``), the
    many-world warning for MPM beside a moving body (``mpm_moving``) and the projection out of colliders after every
    MPM step (``mpm_project``)."""
    from alloy.integration import mpm_bodies, mpm_clock, mpm_moving, mpm_project  # noqa: PLC0415
    from alloy.integration.mpm_solve import attach_notes  # noqa: PLC0415

    notes = mpm_bodies.install_for_pairs(built, solvers) + mpm_moving.warn(built)
    notes += mpm_project.install_for_entries(built, solvers)   # after the body carry: colliders at the step's end
    if getattr(built.model, mpm_clock.MODEL_ATTR, None) is not None:       # sim.routes() names the pair clock
        for e in built.entries:
            solver = solvers.get(e.name)
            if e.capability == "mpm" and solver is not None:
                attach_notes(solver, "clock", mpm_clock.route_notes(built.model))
    for pair in _wall_pairs(built):
        solver = solvers.get(pair.source)
        if solver is not None and install(solver):
            line = f"mpm entry {pair.source}: takes {pair.destination}'s force on its particles (SPH-MPM pair)"
            log.info("%s", line)
            notes.append(line)
    return notes


def _wall_pairs(built: Any) -> list[Any]:
    cap = {e.name: e.capability for e in built.entries}
    return [p for p in getattr(built.coupler, "pairs", ()) or ()
            if not p.one_way and cap.get(p.source) == "mpm" and cap.get(p.destination) == "sph"]


def wall_policy(policy: SchedulerPolicy, built: Any) -> SchedulerPolicy:
    """``policy`` with the L2 interface force norm when ``built`` couples an MPM source two-way into SPH and the
    policy keeps the default per-endpoint gate (module doc); unchanged otherwise."""
    if policy.tol.force_norm != "max" or not _wall_pairs(built):
        return policy
    log.info("SPH-MPM pair: the scene's interface force gate is the L2 norm per world (particle walls)")
    return dataclasses.replace(policy, tol=dataclasses.replace(policy.tol, force_norm="l2"))
