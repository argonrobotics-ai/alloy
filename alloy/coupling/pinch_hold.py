# SPDX-License-Identifier: Apache-2.0
"""A coupled pinch-hold rig (``run_hold``): does a two-pad pinch hold a hanging load, or does the cloth creep out? The
pads, the cloth and the coupled Scheduler are those of ``alloy.coupling.pinch_rigs`` (``gripper_model``,
``_cloth_scheduler``).
Laws: tests/coupling/test_cloth_grasp_hold_law.py and tests/envs/test_cloth_attach.py."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy._vendor import newton
from alloy.coupling.pinch_rigs import (PAD_HALF, PAD_MASS, PAD_Z, ROUTE_ITERATIONS, STICK_KEY, TOWEL, Cloth,
                                       _cloth_scheduler, _pad_targets, _pass_through_count, gripper_model)
from alloy.coupling.types import SchedulerPolicy


HOLD_MU = math.sqrt(0.45 * 0.4)       # pad-cloth mu: VBD's geometric mean of soft_contact_mu 0.45 and the pad's 0.4


@dataclass(frozen=True)
class HoldResult:
    slide: float              # clamped cloth's total slide down relative to the pads over the hold [m]
    creep: float              # slide speed [m/s], linear fit over the hold's second half
    left_pads: bool           # the clamped particles are no longer between the pads at the end
    gap: float                # inner-face gap at the end of the hold [m]
    load_n: float             # the load W [N] = load_ratio x 2 x mu x grip_n
    pass_through: int
    finite: bool
    wall_s: float             # CPU wall time of the whole run [s]
    sim_s: float              # simulated time [s]
    trace: tuple[tuple[float, float], ...] = ()    # (t since the hold's start [s], slide so far [m]) per control period


def run_hold(load_ratio: float, grip_n: float = 15.0, hold_s: float = 10.0, *,
             friction_epsilon: float = 1.0e-2, cloth: Cloth = TOWEL, iterations: int | None = None,
             full_surface: bool = True, policy: SchedulerPolicy | None = None, solver_kwargs: dict | None = None,
             mu: float = HOLD_MU, settle_s: float = 0.3, close_s: float = 0.15, route: bool = True,
             stick: bool = False, pad_mass: float = PAD_MASS, attach: bool = False) -> HoldResult:
    """Does a pinch hold a hanging load? The pads start on the cloth's top edge (double layer, rows at the pads'
    height, top edge free, bottom free) and squeeze at ``grip_n``; the carriage holds still. After ``close_s`` a
    downward load W = ``load_ratio`` x 2 x ``mu`` x ``grip_n`` (the Coulomb limit's multiple) is ramped on over
    ``settle_s`` and held; the slide is measured over the next ``hold_s``.

    The load is an external particle force (state.particle_f, set every exchange) shared over the cloth's bottom row,
    not extra mass: the cloth must be gripped before the load acts (extra mass would pull a free cloth out of the
    pads at once), and a force ramps. Slide: the mean z of the particles between the pads when the load starts, relative
    to the pads' z. ``route``: the VBD solver as the cloth route builds it (``make_vbd``, 40 iterations by default;
    False: plain Newton SolverVBD, 10 iterations); ``solver_kwargs`` go to its constructor; ``stick``: the opt-in
    stick friction (``body_particle_friction_stick``, route solver only); ``pad_mass`` [kg] of each pad; ``attach``:
    attach-on-grasp (``alloy.envs.cloth_attach``: the cloth between the closed pads is held to pad 1, a shortcut,
    not friction)."""
    import time

    if stick and not route:
        raise ValueError("stick friction needs the route solver (route=True)")
    iterations = iterations or (ROUTE_ITERATIONS if route else 10)
    extra = {**(solver_kwargs or {}), **({STICK_KEY: True} if stick else {})}
    model = gripper_model(cloth=cloth, closed=True, fix_top=False, film_top=PAD_Z + PAD_HALF[2] - 0.001,
                          pad_mass=pad_mass)
    hook = None
    if attach:
        from alloy.envs.cloth_attach import ClothAttach  # noqa: PLC0415

        half = (PAD_HALF[0] + 0.002, PAD_HALF[1], PAD_HALF[2] + 0.002)
        hook = lambda s: ClothAttach(s, np.array([1]), np.array([2]), half)  # noqa: E731  (pads: bodies 1, 2)
    sched = _cloth_scheduler(model, full_surface, iterations, "lagged", cloth.radius, friction_epsilon, policy,
                             extra, route, hook)
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    glob = newton.CollisionPipeline(model)
    contacts = glob.contacts()
    q0 = s0.particle_q.numpy()
    bottom = np.flatnonzero(q0[:, 2] < q0[:, 2].min() + 0.5 * cloth.cell)
    load_n = load_ratio * 2.0 * mu * grip_n
    period, t_load, t_meas = 1.0 / 60.0, close_s, close_s + settle_s
    n_total = int(round((t_meas + hold_s) / period))
    sel, z0, trace, last = None, None, [], None
    wall0 = time.perf_counter()
    for k in range(n_total):
        t = k * period
        q = s0.joint_q.numpy()
        ctl.joint_target_q.assign(np.array([0.0, *_pad_targets(q[1:3].astype(np.float64), True, grip_n, 0.0,
                                                                2.0 * cloth.radius)], np.float32))
        frac = min(1.0, max(0.0, (t - t_load) / max(settle_s, 1e-9)))
        force = np.zeros((model.particle_count, 3), np.float32)
        force[bottom, 2] = -frac * load_n / len(bottom)
        dt, _ = sched.choose_clock(period, last)
        for _ in range(int(round(period / dt))):
            s0.clear_forces()
            s0.particle_f.assign(force)
            glob.collide(s0, contacts)
            sched.step(s0, s1, ctl, contacts, dt)
            s0, s1 = s1, s0
        last = sched.report()
        bq, pq = s0.body_q.numpy(), s0.particle_q.numpy()
        if sel is None and t + period >= t_load - 1e-9:          # the clamped particles, before the load acts
            sel = ((np.abs(pq[:, 0]) < PAD_HALF[0]) & (np.abs(pq[:, 2] - bq[1, 2]) < PAD_HALF[2])
                   & (np.abs(pq[:, 1]) < 0.004))
        if t + period < t_meas - 1e-9:
            continue
        zrel = float(pq[sel, 2].mean() - bq[1, 2])
        z0 = zrel if z0 is None else z0
        trace.append((t + period - t_meas, z0 - zrel))
    bq, pq = s0.body_q.numpy(), s0.particle_q.numpy()
    tr = np.asarray(trace)
    late = tr[len(tr) // 2:]
    creep = float(np.polyfit(late[:, 0], late[:, 1], 1)[0]) if len(late) > 2 else 0.0
    inside = np.abs(pq[sel, 2] - bq[1, 2]) < PAD_HALF[2]
    return HoldResult(slide=float(tr[-1, 1]), creep=creep, left_pads=bool(inside.mean() < 0.5),
                      gap=float(bq[2, 1] - bq[1, 1]) - 2.0 * PAD_HALF[1], load_n=load_n,
                      pass_through=_pass_through_count(sched), finite=bool(np.isfinite(pq).all()),
                      wall_s=time.perf_counter() - wall0, sim_s=n_total * period,
                      trace=tuple((float(a), float(b)) for a, b in trace))


__all__ = ["HOLD_MU", "HoldResult", "run_hold"]
