# SPDX-License-Identifier: Apache-2.0
"""Item 8's film pinch law rig (gap G4; plan [internal note] notes/wp4-thin-film-plan.md step 3): ``run_film_pinch``.

WP4's two-pad gripper (``alloy.coupling.pinch_rigs``) on C16's 2-ply HDPE film, through the lagged MuJoCo->VBD
Scheduler or a plain SolverVBD reference at the same step and VBD iteration count. Full-surface proxy contacts put
several records on each particle, and 10 VBD iterations do not converge the clamp (G4, [internal note] scr/SC4.md "G4 root
cause"): ``FilmPinch.iterations`` None runs the rig's 10 per-vertex and, full-surface, the engine's floor for film
entries with full-surface proxy contacts (``alloy.integration.stepper.FULL_SURFACE_MIN_ITERATIONS``, 40; WP4).
The plain reference steps VBD once per ``FilmPinch.dt_x``, the Scheduler's clock on this rig (``scheduler_clock``: its
step, one substep per entry; a fast test checks both rigs agree, so a clock change fails in the merge gate, not only
in the slow law).
Test: tests/coupling/test_film_pinch_law.py; rows: docs/notes/sc8-film-laws.md (Film pinch law).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton
from alloy.coupling.pinch_rigs import (FILM, PAD_HALF, PAD_Z, Cloth, PinchPhase, _cloth_scheduler, _pass_through_count,
                                       _run_script, _vbd, gripper_model)
from alloy.coupling.proxy_contacts import ProxyContactPipeline

RIG_ITERATIONS = 10            # the pinch rigs' VBD iterations (pinch_rigs._cloth_scheduler's default)


def law_iterations(law: "FilmPinch", full_surface: bool) -> int:
    """VBD iterations for both runs: ``law.iterations``, else the engine's full-surface film floor (G4: 10 creep out
    of the pinch, 40 hold), else the rig's 10."""
    if law.iterations is not None:
        return int(law.iterations)
    if not full_surface:
        return RIG_ITERATIONS
    from alloy.integration.stepper import FULL_SURFACE_MIN_ITERATIONS  # noqa: PLC0415 (integration builds on coupling)

    return max(RIG_ITERATIONS, int(FULL_SURFACE_MIN_ITERATIONS["film"]))


class _PlainVBD:
    """The film pinch law's plain reference: one SolverVBD steps the whole rig (pads: its AVBD bodies on the same joint
    servos) with its own contacts at a fixed step; VBD writes body poses, so the servo's joint q comes from eval_ik."""
    report, capacity_flags = staticmethod(lambda: None), staticmethod(tuple)     # no clock report, no detector

    def __init__(self, model: newton.Model, full_surface: bool, dt_x: float, radius: float, iterations: int) -> None:
        self.model, self.solver = model, _vbd(model, radius, iterations)
        self.choose_clock = lambda period, last: (dt_x, None)               # _run_script's Scheduler clock call
        moving = np.flatnonzero(model.shape_body.numpy() >= 0)
        self.pipe = (ProxyContactPipeline(full_surface=True, full_surface_shapes=moving, soft_contact_gap=0.002)(model)
                     if full_surface else newton.CollisionPipeline(model, soft_contact_gap=0.002))
        self.contacts = self.pipe.contacts()

    def step(self, s0: newton.State, s1: newton.State, ctl: newton.Control, contacts: Any, dt: float) -> None:
        self.pipe.collide(s0, self.contacts)
        self.solver.step(s0, s1, ctl, self.contacts, dt)
        newton.eval_ik(self.model, s1, s1.joint_q, s1.joint_qd)


@dataclass(frozen=True)
class FilmPinch:
    """Item 8's film pinch law (G4, plan notes/wp4-thin-film-plan.md step 3): pads close at ``grip_n`` on C16's 2-ply
    HDPE film (1 cm cells, r 0.5 mm, vertex mass x25: its 3.4 g bag), top edge ``overlap`` above the pads' bottom edge
    (0.020: the full pad height); settle, lift, hold. Two stages (top row fixed for the ``squeeze``, then a free model
    from that state): VBD never solves particles held inactive at construction, so no flag release."""

    overlap: float = 0.020
    soft_ke: float = FILM.soft_ke
    grip_n: float = 15.0
    lift: float = 0.050
    speed: float = 0.050
    settle: float = 0.5
    hold: float = 2.0
    squeeze: float = 0.4
    dt_x: float = 1.0 / 1920.0           # the plain reference's step: the Scheduler's on this rig at 1/60 s periods
    #   (``scheduler_clock``; 1/960 until 3172e6f3e sized the clock bound from the pads' 0.56 kg axis mass: 0.75 ms)
    iterations: int | None = None        # VBD iterations of both runs; None: ``law_iterations``


@dataclass(frozen=True)
class FilmPinchResult:
    carry: float              # film top-row rise from the lift's start to the end [m]
    pad_rise: float           # the pads' rise over the same span [m]
    gap_min: float            # smallest pad gap after the squeeze [m] (0: the film left the pinch)
    top_above_edge: float     # film top row above the pads' bottom edge at the end [m]
    pass_through: int         # pass-through detector flags (coupled; the plain pipeline has no detector)
    pen_fail: bool            # PEN_FAIL or PAIR_PEN_FAIL from a HealthMonitor checking every exchange
    max_pen: float            # deepest penetration health measured [m]
    dt_x: float               # exchange step used [s]
    iterations: int           # VBD iterations used (both runs)
    finite: bool


def run_film_pinch(coupled: bool, full_surface: bool, law: FilmPinch = FilmPinch()) -> FilmPinchResult:
    """The film pinch law's rig (``FilmPinch``) through the lagged MuJoCo->VBD Scheduler (``coupled``) or the plain
    reference (``_PlainVBD`` at ``law.dt_x``), per-vertex or ``full_surface`` proxy contacts."""
    from alloy.precision.types import HealthFlags
    cloth, top = _law_cloth(law), _film_top(law)
    face = 2.0 * cloth.radius

    it = law_iterations(law, full_surface)
    stepper = ((lambda m: _cloth_scheduler(m, full_surface, it, radius=cloth.radius)) if coupled
               else (lambda m: _PlainVBD(m, full_surface, law.dt_x, cloth.radius, it)))
    m1 = gripper_model(cloth=cloth, closed=True, film_top=top)
    s1 = _run_script(stepper(m1), m1, (PinchPhase("squeeze", law.squeeze, 0.0, True),), lambda *a: None,
                     law.grip_n, half_gap=face)
    m2 = gripper_model(cloth=cloth, closed=True, film_top=top, fix_top=False)
    for name in ("particle_q", "particle_qd", "joint_q"):
        getattr(m2, name).assign(getattr(s1, name).numpy())
    m2.joint_qd.zero_()
    pq0 = m2.particle_q.numpy()
    row = np.abs(pq0[:, 2] - pq0[:, 2].max()) < 2.0e-3                     # the film's top row
    sched = stepper(m2)
    monitor = law_monitor(m2, sched)
    t_lift = law.settle + law.lift / law.speed
    script = (PinchPhase("settle", law.settle, 0.0, True),
              PinchPhase("lift", t_lift, law.lift, True, ramp=law.lift / law.speed),
              PinchPhase("hold", t_lift + law.hold, law.lift, True))
    acc: dict[str, Any] = {"gap": [], "n": 0, "t": 0.0}

    def observe(ph: PinchPhase, k: int, s: newton.State) -> None:
        bq, pq = s.body_q.numpy(), s.particle_q.numpy()
        acc["gap"].append(float(bq[2, 1] - bq[1, 1]) - 2.0 * PAD_HALF[1])
        acc[ph.name] = (float(bq[1, 2]), float(pq[row, 2].mean()))

    s = _run_script(sched, m2, script, observe, law.grip_n, on_exchange=lambda ph, t: acc.update(n=acc["n"] + 1, t=t),
                    half_gap=face, monitor=monitor)
    rep, (pz0, fz0), (pz1, fz1) = monitor.poll(), acc["settle"], acc["hold"]
    return FilmPinchResult(carry=fz1 - fz0, pad_rise=pz1 - pz0, gap_min=float(min(acc["gap"])),
                           top_above_edge=fz1 - (pz1 - PAD_HALF[2]), pass_through=_pass_through_count(sched),
                           pen_fail=bool(int(rep.flags[0]) & int(HealthFlags.PEN_FAIL | HealthFlags.PAIR_PEN_FAIL)),
                           max_pen=float(np.nan_to_num(rep.max_pen_m[0])), dt_x=acc["t"] / acc["n"], iterations=it,
                           finite=bool(np.isfinite(s.particle_q.numpy()).all()))


def law_monitor(m: newton.Model, stepper: Any) -> Any:
    """The law's HealthMonitor (every exchange, full level), with the stepper's solvers. Health reads a solver's own
    overflow counters only from the solvers it is given: a rig that left them out dropped MuJoCo constraint rows
    3,171 times in one run and read clean (GAP-10A r8, 2026-11-10). With them a dropped row is ROW_OVERFLOW."""
    from alloy.integration.stepper import entry_solvers
    from alloy.precision.health import HealthMonitor
    from alloy.precision.types import CheckPolicy

    return HealthMonitor(m, CheckPolicy(level="full", every=1, full_every=1), capacity_flags=stepper.capacity_flags(),
                         solvers=list(entry_solvers(stepper).values()))


def _law_cloth(law: FilmPinch) -> Cloth:
    """C16's film at the law's contact stiffness, vertex mass x25 (its 3.4 g bag)."""
    return Cloth(FILM.cell, FILM.radius, FILM.mass * 25.0, law.soft_ke)


def _film_top(law: FilmPinch) -> float:
    """The film's top edge: ``law.overlap`` above the pads' bottom edge."""
    return PAD_Z - PAD_HALF[2] + law.overlap


def scheduler_clock(law: FilmPinch, full_surface: bool, fix_top: bool = False,
                    period: float = 1.0 / 60.0) -> tuple[float, dict[str, int]]:
    """The clock the lagged Scheduler picks on the law's rig (pads closed; ``fix_top``: the squeeze stage's model with
    the film's top row fixed, else the lift stage's free one) for one control ``period``: its first clock call, as in
    ``run_film_pinch``, i.e. (exchange step, substeps per entry). The plain reference mirrors ``FilmPinch.dt_x`` with
    one substep per entry."""
    cloth = _law_cloth(law)
    m = gripper_model(cloth=cloth, closed=True, film_top=_film_top(law), fix_top=fix_top)
    sched = _cloth_scheduler(m, full_surface, law_iterations(law, full_surface), radius=cloth.radius)
    dt_x, substeps = sched.choose_clock(period, None)
    return float(dt_x), dict(substeps)
