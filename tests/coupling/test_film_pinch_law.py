# SPDX-License-Identifier: Apache-2.0
"""Item 8's film pinch law (gap G4; plan [internal note] notes/wp4-thin-film-plan.md step 3, WP4 hand-off
notes/wp4-film-pinch-law/HANDOFF.md), on ``alloy.coupling.film_pinch.run_film_pinch``: two pads close at 15 N on C16's
2-ply HDPE film (1 cm cells, r 0.5 mm, its 3.4 g bag) gripping the film's top 20 mm, lift 50 mm at 0.05 m/s and hold
2 s, through the lagged MuJoCo->VBD Scheduler and through plain VBD (the pads its AVBD bodies) at the same step and
the same VBD iterations: the rig's 10 per-vertex; full-surface, the engine's floor for film entries with full-surface
proxy contacts (``FULL_SURFACE_MIN_ITERATIONS["film"]``, 40: 10 do not converge the clamp's several records per
particle, G4).

Pass, per contact mode: the carry is >= 95 % of plain VBD's; the film stays in the pinch (the pad gap stays > 0, and
the film's top row ends above the pads' bottom edge: in the coupled full-surface run the film fell 670 mm out of the
pinch while the gap stayed 0.97 mm, so the gap alone does not see a drop); no pass-through flags, no PEN_FAIL. WP10's
caps stay off (opt-in, ruling 2026-11-06 07:27Z). The film is SolverVBD cloth with C16's contact values (no film:*
attributes, i.e. SolverFilm bitwise per FILM-6): the law checks contact and coupling, not the film hinges.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.coupling import film_pinch as F
from alloy.coupling import pinch_rigs as P

LAW = F.FilmPinch()


def _law(full_surface: bool) -> None:
    plain = F.run_film_pinch(False, full_surface, LAW)
    coupled = F.run_film_pinch(True, full_surface, LAW)
    assert plain.finite and coupled.finite
    assert coupled.dt_x == pytest.approx(plain.dt_x, rel=1e-6), (coupled, plain)     # the same exchange step
    assert coupled.iterations == plain.iterations == F.law_iterations(LAW, full_surface), (coupled, plain)
    assert plain.gap_min > 0.0 and plain.top_above_edge > 0.0 and plain.carry > 0.5 * LAW.lift, plain   # null case
    assert coupled.gap_min > 0.0 and coupled.top_above_edge > 0.0, coupled
    assert coupled.carry >= 0.95 * plain.carry, (coupled, plain)
    assert coupled.pass_through == 0 and not coupled.pen_fail, coupled


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)                              # ~12 min on one CPU core: plain ~3 min + coupled ~8.5 min
def test_film_pinch_law_per_vertex():
    """Rows (CPU, main 81065d424 + this step, Newton pin 4f3b41ef, both at 1/1920 s): coupled carries 49.2 of 50.0 mm
    vs plain 49.5 (99.5 %), gap >= 0.98 / 0.98 mm, top row 17.3 / 15.2 mm above the pad edge, worst depth 0.015 /
    0.010 mm, no flags. At 1/960 s (the Scheduler's step before 3172e6f3e): 49.1 vs 48.0 (102 %)."""
    _law(False)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(5400)                         # ~55 min on one (loaded) CPU core: plain ~15 min + coupled ~39 min
def test_film_pinch_law_full_surface():
    """Rows (CPU, main 81065d424 + this step, both at 1/1920 s and 40 VBD iterations): coupled carries 49.9 of 50.0 mm
    vs plain 49.9 (100 %), gap >= 1.15 / 1.14 mm, top row 19.7 / 19.9 mm above the pad edge, worst depth 0.006 /
    0.006 mm, no flags. At 1/960 s: 45.7 vs 43.9 (104 %). At the rig's former 10 iterations (1/960 s) both dropped
    (coupled fell 591 mm, plain 800 mm): G4, [internal note] scr/SC4.md "G4 root cause"."""
    _law(True)


def test_both_runs_take_the_engines_iterations():
    """Plain and coupled run at the rig's 10 iterations per-vertex and at the engine's full-surface film floor
    (``FULL_SURFACE_MIN_ITERATIONS["film"]``, WP4) full-surface; ``FilmPinch.iterations`` overrides both. The
    full-surface rows are at 40: a new floor needs new rows."""
    from alloy.integration.stepper import FULL_SURFACE_MIN_ITERATIONS

    assert F.law_iterations(LAW, False) == F.RIG_ITERATIONS == 10
    assert F.law_iterations(LAW, True) == FULL_SURFACE_MIN_ITERATIONS["film"] == 40
    fixed = F.FilmPinch(iterations=20)
    assert F.law_iterations(fixed, True) == F.law_iterations(fixed, False) == 20


def test_law_step_is_the_schedulers():
    """The plain reference's clock (one VBD step per ``FilmPinch.dt_x``) is the one the Scheduler picks on the law's
    rigs: the same exchange step and one substep per entry, on the squeeze stage's model (top row fixed) and the lift
    stage's, in both contact modes, so a clock change fails here in the merge gate. The slow law asserts dt_x at run
    time, but gates skip it: 3172e6f3e (the clock bound from the pads' 0.56 kg axis mass, 0.75 ms) moved the coupled
    run to 1/1920 s and left the plain one at 1/960 unnoticed; a VBD rung of 2 would differ with dt_x equal (WP4's
    review). A new clock needs new law rows (docs/notes/sc8-film-laws.md)."""
    for full_surface in (False, True):
        for fix_top in (True, False):
            dt_x, substeps = F.scheduler_clock(LAW, full_surface, fix_top)
            assert dt_x == pytest.approx(LAW.dt_x, rel=1e-9), (full_surface, fix_top, dt_x)
            assert substeps and set(substeps.values()) == {1}, (full_surface, fix_top, substeps)


def test_free_film_falls_at_the_vbd_motion_budget():
    """Null case for the law's drops: a free film with self-contact on, nothing near it, falls freely until VBD's
    divide-and-truncate budget (0.5 x 0.85 x the 1 mm query radius per step) caps it at 0.425 mm per exchange. That
    is why a film that leaves the pinch descends at a steady speed (0.41 m/s at 1/960 s), not friction creep. The
    constants are Newton's (pin 4f3b41ef): newton/_src/solvers/vbd/solver_vbd.py:433 (dat_conservative_bound_relaxation
    0.85) and its docstring at :497-500 ("each side's motion budget between detections is 0.5 x relaxation x the
    detection query radius"); a pin that changes them fails here."""
    from alloy._vendor import newton

    cloth = P.Cloth(P.FILM.cell, P.FILM.radius, P.FILM.mass * 25.0, P.FILM.soft_ke)
    model = P.gripper_model(cloth=cloth, film_top=P.PAD_Z - 0.05, fix_top=False)        # 4 cm below the open pads
    dt = 1.0 / 960.0
    solver = P._vbd(model, cloth.radius)
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model, soft_contact_margin=0.002)
    contacts = pipe.contacts()
    for _ in range(60):                                                                  # 62.5 ms: 0.61 m/s free
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, dt)
        s0, s1 = s1, s0
    vz = s0.particle_qd.numpy()[:, 2]
    budget = 0.5 * 0.85 * 2.0 * cloth.radius / dt                                          # 0.408 m/s
    assert np.allclose(vz, -budget, rtol=0.02), (vz.min(), vz.max(), budget)
