# SPDX-License-Identifier: Apache-2.0
"""Self-contact gains apart from body contact (alloy.solvers.vbd.base.SelfContactGains; opt-in
``alloy_self_contact_ke_ratio`` / ``alloy_self_contact_kd``): VBD's self-contact reads the capped stiffness and the
replaced damping inside the particle iteration, while body contacts keep the model's values (known gap
cloth-fold-unsettled-default: the thirds fold rests with a softer, undamped self-contact)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.solvers.vbd import SolverVBDAlloy
from alloy.solvers.vbd import scenes as S
from alloy.solvers.vbd.base import SelfContactGains

KE, KD, DT = 1.0e6, 10.0, 1.0e-3
SPLIT = {"alloy_self_contact_ke_ratio": 400.0, "alloy_self_contact_kd": 0.0}


def _stack_solver(**extra):
    """The pinned-sheet stack of scenes.py (its only support is self contact) at a route-like contact ke / kd."""
    sc = S.build("stack")
    sc.model.soft_contact_ke, sc.model.soft_contact_kd = KE, KD
    return sc, SolverVBDAlloy(sc.model, **S.solver_kwargs(sc, 10), **extra)


def _step(sc, solver, steps: int):
    m = sc.model
    s0, s1, ctrl, contacts = m.state(), m.state(), m.control(), sc.pipeline.contacts()
    for _ in range(steps):
        s0.clear_forces()
        sc.pipeline.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, DT)
        s0, s1 = s1, s0
    return s0


def test_off_by_default():
    sc, solver = _stack_solver()
    assert solver.self_contact_gains is None
    assert SelfContactGains.for_model(sc.model, None, None) is None


def test_values_cap_the_stiffness_and_never_stiffen():
    g = SelfContactGains(ke_ratio=400.0, kd=0.0, m_ref=1.0e-4)
    cap = 400.0 * 1.0e-4 / DT**2                                         # 4e4 N/m
    assert g.values(1.0e6, 10.0, DT) == pytest.approx((cap, 0.0))
    assert g.values(1.0e3, 10.0, DT) == (1.0e3, 0.0)                     # a softer model ke stays
    assert g.values(1.0e6, 10.0, DT / 2)[0] == pytest.approx(4.0 * cap)  # the cap follows the step
    assert SelfContactGains(None, 0.0, 1.0e-4).values(1.0e6, 10.0, DT) == (1.0e6, 0.0)
    assert SelfContactGains(400.0, None, 1.0e-4).values(1.0e6, 10.0, DT)[1] == 10.0


@pytest.mark.parametrize(("ratio", "kd"), [(0.0, None), (-1.0, None), (None, -0.1)])
def test_bad_values_raise(ratio, kd):
    sc = S.build("stack")
    with pytest.raises(ValueError, match="alloy_self_contact"):
        SelfContactGains.for_model(sc.model, ratio, kd)


def test_self_contact_reads_the_split_and_body_contact_the_model():
    """Inside the particle iteration (where Newton's self-contact launch reads them) the model's ke / kd are the
    self-contact values; at the body-contact refresh and after the step they are the model's own."""
    sc, solver = _stack_solver(**SPLIT)
    m_ref = float(np.median(sc.model.particle_mass.numpy()[sc.model.particle_mass.numpy() > 0]))
    seen_iter, seen_refresh = set(), set()
    solver.add_colour_callback(lambda s, c, st, ct: seen_iter.add((s.model.soft_contact_ke, s.model.soft_contact_kd)),
                               external=False)
    refresh0 = solver._refresh_body_particle_contact_state

    def refresh(contacts, refresh_):
        seen_refresh.add((solver.model.soft_contact_ke, solver.model.soft_contact_kd))
        return refresh0(contacts, refresh_)

    solver._refresh_body_particle_contact_state = refresh
    _step(sc, solver, 3)
    (ke, kd), = seen_iter
    assert ke == pytest.approx(400.0 * m_ref / DT**2) and ke < KE and kd == 0.0
    assert seen_refresh == {(KE, KD)}
    assert (sc.model.soft_contact_ke, sc.model.soft_contact_kd) == (KE, KD)


def _upper_sheet_speed(extra: dict, steps: int = 100) -> tuple[float, float]:
    """Mean speed of the stack's upper sheet over the last 20 steps, and its lowest point above the pinned sheet."""
    sc, solver = _stack_solver(alloy_initial_guess="adaptive", alloy_chebyshev_rho=0.9,
                               alloy_chebyshev_hold_self_contact=True, **extra)
    m = sc.model
    s0, s1, ctrl, contacts = m.state(), m.state(), m.control(), sc.pipeline.contacts()
    lo, hi = sc.sheets
    speeds = []
    for _ in range(steps):
        s0.clear_forces()
        sc.pipeline.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, DT)
        s0, s1 = s1, s0
        speeds.append(float(np.linalg.norm(s0.particle_qd.numpy()[hi[0]:hi[1]], axis=1).mean()))
    return float(np.mean(speeds[-20:])), float(s0.particle_q.numpy()[hi[0]:hi[1], 2].min() - S.STACK_Z)


def test_the_split_rests_a_stack_that_the_route_gains_keep_moving():
    """At a route-like ke 1e6 N/m, kd 10 N s/m (the cloth route's ke is the material's bulk K, kd Newton's default),
    the stack's upper sheet keeps moving through the default solve (adaptive start, Chebyshev, hold; CPU, 10
    iterations: 9.6e-3 m/s over steps 80-100); with self-contact capped at 400 m / dt^2 and undamped it rests
    (3.6e-5 m/s), still on the lower sheet at the contact distance (0.5 mm). ~11 s."""
    moving, z_moving = _upper_sheet_speed({})
    rest, z_rest = _upper_sheet_speed(SPLIT)
    assert moving > 3.0e-3 and rest < 2.0e-4 and rest < moving / 50.0, (moving, rest)
    assert 0.4e-3 < z_rest < 0.6e-3 and 0.4e-3 < z_moving < 0.6e-3, (z_moving, z_rest)


def test_without_self_contact_nothing_is_split():
    sc = S.build("sheet_rest")
    solver = SolverVBDAlloy(sc.model, **S.solver_kwargs(sc, 10), **SPLIT)
    assert not solver.particle_enable_self_contact and solver.self_contact_gains is None
