# SPDX-License-Identifier: Apache-2.0
"""Scheduler construction rules and equivalence with Newton's stock proxy (SPEC §7 WP4 acceptance, CPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.core.errors import CouplingError
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.scheduler import Scheduler, strong_pair
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.testing.coupling import (ClothSceneSpec, cloth_entries, cloth_proxy, cloth_scheduler,
                                    coupling_box_on_cloth, stock_proxy_solver)

SPEC = ClothSceneSpec(box_mass=0.2, cells=8)
DT = 1.0 / 480.0


@pytest.fixture(scope="module")
def model2():
    return coupling_box_on_cloth(2, SPEC, device="cpu")


def _run(model, solver, steps, dt=DT, between=None):
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    out = []
    for k in range(steps):
        if between is not None:
            between(k)
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        out.append(np.concatenate([s0.body_q.numpy()[:, :3].ravel(), s0.particle_q.numpy().ravel()]))
    return np.array(out)


def test_strong_pair_rule():
    p = PairPolicy("a", "b")
    assert not strong_pair(0.4, 1.0, p) and strong_pair(0.6, 1.0, p)        # m_a/m_s > 0.5 (D5)
    assert strong_pair(0.0, 1.0, dataclasses.replace(p, saturating_actuator=True))
    assert strong_pair(0.0, 1.0, dataclasses.replace(p, static_hold=True))
    assert strong_pair(0.0, 0.08, dataclasses.replace(p, added_mass=0.3))   # floating cup: C_a·ρ_f·V_sub wins


def test_staggered_fixed_relaxation_rejected(model2):
    e_m, e_v = cloth_entries(model2, SPEC)
    pair = PairPolicy("mjc", "vbd", mode="staggered", proxy=cloth_proxy(model2, "staggered", "fixed"))
    with pytest.raises(CouplingError, match="fixed"):
        Scheduler(model2, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)], [pair])


def test_mode_and_mapping_rules(model2):
    e_m, e_v = cloth_entries(model2, SPEC)
    entries = [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)]
    with pytest.raises(CouplingError, match="SchedulerADMM"):
        Scheduler(model2, entries, [PairPolicy("mjc", "vbd", mode="admm", proxy=cloth_proxy(model2))])
    with pytest.raises(CouplingError, match="Proxy mapping"):
        Scheduler(model2, entries, [PairPolicy("mjc", "vbd")])
    with pytest.raises(CouplingError, match="max rung"):                      # D21: adaptive needs the scratch
        Scheduler(model2, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=True)],
                  [PairPolicy("mjc", "vbd", proxy=cloth_proxy(model2))])


def test_lagged_equals_stock_proxy(model2):
    """Fixed clock: Scheduler (lagged, 2 passes, ω = 1) == SolverCoupledProxy(iterations=2) within 1e-6 m."""
    steps = 40
    ref = _run(model2, stock_proxy_solver(model2, SPEC, iterations=2), steps)
    ours = _run(model2, cloth_scheduler(model2, SPEC, vbd_adaptive=False), steps)
    err = float(np.max(np.abs(ref - ours)))
    print(f"lagged scheduler vs stock proxy: max |dq| = {err:.3e} m over {steps} steps")
    assert err <= 1e-6


def test_runtime_rung_change_takes_effect(model2):
    """D21: VBD built at the max rung; switching to 4 at runtime == a stock entry built with substeps=4; switching
    again mid-run never raises."""
    steps = 20
    e_m, e_v = cloth_entries(model2, SPEC, vbd_substeps=4)
    stock4 = SolverCoupledProxy(model=model2, entries=[e_m, e_v], coupling=SolverCoupledProxy.Config(
        proxies=[cloth_proxy(model2)], iterations=2))
    ref = _run(model2, stock4, steps)
    sched = cloth_scheduler(model2, SPEC, vbd_adaptive=True, vbd_initial=1)
    sched.set_substeps({"vbd": 4})
    ours = _run(model2, sched, steps)
    assert sched._entries["vbd"].substeps == 4
    assert float(np.max(np.abs(ref - ours))) <= 1e-6

    sched2 = cloth_scheduler(model2, SPEC, vbd_adaptive=True, vbd_initial=1)
    rungs = {5: 2, 10: 64}
    traj = _run(model2, sched2, 15, between=lambda k: sched2.set_substeps({"vbd": rungs[k]}) if k in rungs else None)
    assert np.all(np.isfinite(traj))
    assert sched2._entries["vbd"].substeps == 64
    with pytest.raises(CouplingError):
        sched2.set_substeps({"nope": 2})


def test_report_notes_carry_per_world_stable_dt(model2):
    """SCR-JOB2-1 (WP12 reads it): ``report().notes["stable_dt"]`` = each estimated entry's per-world stability bound
    [s], whose minimum is the host value the clock uses."""
    sched = cloth_scheduler(model2, SPEC, vbd_adaptive=False)
    _run(model2, sched, 3)
    rep = sched.report()
    stable = rep.notes["stable_dt"]
    assert set(stable) == set(sched._stable_host) and stable
    for name, per_world in stable.items():
        assert per_world.shape == (2,) and np.all(per_world > 0.0)
        assert float(per_world.min()) == sched._stable_host[name]
