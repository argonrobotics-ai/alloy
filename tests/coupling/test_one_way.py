# SPDX-License-Identifier: Apache-2.0
"""One-way followers (SCR-DC5-6): an SPH entry on its own clock (rung 4 -> 4 x 0.15625 ms) inside a 0.625 ms MuJoCo
exchange, following the bodies and never feeding back. Beside a coupled MuJoCo <-> VBD Scheduler (the DC5
composition, for a partner SPH does not couple to) the follower leaves the coupled pair bitwise unchanged and steps
SPH rung times per exchange."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import CouplingError
from alloy.coupling.one_way import OneWayFollower, OneWayLeaf
from alloy.coupling.scheduler import Scheduler, SchedulerADMM
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.types import AlloyEntry, ClockPolicy, CouplingReport, PairPolicy, SchedulerPolicy
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.coupling import ClothSceneSpec, cloth_sph_model, cloth_sph_stepper
from alloy.testing.sph import default_config, floating_box, water_column

DT = 0.625e-3                     # the MuJoCo period of SCR-DC5-6
RUNG = 4                          # SPH at 0.15625 ms (0.16 ms on the D4 grid)
BOX = {"nx": 16, "ny": 16, "nz": 10, "box_cells": 8, "box_height_cells": 6.25, "box_density": 800.0}
POLICY = SchedulerPolicy(clock=ClockPolicy(dt_x_max=DT, omega_dt_max=float("inf")))


class _SolverStepper:
    """Minimal StepperLike over one solver on the full model (the single-entry base of a follower)."""

    def __init__(self, solver, model):
        self._solver, self.world_count, self.steps = solver, int(model.world_count), 0

    def step(self, state_in, state_out, control, contacts, dt):
        self.steps += 1
        self._solver.step(state_in, state_out, control, contacts, dt)

    def choose_clock(self, control_dt, last):
        return DT, {"mjc": 1}

    def report(self):
        return CouplingReport.quiet(DT, {"mjc": 1}, self.world_count)

    def state_arrays(self):
        return {}

    def state_world_maps(self):
        return {}


def _mujoco(model):
    from alloy._vendor.newton.solvers import SolverMuJoCo
    return SolverMuJoCo(model=model, use_mujoco_contacts=False, njmax=100, nconmax=100)


def _count_steps(solver) -> dict:
    calls = {"n": 0}
    step = solver.step

    def counted(*args, **kwargs):
        calls["n"] += 1
        return step(*args, **kwargs)
    solver.step = counted
    return calls


def _drive(model, stepper, steps: int, contacts: bool = True) -> tuple[np.ndarray, newton.State]:
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model, soft_contact_max=0) if contacts else None
    c, control = (pipe.contacts() if pipe else None), model.control()
    traj = []
    for _ in range(steps):
        s0.clear_forces()
        if pipe:
            pipe.collide(s0, c)
        stepper.step(s0, s1, control, c, DT)
        s0, s1 = s1, s0
        traj.append(s0.body_q.numpy().copy())
    return np.array(traj), s0


def test_one_way_rules():
    with pytest.raises(CouplingError, match="lagged"):
        Scheduler(None, [], [PairPolicy("a", "b", mode="staggered", proxy=object(), one_way=True)])
    with pytest.raises(CouplingError, match="one-way stage"):
        SchedulerADMM(None, [], [PairPolicy("a", "b", mode="admm", one_way=True)])
    sc = water_column(2, 2, 2, cfg=default_config(coupling="one_way"))
    base = _SolverStepper(None, sc.model)
    with pytest.raises(CouplingError, match="integrate"):
        OneWayFollower(base, sc.model, [OneWayLeaf("sph", SolverDFSPH(sc.model, default_config(
            integrate_bodies="dynamic")), [0])])
    with pytest.raises(CouplingError, match="ladder"):
        OneWayFollower(base, sc.model, [OneWayLeaf("sph", SolverDFSPH(sc.model, sc.config), [0], rung=3)])


def test_follower_sph_follows_mujoco_on_its_own_clock():
    """MuJoCo box sinking into a pool (no buoyancy: one-way): the box trajectory is bitwise MuJoCo alone, MuJoCo
    steps once and SPH RUNG times per exchange, the liquid moves and stays in the tank."""
    cfg = default_config(coupling="one_way")
    sc = floating_box(**BOX, cfg=cfg)
    m, box = sc.model, int(sc.bodies["box"])
    sph = SolverDFSPH(m, cfg)
    sph_calls = _count_steps(sph)
    base = _SolverStepper(_mujoco(m), m)
    ids = np.flatnonzero(m.sph.phase.numpy() >= 0)
    follower = OneWayFollower(base, m, [OneWayLeaf("sph", sph, ids, rung=RUNG, adaptive=False)])
    dt_x, substeps = follower.choose_clock(0.02, None)
    assert dt_x == DT and substeps == {"mjc": 1, "sph": RUNG}
    steps = 40
    coupled, state = _drive(m, follower, steps)
    assert base.steps == steps and sph_calls["n"] == RUNG * steps
    alone, _ = _drive(m, _SolverStepper(_mujoco(m), m), steps)
    assert coupled.tobytes() == alone.tobytes()
    assert coupled[-1, box, 2] < coupled[0, box, 2] - 1.0e-3        # it sank: the liquid did not hold it
    assert np.max(np.linalg.norm(state.particle_qd.numpy()[ids], axis=1)) > 1.0e-2
    assert sph.report().escaped[0] == 0
    assert follower.report().substeps == {"mjc": 1, "sph": RUNG}


def test_follower_leaves_the_coupled_pair_unchanged():
    """DC5 composition: MuJoCo <-> VBD Scheduler plus a one-way SPH follower. Box and cloth are bitwise the run
    without the follower; MuJoCo steps twice per exchange either way (the 2 lagged passes, no extra re-step)."""
    spec = ClothSceneSpec()
    model, sph_ids = cloth_sph_model(spec)
    runs = {}
    for with_sph in (True, False):
        stepper = cloth_sph_stepper(model, sph_ids, spec, with_sph=with_sph, policy=POLICY)
        calls = _count_steps(stepper.solver("mjc"))
        sph_calls = _count_steps(stepper.solver("sph")) if with_sph else {"n": 0}
        assert stepper.choose_clock(0.02, None)[0] == DT
        traj, state = _drive(model, stepper, 30)
        runs[with_sph] = (traj, state.particle_q.numpy().copy(), calls["n"], sph_calls["n"])
    (t1, p1, m1, s1), (t0, p0, m0, s0) = runs[True], runs[False]
    pool = set(sph_ids)
    cloth = [i for i in range(model.particle_count) if i not in pool]
    assert t1.tobytes() == t0.tobytes()
    assert p1[cloth].tobytes() == p0[cloth].tobytes()
    assert m1 == m0 == 2 * 30 and s1 == RUNG * 30 and s0 == 0
    assert np.all(np.isfinite(p1[sph_ids]))


def test_follower_rung_follows_the_sph_cfl():
    """Adaptive leaf: a particle at 2 m/s needs dt <= safety·cfl·2r/v = 0.5·0.2·2e-3/2 = 1e-4 s -> rung 8 at 0.625 ms;
    raised at once, then lowered one ladder step after the quiet periods once the liquid is slow."""
    cfg = default_config(coupling="one_way")
    sc = water_column(4, 4, 4, cfg=cfg)
    m = sc.model
    clock = ClockPolicy(downshift_quiet_periods=2)
    leaf = OneWayLeaf("sph", SolverDFSPH(m, cfg), list(range(m.particle_count)), rung=1, adaptive=True)
    follower = OneWayFollower(_SolverStepper(_NoBodies(), m), m, [leaf], clock)
    assert follower.choose_clock(0.02, None)[1]["sph"] == 1          # no estimate yet: hold
    s0, s1 = m.state(), m.state()
    qd = np.zeros((m.particle_count, 3), np.float32)
    qd[0] = (0.0, 0.0, -2.0)
    s0.particle_qd.assign(qd)
    follower.step(s0, s1, None, None, DT)
    stable = follower._leaves[0].stable.numpy()[0]
    assert stable <= cfg.cfl * cfg.spacing / 1.5                     # the 2 m/s particle sets the step
    assert follower.choose_clock(0.02, None)[1]["sph"] == 8
    follower._leaves[0].stable.fill_(cfg.dt_max)                     # liquid at rest
    rungs = [follower.choose_clock(0.02, None)[1]["sph"] for _ in range(4)]
    assert rungs == [8, 4, 4, 2]


class _NoBodies:
    def step(self, state_in, state_out, control, contacts, dt):
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)


def test_sph_estimator_is_the_solver_cfl_step():
    from alloy.core.registry import estimator_for

    cfg = default_config(coupling="one_way")
    sc = water_column(4, 4, 4, worlds=2, cfg=cfg)
    est = estimator_for(SolverDFSPH(sc.model, cfg))
    assert est is not None
    state = sc.model.state()
    qd = np.zeros((sc.model.particle_count, 3), np.float32)
    qd[np.flatnonzero(sc.model.particle_world.numpy() == 1)[0]] = (0.0, 0.0, -1.0)   # world 1: one particle, 1 m/s
    state.particle_qd.assign(qd)
    out = wp.zeros(2, dtype=float, device=sc.model.device)
    est.alloy_stable_dt(state, None, out)
    got = out.numpy()
    assert got[0] == pytest.approx(cfg.dt_max)                     # at rest: the solver's dt_max clamp
    assert got[1] == pytest.approx(cfg.cfl * cfg.spacing / 1.0, rel=1e-6)


def _floating_box_scheduler(calls: dict, pipeline=lambda _m: None):
    cfg = default_config(coupling="one_way")
    sc = floating_box(**BOX, cfg=cfg)
    m, box = sc.model, int(sc.bodies["box"])

    def counted(name, make):
        def build(view):
            solver = make(view)
            c = _count_steps(solver)
            calls[name] = c
            return solver
        return build
    e_m = SolverCoupledProxy.Entry(name="mjc", bodies=[box], joints=list(range(m.joint_count)),
                                   solver=counted("mjc", _mujoco))
    e_s = SolverCoupledProxy.Entry(name="sph", particles=list(range(m.particle_count)), substeps=64,
                                   solver=counted("sph", lambda v: SolverDFSPH(v, cfg)))
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="sph", bodies=[box], mode="lagged",
                                     collision_pipeline=pipeline)
    sched = Scheduler(m, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_s, initial_rung=RUNG, adaptive=False)],
                      [PairPolicy("mjc", "sph", proxy=proxy, one_way=True)], POLICY)
    return m, sched


def test_one_way_pair_skips_the_second_pass_bitwise():
    """SCR-DC5-8: the two-entry Scheduler with a one-way MuJoCo -> SPH pair runs one pass per exchange; forcing the
    second pass reproduces the destination (and the source) bitwise, at twice the steps."""
    runs = {}
    for passes in (1, 2):
        calls: dict = {}
        m, sched = _floating_box_scheduler(calls)
        sched._one_way_passes = passes
        assert sched.choose_clock(0.02, None) == (DT, {"mjc": 1, "sph": RUNG})
        sph = sched.solver("sph")
        traj, state = _drive(m, sched, 20)
        runs[passes] = (traj, state.particle_q.numpy().copy(), state.particle_qd.numpy().copy(),
                        calls["mjc"]["n"], calls["sph"]["n"], sph.report().escaped[0])
    one, two = runs[1], runs[2]
    assert one[0].tobytes() == two[0].tobytes()                     # source bodies
    assert one[1].tobytes() == two[1].tobytes() and one[2].tobytes() == two[2].tobytes()   # destination
    assert (one[3], one[4]) == (20, RUNG * 20) and (two[3], two[4]) == (40, 2 * RUNG * 20)
    assert one[5] == 0 and two[5] == 0
    assert int(sched.report().iters_used[0]) == 1


def test_sph_destination_ignores_proxy_contacts():
    """SCR-DC5-19: SolverDFSPH.step never reads ``contacts``, so make_stepper builds no proxy pipeline for an SPH
    destination (``stepper.CONTACT_FREE_DESTINATIONS``): with Newton's pipeline the run is bitwise the same."""
    from alloy._vendor import newton

    runs = []
    for pipeline in (lambda _m: None, lambda view: newton.CollisionPipeline(view)):
        m, sched = _floating_box_scheduler({}, pipeline)
        traj, state = _drive(m, sched, 12)
        runs.append((traj.tobytes(), state.particle_q.numpy().tobytes(), state.particle_qd.numpy().tobytes()))
    assert runs[0] == runs[1]


def test_compose_hub_vbd_sph_and_refusals(monkeypatch):
    """SCR-DC5-7: hub + VBD (two-way) + SPH (one-way) composes to Scheduler + OneWayFollower, whose ledger records
    zero work on the hub; two-way pairs over three entries need Newton's N-entry proxy coupling (P3, WP11)."""
    from alloy.coupling.compose import build_coupled_stepper
    from alloy.testing.coupling import cloth_entries, cloth_proxy

    spec = ClothSceneSpec()
    model, sph_ids = cloth_sph_model(spec)
    pool = set(sph_ids)
    e_m, _ = cloth_entries(model, spec)
    from alloy._vendor.newton.solvers import SolverVBD
    e_v = SolverCoupledProxy.Entry(name="vbd", solver=lambda v: SolverVBD(model=v, iterations=spec.vbd_iterations),
                                   particles=[i for i in range(model.particle_count) if i not in pool])
    cfg = default_config(coupling="one_way")
    e_s = SolverCoupledProxy.Entry(name="sph", solver=lambda v: SolverDFSPH(v, cfg), particles=list(sph_ids))
    entries = [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False),
               AlloyEntry(e_s, initial_rung=RUNG, adaptive=False)]
    pairs = [PairPolicy("mjc", "vbd", proxy=cloth_proxy(model)), PairPolicy("mjc", "sph", one_way=True)]
    stepper = build_coupled_stepper(model, entries, pairs, POLICY)
    assert isinstance(stepper, OneWayFollower) and isinstance(stepper.base, Scheduler)
    assert stepper.choose_clock(0.02, None) == (DT, {"mjc": 1, "vbd": 1, "sph": RUNG})
    _drive(model, stepper, 10)
    ledger = stepper.energy_ledger()
    assert np.all(ledger["sph.hub_work"] == 0.0) and np.all(np.isfinite(ledger["sph.wall_work"]))
    import alloy.coupling.compose as compose

    monkeypatch.setattr(compose, "proxy_dag_supported", lambda: False)
    with pytest.raises(CouplingError, match="N-entry proxy coupling"):
        build_coupled_stepper(None, entries, pairs[:1] + [PairPolicy("mjc", "mpm"), pairs[1]], POLICY)
