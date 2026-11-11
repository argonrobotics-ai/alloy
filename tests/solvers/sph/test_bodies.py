# SPDX-License-Identifier: Apache-2.0
"""Fluid-body coupling on CPU: DFSPH-owned floating box (D5 monolithic), MuJoCo-owned box through the stock proxy
coupler with embedded proxy integration (SCR-WP9-2), the hydrostatic floor load (momentum balance) and the walls'
work on the liquid (energy balance, SCR-WP9-6)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.sph import default_config, floating_box, lifting_floor_energy, run, water_column

DT = 1.0e-3


BOX = {"nx": 16, "ny": 16, "nz": 10, "box_cells": 8, "box_height_cells": 6.25, "box_density": 800.0}


def _draft(scene, body_q: np.ndarray) -> float:
    """Draft with the surface rise: d = (V_liquid / A_tank - z_bottom) / (1 - A_box / A_tank)."""
    s = scene.config.spacing
    nx, ny, nz = scene.fill[0]
    a_tank, a_box = nx * ny * s * s, (BOX["box_cells"] * s) ** 2
    v_liq = nx * ny * nz * s ** 3 - a_box * scene.bodies["draft"]  # lattice cells minus the submerged ones
    bottom = float(body_q[2]) - 0.5 * BOX["box_height_cells"] * s
    return (v_liq / a_tank - bottom) / (1.0 - a_box / a_tank)


@pytest.mark.timeout(240)  # ~56 s on CPU at the resting-jitter rule's tolerance (17 s at 1e-3), SCR-SC-C2
def test_floating_box_monolithic_draft():
    """rho_b = 800, side 1.28 x height (positive metacentric height), draft/(2r) = 5: the box floats stably within
    10 % of m/(rho A) (A13's 5 % at draft/(2r) >= 10 is the law run)."""
    sc = floating_box(**BOX)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    box = int(sc.bodies["box"])
    drafts = []
    run(s, a, b, 500, DT, record=lambda k, st, r: drafts.append(_draft(sc, st.body_q.numpy()[box])) if k >= 300
        else None)
    expect = sc.bodies["draft"]
    assert abs(np.mean(drafts) - expect) / expect < 0.10
    assert np.ptp(drafts) < 0.1 * expect
    assert s.report().escaped[0] == 0


@pytest.mark.slow
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_floating_box_through_mujoco_proxy():
    """The box belongs to SolverMuJoCo; SolverDFSPH is the proxy destination and integrates the proxy at the
    source's effective mass, so the lagged coupler holds a strong pair (m_a/m_s ~ 1) without ADMM (SCR-WP9-2): the
    heave stays bounded and the box floats. Measured over steps 300-1000 (docs/notes/wp9-results.md): mean draft
    -1.9 % with a 0.13 mm heave (pressure_solver "cg"), -2.8 % / 0.09 mm with Jacobi."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    cfg = default_config()
    sc = floating_box(**BOX, cfg=cfg)
    m = sc.model
    box = int(sc.bodies["box"])
    solver = SolverCoupledProxy(
        model=m,
        entries=[
            SolverCoupledProxy.Entry(name="mujoco", bodies=[box], joints=list(range(m.joint_count)),
                                     solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=100,
                                                                   nconmax=100)),
            SolverCoupledProxy.Entry(name="sph", solver=lambda v: SolverDFSPH(v, cfg),
                                     particles=list(range(m.particle_count))),
        ],
        coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(
            source="mujoco", destination="sph", bodies=[box], mode="lagged",
            collision_pipeline=lambda _m: None)], iterations=2),
    )
    sph = solver.solver("sph")
    assert sph._body_mode.numpy()[box] == 1  # the proxy is integrated inside SolverDFSPH
    st0, st1 = m.state(), m.state()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st0)
    sph.prime(st0, DT)
    pipe = newton.CollisionPipeline(m, soft_contact_max=0)
    contacts, ctrl = pipe.contacts(), m.control()
    drafts = []
    for k in range(700):
        st0.clear_forces()
        pipe.collide(st0, contacts)
        solver.step(st0, st1, ctrl, contacts, DT)
        st0, st1 = st1, st0
        if k >= 200:
            drafts.append(_draft(sc, st0.body_q.numpy()[box]))
    expect = sc.bodies["draft"]
    assert np.all(np.isfinite(drafts)) and sph.report().escaped[0] == 0
    assert np.ptp(drafts[-300:]) < 0.15 * expect  # bounded heave
    assert abs(np.mean(drafts) - expect) / expect < 0.15  # floats (the bias is reported, not hidden)


def test_floor_load_equals_liquid_weight():
    """Momentum balance at rest: time-averaged fluid impulse on the tank floor = liquid weight (within 3 %)."""
    sc = water_column(6, 6, 16)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    a, b = run(s, a, b, 100, DT)
    s.reset_shape_meters()
    run(s, a, b, 200, DT)
    floor = sc.tank[0]["floor"]
    force = s.shape_impulse()[floor] / (200 * DT)
    weight = float(sc.model.particle_mass.numpy().sum()) * 9.81
    assert force[2] < 0.0  # the liquid pushes the floor down
    assert abs(-force[2] - weight) / weight < 0.03
    side = s.shape_impulse()[sc.tank[0]["x_lo"]] / (200 * DT)
    assert side[0] < 0.0  # and the x_lo wall outward (-x)


@pytest.mark.parametrize("speed", [0.05, 0.2])
def test_moving_wall_work_balances_the_liquid_energy(speed):
    """A kinematic floor lifting a resting column: its reported work (``alloy_shape_work``) pays for the liquid's
    energy gain, less what the start of the lift dissipates (measured 2 % at 0.05 m/s, 7 % at 0.2 m/s); static walls
    do none. Without this term the bully ledger read a driven cup's work on the liquid as unexplained gain."""
    o = lifting_floor_energy(speed)
    assert o["w_static"] == 0.0
    assert 0.9 * o["w_floor"] < o["de"] <= o["w_floor"], o
