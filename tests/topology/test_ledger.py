# SPDX-License-Identifier: Apache-2.0
"""TOP-1: mass after/before per event, from the ledger, ≤ 1e-6 relative, for tears, cuts, weld breaks, crumbles.

``ledger()`` is read after every step, so each period with events holds the events of one step. The world totals
are float64 sums over active particles and bodies. Bond breaks do not change the state, so their momentum
balance must be exact. Crumble moves the chunk's momentum onto its particles (TOP-5 checks that against the
body); here only the mass balance is asserted for it.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.cards.types import Failure
from alloy.core.quantity import Q
from alloy.testing.topology import blade_pose, clamp_particles, cloth_kwargs, grid_tri_mesh, presplit_block
from alloy.topology.bonds import add_weld_bond, bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.crumble import add_crumble_source, box_hull
from alloy.topology.emit import add_presplit_cloth
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget

TOL = 1.0e-6


class Balance:
    """Collects per-period mass ratios and momentum differences for periods with events."""

    def __init__(self) -> None:
        self.mass_err: list[float] = []
        self.mom_err: list[float] = []
        self.events = 0

    def read(self, ctl: TopologyController) -> None:
        led = ctl.ledger()
        for w in np.flatnonzero(led.events > 0):
            self.events += int(led.events[w])
            self.mass_err.append(abs(led.mass_after[w] / led.mass_before[w] - 1.0))
            self.mom_err.append(float(np.abs(led.momentum_after[w] - led.momentum_before[w]).max()))

    def worst(self) -> tuple[float, float]:
        return max(self.mass_err), max(self.mom_err)


def test_ledger_tears(device):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, t = grid_tri_mesh(12, 6, 5e-3)
    add_presplit_cloth(b, v, t, SplitPlan(spacing=1.5e-2, axes=(0, 1)), cloth_kwargs(),
                       budget=TopologyBudget(spare_vertices=1000),
                       failure=Failure(fracture_strain=Q(0.05, "model", note="test")))
    b.color()
    m = b.finalize(device=device)
    x0 = m.particle_q.numpy()
    left, right = np.flatnonzero(np.isclose(x0[:, 0], 0.0)), np.flatnonzero(np.isclose(x0[:, 0], 0.06))
    clamp_particles(m, np.concatenate([left, right]))
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=4))
    solver = newton.solvers.SolverVBD(m, iterations=10)
    s0, s1, c = m.state(), m.state(), m.control()
    bal = Balance()
    for k in range(400):
        q = s0.particle_q.numpy()
        q[right, 0] = x0[right, 0] * (1.0 + 2.5e-4 * (k + 1))
        s0.particle_q.assign(q)
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
        bal.read(ctl)
    mass, mom = bal.worst()
    print(f"TOP-1 tears: events={bal.events} worst_mass_rel={mass:.3e} worst_momentum_abs={mom:.3e}")
    assert bal.events >= 10 and mass <= TOL and mom == 0.0


def test_ledger_cuts(device):
    sc = presplit_block(device, worlds=2, cells=(4, 3, 3), k_mu=5e4, k_lambda=5e4,
                        failure=Failure(fracture_strain=Q(0.03, "model", note="test")))
    m = sc.model
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=4), sc.blades)
    solver = newton.solvers.SolverVBD(m, iterations=4)
    s0, s1, c = m.state(), m.state(), m.control()
    bal = Balance()
    for k in range(60):
        bq = s0.body_q.numpy()
        for w, body in enumerate(sc.blade_bodies):
            p = np.array([0.01, 0.0075, 0.03 - 5e-4 * k])
            bq[body] = blade_pose(p, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        s0.body_q.assign(bq)
        ctl.pre_step(s0, None, 1e-3)
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
        bal.read(ctl)
    mass, mom = bal.worst()
    print(f"TOP-1 cuts: events={bal.events} worst_mass_rel={mass:.3e} worst_momentum_abs={mom:.3e}")
    assert bal.events >= 10 and mass <= TOL and mom == 0.0


def test_ledger_weld_breaks_and_crumble(device):
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(t)
    body = t.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    t.add_shape_box(body, hx=0.02, hy=0.02, hz=0.01)
    add_weld_bond(t, body, -1, break_force=1.0)                       # 0.016 kg box: weight 0.157 N
    add_crumble_source(t, body, box_hull(0.02, 0.02, 0.01), trigger="release",
                       budget=TopologyBudget(spare_particles=5000))
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
    b.replicate(t, 2)
    m = b.finalize(device=device)
    bonds = bonds_from_model(m)
    bonds.break_force.assign(np.array([1.0, 0.1], np.float32))      # world 1's weld breaks, then it crumbles
    solver = newton.solvers.SolverMuJoCo(m)
    ctl = TopologyController(m, bonds, TopologyBudget(spare_particles=5000), entries={"mujoco": solver})
    s0, s1, c = m.state(), m.state(), m.control()
    bal = Balance()
    for _ in range(40):
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
        bal.read(ctl)
    print(f"TOP-1 weld+crumble: events={bal.events} worst_mass_rel={max(bal.mass_err):.3e}")
    assert bal.events == 2                                           # one weld break, one crumble, world 1
    assert max(bal.mass_err) <= TOL
    np.testing.assert_array_equal(ctl.body_active.numpy(), [1, 0])
