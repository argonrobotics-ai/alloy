# SPDX-License-Identifier: Apache-2.0
"""emit.py: seam thresholds, mass conservation, seam fidelity against uncut cloth (spike 1), cross-seam bending."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.cards.types import Elastic, Failure
from alloy.core.errors import CapacityError
from alloy.core.quantity import Q
from alloy.testing.topology import clamp_particles, cloth_kwargs, grid_tet_mesh, grid_tri_mesh, soft_kwargs
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_cloth, add_presplit_soft, seam_thresholds
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget


def _q(x: float) -> Q:
    return Q(x, "model", note="test")


def test_seam_thresholds_table():
    meas = np.array([0.5e-3, 1.0e-3])
    f = Failure(fracture_strain=_q(0.2), fracture_stress=_q(4.0e6), tear_energy=_q(100.0), cut_min_strain=_q(0.3),
                cut_min_force=_q(2.0))
    e = Elastic(youngs=_q(1.0e7), poisson=_q(0.3))
    out = seam_thresholds(f, e, thickness=1e-3, measure=meas, element_size=5e-3, cloth=True)
    griffith = math.sqrt(2.0 * 100.0 / (1.0e7 * 1e-3 * 5e-3))       # 2.0 > 0.2, so fracture_strain wins
    np.testing.assert_allclose(out["break_strain"], min(0.2, griffith))
    np.testing.assert_allclose(out["break_force"], 4.0e6 * 1e-3 * meas)   # σ·h·tributary length [N]
    np.testing.assert_allclose(out["cut_strain"], 0.3)
    np.testing.assert_allclose(out["cut_force"], 2.0)
    tet = seam_thresholds(Failure(fracture_stress=_q(1.0e5)), None, thickness=None, measure=meas, element_size=0.0,
                          cloth=False)
    np.testing.assert_allclose(tet["break_force"], 1.0e5 * meas)          # σ·area [N]
    assert np.isinf(tet["break_strain"]).all() and np.isinf(tet["cut_force"]).all()
    nothing = seam_thresholds(None, None, thickness=None, measure=meas, element_size=0.0, cloth=True)
    assert all(np.isinf(v).all() for v in nothing.values())
    no_h = seam_thresholds(Failure(fracture_stress=_q(1.0e5)), None, thickness=None, measure=meas, element_size=0.0,
                           cloth=True)
    assert np.isinf(no_h["break_force"]).all()                           # no thickness, no cloth force criterion


def _mass(builder_fn) -> float:
    b = newton.ModelBuilder()
    builder_fn(b)
    return float(np.sum(np.asarray(b.particle_mass, dtype=np.float64)))


def test_presplit_conserves_mass_and_respects_budget():
    v, t = grid_tet_mesh(4, 2, 2, 5e-3)
    plan = SplitPlan(spacing=1e-2, axes=(0,))
    uncut = _mass(lambda b: b.add_soft_mesh(vertices=[wp.vec3(*p) for p in v], indices=t.ravel().tolist(),
                                            **soft_kwargs()))
    split = _mass(lambda b: add_presplit_soft(b, v, t, plan, soft_kwargs(), budget=TopologyBudget(spare_vertices=100)))
    assert split == pytest.approx(uncut, rel=1e-6)
    with pytest.raises(CapacityError):
        add_presplit_soft(newton.ModelBuilder(), v, t, plan, soft_kwargs(), budget=TopologyBudget(spare_vertices=2))


def _pulled_strip(device: str, split: bool, eps: float = 0.05, steps: int = 300) -> np.ndarray:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, t = grid_tri_mesh(24, 6, 5e-3)
    kw = cloth_kwargs(tri_ka=1e3, tri_kd=1.0)
    if split:
        add_presplit_cloth(b, v, t, SplitPlan(planes=(((1.0, 0.0, 0.0), 0.06),)), kw,
                           budget=TopologyBudget(spare_vertices=10))
    else:
        b.add_cloth_mesh(vertices=[wp.vec3(*p) for p in v], indices=t.ravel().tolist(), **kw)
    b.color()
    m = b.finalize(device=device)
    x0 = m.particle_q.numpy()
    right = np.flatnonzero(np.isclose(x0[:, 0], 0.12))
    clamp_particles(m, np.concatenate([np.flatnonzero(np.isclose(x0[:, 0], 0.0)), right]))
    solver = newton.solvers.SolverVBD(m, iterations=10)
    s0, s1, c = m.state(), m.state(), m.control()
    for k in range(steps):
        q = s0.particle_q.numpy()
        q[right, 0] = x0[right, 0] * (1.0 + eps * min(1.0, (k + 1) / (0.6 * steps)))
        s0.particle_q.assign(q)
        solver.step(s0, s1, c, None, 1e-3)
        s0, s1 = s1, s0
    return s0.particle_q.numpy()[: len(v)]


def test_seam_matches_uncut_cloth(device):
    """Twin springs at 100 x tri_ke: the pre-split strip follows the uncut one (spike 1: 0.4 % of displacement)."""
    uncut, split = _pulled_strip(device, False), _pulled_strip(device, True)
    err = np.abs(split - uncut).max()
    print(f"seam fidelity: max |dx| {err:.3e} m at 3 mm mid-strip displacement")
    assert err < 0.01 * 3e-3 * 2                       # < 2 % of the 3 mm mid-strip displacement


def test_cross_seam_bending_is_dropped_on_break(device):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, t = grid_tri_mesh(4, 2, 0.01)
    rec = add_presplit_cloth(b, v, t, SplitPlan(spacing=0.02, axes=(0,)), cloth_kwargs(edge_ke=1e-3),
                             budget=TopologyBudget(spare_vertices=10))
    assert rec.cross_seam_edges == 2                    # two seam edges between three seam vertices
    m = b.finalize(device=device)
    tagged = np.flatnonzero(m.topology.edge_bond0.numpy() >= 0)
    assert len(tagged) == 2 and np.all(m.edge_bending_properties.numpy()[tagged, 0] > 0)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget())
    middle = int(np.argmax(m.topology.bond_measure.numpy()))          # the middle seam vertex touches both edges
    ctl.preset_broken([middle])
    assert np.all(m.edge_bending_properties.numpy()[tagged, 0] == 0.0)
    others = np.setdiff1d(np.arange(m.edge_count), tagged)
    assert np.all(m.edge_bending_properties.numpy()[others, 0] > 0)
