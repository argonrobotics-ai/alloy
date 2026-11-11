# SPDX-License-Identifier: Apache-2.0
"""WP10 acceptance: after ≥ 100 tears and ≥ 100 blade cuts, the VBD adjacency and colouring arrays are bitwise
unchanged, so no rebuild is needed (D19). Covered arrays: Model.soft_mesh_adjacency_device, the solver's
particle_adjacency, particle_color_groups, particle_colors, and the element index arrays."""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton
from alloy.cards.types import Failure
from alloy.core.quantity import Q
from alloy.testing.topology import clamp_particles, cloth_kwargs, grid_tri_mesh, run_cut_fuzz, vbd_topology_arrays
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_cloth
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import BondKind, TopologyBudget


def test_vbd_arrays_unchanged_after_100_tears(device):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, t = grid_tri_mesh(24, 12, 5e-3)
    add_presplit_cloth(b, v, t, SplitPlan(spacing=1.0e-2, axes=(0, 1)), cloth_kwargs(edge_ke=1e-4),
                       budget=TopologyBudget(spare_vertices=10_000),
                       failure=Failure(fracture_strain=Q(0.03, "model", note="test")))
    b.color(include_bending=True)
    m = b.finalize(device=device)
    x0 = m.particle_q.numpy()
    edge = np.flatnonzero(np.isclose(x0[:, 0], 0.0) | np.isclose(x0[:, 0], 0.12) | np.isclose(x0[:, 1], 0.0)
                          | np.isclose(x0[:, 1], 0.06))
    clamp_particles(m, edge)
    centre = np.array([0.06, 0.03, 0.0])
    bonds = bonds_from_model(m)
    assert (bonds.kind.numpy() == int(BondKind.CLOTH_EDGE)).sum() >= 100
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=16))
    solver = newton.solvers.SolverVBD(m, iterations=10)
    before = vbd_topology_arrays(m, solver)
    s0, s1, c = m.state(), m.state(), m.control()
    for k in range(400):                                # equibiaxial stretch: every seam loads at once
        q = s0.particle_q.numpy()
        q[edge] = centre + (x0[edge] - centre) * (1.0 + 2.0e-4 * (k + 1))
        s0.particle_q.assign(q)
        solver.step(s0, s1, c, None, 1e-3)
        ctl.apply(s1)
        s0, s1 = s1, s0
        if (bonds.intact.numpy() == 0).sum() >= 100:
            break
    tears = int((bonds.intact.numpy() == 0).sum())
    for _ in range(20):                                 # VBD keeps stepping on the same arrays
        solver.step(s0, s1, c, None, 1e-3)
        s0, s1 = s1, s0
    after = vbd_topology_arrays(m, solver)
    print(f"fixed arrays: tears={tears}, arrays compared={len(before)}")
    assert tears >= 100
    assert before.keys() == after.keys()
    for k in before:
        np.testing.assert_array_equal(before[k], after[k], err_msg=k)
    assert np.isfinite(s0.particle_q.numpy()).all()


def test_vbd_arrays_unchanged_after_100_cuts(device):
    stats = run_cut_fuzz(device, 400, seed=7)
    print(f"fixed arrays: blade_cuts={stats['blade_cuts']}, events={stats['events']}")
    assert stats["blade_cuts"] >= 100
    assert stats["vbd_arrays_unchanged"] == 1.0
    assert stats["nan_strokes"] == 0
