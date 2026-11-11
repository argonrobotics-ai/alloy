# SPDX-License-Identifier: Apache-2.0
"""TOP-4: tear onset strain vs ``fracture_strain`` on a pre-split cloth strip pulled across and along the split lines.

The strip is 24 x 12 cells of 5 mm, with seams every 3 cells. Pattern axes (0,) give lines x = const, across the
pull; (1,) gives lines y = const, along it. Both ends are clamped and the right end moves at a constant strain
rate. tri_kd = 0: VBD damps the in-step displacement from state_in, which a clamp written into state_in
lacks (it would load the last column). With tri_ka = 0 (λ = 0, Poisson ratio 0) the
field is uniform, so the onset is read as the global strain ΔL/L at the first break. The target is ≤ 10 %
(SPEC §5.2). A λ = μ variant reports the local strain at the break.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.cards.types import Failure
from alloy.core.quantity import Q
from alloy.testing.topology import clamp_particles, cloth_kwargs, grid_tri_mesh
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_cloth
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import TopologyBudget

log = logging.getLogger("alloy.topology.tests")

FRACTURE = 0.10
NX, NY, CELL, SPACING = 24, 12, 5.0e-3, 3


def tri_principal_strain(model: newton.Model, x: np.ndarray) -> np.ndarray:
    """Largest engineering principal strain per triangle (numpy, same rest frame as geometry.tri_stretch_range)."""
    ti, poses = model.tri_indices.numpy(), model.tri_poses.numpy()
    d1, d2 = x[ti[:, 1]] - x[ti[:, 0]], x[ti[:, 2]] - x[ti[:, 0]]
    f1 = d1 * poses[:, 0, 0, None] + d2 * poses[:, 1, 0, None]
    f2 = d1 * poses[:, 0, 1, None] + d2 * poses[:, 1, 1, None]
    a, b, c = (f1 * f1).sum(1), (f1 * f2).sum(1), (f2 * f2).sum(1)
    return np.sqrt(0.5 * (a + c + np.sqrt((a - c) ** 2 + 4 * b * b))) - 1.0


def build_strip(device: str, axes: tuple[int, ...], tri_ka: float = 0.0):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    v, t = grid_tri_mesh(NX, NY, CELL)
    fail = Failure(fracture_strain=Q(FRACTURE, "model", note="TOP-4 test value"))
    add_presplit_cloth(b, v, t, SplitPlan(spacing=SPACING * CELL, axes=axes), cloth_kwargs(tri_ka=tri_ka, tri_kd=0.0),
                       budget=TopologyBudget(spare_vertices=10_000), failure=fail)
    b.color()
    m = b.finalize(device=device)
    x0 = m.particle_q.numpy()
    left, right = np.flatnonzero(np.isclose(x0[:, 0], 0.0)), np.flatnonzero(np.isclose(x0[:, 0], NX * CELL))
    clamp_particles(m, np.concatenate([left, right]))
    return m, x0, right


def pull_until_tear(device: str, axes: tuple[int, ...], tri_ka: float = 0.0, rate: float = 0.2, dt: float = 1.0e-3):
    """(global strain at first break, interior material strain then, bond strain at break, broken bond sources)."""
    m, x0, right = build_strip(device, axes, tri_ka)
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=64))
    solver = newton.solvers.SolverVBD(m, iterations=10)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    length = NX * CELL
    inner = np.abs(x0[m.tri_indices.numpy()].mean(1)[:, 0] - 0.5 * length) < 0.3 * length
    for k in range(int(0.15 / rate / dt)):
        eps = rate * (k + 1) * dt
        q, qd = s0.particle_q.numpy(), s0.particle_qd.numpy()
        q[right, 0], qd[right, 0] = x0[right, 0] * (1.0 + eps), rate * length
        s0.particle_q.assign(q)
        s0.particle_qd.assign(qd)
        ctl.pre_step(s0, None, dt)
        solver.step(s0, s1, ctrl, None, dt)
        ctl.apply(s1)
        s0, s1 = s1, s0
        if ctl.step_events.numpy().sum() > 0:
            broken = np.flatnonzero(bonds.intact.numpy() == 0)
            interior = float(tri_principal_strain(m, s0.particle_q.numpy())[inner].mean())
            return eps, interior, float(ctl.strain.numpy()[broken].max()), x0[bonds.a.numpy()[broken]]
    pytest.fail(f"no tear up to global strain {eps:.3f} (axes {axes})")


@pytest.mark.parametrize("axes,label", [((0,), "across"), ((1,), "along")])
def test_tear_onset_matches_fracture_strain(device, axes, label):
    eps, interior, at_break, where = pull_until_tear(device, axes)
    err = abs(eps - FRACTURE) / FRACTURE
    log.info("TOP-4 %s: onset global %.4f, interior %.4f, bond %.4f, error %.2f %%", label, eps, interior, at_break,
             100 * err)
    print(f"TOP-4 {label}: onset_global={eps:.4f} interior={interior:.4f} bond_at_break={at_break:.4f} "
          f"err={100 * err:.2f}%")
    assert err <= 0.10
    assert at_break == pytest.approx(FRACTURE, rel=0.02)      # the criterion fires at the card value
    if label == "along":
        rows = np.round(where[:, 1] / CELL).astype(int)
        assert np.all(rows % SPACING == 0)                    # only lines y = const (the along family) opened


def test_tear_poisson_reports_local_strain(device):
    """λ = μ: the clamps make the field non-uniform; the break still fires at the local fracture strain."""
    eps, interior, at_break, _ = pull_until_tear(device, (0,), tri_ka=1.0e3)
    print(f"TOP-4 poisson: onset_global={eps:.4f} interior={interior:.4f} bond_at_break={at_break:.4f}")
    assert at_break == pytest.approx(FRACTURE, rel=0.02)
    assert 0.5 * FRACTURE < eps < 1.5 * FRACTURE
