# SPDX-License-Identifier: Apache-2.0
"""The opt-in self-contact hold of the Chebyshev step (alloy.solvers.vbd.accel, ``hold_self_contact``;
``alloy_chebyshev_hold_self_contact=True``): particles of a self-contact pair keep the plain iterate, as body-contact
particles do. Extrapolated fold plies keep a folded towel moving (known gap cloth-fold-unsettled-default)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.solvers.vbd import SolverVBDAlloy
from alloy.solvers.vbd import scenes as S
from alloy.solvers.vbd.accel import AccelConfig

CHEB = {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9}


def _stack(hold: bool, steps: int, iterations: int = 10):
    """A smaller sheet resting on a pinned one 5 cm above the ground: its only support is self contact."""
    sc = S.build("stack")
    solver = SolverVBDAlloy(sc.model, **S.solver_kwargs(sc, iterations), **CHEB, alloy_chebyshev_hold_self_contact=hold)
    m = sc.model
    s0, s1, ctrl, contacts = m.state(), m.state(), m.control(), sc.pipeline.contacts()
    lo, hi = sc.sheets
    speeds = []
    for _ in range(steps):
        s0.clear_forces()
        sc.pipeline.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, 1.0e-3)
        s0, s1 = s1, s0
        speeds.append(float(np.linalg.norm(s0.particle_qd.numpy()[hi[0]:hi[1]], axis=1).mean()))
    return sc, solver, s0, contacts, np.array(speeds)


def _pair_particles(solver) -> np.ndarray:
    """Particles of the self-contact pairs of the solver's last detection (vertex, edge and triangle counts)."""
    info, m = solver.trimesh_collision_info, solver.model
    out = np.zeros(m.particle_count, bool)
    out[np.flatnonzero(info.vertex_colliding_triangles_count.numpy()[:m.particle_count] > 0)] = True
    for c, rows, first in ((info.edge_colliding_edges_count, m.edge_indices, 2),
                           (getattr(info, "triangle_colliding_vertices_count", None), m.tri_indices, 0)):
        if c is not None:
            r = rows.numpy()
            out[r[np.flatnonzero(c.numpy()[:len(r)] > 0), first:].ravel()] = True
    return out


def test_config_default_is_off():
    assert AccelConfig().hold_self_contact is False
    assert AccelConfig("adaptive", 0.9, True).hold_self_contact is True


def test_cloth_default_and_super_hold_self_contact():
    """The solver's own default stays off; the cloth settings that run the Chebyshev step turn the hold on (thirds
    fold, 8 CPU starts: 0 of 8 over the 25 mm gate, against 5 of 8 without; known gap cloth-fold-unsettled-default)."""
    from alloy.scene import cloth_accuracy as CA

    for name in ("default", "super"):
        assert CA.KWARGS[name]["alloy_chebyshev_rho"] and CA.KWARGS[name]["alloy_chebyshev_hold_self_contact"] is True
    assert CA.KWARGS["fast"]["alloy_chebyshev_rho"] is None          # no Chebyshev step: nothing to hold


@pytest.mark.parametrize("hold", [False, True], ids=["off", "hold"])
def test_the_hold_marks_every_self_contact_particle(hold):
    """With the hold, every particle of a pair from the step's last detection is held (the held set also keeps the
    step's earlier detections' pairs); without it, a stack touching no body holds none."""
    sc, solver, _, contacts, _ = _stack(hold, 40)
    pairs = _pair_particles(solver)
    held = solver._accel.held.numpy().astype(bool)
    assert int(contacts.soft_contact_count.numpy()[0]) == 0 and pairs.any()
    if hold:
        assert held[pairs].all()
    else:
        assert not held.any()


def test_the_hold_settles_a_resting_stack_and_stays_penetration_free():
    """The upper sheet of the stack moves 4x less with the hold (CPU: mean speed over the last 20 of 150 steps
    7.2e-7 vs 2.8e-6 m/s); the held plies still go through the truncation."""
    v = {}
    for hold in (False, True):
        sc, _, state, _, speeds = _stack(hold, 150)
        lo, hi = sc.sheets
        q = state.particle_q.numpy()
        assert np.isfinite(q).all() and q[hi[0]:hi[1], 2].min() > S.STACK_Z
        v[hold] = speeds[-20:].mean()
    assert v[True] < 0.5 * v[False], v
