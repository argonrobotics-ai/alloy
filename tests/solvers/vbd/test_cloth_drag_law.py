# SPDX-License-Identifier: Apache-2.0
"""Law (SCR-WP11-6 step 2): a pad pressed on a cloth that lies on a table drags it as Coulomb says.

The pad's friction on the vertices under it (mu 0.95, pad mu 2 against the cloth's 0.45) is far above the table's (mu
0.42), so Coulomb has the sheet follow the pad; the law asks for 90 % of the pad's travel. Built through ``make_vbd``
(the cloth entry's solver: log-barrier body contact, momentum coarse step), 4 ms step, 10 iterations, CPU. The pad is
pressed 0.1 mm into the cloth: under the barrier's onset (k p^2 (2w - p) / w^2, w = r / 2) that is about the force
the linear penalty gives at 10 um.

Share of the pad's travel, make_vbd / plain SolverVBD with the linear penalty at 10 um:
- a pad (C7's 16 x 6.4 mm) on a frictionless table: 0.98 / 0.97. Passes.
- a plate over the whole patch, every vertex between plate and table: 0.87 / 0.93. The vertex block converges slowly
  between a sliding and a static friction. Known gap.
- a C7-size pad on the frictional table (2 vertices under it): 0.17 / 0.08; 160 iterations reach 0.98. The few
  vertices must pull a stiff sheet across the table, and the sweep converges slowly on that near-rigid translation
  (membrane tri_ke 1e2: 0.94, 1e3: 0.07, 1e4: 0.04, 1e5: 0.005; plain VBD). Known gap.
The momentum coarse step, which corrects the translation mode, skips components in contact. Both gaps stay strict
xfails until SCR-WP11-6 step 3 (scr/WP11.md).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import make_vbd

CELL, RADIUS, VERTEX_MASS = 0.0085, 1.0e-3, 2.0e-5     # C7's towel grid (alloy.coupling.pinch_rigs.TOWEL)
PAD_HALF = (0.008, 0.0032, 0.010)                      # C7's pad
SPEED, DT, DRAG_S = 0.0375, 4.0e-3, 0.2                # 7.5 mm of pad travel
PRESS_DEPTH = 1.0e-4                                   # [m] the pad's bottom below the cloth's top


def _model(half_xy: tuple[float, float], mu_table: float) -> tuple[newton.Model, int]:
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke, b.default_shape_cfg.kd = 2.5e5, 1.0e3
    table = b.default_shape_cfg.copy()
    table.mu = mu_table
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.01, cfg=table)
    pad_cfg = b.default_shape_cfg.copy()
    pad_cfg.mu = 2.0
    z = PAD_HALF[2] + 2.0 * RADIUS - PRESS_DEPTH
    pad = b.add_body(xform=wp.transform((0.0, -0.0195, z), wp.quat_identity()), mass=0.5, is_kinematic=True)
    b.add_shape_box(pad, hx=half_xy[0], hy=half_xy[1], hz=PAD_HALF[2], cfg=pad_cfg)
    b.add_cloth_grid(pos=wp.vec3(-0.03, -0.045, RADIUS), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=7, dim_y=11,
                     cell_x=CELL, cell_y=CELL, mass=VERTEX_MASS, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e-2,
                     edge_ke=1.0e-3, edge_kd=1.0e-5, particle_radius=RADIUS)
    b.color()
    m = b.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 1.0e4, 1.0e-2, 0.45
    return m, pad


def _drag(half_xy: tuple[float, float], mu_table: float = 0.4) -> float:
    """Travel of the vertices under the pad over the travel of the pad (the sheet's share of the pad's motion)."""
    m, pad = _model(half_xy, mu_table)
    solver = make_vbd(m, {"iterations": 10, "friction_epsilon": 1.0e-3})
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1, ctl = m.state(), m.state(), m.control()
    q0, b0 = m.particle_q.numpy().copy(), m.body_q.numpy()[pad].copy()
    under = np.flatnonzero((np.abs(q0[:, 0] - b0[0]) < half_xy[0]) & (np.abs(q0[:, 1] - b0[1]) < half_xy[1]))
    settle, steps = int(round(0.05 / DT)), int(round(DRAG_S / DT))
    for k in range(settle + steps):
        bq = s0.body_q.numpy()
        bq[pad, 1] = b0[1] + SPEED * max(0, k - settle) * DT
        s0.body_q.assign(bq)
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, DT)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    assert np.isfinite(q).all() and len(under) > 0
    return float((q[under, 1] - q0[under, 1]).mean()) / (SPEED * steps * DT)


def test_a_pad_on_a_frictionless_table_drags_the_sheet():
    assert _drag(PAD_HALF[:2], mu_table=0.0) > 0.9


@pytest.mark.xfail(strict=True, reason="known gap (SCR-WP11-6): a vertex between a sliding pad and a static table "
                   "converges slowly at 10 iterations (0.87 of the pad's travel)")
def test_a_plate_drags_the_sheet_across_a_frictional_table():
    assert _drag((0.06, 0.06)) > 0.9


@pytest.mark.xfail(strict=True, reason="known gap (SCR-WP11-6): a small pad's few vertices pull a stiff sheet across "
                   "a frictional table and VBD's sweep converges slowly on that translation (0.17 of the pad's travel "
                   "at 10 iterations); the momentum coarse step skips components in contact (step 3)")
def test_a_small_pad_drags_the_sheet_across_a_frictional_table():
    assert _drag(PAD_HALF[:2]) > 0.9
