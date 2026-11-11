# SPDX-License-Identifier: Apache-2.0
"""Laws of the contact-aware coarse step (``make_vbd(..., coarse_contact=True)``, SCR-WP11-6 step 3 (D), opt-in).

- The cloth drag law's two known gaps pass with it: a plate and a C7-size pad drag the sheet across a frictional
  table (stock 0.87 and 0.17 of the pad's travel at 10 iterations; with it 0.97 and 0.97), also on a table top at
  0.75 m.
- Coulomb on a slope: a sheet lying on a table tilted below its friction angle stays put, and above it slides as a
  block with acceleration g (sin a - mu cos a). The slope is a flat table under tilted gravity (the same physics with
  an axis-aligned contact normal). With the step: 128.0 mm against the block's 128.7 mm (35 deg, mu 0.5, 0.4 s).
  Stock VBD holds the sheet at any friction (0.04 mm): iteration 1 meets the inertial guess's deep normal, so friction
  takes the slide back, and the sweep then recovers the near-rigid translation about 0.1 um per iteration (the drag
  law's mechanism). A strict xfail.
- On a table top at 0.75 m the slide fails with the step too (0.6 mm): docs/known_gaps/cloth-contact-depth-float32.md.
  A strict xfail.
- A sheet resting on a table stays put (the step adds no drift).
- Off (the default), the solver is the stock SolverVBDMomentum bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import make_vbd

from . import test_cloth_drag_law as drag

MU_EFF = 0.5                      # sheet-plane friction: VBD uses sqrt(cloth mu x shape mu)
CLOTH_MU = 0.45


def test_the_drag_law_gaps_pass_with_the_contact_step(monkeypatch):
    monkeypatch.setattr(drag, "make_vbd", lambda m, kw: make_vbd(m, {**kw, "coarse_contact": True}))
    assert drag._drag((0.06, 0.06)) > 0.9
    assert drag._drag(drag.PAD_HALF[:2]) > 0.9
    assert drag._drag(drag.PAD_HALF[:2], mu_table=0.0) > 0.9


def _slope(angle_deg: float, contact: bool, top_z: float = 0.0, seconds: float = 0.4,
           dt: float = 4.0e-3) -> tuple[float, float]:
    """Mean travel of a 0.06 x 0.06 m sheet down a slope of ``angle_deg`` over ``seconds``, and the block estimate
    1/2 g (sin a - mu cos a) t^2 (0 below the friction angle). The sheet lies on a flat table whose top is at ``top_z``;
    gravity is tilted by ``angle_deg`` about x."""
    a = math.radians(angle_deg)
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke, b.default_shape_cfg.kd = 2.5e5, 1.0e3
    cfg = b.default_shape_cfg.copy()
    cfg.mu = MU_EFF ** 2 / CLOTH_MU
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, top_z - 0.01), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.01,
                    cfg=cfg)
    b.add_cloth_grid(pos=wp.vec3(-0.03, -0.03, top_z + drag.RADIUS), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                     dim_x=6, dim_y=6, cell_x=0.01, cell_y=0.01, mass=drag.VERTEX_MASS, tri_ke=1.0e3,
                     tri_ka=1.0e3, tri_kd=1.0e-2, edge_ke=1.0e-3, edge_kd=1.0e-5, particle_radius=drag.RADIUS)
    b.color()
    m = b.finalize(device="cpu")
    m.gravity.assign(np.array([[0.0, -9.81 * math.sin(a), -9.81 * math.cos(a)]], dtype=np.float32))
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 1.0e4, 1.0e-2, CLOTH_MU
    solver = make_vbd(m, {"iterations": 10, "friction_epsilon": 1.0e-3, "coarse_contact": contact})
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1, ctl = m.state(), m.state(), m.control()
    q0 = m.particle_q.numpy().copy()
    for _ in range(int(round(seconds / dt))):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, dt)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    assert np.isfinite(q).all()
    travel = float(-(q - q0)[:, 1].mean())                              # down the slope: -y
    block = max(0.0, 0.5 * 9.81 * (math.sin(a) - MU_EFF * math.cos(a)) * seconds ** 2)
    return travel, block


@pytest.mark.parametrize("contact", [False, True], ids=["stock", "coarse_contact"])
def test_a_sheet_on_a_slope_holds_below_its_friction_angle(contact):
    travel, block = _slope(20.0, contact)                               # friction angle 26.6 deg
    assert block == 0.0 and abs(travel) < 1.0e-3, travel


def test_a_sheet_on_a_slope_slides_as_a_block_above_it():
    travel, block = _slope(35.0, True)
    assert travel == pytest.approx(block, rel=0.15), (travel, block)


@pytest.mark.xfail(strict=True, reason="known gap (SCR-WP11-6): stock VBD holds a sheet on a slope at any friction; "
                   "iteration 1 takes the slide back and the sweep recovers the translation ~0.1 um per iteration")
def test_a_sheet_on_a_slope_slides_as_a_block_above_it_stock():
    travel, block = _slope(35.0, False)
    assert travel == pytest.approx(block, rel=0.15), (travel, block)


@pytest.mark.xfail(strict=True, reason="known gap docs/known_gaps/cloth-contact-depth-float32.md: at 0.75 m a light "
                   "vertex's contact depth is below float32 position resolution, so friction loads are many times "
                   "the weight")
def test_a_sheet_on_a_slope_slides_as_a_block_on_a_table_top():
    travel, block = _slope(35.0, True, top_z=0.75)
    assert travel == pytest.approx(block, rel=0.15), (travel, block)


def _raise(m: newton.Model, dz: float) -> None:
    """Move the whole drag rig up by ``dz``: particles, bodies and the static shapes (body shapes follow their body)."""
    q = m.particle_q.numpy()
    q[:, 2] += dz
    m.particle_q.assign(q)
    bq = m.body_q.numpy()
    bq[:, 2] += dz
    m.body_q.assign(bq)
    st = m.shape_transform.numpy()
    st[m.shape_body.numpy() < 0, 2] += dz
    m.shape_transform.assign(st)


def test_a_pad_drags_the_sheet_on_a_table_top_with_the_contact_step(monkeypatch):
    real = drag._model

    def raised(half_xy, mu_table):
        m, pad = real(half_xy, mu_table)
        _raise(m, 0.75)
        return m, pad

    monkeypatch.setattr(drag, "make_vbd", lambda m, kw: make_vbd(m, {**kw, "coarse_contact": True}))
    monkeypatch.setattr(drag, "_model", raised)
    assert drag._drag(drag.PAD_HALF[:2]) > 0.9


def test_a_resting_sheet_stays_put_with_the_contact_step():
    m, _ = drag._model(drag.PAD_HALF[:2], 0.4)
    solver = make_vbd(m, {"iterations": 10, "friction_epsilon": 1.0e-3, "coarse_contact": True})
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1, ctl = m.state(), m.state(), m.control()
    for _ in range(25):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, drag.DT)
        s0, s1 = s1, s0
    q1 = s0.particle_q.numpy().copy()
    for _ in range(50):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, drag.DT)
        s0, s1 = s1, s0
    assert np.abs(s0.particle_q.numpy() - q1)[:, :2].max() < 1.0e-6


def test_off_is_the_stock_solver_bit_for_bit():
    def run(kw):
        m, pad = drag._model(drag.PAD_HALF[:2], 0.4)
        solver = make_vbd(m, {"iterations": 10, "friction_epsilon": 1.0e-3, **kw})
        pipe = newton.CollisionPipeline(m)
        contacts = pipe.contacts()
        s0, s1, ctl = m.state(), m.state(), m.control()
        for k in range(20):
            bq = s0.body_q.numpy()
            bq[pad, 1] += drag.SPEED * drag.DT
            s0.body_q.assign(bq)
            s0.clear_forces()
            pipe.collide(s0, contacts)
            solver.step(s0, s1, ctl, contacts, drag.DT)
            s0, s1 = s1, s0
        return s0.particle_q.numpy()

    np.testing.assert_array_equal(run({}), run({"coarse_contact": False}))
    with pytest.raises(ValueError, match="coarse_step"):
        make_vbd(drag._model(drag.PAD_HALF[:2], 0.4)[0], {"coarse_step": False, "coarse_contact": True})
