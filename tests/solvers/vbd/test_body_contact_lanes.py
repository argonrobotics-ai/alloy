# SPDX-License-Identifier: Apache-2.0
"""Law (SCR-WP10-4, lever 4b): VBD's per-body sum of body-particle contacts gives the same body wrench at 4 and 128
lanes per body, also when the body's contact list overflows.

The vendored ``SolverVBD`` runs 128 lanes per rigid body on CUDA (``_body_particle_contact_threads``; Newton chose 128
only when a body owns >= 4,096 particles) and 4 on the CPU. A body whose contacts overflow its per-body list
(``rigid_body_particle_contact_buffer_size``, 256) makes every lane scan every soft contact of the model. Here a box
rests on a cloth lying on a table, with about 400 cloth particles under it. Three runs step it from one state:
- 4 lanes, list of 256 (overflow);
- 128 lanes, list of 256 (overflow; forced on the CPU);
- 4 lanes, list of 4,096 (no overflow): the reference.
The box's wrench (to 1e-5 of the contact force's size), pose, velocity and the cloth agree to float-sum order: on the
CPU the three runs differ by 1e-7 to 1e-6 of the force. A sum that missed one of the box's 529 contacts would be off by
about 1/529 of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import make_vbd

CELL, RADIUS, VERTEX_MASS, N = 0.006, 1.0e-3, 2.0e-5, 25    # a 15 cm sheet of 25 x 25 vertices
BOX_HALF, BOX_MASS = (0.06, 0.06, 0.02), 0.2                # about 21 x 21 vertices under the box
DT, STEPS = 2.0e-3, 4


def _model() -> tuple[newton.Model, int]:
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke, b.default_shape_cfg.kd = 2.5e5, 1.0e3
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.01)
    z = BOX_HALF[2] + 2.0 * RADIUS - 1.0e-4                  # the box's bottom 0.1 mm into the cloth's top
    box = b.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()), mass=BOX_MASS)
    b.add_shape_box(box, hx=BOX_HALF[0], hy=BOX_HALF[1], hz=BOX_HALF[2])
    half = 0.5 * CELL * (N - 1)
    b.add_cloth_grid(pos=wp.vec3(-half, -half, RADIUS), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=N - 1,
                     dim_y=N - 1, cell_x=CELL, cell_y=CELL, mass=VERTEX_MASS, tri_ke=1.0e3, tri_ka=1.0e3,
                     tri_kd=1.0e-2, edge_ke=1.0e-3, edge_kd=1.0e-5, particle_radius=RADIUS)
    b.color()
    m = b.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 1.0e4, 1.0e-2, 0.45
    return m, box


def _run(lanes: int, buffer: int) -> dict[str, np.ndarray]:
    m, box = _model()
    solver = make_vbd(m, {"iterations": 5, "rigid_body_particle_contact_buffer_size": buffer})
    solver._body_particle_contact_threads = lanes            # CPU default is 4; the launch reads it every iteration
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1, ctl = m.state(), m.state(), m.control()
    for _ in range(STEPS):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, DT)
        s0, s1 = s1, s0
    return {"count": solver.body_particle_contact_counts.numpy()[box],
            "force": solver.body_forces.numpy()[box], "torque": solver.body_torques.numpy()[box],
            "body_q": s0.body_q.numpy()[box], "body_qd": s0.body_qd.numpy()[box], "particle_q": s0.particle_q.numpy()}


@pytest.mark.timeout(600)
def test_an_overflowing_body_gets_the_same_wrench_at_4_and_128_lanes() -> None:
    four, many, ref = _run(4, 256), _run(128, 256), _run(4, 4096)
    assert four["count"] > 256 and many["count"] > 256, "the box's list must overflow"
    assert ref["count"] == four["count"] <= 4096
    f = float(np.linalg.norm(ref["force"]))   # one contact is ~f / 529: a sum missing one is off by 200x the tolerance
    assert f > BOX_MASS * 9.81, "the cloth carries the box"
    for run in (four, many):
        np.testing.assert_allclose(run["force"], ref["force"], rtol=0.0, atol=1.0e-5 * f)
        np.testing.assert_allclose(run["torque"], ref["torque"], rtol=0.0, atol=1.0e-5 * f * BOX_HALF[0])
        np.testing.assert_allclose(run["body_q"], ref["body_q"], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(run["body_qd"], ref["body_qd"], rtol=0.0, atol=1.0e-5)
        np.testing.assert_allclose(run["particle_q"], ref["particle_q"], rtol=0.0, atol=1.0e-7)


def test_the_lane_count_is_128_on_cuda_and_4_on_the_cpu() -> None:
    m, _ = _model()
    assert make_vbd(m, {"iterations": 1})._body_particle_contact_threads == 4
    if wp.is_cuda_available():
        gm = _model_on("cuda:0")
        assert make_vbd(gm, {"iterations": 1})._body_particle_contact_threads == 128


def _model_on(device: str) -> newton.Model:
    b = newton.ModelBuilder()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=BOX_MASS)
    b.add_shape_box(box, hx=BOX_HALF[0], hy=BOX_HALF[1], hz=BOX_HALF[2])
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=3, dim_y=3,
                     cell_x=CELL, cell_y=CELL, mass=VERTEX_MASS, particle_radius=RADIUS)
    b.color()
    return b.finalize(device=device)
