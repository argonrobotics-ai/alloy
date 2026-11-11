# SPDX-License-Identifier: Apache-2.0
"""SCR-SC8-3: with VBD self-contact on, a piled film must not sink through a static ground plane.

Fixture `data/scrunched_pile.npz`: the C16 film bag (2,009 particles, 3,936 triangles, 15 um HDPE, 2.7 g) scrunched
by hand and saved at the release (sc8-dev d406c5a2, bench/clips/bag/c16_scrunch.py, 5 ms step), many layers deep.
Here it is lifted 5 mm clear of the ground and settled with plain cloth VBD: ground plane with C16's plywood material
(ke 2.3e5, kd 960, mu 0.28), soft_contact_ke 300, C16's self-contact settings, 20 iterations, 5 ms steps.
Stock SolverVBD sank it before Newton release-1.8 (the self-contact truncation cut the bottom layer's climb out of the
ground, nothing cut its descent); SolverVBDAlloy truncates against the static plane as well and keeps every particle
out. On release-1.8 stock keeps it out too (first test).
docs/notes/wp11-results.md (SCR-SC8-3).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import SolverVBDAlloy, SolverVBDMomentum

wp.config.quiet = True
DATA = Path(__file__).parent / "data" / "scrunched_pile.npz"
LIFT, DT, STEPS = 5.0e-3, 5.0e-3, 20
SELF_CONTACT = dict(particle_enable_self_contact=True, particle_self_contact_radius=5.0e-4,
                    particle_self_contact_margin=1.0e-3, particle_topological_contact_filter_threshold=2,
                    particle_rest_shape_contact_exclusion_radius=0.0, particle_collision_detection_interval=0)


def _model(device: str = "cpu") -> tuple[newton.Model, np.ndarray]:
    p = np.load(DATA)
    b = newton.ModelBuilder()
    cfg = b.default_shape_cfg.copy()
    cfg.ke, cfg.kd, cfg.mu = 2.304e5, 960.0, 0.28
    b.add_ground_plane(cfg=cfg)
    b.add_cloth_mesh(pos=wp.vec3(0.0, 0.0, LIFT), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(), vertices=p["q"],
                     indices=p["tri"].reshape(-1).tolist(), density=1.0, tri_ke=1.8, tri_ka=0.0, tri_kd=0.0,
                     edge_ke=1.0e-6, edge_kd=0.0, particle_radius=5.0e-4)
    b.color()
    m = b.finalize(device=device)
    m.particle_mass.assign(p["mass"])
    m.particle_inv_mass.assign(np.where(p["mass"] > 0.0, 1.0 / np.maximum(p["mass"], 1e-30), 0.0).astype(np.float32))
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = 300.0, 1.0e-2, 0.3
    return m, p["qd"]


def _settle(cls: type, steps: int = STEPS, self_contact: bool = True) -> np.ndarray:
    m, qd = _model()
    kw = dict(iterations=20, friction_epsilon=1.0e-3, rigid_body_particle_contact_buffer_size=8192)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)         # Newton's self-contact radius rename
        solver = cls(m, **kw, **(SELF_CONTACT if self_contact else {}))
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1 = m.state(), m.state()
    s0.particle_qd.assign(qd)
    for _ in range(steps):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, None, contacts, DT)
        s0, s1 = s1, s0
    return s0.particle_q.numpy()


def test_stock_vbd_keeps_the_pile_out_since_newton_1p8():
    """Flipped at the Newton release-1.8 bump (NEWTON-BUMP, 2026-11-09): stock SolverVBD no longer sinks the pile
    (lowest particle 0.47 mm above the plane, none below), where the previous pin (ac7f7830 + our commits) left
    particles below it. Which upstream change did it is not yet traced; SolverVBDAlloy's static-plane truncation
    stays until it is (the next test)."""
    z = _settle(newton.solvers.SolverVBD)[:, 2]
    assert z.min() >= 0.0, (int((z < 0.0).sum()), float(z.min()))


def test_alloy_vbd_keeps_the_pile_out_of_the_ground():
    for cls in (SolverVBDAlloy, SolverVBDMomentum):
        z = _settle(cls)[:, 2]
        assert z.min() >= 0.0, (cls.__name__, int((z < 0.0).sum()), float(z.min()))


def test_without_self_contact_it_is_bitwise_stock():
    a = _settle(newton.solvers.SolverVBD, steps=4, self_contact=False)
    b = _settle(SolverVBDAlloy, steps=4, self_contact=False)
    assert np.array_equal(a, b)
