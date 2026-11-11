# SPDX-License-Identifier: Apache-2.0
"""Bonds on curved substrates (SCR-WP8-11 Part B3): cylinder and capsule distances for pre-bonds, the flat rest layout
of a label posed wrapped, a roller's Hertz contact on a parallel tube (1/R_e = 1/R + 1/R_s, offsets along the tube),
and law C-1 at two cell sizes (laws.py). CPU."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.bonds import kernels as K
from alloy.plugins.bonds.layout import shape_distance, unfold, vertex_areas
from alloy.plugins.bonds.rig import TubeRig, model
from alloy.plugins.bonds.wetout import STANDARD_ROLLER, roller_pressure, rollers_struct

R_S, ALONG_Y = 0.025, wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)


def _shapes() -> newton.Model:
    b = newton.ModelBuilder()
    b.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), ALONG_Y), radius=R_S, half_height=0.03,
                         label="tube")
    b.add_shape_capsule(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.01,
                        half_height=0.02, label="cap")
    return b.finalize(device="cpu")


def test_cylinder_and_capsule_distances():
    m = _shapes()
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.05, 0.05, (300, 3))
    d, n = shape_distance(m, np.zeros((0, 7)), 0, pts)                # axis along world y
    rr, dy = np.hypot(pts[:, 0], pts[:, 2]) - R_S, np.abs(pts[:, 1]) - 0.03
    want = np.where((rr <= 0) & (dy <= 0), np.maximum(rr, dy), np.hypot(np.maximum(rr, 0), np.maximum(dy, 0)))
    np.testing.assert_allclose(d, want, atol=2e-8)                      # float32 shape frames
    d, n = shape_distance(m, np.zeros((0, 7)), 1, pts)                # capsule along z
    seg = np.stack([np.zeros(300), np.zeros(300), np.clip(pts[:, 2], -0.02, 0.02)], 1)
    np.testing.assert_allclose(d, np.linalg.norm(pts - seg, axis=1) - 0.01, atol=2e-8)
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)


def test_a_wrapped_label_unfolds_to_its_flat_layout():
    """The tube rig poses its label wrapped; unfolding the rest frames gives back the flat grid's distances."""
    rig = TubeRig(dx=2.5e-3, length=0.010, width=0.010)
    m = model(rig, 1)
    flat = unfold(m.tri_indices.numpy(), m.tri_poses.numpy(), int(m.particle_count))
    nx, ny = int(round(rig.length / rig.dx)) + 1, int(round(rig.width / rig.dx)) + 1
    grid = np.stack(np.meshgrid(np.arange(nx) * rig.dx, np.arange(ny) * rig.dx, indexing="ij"), -1).reshape(-1, 2)
    d_flat = np.linalg.norm(flat[:, None, :2] - flat[None, :, :2], axis=2)
    d_grid = np.linalg.norm(grid[:, None] - grid[None], axis=2)
    np.testing.assert_allclose(d_flat, d_grid, atol=1e-7)
    posed = m.particle_q.numpy().astype(np.float64)
    assert np.ptp(posed[:, 2]) > 1e-4                                  # it is wrapped: not a plane
    np.testing.assert_allclose(np.hypot(posed[:, 0] - 0.5 * rig.length, posed[:, 2] + R_S), R_S + rig.z_film,
                               rtol=1e-6)
    np.testing.assert_allclose(vertex_areas(flat, m.tri_indices.numpy()).sum(), rig.length * rig.width, rtol=1e-9)


def test_a_label_wrapped_on_a_tube_starts_bonded_there():
    from alloy.plugins.bonds.plugin import Bonds
    from alloy.plugins.bonds.rig import simulation

    rig = TubeRig(dx=2.5e-3, length=0.010, width=0.010)
    bonds = Bonds(substrates=("tube",), adhesives=("3m_7818/stainless_steel",), grip="", prebond=True,
                  prebond_reach=1.0e-4)
    simulation(rig, [bonds], worlds=1)
    d = bonds.read()
    assert np.all(d["bond_state"] == 1)
    a = d["anchor"][0].astype(np.float64)                             # static tube: world frame
    np.testing.assert_allclose(np.hypot(a[:, 0] - 0.5 * rig.length, a[:, 2] + R_S), R_S + rig.z_film, rtol=1e-5)


@pytest.mark.parametrize("dx", [2.5e-3, 1.25e-3])
def test_the_tube_roller_starts_just_above_the_labels_leading_edge(dx):
    """WP5's red (2026-11-10): the roller started where it would rest on a label at x = -lead, where there is none, and
    fell 0.53 mm onto the posed label's edge; at 1.25 mm cells the knock slid the label around the tube and off it
    (C-1 read no load). It now drops LAND_GAP onto the leading edge, the first row it meets."""
    from alloy.plugins.bonds.rig import LAND_GAP

    rig = TubeRig(dx=dx, length=0.010, width=0.010, dead_weight=True, link_mass=2.04, dt=1.0e-3)
    m = model(rig, 1)
    q = m.particle_q.numpy().astype(np.float64)
    zb = m.body_q.numpy()[:, 2].astype(np.float64)
    np.testing.assert_allclose(zb, rig.roller_z, atol=1e-7)             # both gantry links start at the roller
    touch, off = rig.card.radius + rig.z_film, q[:, 0] + rig.lead
    drop = rig.roller_z - (q[:, 2] + np.sqrt(touch * touch - off * off))  # fall before it touches each vertex
    np.testing.assert_allclose(drop.min(), LAND_GAP, atol=1e-6)
    assert q[np.argmin(drop), 0] == pytest.approx(q[:, 0].min(), abs=1e-9)  # the leading edge is met first


@wp.kernel
def _pressure(rl: K.Rollers, x: wp.array[wp.vec3], nrm: wp.array[wp.vec3], sub: int, rp: float,
              body_q: wp.array[wp.transform], shape_body: wp.array[wp.int32], xf: wp.array[wp.transform],
              out: wp.array[float]):
    i = wp.tid()
    out[i] = roller_pressure(rl, 0, x[i], nrm[i], sub, rp, body_q, shape_body, xf)


@pytest.mark.parametrize("parallel", [True, False])
def test_a_roller_on_a_parallel_tube_presses_with_the_tubes_hertz_contact(parallel):
    """On a tube parallel to it the roller's contact is Hertz with R_e and offsets along the tube's surface: narrower
    and higher than on a flat face. Across the tube (a point contact, not modelled) it keeps R."""
    card, q, rp = STANDARD_ROLLER, 450.0, 5.0e-4
    reach = R_S + 2.0 * rp + card.radius
    roller_rot = ALONG_Y if parallel else wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)  # along x
    xf = wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.0), ALONG_Y),
                   wp.transform(wp.vec3(0.0, 0.0, reach), roller_rot)], dtype=wp.transform, device="cpu")
    load = wp.full((1, 1), q, dtype=float, device="cpu")
    rl = rollers_struct((card,), np.array([[1]]), 2, load, wp.zeros_like(load), 2.0e-3, "cpu",
                        sub_radius=np.array([R_S, 0.0]))
    psi = np.linspace(-0.06, 0.06, 241)                                # angle on the tube from its top
    x = np.stack([(R_S + rp) * np.sin(psi), np.zeros_like(psi), (R_S + rp) * np.cos(psi)], 1)
    nrm = np.stack([np.sin(psi), np.zeros_like(psi), np.cos(psi)], 1)
    out = wp.zeros(len(psi), dtype=float, device="cpu")
    wp.launch(_pressure, dim=len(psi), inputs=[rl, wp.array(x, dtype=wp.vec3, device="cpu"),
                                               wp.array(nrm, dtype=wp.vec3, device="cpu"), 0, rp,
                                               wp.zeros(1, dtype=wp.transform, device="cpu"),
                                               wp.array([-1, -1], dtype=wp.int32, device="cpu"), xf],
              outputs=[out], device="cpu")
    got = out.numpy()
    if not parallel:                                                   # across: the line law with R at the top
        np.testing.assert_allclose(got[len(psi) // 2], card.pressure(0.0, q), rtol=1e-5)
        return
    want = card.pressure((R_S + rp) * np.sin(psi), q, R_S)
    np.testing.assert_allclose(got, want, atol=2e-3 * want.max())
    assert want.max() > 1.6 * card.pressure(0.0, q)                    # 1.7x the flat peak at R_e = 16.4 mm


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("name", ["C-1_tube_dx2.5", "C-1_tube_dx1.25", "GA_standard_flat", "GA_standard_tube"])
def test_c1_a_label_rolled_onto_a_tube(name):
    """Rows (CPU, 1 world, main 2026-11-10 with the LAND_GAP start): load 441.7 / 438.6 vs 450.2 N/m, mean w 0.4417 /
    0.4415 vs the tube's Hertz 0.4488 / 0.4465 at 2.5 / 1.25 mm cells (flat would be 0.4308 / 0.4283), all bonded;
    at 1.25 mm also with the roller started 1 mm nearer or farther (lead 14 / 16 mm: w 0.4445 / 0.4404)."""
    from alloy.plugins.bonds.laws import bonds_laws
    from alloy.plugins.bonds.plugin import Bonds
    from alloy.plugins.harness.run import run_scene

    law = next(x for x in bonds_laws() if x.name == name)
    run = run_scene(law.scene, [Bonds(**dict(law.config))], "cpu")
    assert run.error is None, run.error
    res = law.check(run)
    assert res.passed, res
