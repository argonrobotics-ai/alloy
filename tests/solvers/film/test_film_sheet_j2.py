# SPDX-License-Identifier: Apache-2.0
"""J2 metal sheets on the film route (SCR-WP7-15 P3): the hinge's fibre section (sheet_j2.py), law S1, its state
(reset, rollback) and default off. CPU, small."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.film import sheet_laws as L
from alloy.solvers.film.emit import add_film_mesh, hinge_length
from alloy.solvers.film.materials import FilmSpec
from alloy.solvers.film.rigs import FilmRig, grid_mesh, pin

pytestmark = pytest.mark.compile_heavy


def test_s1_moment_follows_the_plane_strain_section_and_the_rest_keeps_the_springback():
    """6061-T6, 0.5 mm, bent isometrically to 20 x the first-yield curvature: each bending hinge's moment is the
    plane-strain section's (onset at σ_y' t²/6 exact, the 21-layer quadrature within 0.4 % to 0.98 M_p and past it
    with hardening), all hinges alike, and its rest angle keeps κ - M/D."""
    rows = L.s1_pure_bending(steps=20)
    spec = L.metal_spec("aluminium_6061_t6", 5.0e-4)
    Ep, syp, _, _ = L.section(spec)
    onset = rows[0]                                                    # k_max / steps = 1: exactly at first yield
    assert onset.moment == pytest.approx(syp * 5.0e-4**2 / 6.0, rel=1e-5) and onset.kept < 1e-3
    for r in rows:
        assert r.moment == pytest.approx(r.moment_cf, rel=4e-3), r
        assert r.spread < 1e-4, r
        if r.kept_cf > 1.0:
            assert r.kept == pytest.approx(r.kept_cf, rel=5e-3), r


def test_s1_springback_is_gardiners_for_a_perfectly_plastic_card():
    """Brass C360 has no hardening: the curvature kept after release from κ0 is Gardiner's (R0/Rf = 4a³ - 3a + 1)."""
    rows = L.s1_pure_bending(material="brass_c360", steps=8, k_max=8.0)
    for r in rows[1:]:
        assert r.kept == pytest.approx(L.gardiner_kept("brass_c360", 5.0e-4, r.kappa), rel=5e-3), r


def test_s1_off_axis_measures_the_hinge_networks_anisotropy():
    """A measured row (WP11's review), pinned so a change to the network shows. Bent along the grid the network is the
    plate. Off the grid the alternating-diagonal hinges read twist with weight 4 where a plate's cylinders take 2: the
    elastic network energy is κxx² + κyy² + 4 κxy², i.e. 1 + sin²2φ/2 x the plate's (30°: 1.375, 45°: 1.5), and past
    yield (8 κ_y) it carries 1.80 / 1.94 x the plate's moment and keeps 0.92 of its curvature
    (docs/notes/plasticity-sheets.md)."""
    for angle, want in ((0.0, 1.0), (30.0, 1.375), (45.0, 1.5)):
        rows = L.s1_off_axis(angle, steps=8)
        assert rows[0].stiffness == pytest.approx(want, rel=2e-3), (angle, rows[0])
        assert rows[0].moment == pytest.approx(want * rows[0].moment_cf, rel=3e-3), (angle, rows[0])
        last = rows[-1]
        if angle == 0.0:
            assert last.moment == pytest.approx(last.moment_cf, rel=4e-3) and last.kept == pytest.approx(last.kept_cf, rel=2e-3)
        else:
            assert 1.7 < last.moment / last.moment_cf < 2.0 and 0.88 < last.kept / last.kept_cf < 0.95, last


def test_the_hinge_length_reads_the_true_curvature_and_the_plate_energy():
    """l_h = (A1 + A2)/|e|: a grid strip bent about either axis turns each bending hinge by κ l_h, and ke = D/l_h
    sums to the plate's ½ D κ² over the bent area."""
    for nx, ny, axis in ((24, 4, "y"), (4, 24, "x")):
        v, tris = grid_mesh(nx, ny, 2.0e-3)
        x = L.arc(v, 5.0) if axis == "y" else L.arc(v[:, [1, 0, 2]], 5.0)[:, [1, 0, 2]]
        b = _builder()
        add_film_mesh(b, v, tris, L.metal_spec("aluminium_6061_t6", 5.0e-4))
        ei = np.asarray(b.edge_indices, np.int64)
        lh = hinge_length(v, ei)
        th = np.array([L._angle(x, r) if lh[n] > 0 else 0.0 for n, r in enumerate(ei)])
        bent = np.abs(th) > 1e-9
        assert np.allclose(np.abs(th[bent]) / lh[bent], 5.0, rtol=1e-6, atol=0.0)
        ke = np.asarray([p[0] for p in b.edge_bending_properties])
        energy = float(np.sum(0.5 * ke * np.linalg.norm(v[ei[:, 3]] - v[ei[:, 2]], axis=1) * th**2))
        D = L.section(L.metal_spec("aluminium_6061_t6", 5.0e-4))[3]
        n_b, w = (nx, ny) if axis == "y" else (ny, nx)
        assert energy == pytest.approx(0.5 * D * 25.0 * (n_b - 1) * 2.0e-3 * w * 2.0e-3, rel=1e-6)


def test_a_j2_sheet_refuses_fold_lines():
    """A crease hinge's or a dead fold's yield angle and the J2 rest-angle write would both own the hinge (SC8's and
    WP7's reviews): refused."""
    v, tris = grid_mesh(4, 2, 2.0e-3)
    with pytest.raises(ValueError, match="fold lines"):
        add_film_mesh(_builder(), v, tris, L.metal_spec("aluminium_6061_t6", 5.0e-4), creases=np.array([[3, 4]]))
    with pytest.raises(ValueError, match="dead-fold"):                            # WP7's review: nor a dead fold
        add_film_mesh(_builder(), v, tris, L.metal_spec("aluminium_6061_t6", 5.0e-4).with_extras(yield_angle=0.1))


def test_a_film_without_j2_cards_builds_no_j2_state():
    """Default off: an HDPE sheet gets no J2 section, no J2 arrays and the film state it had before."""
    from alloy.solvers.film.materials import HDPE_15UM

    rig = _strip_rig(HDPE_15UM, pinned=False)
    assert rig.solver._j2 is None and rig.solver.film_topology.j2 == 0
    assert not any(k.startswith("film_j2") for k in rig.solver.film_state_arrays())
    assert float(np.abs(rig.model.film.j2_yield.numpy()).max()) == 0.0


def test_reset_and_rollback_restore_a_yielded_sheet_bitwise():
    """Mid-yield: the film's state arrays (rest angles, layer ε_p and ε̄p) snapshot and restore, so re-running the
    same steps gives the same bits; reset puts the authored rest angles back and zeroes the layers."""
    from alloy.solvers.film.plugin import _state_arrays

    spec = L.metal_spec("aluminium_6061_t6", 5.0e-4)
    rig = _strip_rig(spec, pinned=True)
    rest0 = rig.model.edge_rest_angle.numpy().copy()
    k_y = L.section(spec)[1] / (L.section(spec)[0] * 2.5e-4)
    v = rig.v

    def bend(k: float) -> None:
        x = wp.array(L.arc(v, k).astype(np.float32), dtype=wp.vec3, device="cpu")
        wp.copy(rig.s0.particle_q, x)
        wp.copy(rig.s1.particle_q, x)
        rig.step(1)

    for k in np.linspace(0.5, 4.0, 6) * k_y:
        bend(k)
    snap = {k: a.numpy().copy() for k, a in _state_arrays(rig.solver).items()}
    later = np.linspace(4.5, 7.0, 4) * k_y
    for k in later:
        bend(k)
    first = {k: a.numpy().copy() for k, a in rig.solver.film_state_arrays().items()}
    for k, a in _state_arrays(rig.solver).items():
        a.assign(snap[k])
    for k in later:
        bend(k)
    again = {k: a.numpy().copy() for k, a in rig.solver.film_state_arrays().items()}
    assert first.keys() == again.keys() and {"film_j2_eps_p", "film_j2_ebar", "film_edge_rest_angle"} <= first.keys()
    for k in first:
        assert np.array_equal(first[k], again[k]), k
    assert float(first["film_j2_ebar_max"].max()) > 0.0                 # it did yield
    rig.solver.reset(rig.s0)
    assert np.array_equal(rig.model.edge_rest_angle.numpy(), rest0)
    assert float(np.abs(rig.solver.film_state_arrays()["film_j2_ebar"].numpy()).max()) == 0.0


def _builder():
    from alloy._vendor import newton
    from alloy.solvers.film.config import register_custom_attributes

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    return b


def _strip_rig(spec: FilmSpec, pinned: bool) -> FilmRig:
    v, tris = grid_mesh(16, 4, 2.0e-3)

    def build(b):
        p0, _ = add_film_mesh(b, v, tris, spec)
        if pinned:
            pin(b, range(p0, p0 + len(v)))
        return {"p0": [p0]}

    rig = FilmRig(build=build, gravity=0.0)
    rig.v = v
    return rig
