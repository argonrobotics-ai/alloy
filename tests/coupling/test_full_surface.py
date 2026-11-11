# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4 (CPU): a rigid feature narrower than the cloth's vertex gap threads the cloth under per-vertex contact and
is stopped by the cloth under full-surface (edge/face) proxy contacts (``alloy.coupling.pinch_rigs``)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.coupling.pinch_rigs import run_blade, run_pinch

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]


def test_thin_blade_threads_the_cloth_with_per_vertex_contact_only():
    r = run_blade(full_surface=False)
    assert r.finite and r.tip_x > 0.010                     # the tip went 10+ mm through the cloth plane
    assert r.threaded >= 1 and r.cloth_x_max < 1e-4         # through a cell, the cloth never touched


def test_full_surface_contact_pushes_the_cloth_ahead_of_the_blade():
    r = run_blade(full_surface=True)
    assert r.finite and r.threaded == 0
    assert r.cloth_x_max > r.tip_x - 0.002                  # the cloth stays on the blade's tip


def test_pinch_open_rise_with_full_surface_contact():
    """The ruling's CPU regression on the Scheduler's own clock (lagged, make_stepper's proxy contact view): two
    0.56 kg pads close on a hanging double layer at the 15 N cap and stop on it (>= 2 mm), open at 40 N to the stop
    (no stall), rise 30 mm with the cloth left in place (< 2 mm), no pass-through, bounded interface work."""
    r = run_pinch(full_surface=True, mode="lagged")
    assert r.finite and r.pass_through == 0
    assert 0.002 <= r.gap_hold < 0.004, r                 # two 2 mm layers, compressed at 15 N (3.6 mm)
    assert r.open_left < 0.001 and r.cloth_rise < 0.002, r
    assert abs(r.work) < 6.0, r                            # 3.7 J; an earlier tree kicked to 30 J at first contact


def film_contact_stack() -> float:
    """The film pinch's equilibrium gap as the clamp force goes to 0: each pad rests at the body-particle contact
    radius (particle radius + pad shape margin) from its layer, and the layers at VBD's self-contact distance, which
    the rig sets to the particle radius (``_cloth_scheduler``) [m]."""
    from alloy.coupling.pinch_rigs import FILM, gripper_model

    model = gripper_model(cloth=FILM)
    pads = np.flatnonzero(model.shape_body.numpy() >= 1)                  # bodies 1, 2: the pads
    margin = float(model.shape_margin.numpy()[pads].max())
    return 2.0 * (FILM.radius + margin) + FILM.radius


def test_pinch_of_a_film_double_layer_at_its_soft_ke():
    """C16's film (1 cm cells, 0.5 mm particles, 1.4 mg vertices, soft ke 300): records at 0.5 (300 + pad ke),
    stable on the light vertices. The double layer settles at its contact stack (1.5 mm) less a penalty compression.
    The grip servo's goal is the closed stop, so at this gap it pushes kp x gap / 2 below its 15 N cap: 7.24 N here.
    That load is carried by self-contact records at 300 N/m each (on 65b7c92e: 144 vertex-triangle records carry
    1.9 N, edge-edge records the rest), which compresses the stack by 0.03-0.06 mm (film-floor ruling, scr/WP4.md
    2026-11-04 23:48Z: the stack is the reference). Measured: 1.445 mm (Newton 65b7c92e) and
    1.467 mm (25f406c7 + corner fix). The 1.71 mm on ac7f7830 was a chattering corner contact; it is 0.21 mm above
    the stack and fails the main assertion."""
    from alloy.coupling.pinch_rigs import FILM

    stack = film_contact_stack()
    r = run_pinch(full_surface=True, mode="lagged", cloth=FILM)
    assert r.finite and r.pass_through == 0
    assert abs(r.gap_hold - stack) <= 1.0e-4, (r, stack)    # at the computed equilibrium
    assert 0.00125 <= r.gap_hold <= stack + 2.0e-4, r        # above the self-contact barrier band; chatter fails
    assert r.open_left < 0.001 and r.cloth_rise < 0.002, r
    assert abs(r.work) < 2.0, r


def _fine_pad_model():
    """A 26 x 16 x 6 mm convex pad of 600 hull vertices (rounded box) on a body, 0.5 mm into a flat 8.5 mm cloth."""
    import warp as wp
    from scipy.spatial import ConvexHull

    from alloy._vendor import newton

    rng = np.random.default_rng(0)
    d = rng.normal(size=(600, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = np.sign(d) * np.abs(d) ** 0.3 * np.array([0.013, 0.008, 0.003])      # a rounded box surface
    hull = ConvexHull(pts)
    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    b.add_shape_mesh(body, mesh=newton.Mesh(pts[hull.vertices].astype(np.float32), _hull_faces(pts[hull.vertices])))
    b.add_cloth_grid(pos=wp.vec3(-0.04, -0.04, pts[:, 2].min() + 0.0005), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                     dim_x=10, dim_y=10, cell_x=0.0085, cell_y=0.0085, mass=2e-5, tri_ke=1e3, tri_ka=1e3, tri_kd=1e-2,
                     edge_ke=1e-3, edge_kd=1e-5, particle_radius=0.001)
    return b.finalize(device="cpu")


def _hull_faces(pts):
    from alloy.coupling.full_surface import reduced_hull

    return reduced_hull(pts, len(pts))[1]


def test_contact_view_reduces_a_fine_hull_in_the_view_only():
    """A 64-vertex hull of a fine pad in the destination view, enclosing the original (SCR-SC4-4b): far fewer
    full-surface records; the model keeps the original mesh; the stiffness is the shape's own."""
    from alloy.coupling.full_surface import LOD_VERTICES, contact_view, reduced_hull
    from alloy._vendor.newton._src.solvers.coupled.model_view import ModelView

    model = _fine_pad_model()
    original = int(model.shape_source_ptr.numpy()[0])
    view = ModelView(model, "vbd")
    ke = float(model.shape_material_ke.numpy()[0])
    contact_view([0])(view)
    assert int(view.shape_source_ptr.numpy()[0]) != original and int(model.shape_source_ptr.numpy()[0]) == original
    assert float(view.shape_material_ke.numpy()[0]) == ke
    assert len(view.shape_source[0].vertices) == LOD_VERTICES
    src = np.asarray(model.shape_source[0].vertices, np.float64)
    v, _ = reduced_hull(src, LOD_VERTICES)
    from scipy.spatial import ConvexHull

    eq = ConvexHull(v.astype(np.float64)).equations
    assert (src @ eq[:, :3].T + eq[:, 3]).max() < 0.0                        # every original vertex inside
    state = model.state()
    full, lod = _records_on(model, state), _records_on(view, state)
    assert lod < full / 4, (full, lod)


def _records_on(view, state):
    from alloy.coupling.proxy_contacts import ProxyContactPipeline

    pipeline = ProxyContactPipeline(rigid_shapes=(), full_surface=True, soft_contacts_per_world=200_000)(view)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return int(contacts.soft_contact_count.numpy()[0])


def test_contact_view_keeps_a_non_convex_mesh():
    """A cup (non-convex) keeps its own triangles in the view, its cavity open."""
    from alloy._vendor import newton
    import warp as wp

    from alloy.coupling.full_surface import contact_view, is_convex
    from alloy.coupling.pinch_rigs import cup_mesh
    from alloy._vendor.newton._src.solvers.coupled.model_view import ModelView

    b = newton.ModelBuilder()
    body = b.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    b.add_shape_mesh(body, mesh=newton.Mesh(*cup_mesh()))
    model = b.finalize(device="cpu")
    assert not is_convex(model.shape_source[0])
    view = ModelView(model, "vbd")
    contact_view([0])(view)
    assert int(view.shape_source_ptr.numpy()[0]) == int(model.shape_source_ptr.numpy()[0])


def test_reduced_hull_encloses_the_yam_pad_tips():
    """SCR-SC4-4b: health measures film depth against the original pad mesh, so the view hull must enclose it.
    Farthest-point sampling alone left 0.04-0.42 mm of the YAM tip hulls outside; the enclosing pass clears every
    original vertex by >= ENCLOSE_PAD/2 and grows the hull by about as much as it was short (< 0.5 mm)."""
    from pathlib import Path

    from scipy.spatial import ConvexHull

    from alloy.coupling.full_surface import ENCLOSE_PAD, LOD_VERTICES, reduced_hull
    from alloy.tasks.msh import read_msh

    files = sorted(Path("library/robots/meshes").glob("hull_linear_tip_*_9-*.msh"))
    assert len(files) == 2                                                     # left and right pad tips
    for f in files:
        pts = np.asarray(read_msh(f).vertices, np.float64)
        v, tris = reduced_hull(pts)
        assert len(v) == LOD_VERTICES and tris.size % 3 == 0
        eq = ConvexHull(v.astype(np.float64)).equations
        assert (pts @ eq[:, :3].T + eq[:, 3]).max() <= -ENCLOSE_PAD / 2, f.name
        orig = ConvexHull(pts).equations
        grow = (v.astype(np.float64) @ orig[:, :3].T + orig[:, 3]).max()
        assert grow < 5.0e-4, (f.name, grow)


def test_enclose_leaves_a_vertex_inside_the_hull_alone():
    """A lift pass can move a sampled vertex inside the hull of the others (24 of 64 on a 264-point Panda scene
    mesh after pass 0): it has no hull face, so it gets no lift (``_lift`` of no faces raised on an empty array and
    the robot + cloth scene failed to build). It stays in the mesh, unused by any face, as the
    passes before the fix already left such vertices."""
    from scipy.spatial import ConvexHull

    from alloy.coupling.full_surface import ENCLOSE_PAD, _enclose

    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64) * 0.01
    v = np.vstack([corners, [[0.0, 0.0, 0.002]]])                 # the last vertex is inside the cube's hull
    pts = corners * 1.02                                          # every corner a little outside: all need a lift
    out = _enclose(v, pts)
    np.testing.assert_array_equal(out[-1], v[-1])                 # no face, no lift
    eq = ConvexHull(out).equations
    assert (pts @ eq[:, :3].T + eq[:, 3]).max() <= -ENCLOSE_PAD / 2
