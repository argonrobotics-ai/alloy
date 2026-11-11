# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-10: curvature-aware boundary volume for mesh walls and the conforming round-bore fill (CPU).

- The curvature ratio gives the boundary volume a conforming ring fill needs at its first layer.
  - Concave bores: within 2 % at R/h 1.67 and 3.5. The flat volume is 9 % / 5 % low.
  - Convex rods: within 4 %.
- The fill's ring radii put every ring at rest density.
- Mesh curvature from the BVH probes.
  - A round bore reads +1/R and a round rod -1/R, within 5 %.
  - A mesh box reads exactly 0 on its faces and at its edges.
- Thin faces: on a tube whose bore faces are 30 mm x 0.05 mm (SC10's lathe ring), Warp's closest point tilts the
  wall normal; ``face_closest`` brings it back to the face normal.
- Bitwise: a tank of mesh boxes steps the same bytes with boundary_curvature_order 1 and 0, so flat meshes never take
  the curved path. Planes, boxes and primitives never reach it at all: it is gated on GeoType.MESH.
- A resting column in a round mesh bore at R/h 1.67, at the C18 step (dt_min).
  - With the fill it stays at rest: no leak, drift under 0.5 spacing, near-wall density at least 0.97.
  - With rings at (k + 1/2) spacings it starts below 0.9 at the wall and sags.
"""

from __future__ import annotations

import math

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.boundary import ShapeArrays, shape_sdf
from alloy.solvers.sph.boundary_curvature import curved_volume, mesh_mean_curvature
from alloy.solvers.sph.boundary_curvature_fill import (BoreModel, _layer, _ring_counts, ring_densities,
                                                       round_bore_points, solve_ring_radii, wall_psi)
from alloy.solvers.sph.build import add_box_tank, add_fluid_points
from alloy.solvers.sph.config import BOUNDARY_SDF, DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.kernels import boundary_fit, halfspace_volume
from alloy.solvers.sph.layout import shape_table
from alloy.solvers.sph.solver import SolverDFSPH

R_SMALL = 0.075e-3  # C18's particle radius [m]
MODEL = BoreModel(spacing=2.0 * R_SMALL, h=4.0 * R_SMALL)


def tube_mesh(r_in: float, r_out: float, length: float, n_seg: int = 64, n_z: int = 60) -> newton.Mesh:
    """Closed tube (bore radius r_in, outer radius r_out, z in [0, length]), n_z axial segments, outward normals."""
    ang = 2.0 * math.pi * np.arange(n_seg) / n_seg
    zs = np.linspace(0.0, length, n_z + 1)
    rings = [(r, z) for r in (r_in, r_out) for z in zs]
    v = np.array([(r * math.cos(a), r * math.sin(a), z) for r, z in rings for a in ang], dtype=np.float64)

    def vid(which: int, iz: int, k: int) -> int:
        return (which * (n_z + 1) + iz) * n_seg + k % n_seg

    faces, want = [], []
    for iz in range(n_z):
        for k in range(n_seg):
            for which, sign in ((0, -1.0), (1, 1.0)):  # bore wall faces the axis, outer wall faces away
                a, b, c, d = vid(which, iz, k), vid(which, iz, k + 1), vid(which, iz + 1, k + 1), vid(which, iz + 1, k)
                mid = v[[a, b, c, d]].mean(0)
                out = sign * np.array([mid[0], mid[1], 0.0])
                faces += [(a, b, c), (a, c, d)]
                want += [out, out]
    for iz, sz in ((0, -1.0), (n_z, 1.0)):  # end annuli
        for k in range(n_seg):
            a, b, c, d = vid(0, iz, k), vid(0, iz, k + 1), vid(1, iz, k + 1), vid(1, iz, k)
            faces += [(a, b, c), (a, c, d)]
            want += [np.array([0.0, 0.0, sz])] * 2
    f = np.array(faces, dtype=np.int64)
    for j, (a, b, c) in enumerate(f):
        if np.dot(np.cross(v[b] - v[a], v[c] - v[a]), want[j]) < 0.0:
            f[j] = (a, c, b)
    return newton.Mesh(v.astype(np.float32), f.reshape(-1).astype(np.int32))


def box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    v = np.array([[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)], dtype=np.float32)
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                  [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]], dtype=np.int32)
    for k, (a, b, c) in enumerate(f):
        if np.dot(np.cross(v[b] - v[a], v[c] - v[a]), (v[a] + v[b] + v[c]) / 3.0) < 0.0:
            f[k] = (a, c, b)
    return newton.Mesh(v, f.reshape(-1))


# ------------------------------------------------------------------------------------------------ boundary volume


@wp.kernel
def _volume_kernel(phi: wp.array[wp.float32], H: float, offset: float, amplitude: float, h: float,
                   curved: wp.array[wp.vec2], flat: wp.array[wp.vec2]):
    i = wp.tid()
    curved[i] = curved_volume(phi[i], offset, amplitude, h, H, 0)
    flat[i] = halfspace_volume(phi[i] + offset, h, 0.0, 0) * amplitude


def test_curved_volume_with_zero_curvature_is_the_flat_volume():
    h = MODEL.h
    off, amp = boundary_fit(MODEL.spacing, h)
    phi = wp.array(np.linspace(-0.5 * h, 1.2 * h, 101, dtype=np.float32), dtype=wp.float32, device="cpu")
    c, f = (wp.zeros(101, dtype=wp.vec2, device="cpu") for _ in range(2))
    wp.launch(_volume_kernel, dim=101, inputs=[phi, 0.0, off, amp, h, c, f], device="cpu")
    assert c.numpy().tobytes() == f.numpy().tobytes()


def _first_ring_need(model: BoreModel, radius: float) -> tuple[float, float]:
    """(boundary volume the conforming ring fill needs at its first ring, that ring's wall distance)."""
    radii, counts, centre = _ring_counts(radius, model.spacing)
    rho = ring_densities(model, radius, radii, counts, centre)[0]
    psi = wall_psi(model, radius, np.array([radius - radii[0]]))[0]
    return 1.0 - (rho - psi), radius - radii[0]


@pytest.mark.parametrize("r_over_h, flat_low", [(1.6667, 0.08), (3.5, 0.04)])
def test_curvature_ratio_gives_the_volume_a_conforming_fill_needs(r_over_h, flat_low):
    R = r_over_h * MODEL.h
    need, d = _first_ring_need(MODEL, R)
    model = wall_psi(MODEL, R, np.array([d]))[0]
    flat = wall_psi(BoreModel(MODEL.spacing, MODEL.h, curvature=0), R, np.array([d]))[0]
    assert abs(model / need - 1.0) < 0.025, (model, need)
    assert flat / need < 1.0 - flat_low, (flat, need)  # what the mesh wall had before


@pytest.mark.parametrize("r_over_h", [1.6667, 2.5, 3.5])
def test_round_bore_fill_sits_at_rest_density(r_over_h):
    R = r_over_h * MODEL.h
    radii, counts, centre, dens = solve_ring_radii(MODEL, R)
    assert np.all(np.abs(dens - 1.0) < 1.0e-4), dens
    area = math.pi * R * R / MODEL.spacing**2
    assert abs(sum(counts) + centre - area) <= 1.5, (counts, area)
    pts = round_bore_points(MODEL, R, 1.0e-3, 1.0e-3 + 10 * MODEL.spacing)
    per = sum(counts) + centre
    assert pts.shape == (10 * per, 3)
    assert np.hypot(pts[:, 0], pts[:, 1]).max() < R - 0.4 * MODEL.spacing
    assert np.allclose(np.unique(np.round(pts[:, 2], 9)), 1.0e-3 + (np.arange(10) + 0.5) * MODEL.spacing)
    assert np.allclose(pts[:per, :2], _layer(radii, counts, centre, 0.0))


# ------------------------------------------------------------------------------------------------ mesh curvature


def _table(mesh: newton.Mesh, cfg: DFSPHConfig) -> tuple[newton.Model, ShapeArrays, list]:
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.add_shape_mesh(-1, mesh=mesh, custom_attributes={"sph:shape_boundary": BOUNDARY_SDF})
    m = b.finalize(device="cpu")
    t = shape_table(m, cfg, m.device)
    return m, t.arrays, t.keep


@wp.kernel
def _curv_kernel(shapes: ShapeArrays, x: wp.array[wp.vec3], h: float, phi: wp.array[wp.float32],
                 n: wp.array[wp.vec3], H: wp.array[wp.float32]):
    i = wp.tid()
    _lo, p, g = shape_sdf(shapes, 0, x[i])
    nl = wp.normalize(g)
    phi[i] = p
    n[i] = nl
    H[i] = mesh_mean_curvature(shapes.mesh[0], shapes.scale[0], x[i] - p * nl, nl, h, 1 - shapes.bvh[0])


def _probe(mesh: newton.Mesh, pts: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = DFSPHConfig(particle_radius=0.25 * h)
    _m, shapes, _keep = _table(mesh, cfg)
    x = wp.array(pts.astype(np.float32), dtype=wp.vec3, device="cpu")
    phi, H = (wp.zeros(len(pts), dtype=wp.float32, device="cpu") for _ in range(2))
    n = wp.zeros(len(pts), dtype=wp.vec3, device="cpu")
    wp.launch(_curv_kernel, dim=len(pts), inputs=[shapes, x, h, phi, n, H], device="cpu")
    return phi.numpy(), n.numpy(), H.numpy()


def _ring(radius: float, z: float, count: int = 48) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False) + 0.013
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(count, z)], 1)


def test_mesh_curvature_of_a_round_bore_and_rod():
    h, R, Ro = MODEL.h, 1.6667 * MODEL.h, 2.5 * MODEL.h
    mesh = tube_mesh(R, Ro, 6.0e-3)
    _phi, _n, H_in = _probe(mesh, _ring(R - 0.25 * h, 3.0e-3), h)
    assert np.all(np.abs(H_in * R - 1.0) < 0.05), H_in * R  # concave bore: +1/R
    _phi, _n, H_out = _probe(mesh, _ring(Ro + 0.25 * h, 3.0e-3), h)
    assert np.all(np.abs(H_out * Ro + 1.0) < 0.05), H_out * Ro  # convex rod: -1/Ro


def test_mesh_box_reads_zero_curvature_on_faces_and_edges():
    h = MODEL.h
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.4e-3, 1.4e-3, (600, 3))  # around a 2 mm box: faces, edges and corners
    phi, _n, H = _probe(box_mesh(1.0e-3, 1.0e-3, 1.0e-3), pts, h)
    near = phi < 0.75 * h  # where the solver evaluates it (phi < taper1 = 1.5 spacings)
    assert near.sum() > 300
    assert np.all(H[near] == 0.0), H[near & (H != 0.0)]


def test_thin_bore_faces_keep_the_wall_normal_radial():
    """SC10's lathe ring has one 30 mm face strip per segment along the bore. The float32 closest point on such faces
    tilted the wall normal by up to 15 deg (0.26 axial) before face_closest."""
    h, R = MODEL.h, 1.6667 * MODEL.h
    pts = np.concatenate([_ring(R - 0.25 * h, z, 64) for z in np.linspace(2.0e-3, 8.0e-3, 9)])
    phi, n, H = _probe(tube_mesh(R, R + 0.25e-3, 30.0e-3, n_z=1), pts, h)
    assert np.abs(n[:, 2]).max() < 0.01, np.abs(n[:, 2]).max()
    assert np.mean(np.abs(H * R - 1.0) < 0.05) > 0.95


# ------------------------------------------------------------------------------------------------ bitwise


def _mesh_tank_column(curvature: int) -> bytes:
    cfg = DFSPHConfig(particle_radius=2.0e-3, boundary_curvature_order=curvature)
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    add_mesh = b.add_shape_mesh

    def as_mesh(body, *, xform=None, hx=0.5, hy=0.5, hz=0.5, cfg=None, custom_attributes=None, **_kw):
        return add_mesh(body, xform=xform, mesh=box_mesh(hx, hy, hz), cfg=cfg, custom_attributes=custom_attributes)

    b.add_shape_box = as_mesh
    b.begin_world()
    s_ = cfg.spacing
    add_box_tank(b, (0.0, 0.0, 0.0), (6 * s_, 6 * s_, 12 * s_), 2 * s_)
    g = (np.arange(6) + 0.5) * s_
    add_fluid_points(b, cfg, np.array([(x, y, (k + 0.5) * s_) for k in range(8) for y in g for x in g]))
    b.end_world()
    m = b.finalize(device="cpu")
    assert np.all(m.shape_type.numpy() == int(newton.GeoType.MESH))
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    for _ in range(30):
        s.step(a, c, None, None, 1.25e-3)
        a, c = c, a
    return a.particle_q.numpy().tobytes() + a.particle_qd.numpy().tobytes()


def test_flat_mesh_walls_are_bitwise_with_and_without_mesh_curvature():
    assert _mesh_tank_column(1) == _mesh_tank_column(0)


# ------------------------------------------------------------------------------------------------ resting bore


def _bore_column(fill: str, steps: int, layers: int = 12) -> dict[str, float]:
    """A column in a round mesh bore (R/h 1.67, r 0.075 mm) on a box floor, stepped at dt_min (the C18 step)."""
    cfg = DFSPHConfig(particle_radius=R_SMALL, viscosity="weiler2018", xsph_c=0.3, boundary_viscosity=0.00089,
                      integrate_bodies="none", dt_max=1.953125e-5, world_bounds=((-0.01,) * 3, (0.01,) * 3))
    sp, R, z0 = cfg.spacing, 1.6667 * cfg.h, 1.0e-3
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    b.add_shape_mesh(-1, mesh=tube_mesh(R, R + 0.25e-3, 4.0e-3, n_z=8),
                     custom_attributes={"sph:shape_boundary": BOUNDARY_SDF})
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.5 * z0), wp.quat_identity()), hx=1.0e-3, hy=1.0e-3,
                    hz=0.5 * z0, custom_attributes={"sph:shape_boundary": BOUNDARY_SDF})
    if fill == "rings":
        pts = round_bore_points(BoreModel.of(cfg), R, z0, z0 + layers * sp)
    else:  # SC10's ring_layer: rings at (k + 1/2) spacing
        xy = []
        k = 0
        while (k + 0.5) * sp < R:
            rk, nk = (k + 0.5) * sp, max(1, round(2 * math.pi * (k + 0.5)))
            xy += [(rk * math.cos(2 * math.pi * j / nk), rk * math.sin(2 * math.pi * j / nk)) for j in range(nk)]
            k += 1
        pts = np.array([(x, y, z0 + (i + 0.5) * sp) for i in range(layers) for x, y in xy])
    add_fluid_points(b, cfg, pts, 997.05, 0.00089)
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    s.enable_leak_monitor()
    a, c = m.state(), m.state()
    z_start = a.particle_q.numpy()[:, 2].mean()
    wall0 = _wall_ratio(s, a, R, z0, layers, sp)
    vmax = 0.0
    for _ in range(steps):
        s.step(a, c, None, None, cfg.dt_min)
        a, c = c, a
        vmax = max(vmax, float(np.abs(a.particle_qd.numpy()).max()))
    q = a.particle_q.numpy()
    lk = s.leaks.counts()
    return {"drift": float((q[:, 2].mean() - z_start) / sp), "wall": _wall_ratio(s, a, R, z0, layers, sp),
            "wall0": wall0, "leaks": int(lk["leaks"].sum() + lk["in_solid"].sum()), "vmax": vmax}


def _wall_ratio(s: SolverDFSPH, state, R: float, z0: float, layers: int, sp: float) -> float:
    """Median density ratio of the liquid within one spacing of the bore wall, away from the floor and the top."""
    ratio, act = s.density_ratio(state)
    q = state.particle_q.numpy()
    rr = np.hypot(q[:, 0], q[:, 1])
    mid = act & (q[:, 2] > z0 + 3 * sp) & (q[:, 2] < z0 + (layers - 3) * sp)
    return float(np.median(ratio[mid & (rr > R - sp)]))


def test_round_mesh_bore_column_rests_at_the_c18_step():
    got = _bore_column("rings", steps=600)
    assert got["leaks"] == 0, got
    assert got["wall0"] >= 0.97 and got["wall"] >= 0.97, got
    assert abs(got["drift"]) < 0.5 and got["vmax"] < 0.2, got


@pytest.mark.slow
def test_half_spacing_rings_sag_in_the_bore():
    """The control: SC10's former ring seeding is 20 % short of the bore's count at R/h 1.67 and sags."""
    got = _bore_column("half", steps=2000)
    assert got["wall0"] < 0.9 and got["drift"] < -0.5, got
