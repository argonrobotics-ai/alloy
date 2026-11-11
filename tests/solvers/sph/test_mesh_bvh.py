# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-9: particle-colliding mesh shapes without a provisioned volume SDF (sdf_index -1). Newton's eval_shape_sdf
reads texture_sdf_table[-1] for them, which segfaulted SolverDFSPH on CPU in DC5's drip variant (162 such meshes);
boundary.shape_sdf queries the mesh BVH instead, as Newton's own particle contacts do. A mesh box must give the
analytic box's distance and normal, and a tank of mesh boxes must hold a column like the same tank of box shapes."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.boundary import ShapeArrays, shape_sdf
from alloy.solvers.sph.build import add_box_tank, add_fluid_block, add_fluid_points, wall_box
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.layout import shape_table
from alloy.solvers.sph.solver import SolverDFSPH

DT = 1.25e-3


def _box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    """Closed box mesh centred on its origin, faces wound outward."""
    v = np.array([[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)], dtype=np.float32)
    f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                  [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]], dtype=np.int32)
    for k, (a, b, c) in enumerate(f):
        if np.dot(np.cross(v[b] - v[a], v[c] - v[a]), (v[a] + v[b] + v[c]) / 3.0) < 0.0:
            f[k] = (a, c, b)
    return newton.Mesh(v, f.reshape(-1))


def _mesh_boxes(b: newton.ModelBuilder) -> None:
    """Make add_shape_box on this builder add the same box as a mesh (no volume SDF is provisioned on CPU)."""
    add_mesh = b.add_shape_mesh

    def as_mesh(body, *, xform=None, hx=0.5, hy=0.5, hz=0.5, cfg=None, custom_attributes=None, **_kw):
        return add_mesh(body, xform=xform, mesh=_box_mesh(hx, hy, hz), cfg=cfg, custom_attributes=custom_attributes)

    b.add_shape_box = as_mesh


@wp.kernel(enable_backward=False)
def _eval_kernel(shapes: ShapeArrays, s: int, x: wp.array[wp.vec3], phi: wp.array[wp.float32],
                 grad: wp.array[wp.vec3]):
    i = wp.tid()
    _lo, p, g = shape_sdf(shapes, s, x[i])
    phi[i] = p
    grad[i] = g


def test_mesh_without_sdf_gives_the_box_distance_and_normal():
    half = (0.02, 0.03, 0.01)
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.add_shape_mesh(-1, mesh=_box_mesh(*half))
    b.add_shape_box(-1, hx=half[0], hy=half[1], hz=half[2])
    m = b.finalize(device="cpu")
    assert m._shape_sdf_index.numpy()[0] == -1  # the case under test: no volume SDF for the mesh
    table = shape_table(m, DFSPHConfig(), m.device)
    x = (np.random.default_rng(0).uniform(-1.6, 1.6, (400, 3)) * np.array(half)).astype(np.float32)
    out = {}
    for s in (0, 1):
        # CPU arrays: on a CUDA host the default device is cuda:0, and a CPU launch on device memory segfaults
        phi, grad = wp.zeros(len(x), dtype=wp.float32, device="cpu"), wp.zeros(len(x), dtype=wp.vec3, device="cpu")
        wp.launch(_eval_kernel, dim=len(x), inputs=[table.arrays, s, wp.array(x, dtype=wp.vec3, device="cpu"), phi,
                                                    grad],
                  device="cpu")
        out[s] = phi.numpy(), grad.numpy()
    (pm, gm), (pb, gb) = out[0], out[1]
    near = pb < table.arrays.bvh_dist  # the BVH search is bounded at 2 h; farther points read +2 h (outside)
    assert near.sum() > 200 and (~near).sum() > 20
    assert np.all(np.abs(pm[near] - pb[near]) < 1.0e-6), np.abs(pm[near] - pb[near]).max()
    assert np.allclose(pm[~near], table.arrays.bvh_dist)
    outside = near & (pb > 1.0e-4)  # outside a convex box the closest point, hence the normal, is unique
    assert outside.sum() > 100 and (pb < 0.0).sum() > 20
    assert np.all(np.einsum("ij,ij->i", gm[outside], gb[outside]) > 0.9999)


def _column(mesh_walls: bool) -> np.ndarray:
    cfg = DFSPHConfig(particle_radius=2.0e-3)
    s_ = cfg.spacing
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    if mesh_walls:
        _mesh_boxes(b)
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (6 * s_, 6 * s_, 14 * s_), wall=cfg.h)
    add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (6, 6, 8))
    b.end_world()
    m = b.finalize(device="cpu")
    if mesh_walls:
        assert np.all(m.shape_type.numpy() == int(newton.GeoType.MESH)) and np.all(m._shape_sdf_index.numpy() == -1)
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    s.prime(a, DT)
    for _ in range(40):
        s.step(a, c, None, None, DT)
        a, c = c, a
    return a.particle_q.numpy()


def test_tank_of_mesh_boxes_holds_the_column_like_box_shapes():
    """The DC5 drip crash's path (compute_slots, the path sweep and projection on mesh walls without an SDF). The
    distances agree to float precision; the resting column's jitter (one-sided solves at the 1e-3 tolerance)
    amplifies that to a few hundredths of a spacing over 40 steps, so the check is physical: same height, inside."""
    s_ = DFSPHConfig(particle_radius=2.0e-3).spacing
    q_mesh, q_box = _column(True), _column(False)
    assert np.all(np.isfinite(q_mesh))
    assert np.abs(q_mesh - q_box).max() < 0.1 * s_, np.abs(q_mesh - q_box).max() / s_
    assert abs(q_mesh[:, 2].mean() - q_box[:, 2].mean()) < 0.01 * s_  # breathing at tol 1e-3: tol H ~ 0.008 s
    inside = (q_mesh[:, :2] > 0.0).all(1) & (q_mesh[:, :2] < 6 * s_).all(1) & (q_mesh[:, 2] > 0.0)
    assert inside.all()


def _lift(mesh_walls: bool, device: str, speeds: tuple[float, ...] = (0.1, 0.05), settle: int = 60,
          steps: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per world a kinematic floor on a body lifts a 4x4x6 column (r = 2 mm) inside static side walls, each world at
    its own speed (per-world body poses); returns particle positions and their world ids after the lift."""
    cfg = DFSPHConfig(particle_radius=2.0e-3, integrate_bodies="none")
    r, dt, n, hw, top = 2.0e-3, 0.02 / 64, 4, 8.0e-3, 4.0e-3
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    if mesh_walls:
        _mesh_boxes(b)
    for _ in speeds:
        b.begin_world()
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # static side walls, 60 mm high
            wall_box(b, (sx * (hw + r), sy * (hw + r), 0.03), (r if sx else hw + 2 * r, r if sy else hw + 2 * r, 0.03))
        floor = b.add_body(xform=wp.transform_identity(), label="floor")
        wall_box(b, (0.0, 0.0, 0.0), (hw, hw, top), body=floor)
        g = (np.arange(n) + 0.5) * 2.0 * r - hw
        add_fluid_points(b, cfg, np.array([(x, y, top + r + 2.0 * r * k) for k in range(6) for y in g for x in g]),
                         1000.0, 1.0e-3, (0.0, 0.0, 0.0))
        b.end_world()
    m = b.finalize(device=device)
    if mesh_walls:
        assert np.all(m._shape_sdf_index.numpy() == -1)
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    z = np.zeros(len(speeds))
    for k in range(settle + steps):
        v = np.array(speeds) if k >= settle else np.zeros(len(speeds))
        a.body_q.assign(np.array([[0.0, 0.0, zw, 0.0, 0.0, 0.0, 1.0] for zw in z], dtype=np.float32))
        a.body_qd.assign(np.array([[0.0, 0.0, vw, 0.0, 0.0, 0.0] for vw in v], dtype=np.float32))
        z += v * dt
        s.step(a, c, None, None, dt)
        a, c = c, a
    return a.particle_q.numpy(), m.particle_world.numpy(), z + top


def _check_lift(device: str) -> None:
    """Mesh walls (no SDF) vs box shapes, per world. The impulsive start of the lift splashes the free surface, which
    amplifies float-level distance differences (single particles part by up to 0.4 spacing; CPU), so the checks are
    physical: mean position and height quantiles within 0.1 spacing (measured <= 0.054), and every particle carried
    above the moving floor (clearance > 0.25 spacing; measured 0.47)."""
    (q_m, world, floor_top), (q_b, _w, _t) = _lift(True, device), _lift(False, device)
    s_ = 4.0e-3
    assert np.all(np.isfinite(q_m))
    for w in range(len(floor_top)):
        a, b = q_m[world == w], q_b[world == w]
        assert np.all(np.abs(a.mean(0) - b.mean(0)) < 0.1 * s_), (w, (a.mean(0) - b.mean(0)) / s_)
        dq = [np.quantile(a[:, 2], p) - np.quantile(b[:, 2], p) for p in (0.1, 0.5, 0.9)]
        assert np.all(np.abs(dq) < 0.1 * s_), (w, np.array(dq) / s_)
        assert (a[:, 2] - floor_top[w]).min() > 0.25 * s_, (w, (a[:, 2] - floor_top[w]).min() / s_)


def test_moving_mesh_floor_lifts_the_column_like_a_box_cpu():
    """SCR-WP9-9 acceptance (moving boundary, CPU): mesh walls without an SDF on a moving body, 2 worlds."""
    _check_lift("cpu")


@pytest.mark.gpu
def test_moving_mesh_floor_lifts_the_column_like_a_box_cuda():
    """SCR-WP9-9 acceptance (moving boundary, CUDA): as the CPU test (an out-of-bounds read need not fault on CUDA)."""
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    _check_lift("cuda:0")


def _band_pair(band: tuple[float, float], r: float = 1.25e-3):
    """CUDA: a box mesh whose texture SDF has the given narrow band (voxel 0.25 mm) and the same box without an SDF;
    returns (shape table, config, model): the model owns the wp.Mesh objects the table's BVH ids point to."""
    half = (0.03, 0.03, 0.05)
    with_sdf = _box_mesh(*half)
    with_sdf.build_sdf(device="cuda:0", narrow_band_range=band, target_voxel_size=2.5e-4, margin=band[1])
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.add_shape_mesh(-1, mesh=with_sdf)
    b.add_shape_mesh(-1, mesh=_box_mesh(*half))
    m = b.finalize(device="cuda:0")
    assert m._shape_sdf_index.numpy()[0] >= 0 and m._shape_sdf_index.numpy()[1] == -1
    cfg = DFSPHConfig(particle_radius=r)
    return shape_table(m, cfg, m.device), cfg, m


def _phi_beyond(table, cfg, s: int) -> np.ndarray:
    """Distances of shape s at points 0.5 mm .. the reach outside the box's +x face (and inside, -1 mm)."""
    d = np.concatenate([[-1.0e-3], np.linspace(5.0e-4, 1.5 * cfg.spacing, 12)])
    x = np.array([[0.03 + di, 0.004 * k, -0.01 * k] for di in d for k in (-1, 0, 1)], np.float32)
    phi, grad = wp.zeros(len(x), dtype=wp.float32, device="cuda:0"), wp.zeros(len(x), dtype=wp.vec3, device="cuda:0")
    wp.launch(_eval_kernel, dim=len(x), inputs=[table.arrays, s, wp.array(x, dtype=wp.vec3, device="cuda:0"), phi, grad],
              device="cuda:0")
    return phi.numpy()


@pytest.mark.gpu
def test_narrow_band_sdf_reads_the_bvh_cuda():
    """SCR-JOB2-2 (2): a precision pair's +-0.5 mm SDF band is narrower than the liquid's 3.75 mm reach (r 1.25 mm):
    SPH reads that shape through the BVH, giving exactly the no-SDF distances; the SDF itself is untouched."""
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    table, cfg, _model = _band_pair((-5.0e-4, 5.0e-4))
    assert list(table.arrays.bvh.numpy()) == [1, 1]
    assert np.array_equal(_phi_beyond(table, cfg, 0), _phi_beyond(table, cfg, 1))


@pytest.mark.gpu
def test_wide_band_sdf_keeps_the_texture_cuda():
    """A band out to 4.5 mm covers the 3.75 mm reach: SPH keeps the texture SDF (within a voxel of the BVH)."""
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    table, cfg, _model = _band_pair((-1.0e-3, 4.5e-3))
    assert list(table.arrays.bvh.numpy()) == [0, 1]
    tex, bvh = _phi_beyond(table, cfg, 0), _phi_beyond(table, cfg, 1)
    assert not np.array_equal(tex, bvh) and np.abs(tex - bvh).max() < 2.5e-4, np.abs(tex - bvh).max()
