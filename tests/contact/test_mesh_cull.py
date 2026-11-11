# SPDX-License-Identifier: Apache-2.0
"""Culled mesh soft-contact candidates (``alloy.contact.mesh_cull``, WP10 mixed-speed lever #4) give Newton's records.

On the CPU the culled path runs the same 32-lane groups one after another, so its records can be compared with
Newton's own path on one model: a cloth draped over a mesh cube on a body, a static mesh cube beside it, and particles
far from both, in two worlds placed differently."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact import mesh_cull

_CUBE_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32) * 0.05
_CUBE_F = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                    1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7], np.int32)
GAP = 0.004


def _model(worlds: int = 2, device: str = "cpu", band: bool = False) -> newton.Model:
    """Per world: a 0.1 m mesh cube on a body (turned and shifted per world), a static mesh cube 0.3 m away, a 12 x 12
    cloth 1 mm above the body cube's top (its rim hangs past the edges), and 20 loose particles 1 m up."""
    b = newton.ModelBuilder()
    for w in range(worlds):
        sub = newton.ModelBuilder()
        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3 * w)
        body = sub.add_body(xform=wp.transform((0.01 * w, 0.0, 0.05), yaw))
        sub.add_shape_mesh(body, mesh=newton.Mesh(_CUBE_V, _CUBE_F))
        sub.add_shape_mesh(-1, xform=wp.transform((0.3, 0.0, 0.05), wp.quat_identity()),
                           mesh=newton.Mesh(_CUBE_V, _CUBE_F))
        sub.add_cloth_grid(pos=wp.vec3(-0.066, -0.066, 0.101), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=11,
                           dim_y=11, cell_x=0.012, cell_y=0.012, mass=1e-3, particle_radius=0.002)
        for k in range(20):
            sub.add_particle((0.05 * k, 0.0, 1.0), (0.0, 0.0, 0.0), 1e-3, radius=0.002)
        if band:                                   # two more particles in the body cube's band, at its -x face
            for y in (-0.02, 0.02):
                sub.add_particle((-0.0525, y, 0.05), (0.0, 0.0, 0.0), 1e-3, radius=0.002)
        b.add_world(sub)
    return b.finalize(device=device)


def _records(model: newton.Model, cull: bool) -> tuple[np.ndarray, newton.CollisionPipeline]:
    """Sorted soft records (particle, shape, mesh feature triple, normal, body point) of one full-surface collide."""
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=GAP)
    if cull:
        assert mesh_cull.attach(pipeline)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    n = int(contacts.soft_contact_count.numpy()[0])
    rows = np.concatenate([
        contacts.soft_contact_particle.numpy()[:n, None].astype(np.float64),
        contacts.soft_contact_shape.numpy()[:n, None].astype(np.float64),
        contacts._soft_contact_mesh_features.numpy()[:n].astype(np.float64),
        contacts.soft_contact_normal.numpy()[:n].astype(np.float64),
        contacts.soft_contact_body_pos.numpy()[:n].astype(np.float64)], axis=1)
    return rows[np.lexsort(rows.T[::-1])], pipeline


@pytest.fixture
def cpu_cull(monkeypatch):
    monkeypatch.setattr(mesh_cull, "CUDA_ONLY", False)


def test_culled_records_equal_newtons(cpu_cull):
    """Same records, bit for bit, as a set; the cloth touches the body cube in both worlds."""
    model = _model()
    newton_rows, _ = _records(model, cull=False)
    culled_rows, pipeline = _records(model, cull=True)
    vt = newton_rows[:, 2] == 0                                    # vertex-face family
    assert vt.sum() >= 50 and len(np.unique(newton_rows[vt, 1])) == 2      # both worlds' body cubes
    assert np.array_equal(newton_rows, culled_rows)
    cull = pipeline._soft_mesh_contact_data._alloy_cull
    pairs = int(pipeline._soft_mesh_contact_data.vertex_pairs.shape[0])
    assert 0 < int(cull.count.numpy()[0]) < pairs // 2              # the cull keeps the cloth near the body cube


def test_pipeline_without_cull_keeps_newtons_launcher(cpu_cull):
    """Installing the launcher changes nothing for a pipeline that was not given a cull."""
    model = _model()
    before, _ = _records(model, cull=False)
    mesh_cull.install()
    after, _ = _records(model, cull=False)
    assert np.array_equal(before, after)


def test_copy_matches_the_newton_pin():
    """The vertex query body and the launcher steps it calls are those of the pinned Newton."""
    assert mesh_cull.newton_drift() == []


def test_cpu_keeps_newtons_path():
    """Without the test switch the CPU gets no cull (one lane per pair there already)."""
    model = _model(1)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=GAP)
    assert not mesh_cull.attach(pipeline)


def test_particle_destination_gets_candidates_for_its_own_particles_only():
    """The stepper's proxy pipeline: a destination with particles (VBD) filters soft candidates to them; one without
    (MuJoCo) keeps every particle (WP10 lever #4: the robot scene's SPH water left VBD's 129-mesh query)."""
    from types import SimpleNamespace

    from alloy.core.registry import isolated_registry
    from alloy.coupling.types import SchedulerPolicy
    from alloy.integration.stepper import _proxy_pipeline
    from alloy.scene.builtin_plugins import register_builtin_solvers

    model = SimpleNamespace(shape_count=2, shape_body=SimpleNamespace(numpy=lambda: np.array([-1, 0], np.int32)),
                            body_count=1, joint_count=0)
    with isolated_registry(empty=True):
        register_builtin_solvers()
        vbd = _proxy_pipeline(model, SimpleNamespace(name="vbd", capability="vbd", bodies=(), particles=(3, 1, 2)),
                              SchedulerPolicy())
        mjc = _proxy_pipeline(model, SimpleNamespace(name="mujoco", capability="mujoco", bodies=(0,), particles=()),
                              SchedulerPolicy())
    assert vbd.particles.tolist() == [1, 2, 3]
    assert mjc.particles is None


@pytest.mark.gpu
def test_culled_records_equal_newtons_on_cuda():
    """On CUDA, where the 32 lanes of a group share the sign query, the culled path gives Newton's records bit for
    bit as a set (the pipelines that use it then put them in a canonical order)."""
    if not wp.get_cuda_device_count():
        pytest.skip("no CUDA device")
    model = _model(device="cuda:0")
    newton_rows, _ = _records(model, cull=False)
    culled_rows, pipeline = _records(model, cull=True)
    assert len(newton_rows) >= 50
    assert np.array_equal(newton_rows, culled_rows)


def test_destination_keeps_its_own_and_proxied_particles():
    """WP4's review: a destination's soft candidates are its own particles and those proxied into it (Newton's coupler
    disables the rest: zero mass, not ACTIVE). A proxied particle in the band keeps its records; a disabled one in the
    band, which Newton's vertex query still reports (it does not read ACTIVE) and VBD drops, has none; every other
    record equals the pipeline without the filter."""
    from alloy._vendor.newton._src.solvers.coupled.model_view import ModelView

    from alloy.coupling.proxy_contacts import ProxyContactPipeline

    model = _model(1, band=True)
    n = int(model.particle_count)
    own, proxied, disabled = np.arange(144), n - 2, np.r_[np.arange(144, n - 2), n - 1]
    view = ModelView(model, "vbd")
    view.mark_proxy_particles(wp.array([proxied], dtype=int))
    view.zero_particle_mass(wp.array(disabled, dtype=int))
    view.disable_particles(wp.array(disabled, dtype=int))

    def rows(**kw):
        pipeline = ProxyContactPipeline(rigid_shapes=(), full_surface=True, soft_contact_gap=GAP, **kw)(view)
        contacts = pipeline.contacts()
        pipeline.collide(view.state(), contacts)
        k = int(contacts.soft_contact_count.numpy()[0])
        r = np.concatenate([contacts.soft_contact_particle.numpy()[:k, None].astype(np.float64),
                            contacts.soft_contact_shape.numpy()[:k, None].astype(np.float64),
                            contacts._soft_contact_mesh_features.numpy()[:k].astype(np.float64),
                            contacts.soft_contact_normal.numpy()[:k].astype(np.float64)], axis=1)
        return r[np.lexsort(r.T[::-1])]

    full, filtered = rows(), rows(particles=own, proxied=True)
    vt = filtered[:, 2] % 8 == 0                                   # vertex-face records: column 0 is the particle
    assert np.any(filtered[vt, 0] == proxied) and np.any(full[full[:, 2] % 8 == 0, 0] == n - 1)
    assert not np.any(filtered[vt, 0] == n - 1)
    keep = ~((full[:, 2] % 8 == 0) & (full[:, 0] == n - 1))
    assert np.array_equal(full[keep], filtered)
