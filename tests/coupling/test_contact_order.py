# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4d stable proxy soft-contact order (alloy/coupling/contact_order.py): real full-surface records from a CPU
collide, shuffled the way GPU slot counters shuffle them, come out bitwise identical; the hook is CUDA only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.coupling.contact_order import RECORD_ARRAYS, SoftContactOrder
from alloy.coupling.proxy_contacts import ProxyContactPipeline

CUBE = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)
CUBE_FACES = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4,
                       1, 5, 7, 1, 7, 3], dtype=np.int32)


def _two_pads(device: str) -> newton.Model:
    """Two 26 x 16 x 6 mm box-mesh pads on two bodies, 0.5 mm into a flat 8.5 mm cloth (VT, TV and EE records)."""
    b = newton.ModelBuilder()
    half = np.array([0.013, 0.008, 0.003], dtype=np.float32)
    for x in (-0.016, 0.016):
        body = b.add_body(xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
        b.add_shape_mesh(body, mesh=newton.Mesh(CUBE * half, CUBE_FACES))
    b.add_cloth_grid(pos=wp.vec3(-0.045, -0.04, -0.0025), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=11,
                     dim_y=10, cell_x=0.0085, cell_y=0.0085, mass=2e-5, tri_ke=1e3, tri_ka=1e3, tri_kd=1e-2,
                     edge_ke=1e-3, edge_kd=1e-5, particle_radius=0.001)
    return b.finalize(device=device)


def _hull_tip(device: str) -> newton.Model:
    """A fingertip of 8 overlapping box geoms on one body (the YAM tip has 8 hull geoms) and a sphere on a second
    body, 0.5 mm into the cloth: per-vertex records, many particles against several shapes of one body."""
    b = newton.ModelBuilder()
    tip = b.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
    for k in range(8):
        x, y = -0.012 + 0.0035 * k, 0.003 * (k % 3 - 1)
        b.add_shape_box(tip, xform=wp.transform((x, y, 0.0), wp.quat_identity()), hx=0.006, hy=0.006, hz=0.003)
    ball = b.add_body(xform=wp.transform((0.0, 0.02, 0.0), wp.quat_identity()))
    b.add_shape_sphere(ball, radius=0.003)
    b.add_cloth_grid(pos=wp.vec3(-0.045, -0.04, -0.0025), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=11,
                     dim_y=10, cell_x=0.0085, cell_y=0.0085, mass=2e-5, tri_ke=1e3, tri_ka=1e3, tri_kd=1e-2,
                     edge_ke=1e-3, edge_kd=1e-5, particle_radius=0.001)
    return b.finalize(device=device)


def _collided(device: str = "cpu", scene=_two_pads, full_surface: bool = True, **kw):
    model = scene(device)
    factory = ProxyContactPipeline(rigid_shapes=(), full_surface=full_surface, soft_contacts_per_world=20_000, **kw)
    pipeline = factory(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    return model, factory, contacts


def _records(contacts) -> dict[str, np.ndarray]:
    return {n: getattr(contacts, n).numpy().copy() for n in RECORD_ARRAYS if getattr(contacts, n, None) is not None}


def _shuffled(model, contacts, seed: int) -> dict[str, np.ndarray]:
    """The valid records in a random slot order (as GPU counters hand them out), then put in the stable order."""
    rec = _records(contacts)
    n = int(contacts.soft_contact_count.numpy()[0])
    perm = np.random.default_rng(seed).permutation(n)
    for name, a in rec.items():
        b = a.copy()
        b[:n] = a[perm]
        getattr(contacts, name).assign(b)
    SoftContactOrder(contacts, model).apply()
    return _records(contacts)


@pytest.fixture(scope="module")
def collided():
    return _collided()


def _identities(rec: dict[str, np.ndarray], n: int) -> np.ndarray:
    """What the key reads per record: shape, soft corners, rigid primitive corners, feature tag (-1 without)."""
    feat = rec.get("_soft_contact_mesh_features", np.full((n, 3), -1, np.int32))
    return np.concatenate([rec["soft_contact_shape"][:n, None], rec["soft_contact_indices"][:n],
                           rec["soft_contact_rigid_indices"][:n], feat[:n]], axis=1)


def test_full_surface_records_have_unique_identities(collided) -> None:
    model, _, contacts = collided
    rec, n = _records(contacts), int(contacts.soft_contact_count.numpy()[0])
    kinds = {int((rec["soft_contact_indices"][i] >= 0).sum()) for i in range(n)}
    assert n > 100 and kinds == {1, 2, 3}, (n, kinds)                      # vertex, edge and face records
    assert len(np.unique(_identities(rec, n), axis=0)) == n


def test_per_vertex_records_on_a_many_geom_body_are_unique_and_ordered() -> None:
    """Equal keys would fall back to slot order: per-vertex records of particles touching several geoms of one body
    are (shape, particle) pairs, distinct, and any slot order of them sorts to one."""
    model, _, contacts = _collided(scene=_hull_tip, full_surface=False)
    rec, n = _records(contacts), int(contacts.soft_contact_count.numpy()[0])
    shapes_per_particle = np.bincount(rec["soft_contact_indices"][:n, 0])
    assert n > 50 and shapes_per_particle.max() >= 3, (n, shapes_per_particle.max())
    assert len(np.unique(rec["soft_contact_shape"][:n])) == 9
    assert len(np.unique(_identities(rec, n), axis=0)) == n
    a, b = _shuffled(model, contacts, 5), _shuffled(model, contacts, 6)
    for name in a:
        assert a[name].tobytes() == b[name].tobytes(), name


def test_any_slot_order_gives_the_same_records_bitwise(collided) -> None:
    model, _, contacts = collided
    before = _records(contacts)
    n = int(contacts.soft_contact_count.numpy()[0])
    a, b = _shuffled(model, contacts, 1), _shuffled(model, contacts, 2)
    for name in before:
        assert a[name].tobytes() == b[name].tobytes(), name
        assert a[name][n:].tobytes() == before[name][n:].tobytes(), name    # unused slots untouched
    rows = lambda r: {tuple(np.concatenate([np.atleast_1d(r[k][i]).view(np.uint32) for k in sorted(r)]))
                      for i in range(n)}
    assert rows(a) == rows(before)                                          # the same records, reordered
    SoftContactOrder(contacts, model).apply()                               # already canonical: unchanged
    for name, x in _records(contacts).items():
        assert x.tobytes() == b[name].tobytes(), name


def test_records_are_grouped_by_body_then_shape_then_first_particle(collided) -> None:
    model, _, contacts = collided
    rec = _shuffled(model, contacts, 3)
    n = int(contacts.soft_contact_count.numpy()[0])
    shape = rec["soft_contact_shape"][:n]
    body = model.shape_body.numpy()[shape]
    first = rec["soft_contact_indices"][:n, 0]
    key = (body.astype(np.int64) << 40) | (shape.astype(np.int64) << 20) | first
    assert np.all(np.diff(key) >= 0) and len(np.unique(body)) == 2


def test_the_cpu_pipeline_keeps_newtons_order() -> None:
    _, factory, contacts = _collided()
    _, _, off = _collided(stable_order=False)
    assert factory.orders == [None]
    for name, a in _records(contacts).items():
        assert a.tobytes() == _records(off)[name].tobytes(), name


def test_the_key_must_fit_63_bits() -> None:
    view = SimpleNamespace(shape_body=None, shape_count=2**21, particle_count=2**31, body_count=2**20)
    contacts = SimpleNamespace(soft_contact_max=4, soft_contact_shape=wp.zeros(4, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="> 63"):
        SoftContactOrder(contacts, view)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA")
def test_cuda_collides_put_the_records_in_one_order() -> None:
    model, factory, contacts = _collided("cuda:0")
    assert factory.orders[0] is not None
    assert _collided("cuda:0", stable_order=False)[1].orders == [None]
    first = _records(contacts)
    a = _shuffled(model, contacts, 4)
    for name in first:
        assert a[name].tobytes() == first[name].tobytes(), name
