# SPDX-License-Identifier: Apache-2.0
"""SCR-WP4-6 (CPU): ``NewtonContacts``' per-world contact budget and its always-on lost-contact counter.

A budget that fits gives the contacts of Newton's automatic sizes; a cap that does not counts the lost contacts on
the device, and the counter as a HealthMonitor capacity flag turns every world INVALID (SC0-1)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags
from alloy.scene.steppers import ContactBudget, NewtonContacts

_CUBE_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32) * 0.1
_CUBE_F = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                    1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7], np.int32)
_RIGID = ("rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_point1",
          "rigid_contact_normal")
_SOFT = ("soft_contact_particle", "soft_contact_shape", "soft_contact_body_pos", "soft_contact_normal")


def _box_on_mesh(worlds: int = 2) -> newton.Model:
    """Per world: a box resting slightly into a mesh cube (box-mesh contacts go through triangle pairs) and four
    particles just inside the box's top face."""
    b = newton.ModelBuilder()
    for _ in range(worlds):
        w = newton.ModelBuilder()
        base = w.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
        w.add_shape_mesh(base, mesh=newton.Mesh(_CUBE_V, _CUBE_F))
        top = w.add_body(xform=wp.transform((0.02, 0.01, 0.299), wp.quat_identity()))
        w.add_shape_box(top, hx=0.1, hy=0.1, hz=0.1)
        for x, y in ((-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)):
            w.add_particle((x, y, 0.402), (0.0, 0.0, 0.0), 0.01, radius=0.005)
        b.add_world(w)
    return b.finalize(device="cpu")


def _collided(model: newton.Model, budget: ContactBudget | None) -> tuple[NewtonContacts, newton.State]:
    layer = NewtonContacts(model, budget=budget)
    state = model.state()
    layer.collide(state, layer.contacts(), dt=1e-3)
    return layer, state


def _rows(layer: NewtonContacts, fields: tuple[str, ...], count: str) -> dict[str, np.ndarray]:
    c = layer.contacts()
    n = int(getattr(c, count).numpy()[0])
    return {f: getattr(c, f).numpy()[:n] for f in fields}


def _triangles(layer: NewtonContacts) -> tuple[int, int]:
    narrow = layer.pipeline.narrow_phase
    return int(narrow.triangle_pairs_count.numpy()[0]), int(narrow.triangle_pairs.shape[0])


def test_budget_that_fits_gives_the_auto_contacts():
    model = _box_on_mesh()
    auto, _ = _collided(model, None)
    fit, _ = _collided(model, ContactBudget(triangle_pairs=64, rigid_contacts=64, soft_contacts=64))
    assert _triangles(auto)[1] == 1_000_000 and _triangles(fit)[1] == 64 * 2     # Newton's minimum auto size
    assert _triangles(fit)[0] > 0 and int(fit.contacts().rigid_contact_max) == 128
    assert int(fit.contacts().soft_contact_max) == 16                   # the candidates (< 64 x 2 worlds)
    for fields, count in ((_RIGID, "rigid_contact_count"), (_SOFT, "soft_contact_count")):
        a, b = _rows(auto, fields, count), _rows(fit, fields, count)
        assert a[fields[0]].size > 0, count
        for f in fields:
            np.testing.assert_array_equal(a[f], b[f], err_msg=f)
    assert int(auto.overflow.numpy()[0]) == 0 and int(fit.overflow.numpy()[0]) == 0


def test_no_budget_keeps_newtons_candidates():
    model = _box_on_mesh(1)
    layer, _ = _collided(model, None)
    assert int(layer.pipeline.soft_contact_pair_count) == model.particle_count * model.shape_count
    assert not layer.budget.is_set


@pytest.mark.parametrize("budget", [ContactBudget(triangle_pairs=1), ContactBudget(rigid_contacts=1),
                                    ContactBudget(soft_contacts=1)], ids=["triangles", "rigid", "soft"])
def test_each_cap_counts_what_it_loses(budget):
    layer, _ = _collided(_box_on_mesh(), budget)
    assert int(layer.overflow.numpy()[0]) > 0


def test_lost_contacts_mark_every_world_invalid():
    model = _box_on_mesh()
    layer, state = _collided(model, ContactBudget(triangle_pairs=1))
    assert int(layer.overflow.numpy()[0]) > 0
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), capacity_flags=[layer.overflow])
    monitor.update(0, state, state, None)
    report = monitor.poll()
    assert report.invalid_worlds == (0, 1)
    assert all(int(f) & int(HealthFlags.CONTACT_OVERFLOW) for f in report.flags)
    layer.reset_overflow()
    assert int(layer.overflow.numpy()[0]) == 0


def test_budget_must_be_positive():
    with pytest.raises(ValueError, match="triangle_pairs_per_world"):
        NewtonContacts(_box_on_mesh(1), budget=ContactBudget(triangle_pairs=0))


def test_soft_only_layer_keeps_the_soft_contacts():
    """SCR-WP4-7: without rigid pairs the layer has no narrow phase buffers and no rigid slots, and its soft contacts
    are the full layer's."""
    model = _box_on_mesh()
    full, _ = _collided(model, None)
    soft = NewtonContacts(model, rigid=False)
    soft.collide(model.state(), soft.contacts(), dt=1e-3)
    assert int(soft.contacts().rigid_contact_max) == 0
    assert getattr(getattr(soft.pipeline, "narrow_phase", None), "triangle_pairs", None) is None
    a, b = _rows(full, _SOFT, "soft_contact_count"), _rows(soft, _SOFT, "soft_contact_count")
    assert a["soft_contact_particle"].size > 0
    for f in _SOFT:
        np.testing.assert_array_equal(a[f], b[f], err_msg=f)
    assert int(soft.overflow.numpy()[0]) == 0


@pytest.mark.parametrize(("entries", "native", "shape_body", "rigid"), [
    ((("mujoco", (0, 1)), ("vbd", ())), "mujoco", (-1, 0, 1, 1), False),     # DC5 full: MuJoCo owns every body
    ((("mujoco", (0, 1)), ("sph", ())), "mujoco", (-1, -1, 0), False),       # DC5 drip
    ((("mujoco", (0,)), ("vbd", (1,))), "mujoco", (0, 1), True),             # a rigid shape on a non-MuJoCo body
    ((("mujoco", (0,)), ("vbd", ())), "mujoco", (0, 2), True),               # a shape on a body no entry owns
    ((("mujoco", (0, 1)), ("vbd", ())), "newton", (0, 1), True),             # MuJoCo on Newton contacts
    ((("mujoco", (0,)), ("mujoco", (1,))), "mujoco", (0, 1), True),          # two MuJoCo entries
])
def test_fallback_needs_rigid(entries, native, shape_body, rigid):
    from types import SimpleNamespace

    from alloy.scene.steppers import fallback_needs_rigid

    specs = [SimpleNamespace(capability=c, bodies=b) for c, b in entries]
    assert fallback_needs_rigid(specs, native, np.asarray(shape_body, np.int32)) is rigid


def test_particle_subset_keeps_only_its_soft_contacts():
    """SCR-JOB2-4: ``particles`` drops the other particles' candidates; the kept particles' contacts are the full
    layer's (same rows, same order)."""
    model = _box_on_mesh()
    full, _ = _collided(model, None)
    keep = np.array([0, 1, 2, 3, 5], np.int64)                         # world 0's four, one of world 1
    sub = NewtonContacts(model, particles=keep)
    sub.collide(model.state(), sub.contacts(), dt=1e-3)
    pairs = sub.pipeline.soft_rigid_contact_pairs.numpy()
    assert set(np.unique(pairs[:, 0]).tolist()) == set(keep.tolist())
    a, b = _rows(full, _SOFT, "soft_contact_count"), _rows(sub, _SOFT, "soft_contact_count")
    mask = np.isin(a["soft_contact_particle"], keep)
    assert mask.any() and not mask.all()
    order = np.argsort(b["soft_contact_particle"], kind="stable")
    ref = np.argsort(a["soft_contact_particle"][mask], kind="stable")
    for f in _SOFT:
        np.testing.assert_array_equal(a[f][mask][ref], b[f][order], err_msg=f)
    none = NewtonContacts(model, rigid=False, particles=np.zeros(0, np.int64))
    none.collide(model.state(), none.contacts(), dt=1e-3)
    assert int(none.contacts().soft_contact_count.numpy()[0]) == 0
