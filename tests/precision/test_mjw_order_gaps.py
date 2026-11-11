# SPDX-License-Identifier: Apache-2.0
"""P3-1 round 4: the two gaps round 3 found in Alloy's CONTACTS/CONSTRAINT fill (CPU).

1. Equality rows keep MuJoCo Warp's layout: connect rows, then weld rows, each by equality id. MuJoCo Warp emits
   them in that kernel order and its equality-force sensors read that layout. Sorting by equality id alone would
   interleave the two kinds when ids are mixed.
2. Contacts imported from Newton (``use_mujoco_contacts=False``: MuJoCo Warp's collision does not run) get a
   canonical order inside the constraint sort, with contact row ids and ``efc_address`` remapped.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import mujoco as mj
from alloy.precision import mjw_order, mjw_rows
from alloy.precision.mjw_det import set_mjw_determinism
from alloy.testing.precision import (assert_mjw_equal, mjw_contacts, mjw_rows_state, shuffle_mjw_contacts,
                                    shuffle_mjw_rows)

SORTS = mjw_order.CONTACTS | mjw_order.CONSTRAINT


def _equality_scene(worlds: int) -> newton.Model:
    """Four free boxes held by a weld (eq 0), a connect (eq 1), a weld (eq 2) and a connect (eq 3)."""
    b = newton.ModelBuilder()
    boxes = []
    for i in range(4):
        body = b.add_body(xform=wp.transform(wp.vec3(0.3 * i, 0.0, 0.5), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05,
                        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, has_shape_collision=False))
        boxes.append(body)
    mj.add_equality_weld(b, -1, boxes[0])
    mj.add_equality_connect(b, -1, boxes[1], anchor=wp.vec3(0.0, 0.0, 0.05))
    mj.add_equality_weld(b, -1, boxes[2])
    mj.add_equality_connect(b, -1, boxes[3], anchor=wp.vec3(0.0, 0.0, 0.05))
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    return s.finalize(device="cpu")


def _step(solver, model, n: int, contacts=None, pipeline=None) -> None:
    s0, s1, ctl = model.state(), model.state(), model.control()
    for _ in range(n):
        if pipeline is not None:
            pipeline.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, 1e-3)
        s0, s1 = s1, s0


def _equality_rows(d, w: int) -> list[tuple[int, int]]:
    rtype, rid = d.efc.type.numpy()[w], d.efc.id.numpy()[w]
    n = int(d.nefc.numpy()[w])
    return [(int(t), int(i)) for t, i in zip(rtype[:n], rid[:n]) if t == 0]


@pytest.fixture
def hooks():
    mjw_order.install_order_hooks()
    yield
    mjw_order.uninstall_order_hooks()


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
def test_connect_rows_before_weld_rows(hooks, jacobian: str) -> None:
    model = _equality_scene(2)
    native = newton.solvers.SolverMuJoCo(model, njmax=64, nconmax=8, jacobian=jacobian)
    _step(native, model, 2)  # Warp CPU runs launches serially: the native layout, unsorted
    sorted_ = newton.solvers.SolverMuJoCo(model, njmax=64, nconmax=8, jacobian=jacobian)
    set_mjw_determinism(sorted_, SORTS)
    _step(sorted_, model, 2)
    for w in range(2):
        got = _equality_rows(sorted_.mjw_data, w)
        assert got == _equality_rows(native.mjw_data, w)
        ids = [i for _, i in got]
        # 3 rows per connect (eq 1, 3), then 6 per weld (eq 0, 2)
        assert ids == [1] * 3 + [3] * 3 + [0] * 6 + [2] * 6
    d = sorted_.mjw_data
    ref = mjw_rows_state(d)
    rng = np.random.default_rng(5)
    for _ in range(3):
        shuffle_mjw_rows(d, rng)
        mjw_rows.sort_constraints(sorted_.mjw_model, d)
        assert_mjw_equal(mjw_rows_state(d), ref)


def _box_pile(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


def test_imported_contacts_are_canonical(hooks) -> None:
    model = _box_pile(3)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=200, nconmax=64, solver="newton")
    set_mjw_determinism(solver, SORTS)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    _step(solver, model, 3, contacts, pipeline)
    m, d = solver.mjw_model, solver.mjw_data
    assert not bool(m.opt.run_collision_detection)
    assert int(d.nacon.numpy()[0]) > 6
    ref_c, ref_r = mjw_contacts(d), mjw_rows_state(d)
    key = np.column_stack([ref_c["worldid"], ref_c["geom"]])
    assert key.tolist() == sorted(key.tolist()), "imported contacts not in (world, geom pair) order"
    rng = np.random.default_rng(7)
    for _ in range(3):
        shuffle_mjw_contacts(d, rng, imported=True)
        shuffle_mjw_rows(d, rng)
        mjw_rows.sort_constraints(m, d)
        got = mjw_contacts(d)
        for f in ref_c:
            assert np.array_equal(got[f], ref_c[f]), f
        assert_mjw_equal(mjw_rows_state(d), ref_r)
