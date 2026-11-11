# SPDX-License-Identifier: Apache-2.0
"""P3-1 route (c): Alloy's contact and constraint-row sorts give one canonical state from any race order (CPU).

Warp CPU runs launches serially, so the race is emulated: the contacts and the rows of a stepped MuJoCo Warp Data
are shuffled the way atomic counters could have ordered them, with the row content left where it was in the sparse
J storage. Sorting must then reproduce the unshuffled sorted state bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision import mjw_order, mjw_rows
from alloy.precision.mjw_det import set_mjw_determinism
from alloy.testing.precision import (assert_mjw_equal, mjw_contacts, mjw_rows_state, shuffle_mjw_contacts,
                                    shuffle_mjw_rows)

def _boxes(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


@pytest.fixture(scope="module", params=["dense", "sparse"])
def stepped(request):
    model = _boxes(3)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, njmax=200, nconmax=64,
                                         jacobian=request.param, solver="newton")
    mjw_order.install_order_hooks()
    set_mjw_determinism(solver, mjw_order.CONTACTS | mjw_order.CONSTRAINT)
    s0, s1, ctl = model.state(), model.state(), model.control()
    for _ in range(3):
        solver.step(s0, s1, ctl, None, 1e-3)
        s0, s1 = s1, s0
    yield solver
    mjw_order.uninstall_order_hooks()


def test_sorted_state_is_canonical(stepped) -> None:
    m, d = stepped.mjw_model, stepped.mjw_data
    n = int(d.nacon.numpy()[0])
    assert n > 6, "scene should make several contacts per world"
    ref_c, ref_r = mjw_contacts(d), mjw_rows_state(d)
    rng = np.random.default_rng(3)
    for _ in range(3):
        shuffle_mjw_contacts(d, rng)
        mjw_order.sort_contacts(m, d)
        got = mjw_contacts(d)
        for f in mjw_order._contact_fields(d):
            assert np.array_equal(got[f], ref_c[f]), f
        shuffle_mjw_rows(d, rng)
        mjw_rows.sort_constraints(m, d)
        assert_mjw_equal(mjw_rows_state(d), ref_r)
        assert np.array_equal(mjw_contacts(d)["efc_address"], ref_c["efc_address"])


def test_sorted_order_and_links(stepped) -> None:
    d = stepped.mjw_data
    c = mjw_contacts(d)
    key = np.stack([c["worldid"], c["geom"][:, 0], c["geom"][:, 1], c["geomcollisionid"]], axis=1)
    assert all(tuple(key[i]) <= tuple(key[i + 1]) for i in range(len(key) - 1))
    rows = mjw_rows_state(d)
    for w, (t, i) in enumerate(zip(rows["type"], rows["id"])):
        pairs = list(zip(t.tolist(), i.tolist()))
        assert pairs == sorted(pairs), f"world {w} rows not in (type, id) order"
    for cid, addr in enumerate(c["efc_address"]):
        w = int(c["worldid"][cid])
        for a in addr[addr >= 0]:
            assert rows["id"][w][a] == cid and rows["type"][w][a] >= 5


def test_radix_path_sorts_like_the_block_path_and_skips_unused_slots(stepped, monkeypatch) -> None:
    """More than SMALL_CONTACTS slots take the radix path and its gather: the same canonical order, and the slots past
    nacon are neither read nor written, in the contacts or in the scratch (WP4's compute-sanitizer initcheck found
    the gather copying them, 2026-11-06: they may never have been written)."""
    m, d = stepped.mjw_model, stepped.mjw_data
    n = int(d.nacon.numpy()[0])
    assert n < int(d.naconmax), "the scene must leave unused slots"
    ref = mjw_contacts(d)
    monkeypatch.setattr(mjw_order, "SMALL_CONTACTS", 0)
    shuffle_mjw_contacts(d, np.random.default_rng(11))
    scratch = mjw_order._contact_scratch(m, d).copies["dist"]
    for arr, mark in ((d.contact.dist, 7.0), (scratch, 9.0)):
        a = arr.numpy()
        a[n:] = mark
        arr.assign(a)
    mjw_order.sort_contacts(m, d)
    got = mjw_contacts(d)
    for f in mjw_order._contact_fields(d):
        assert np.array_equal(got[f], ref[f]), f
    assert (d.contact.dist.numpy()[n:] == 7.0).all() and (scratch.numpy()[n:] == 9.0).all()
