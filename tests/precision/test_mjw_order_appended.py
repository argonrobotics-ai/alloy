# SPDX-License-Identifier: Apache-2.0
"""T09 world-index bug (coordinator ruling 2026-11-03 ~23:15Z): the ordered route's contact sort leaves the contacts
a ``contactfilter`` callback appended where they are (CPU, both sort paths).

Alloy's MixedContactBinding (``alloy/contact/mujoco_append.py``) appends the precision layer's rows at
``[nacon, nacon + count)`` inside MuJoCo Warp's ``contactfilter``, which runs just before the CONTACTS sort. Its
``passive`` callback later finds them by that slot range, with a per-slot clearance array that the sort does not move.
The sort used to order them world-major together with the native contacts. Other worlds' native contacts then sat in
that range and took the speculative targets: on T09's fast hold, worlds 2 and ~0.62 N to N - 1 went INVALID.

The sort now keys appended contacts (``geomcollisionid`` -1; MuJoCo Warp's collision writes >= 0) after every native
contact, in slot order. The native contacts stay canonical: (world, geom pair, geomcollisionid), then the tie hash.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision import mjw_order
from alloy.precision.mjw_det import set_mjw_determinism
from alloy.testing.precision import mjw_contacts

APPENDED = 7


def _pile(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    for i in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(0.03 * i, 0.0, 0.05 + 0.099 * i), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    s.add_ground_plane()
    return s.finalize(device="cpu")


@pytest.fixture
def hooks():
    mjw_order.install_order_hooks()
    yield
    mjw_order.uninstall_order_hooks()


def _native_contacts(worlds: int):
    """A box pile after two steps of MuJoCo Warp's own collision, sorted (the route's CONTACTS bit)."""
    model = _pile(worlds)
    solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=64, solver="newton")
    set_mjw_determinism(solver, mjw_order.CONTACTS)
    s0, s1, ctl = model.state(), model.state(), model.control()
    for _ in range(2):
        solver.step(s0, s1, ctl, None, 1e-3)
        s0, s1 = s1, s0
    return solver.mjw_model, solver.mjw_data


def _append(d, src: np.ndarray) -> tuple[int, int]:
    """As a contactfilter callback would: copies of native contacts ``src`` at [nacon, nacon + len(src)), gcid -1,
    each with its own depth (so a moved row is visible)."""
    n, k = int(d.nacon.numpy()[0]), len(src)
    for f in mjw_order._contact_fields(d):
        a = getattr(d.contact, f).numpy()
        a[n:n + k] = a[src]
        if f == "geomcollisionid":
            a[n:n + k] = -1
        elif f == "dist":
            a[n:n + k] = -1.0e-4 * (1.0 + np.arange(k))
        getattr(d.contact, f).assign(a)
    d.nacon.assign(np.array([n + k], dtype=d.nacon.numpy().dtype))
    return n, n + k


def _shuffle_native(d, end: int, rng) -> None:
    """Reorder slots [0, end) only, as another collision counter order would."""
    p = rng.permutation(end)
    for f in mjw_order._contact_fields(d):
        a = getattr(d.contact, f).numpy()
        a[:end] = a[p]
        getattr(d.contact, f).assign(a)


@pytest.mark.parametrize("path", ["rank", "radix"])
def test_appended_contacts_keep_their_slots(hooks, monkeypatch, path: str) -> None:
    if path == "radix":
        monkeypatch.setattr(mjw_order, "SMALL_CONTACTS", 0)
    m, d = _native_contacts(4)
    native = int(d.nacon.numpy()[0])
    assert native > APPENDED and len(np.unique(d.contact.worldid.numpy()[:native])) == 4
    rng = np.random.default_rng(3)
    start, end = _append(d, rng.permutation(native)[:APPENDED])
    first = None
    for _ in range(3):
        before = mjw_contacts(d)
        _shuffle_native(d, start, rng)
        mjw_order.sort_contacts(m, d)
        got = mjw_contacts(d)
        for f in got:
            if f != "efc_address":
                assert np.array_equal(got[f][start:end], before[f][start:end]), f"appended {f} moved"
        key = np.column_stack([got["worldid"][:start], got["geom"][:start]])
        assert key.tolist() == sorted(key.tolist()), "native contacts not in (world, geom pair) order"
        if first is None:
            first = got
        for f in got:
            assert np.array_equal(got[f], first[f]), f"{f} not canonical"
