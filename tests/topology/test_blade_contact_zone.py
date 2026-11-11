# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB1-2: BladeContact rewrites an active twin's blade contact only while the blade is cutting it.

The scene helpers are test_blade_contact's (a 20 mm tet cube pre-split at x = 10 mm, cells 5 mm, particle radius
1 mm, a 1 mm blade box, so the cut-front cap 2h + r is 2 mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.topology.blade_contact import BladeContact, BladeContactConfig
from alloy.topology.blade_contact_kernels import REENTRY_V_MIN
from alloy.topology.bonds import bonds_from_model
from alloy.topology.types import BondKind
from tests.topology.test_blade_contact import (HALF, _changed, _collide, _edge_through, _frame, _in_zone, _mat, _pose,
                                               _seam_scene, _snap)


@dataclass
class _Seam:
    before: dict[str, np.ndarray]
    c: newton.Contacts
    a_rows: np.ndarray                                   # twin a's contact rows within the cut-front cap of the edge
    model: newton.Model
    x: np.ndarray
    blade_contact: BladeContact


def _blade_by_seam(device: str, sep: float, offset: float = 1.0e-3, broken: bool = True, v_blade=(0.0, 0.0, 0.0),
                   v_parts=(0.0, 0.0, 0.0), lift: float = 0.0, reentry: bool = True) -> _Seam:
    """Twin b moved ``sep`` along +x from twin a (the seam at x = 10 mm); the blade's mid-plane at x = 10 mm -
    ``offset`` (so twin a sits ``offset`` on the +x side, across the mid-plane from its own -x face) with its edge along
    y at z = 10 mm + ``lift`` (the blade above it). Every bond broken, or intact and damaged (an uncut, damaged
    plane). The blade moves at ``v_blade``, every particle at ``v_parts``."""
    m, blades, bodies = _seam_scene(device)
    bonds = bonds_from_model(m)
    if broken:
        bonds.intact.fill_(0)
    else:
        bonds.damage.fill_(0.2)
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    s = m.state()
    x = s.particle_q.numpy()
    x[bonds.b.numpy()[cut]] = x[bonds.a.numpy()[cut]] + np.array([sep, 0.0, 0.0], np.float32)
    s.particle_q.assign(x)
    _pose(s, int(bodies[0]), (0.01 - offset, 0.01, 0.04 + lift), np.eye(3))
    s.particle_qd.fill_(wp.vec3(*v_parts))
    qd = s.body_qd.numpy()
    qd[int(bodies[0])] = (*v_blade, 0.0, 0.0, 0.0)
    s.body_qd.assign(qd)
    c = _collide(m, s)
    before = _snap(c)
    bc = BladeContact(m, bonds, blades, config=BladeContactConfig(reentry=reentry))
    bc.adjust_contacts(s, c)
    n = int(before["soft_contact_count"][0])
    parts = before["soft_contact_particle"][:n]
    a_rows = np.flatnonzero(np.isin(parts, bonds.a.numpy()[cut]) & _in_zone(m, s, blades[0], parts))
    return _Seam(before, c, a_rows, m, s.particle_q.numpy().astype(np.float64), bc)


def test_opened_twin_meeting_the_blade_from_the_other_side_keeps_stock(device):
    """SCR-JOB1-2 (JOB1 f2, particle 10763): a cut round whose twin is 2.6 mm away meets the blade from the side
    away from its own face: Newton's contact pushes it back; the old rule put it on the far face, through the blade.
    The same twins still joined (being cut) do cross to their own face."""
    before, c, a_rows = (r := _blade_by_seam(device, sep=2.6e-3)).before, r.c, r.a_rows
    assert len(a_rows) >= 3
    assert len(np.intersect1d(_changed(before, c), a_rows)) == 0
    assert np.all(c.soft_contact_normal.numpy()[a_rows][:, 0] > 0.0)       # pushed back toward +x, its side now
    before, c, a_rows = (r := _blade_by_seam(device, sep=0.0)).before, r.c, r.a_rows
    assert set(a_rows) <= set(_changed(before, c))
    np.testing.assert_allclose(c.soft_contact_normal.numpy()[a_rows][:, 0], -1.0, atol=1e-6)


def test_an_edge_coming_down_into_an_opened_cut_puts_the_twin_under_it_on_its_own_side(device):
    """Gap #9 (2026-11-10; SB-7's stroke, seed 100000 worlds 1911, 915, 19): the pieces have parted (pair 2.6 mm
    apart, over the cap) and the blade comes down again 0.3 mm off the seam, so twin a lies 0.3 mm across the mid-plane
    from its own face, 0.5 mm ahead of the edge. No face stands between it and its side there: it goes on its own
    (-x) face, by the ramp. The rule before left it the stock record, the blade's bottom pressing straight down on
    it (0.5 mm deep at the mid-plane: PEN_FAIL). A descent while sawing along the edge counts the same.

    It keeps the stock record, as before, when the edge is not coming down on it (WP10's review: an opened, parted
    twin is never pulled across by a blade that sweeps or clears face-first, or holds still): a rising blade, a still
    one, one moving along its face normal either way, one moving more across than down, and with the twin level
    with the edge (beside the faces). The re-entry records are counted (``reentry_count``, ``reentry_cross``)."""
    down, up, still = (0.0, 0.0, -0.05), (0.0, 0.0, 0.05), (0.0, 0.0, 0.0)

    def under_the_edge(v_blade, lift: float = 5.0e-4, reentry: bool = True):
        r = _blade_by_seam(device, sep=2.6e-3, offset=3.0e-4, v_blade=v_blade, lift=lift, reentry=reentry)
        rows = r.a_rows[np.isclose(r.x[r.before["soft_contact_particle"][r.a_rows], 2], 0.010, atol=1e-6)]
        assert len(rows) >= 3
        return r, rows

    for v_blade in (down, (0.0, 0.05, -0.02)):
        r, rows = under_the_edge(v_blade)
        assert set(rows) <= set(_changed(r.before, r.c)), v_blade
        np.testing.assert_allclose(r.c.soft_contact_normal.numpy()[rows][:, 0], -1.0, atol=1e-6)
        assert np.all(np.abs(r.before["soft_contact_normal"][rows][:, 0]) < 0.9)   # the stock record was not a face's
        assert int(r.blade_contact.reentry_count.numpy()[0]) == len(rows)
        assert float(r.blade_contact.reentry_cross.numpy()[0]) == pytest.approx(3.0e-4, abs=2e-6)
    for v_blade, lift in ((up, 5.0e-4), (still, 5.0e-4), ((0.05, 0.0, 0.0), 5.0e-4), ((-0.05, 0.0, 0.0), 5.0e-4),
                          ((0.05, 0.0, -0.02), 5.0e-4), ((0.0, 0.0, -5.0e-4), 5.0e-4), (down, 0.0)):
        r, rows = under_the_edge(v_blade, lift)
        assert len(np.intersect1d(_changed(r.before, r.c), rows)) == 0, (v_blade, lift)
        assert int(r.blade_contact.reentry_count.numpy()[0]) == 0, (v_blade, lift)
    r, rows = under_the_edge(down, reentry=False)                    # the rule before, by its switch: stock again
    assert len(np.intersect1d(_changed(r.before, r.c), rows)) == 0
    assert int(r.blade_contact.reentry_count.numpy()[0]) == 0


def test_joined_pair_the_edge_leaves_keeps_stock(device):
    """SCR-JOB1-2 replays (f2 particle 388: a round falls past the stopped knife; seed 1000 world 0 particle 2113: the
    knife lifts out of the pile): a joined pair on a plane the edge nicks, across the blade from twin a's own face,
    is not being cut when the edge moves out of it: twin a keeps Newton's contact. The rule before put it on the far
    face, 2h + r + 1 mm deep. A descending (or still) blade still carries it across. Broken and damaged bonds."""
    for broken in (False, True):
        for v_blade, v_parts, crosses in (((0.0, 0.0, 0.0), (0.0, 0.0, -0.1), False),
                                          ((0.0, 0.0, 0.05), (0.0, 0.0, 0.0), False),
                                          ((0.0, 0.0, -0.05), (0.0, 0.0, 0.0), True),
                                          ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), True)):
            r = _blade_by_seam(device, sep=0.0, broken=broken, v_blade=v_blade, v_parts=v_parts)
            assert len(r.a_rows) >= 3
            moved = np.intersect1d(_changed(r.before, r.c), r.a_rows)
            label = (broken, v_blade, v_parts)
            if crosses:
                assert len(moved) == len(r.a_rows), label
                np.testing.assert_allclose(r.c.soft_contact_normal.numpy()[r.a_rows][:, 0], -1.0, atol=1e-6)
            else:
                assert len(moved) == 0, label


def test_rewrite_never_crosses_the_blade_beyond_the_cut_front_cap(device):
    """The ruling's invariant: a rewrite that puts a particle on the far face of the blade (across its mid-plane) only
    happens within 2h + r of the edge, only while the twin pair is closer than 2h + r or the particle is ahead of the
    edge (gap #9, 2026-11-10: no face stands there yet), and only while the edge is not moving out of the particle.
    Random blade poses, velocities, pair openings and bond states."""
    rng = np.random.default_rng(11)
    m, blades, bodies = _seam_scene(device)
    bonds = bonds_from_model(m)
    cut = np.flatnonzero(bonds.kind.numpy() == int(BondKind.TET_CUT))
    a_ids, b_ids = bonds.a.numpy()[cut], bonds.b.numpy()[cut]
    radius = m.particle_radius.numpy()
    cap = 2.0 * HALF + float(radius.max())
    partner = dict(zip(a_ids.tolist(), b_ids.tolist())) | dict(zip(b_ids.tolist(), a_ids.tolist()))
    far = together = parted = 0
    for _ in range(24):
        bonds.intact.assign(np.where(rng.random(bonds.count) < 0.5, 0, 1).astype(np.int8))
        bonds.damage.assign((rng.random(bonds.count) < 0.3).astype(np.float32) * 0.2)
        s = m.state()
        x = s.particle_q.numpy()
        x[b_ids] = x[a_ids] + np.outer(rng.uniform(0.0, 4.0e-3, len(cut)), [1.0, 0.0, 0.0]).astype(np.float32)
        s.particle_q.assign(x)
        rb = _frame(np.array([1.0, 0.0, 0.0]), rng.uniform(-0.3, 0.3)) @ _frame(np.array([1.0, 0.0, 0.0]), 0.0).T
        edge = (0.01 + rng.uniform(-2.0e-3, 4.0e-3), 0.01, rng.uniform(0.0, 0.02))
        _pose(s, int(bodies[0]), _edge_through(edge, rb), rb)
        v_blade = rng.uniform(-0.05, 0.05, 3)
        qd = s.body_qd.numpy()
        qd[int(bodies[0])] = (*v_blade, 0.0, 0.0, 0.0)
        s.body_qd.assign(qd)
        c = _collide(m, s)
        before = _snap(c)
        BladeContact(m, bonds, blades).adjust_contacts(s, c)
        xf = s.body_q.numpy()[int(bodies[0])]
        rot = _mat(wp.quat(*xf[3:7]))
        nb, mid = rot @ np.array([1.0, 0.0, 0.0]), xf[:3]
        for k in _changed(before, c):
            p = int(before["soft_contact_particle"][k])
            n = c.soft_contact_normal.numpy()[k]
            d = (x[p] - mid) @ nb
            if np.sign(n @ nb) * np.sign(d) >= 0:
                continue                                                 # its own current side: not a crossing
            far += 1
            assert _in_zone(m, s, blades[0], np.array([p]))[0], p
            spine = rot @ np.array([0.0, 0.0, 1.0])
            ahead = (x[p] - np.asarray(edge)) @ spine < 0.0                    # below the edge line
            closing = -v_blade @ spine                                       # particles still: the edge's speed
            joined = np.linalg.norm(x[p] - x[partner[p]]) < cap               # toward them along the spine
            together += int(joined)
            assert joined or ahead, p
            assert closing >= 0.0, p                                         # the edge advances
            if not joined:                                                   # a parted pair (re-entry): the edge
                parted += 1                                                  # comes down on it (WP10's should-fix)
                assert closing > REENTRY_V_MIN and closing >= abs(-v_blade @ nb), (p, v_blade)
    assert far > 0 and together > 0 and parted > 0                       # the cut-front crossing still happens
