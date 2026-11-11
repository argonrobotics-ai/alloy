# SPDX-License-Identifier: Apache-2.0
"""task_deposit_cans and its rig (alloy.tasks.deposit_bag): mesh, cans, script, scoring; the task end to end (slow)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from alloy.tasks import deposit_cans as D
from alloy.tasks.deposit_bag import BagSpec, CanSpec, DepositRig, bag_mesh


def _edge_counts(t: np.ndarray) -> Counter:
    e: Counter = Counter()
    for a, b, c in t:
        for x, y in ((a, b), (b, c), (c, a)):
            e[(min(x, y), max(x, y))] += 1
    return e


def test_bag_mesh():
    s = BagSpec()
    v, t, g = bag_mesh(s)
    counts = Counter(_edge_counts(t).values())
    seam = 2 * round(s.handle_w / s.cell)            # each band's lower edge, where the strap's 2 plies meet the band
    assert set(counts) <= {1, 2, 3} and counts[3] == seam, counts
    assert len(g["band_l"]) == len(g["band_r"]) == (round(s.handle_w / s.cell) + 1) * (round(s.band / s.cell) + 1)
    assert np.all(v[g["band_l"], 1] == 0.0) and np.all(v[g["band_r"], 1] == 0.0)     # welded 2-ply band: one sheet
    centre = g["mouth"][np.abs(v[g["mouth"], 0]) < 1e-9]
    assert np.ptp(v[centre, 1]) == pytest.approx(2.0 * s.gap)                       # the mouth starts open
    assert v[:, 2].min() == 0.0 and v[:, 2].max() == pytest.approx(s.top)


def test_mesh_rejects_a_cell_that_does_not_divide():
    with pytest.raises(ValueError, match="must divide"):
        bag_mesh(BagSpec(cell=0.03))


def test_can_inertia_is_a_shell():
    c = CanSpec()
    ix, iz = c.inertia()[0][0], c.inertia()[2][2]
    assert c.mass * c.radius ** 2 / 2 < iz < c.mass * c.radius ** 2                   # between disc and pure shell
    assert ix > c.mass * c.height ** 2 / 12


def test_cans_fit_the_mouth():
    rig = DepositRig()
    xs = rig.can_x()
    half_mouth = 0.5 * rig.bag.width - rig.bag.handle_w
    assert max(abs(x) for x in xs) + rig.can.radius < half_mouth
    assert 2.0 * rig.bag.gap > 2.0 * rig.can.radius                                  # the open mouth takes a can


def test_script():
    assert D.targets(0.0) == (0.0, 0.0)
    assert D.targets(0.5 * D.PHASES[0]) == (0.0, pytest.approx(-D.SLACK))
    assert D.targets(D.PHASES[0]) == (0.0, pytest.approx(-D.SLACK))
    x, z = D.targets(D.PHASES[1])
    assert x == 0.0 and z == pytest.approx(D.LIFT)
    x, z = D.targets(D.PHASES[3])
    assert x == pytest.approx(D.CARRY_X) and z == pytest.approx(D.LIFT)


def test_score():
    s = BagSpec()
    v, _, g = bag_mesh(s)
    q0 = np.repeat(v[None], 2, axis=0)
    q = q0 + np.array([0.0, 0.0, D.LIFT])                                             # lifted with the jaws
    cans = np.array([[[0.0, 0.0, 0.2]], [[0.0, 0.0, 0.2]]])
    cans[1, 0, 0] = 0.5                                                                # world 1: a can outside
    row = D._score(q, q0, cans, np.full(2, D.LIFT), g)
    assert row["success"] == 1 and row["cans_inside"] == [1, 0]
    q[0, g["band_l"], 2] -= 0.01                                                       # world 0: a strap slid 1 cm
    row = D._score(q, q0, cans, np.full(2, D.LIFT), g)
    assert row["success"] == 0 and row["strap_slide_mm_max"] == pytest.approx(10.0)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_task_deposit_cans():
    """Row in docs/notes/film-grip-c16.md (CPU, 1 world, 3 cans, 4.3 s)."""
    row = D.run(1, "cpu", cans=3)
    assert row["success"] == 1, row
