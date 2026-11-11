# SPDX-License-Identifier: Apache-2.0
"""The real roller on the stores (CPU): the kinematic cover rolls without slip on the wall and in the tray, loading
turns it more than once in the paint, the lanes cover the strip; on a small wall the scripted roller loads (load-roller
passes) and rolls the strip (roll-strip passes) with the store's and the tray's books exact."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import paint_roller as pr
from alloy.tasks.paint_scene import Band, WallLayout

SMALL = WallLayout(width=0.4, height=0.352, cell=0.008, strip=Band(-0.1, 0.1, 0.076, 0.276))


def test_the_cover_rolls_without_slip() -> None:
    start, moves = pr.script(WallLayout(), passes=2)
    angle, rolling = 0.0, 0
    for p in pr.poses(start, moves):
        assert p.angle == pytest.approx(angle, abs=1e-9)                  # the pose's angle integrates the spin
        angle += p.spin * pr.SUB
        if p.roll is None:
            assert p.spin == 0.0
            continue
        rolling += 1
        contact = p.vel + np.cross((p.spin, 0.0, 0.0), pr.R * np.asarray(pr.CONTACT[p.roll]))
        assert np.allclose(contact, 0.0, rtol=0.0, atol=1e-12)            # the contact line stands still
    assert rolling > 0


def test_loading_turns_every_bin_through_the_paint() -> None:
    lay = WallLayout()
    x, y, z = pr.well_floor(lay)
    level = z + pr.TRAY_PAINT / (pr.WELL[0] * pr.WELL[1])
    start, _ = pr.script(lay)
    dipped = []                                     # the cover's angle while it rolls in the paint
    for p in pr.poses(start, pr.load_moves(lay)):
        if p.roll != "tray":
            continue
        dipped.append(p.angle)
        assert abs(p.pos[1] - y) <= 0.5 * pr.WELL[1] - pr.R + 1e-12 and p.pos[2] - pr.R < level   # in the paint
    assert max(dipped) - min(dipped) >= 2.0 * math.pi                     # every bin came to the bottom


@pytest.mark.parametrize("lay", [WallLayout(), SMALL])
def test_the_lanes_cover_the_strip(lay: WallLayout) -> None:
    s, xs = lay.strip, pr.lanes(lay)
    assert xs[0] - pr.HW <= s.x0 + 1e-12 and xs[-1] + pr.HW >= s.x1 - 1e-12
    assert all(b - a <= 2.0 * pr.HW for a, b in zip(xs[:-1], xs[1:]))
    moves = pr.lane_moves(lay, xs[0])
    ys = [m.to[2] - lay.bottom for m in moves if m.roll == "wall"]
    first, last = s.y0 + 0.5 * lay.cell, s.y1 - 0.5 * lay.cell               # the strip's edge cells' centres
    assert min(ys) < first and max(ys) > last                              # the nip crosses them


def test_the_nip_gets_twice_the_naps_strain() -> None:
    assert pr.express(0.0) == 0.0 and pr.express(pr.NAP) == 1.0
    assert pr.express(1.5e-3) == pytest.approx(2.0 * 1.5e-3 / 9.5e-3, rel=1e-12)


@pytest.mark.compile_heavy
def test_the_scripted_roller_loads_and_rolls_a_small_wall() -> None:
    from alloy.tasks.paint_roll_strip import run_roller

    v = run_roller(SMALL, settle=10.0)
    assert v["dry_film"] == "stage"
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == set()
    assert v["checks"]["hide"]["value"] == 1.0
    # the nip's expressed share and split set the coat: e 0.32 lays 56-67 um dry here (the whole load: 108-152,
    # split 0.7: ~32)
    assert 50.0 < v["dry_strip"]["p5_um"] and v["dry_strip"]["p95_um"] < 75.0
    assert {k for k, c in v["load_roller"]["checks"].items() if c["ok"] is False} == set()
    # AB-gym-6: the release, the marks and the missing ceiling and trim are not judged; the load's four are
    assert v["judged"] == "4 of 7" and list(v["not_judged"]) == ["protected", "untouched", "roller_in_tray"]
    assert v["checks"]["roller_in_tray"]["ok"] is None and v["succeeded"]
    assert v["load_roller"]["judged"] == "4 of 4" and v["load_roller"]["not_judged"] == {}
    assert "one drop" in v["load_roller"]["checks"]["no_drip_10s"]["note"] and "first hold" in v["load_roller"]["scene"]
    assert v["load_roller"]["checks"]["load"]["value"] == pytest.approx(0.75 * pr.CAPACITY * 1295e3, rel=0.01)
    assert v["checks"]["tool_books"]["value"] < 1e-12
