# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's geometry: the brush's reach round the radiator matches Gym-Anything's unscored pocket, the scripted
path keeps every tool part clear, and the brush's store values (CPU, no simulation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import radiator_brush as rb
from alloy.tasks import radiator_scene as rs
from alloy.tasks import radiator_script as scr

OBSTACLES = [rb.box_obb(b) for b in rs.obstacles()]
ANGLES = [0.0, 180.0] + [-float(a) for a in range(5, 180, 5)]


def reachable(x: float, z: float) -> bool:
    """Some attitude puts the tips' centre at wall point (x, z) with the hand and every tool part clear."""
    for a in ANGLES:
        tcp = rb.tcp_for_tip(rb.on_wall(x, z), a)
        if rb.hand_clear(tcp) >= 0.0 and rb.clearance(tcp, a, OBSTACLES) >= 0.0:
            return True
    return False


def test_separation_matches_a_box_pair_by_hand():
    a = rb.OBB(np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.1]))
    b = rb.OBB(np.array([0.25, 0.0, 0.0]), np.eye(3), np.array([0.1, 0.1, 0.1]))
    assert rb.separation(a, b) == pytest.approx(0.05, abs=1e-15)
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    turned = rb.OBB(np.array([0.25, 0.0, 0.0]), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), b.half)
    assert rb.separation(a, turned) == pytest.approx(0.25 - 0.1 - 0.1 * math.sqrt(2.0), abs=1e-15)


def test_the_tips_reach_all_but_gym_anythings_pocket():
    """Along the gap from either end the tips reach to 26.4 mm from the middle; from above, head down, to z 0.343 m.
    What they cannot reach at the bottom of the middle lies inside Gym-Anything's unscored pocket (x -25..30 mm below
    z 350 mm, measured with its phantom-held brush)."""
    x0, x1, z1 = rs.POCKET
    for x, z in [(-0.03, 0.2), (0.031, 0.2), (0.0, 0.351), (-0.0265, 0.12), (0.0, 0.55), (0.0, 0.6816)]:
        assert reachable(x, z), (x, z)
    for x, z in [(0.0, 0.2), (-0.024, 0.2), (0.024, 0.338), (0.0, 0.338)]:
        assert not reachable(x, z), (x, z)
        assert x0 <= x <= x1 and z <= z1


def test_the_script_keeps_the_brush_clear_of_the_radiator_and_the_trim():
    """Every exchange's pose of the whole scripted job: the hand 35 mm outside the radiator's box and the handle,
    ferrule and bristles clear of the radiator, its pipe and the trim (the plug-in would let a brush pass through
    them: an applicator lays only on its own chart, and nothing here collides)."""
    start, moves = scr.script()
    tips, alphas = scr.poses(start, moves)
    poses = np.unique(np.column_stack([tips, alphas]), axis=0)          # the holds repeat a pose
    worst = min(rb.clearance(rb.tcp_for_tip(p[:3], p[3]), float(p[3]), OBSTACLES) for p in poses)
    assert worst >= 0.0


def test_the_lanes_and_sweeps_reach_the_bands_ends_and_the_pocket_edge():
    zs = scr.rows()
    assert zs[0] - 0.5 * rb.WIDTH == pytest.approx(rs.BAND[2] + 1e-3) and zs[1] - zs[0] <= scr.PITCH
    assert zs[-1] + 0.5 * rb.WIDTH == pytest.approx(rs.BAND[3] - 1e-3)
    sweep_low = min(m.tip[2] for m in scr.sweeps())
    assert sweep_low == pytest.approx(scr.TCP_TOP - rb.TIP[0], abs=1e-12) and sweep_low < rs.POCKET[2]


def test_brush_store_values():
    """The dip's wetted pore volume (Gym-Anything's, tuft by tuft: ``test_radiator_brush``); GA's tuft stiffness
    density and the press that meters the brushed build at the paint's high-shear viscosity."""
    assert rb.capacity() == pytest.approx(5.60e-6, rel=1e-3)
    assert rb.PAD_STIFFNESS == pytest.approx(1.406e7, rel=1e-3) and rb.PRESS == pytest.approx(2.0e-4, rel=1e-3)
    assert rb.metered(scr.SPEED, rb.METER_VISCOSITY) == pytest.approx(rb.BRUSHED_WET, rel=5e-3)
    s = rb.brush_store("latex_eggshell")
    assert (s.retained, s.held, s.pickup, s.drip, s.meter, s.meter_max) == (0.25, 0.75, 2.0, 0.5, 0.075, 150e-6)
    tray = rb.pail_tray()
    assert 4.0 * tray.half[0] * tray.half[1] == pytest.approx(math.pi * rb.PAIL_R**2, rel=1e-12)


def test_a_dip_lays_a_lane_or_a_sweep():
    """What one dip can release, (held - retained) of the capacity (2.8 mL), covers the paint a lane lays (0.29 m
    in along the gap at the metered build over the brush's width) and a sweep (its 0.33 m down the middle), each
    within 60 % of it (the lay-off and laps over wet film take a little more)."""
    usable = (rb.HELD - rb.RETAINED) * rb.capacity()
    hm = rb.metered(scr.SPEED, rb.METER_VISCOSITY)
    lane = scr.LANES_PER_DIP * (scr.X_OUT - scr.X_IN) * rb.WIDTH * hm
    sweep = float(rb.TIP[0]) * rb.WIDTH * hm
    assert lane <= 0.6 * usable and sweep <= 0.6 * usable
    assert len([m for m in scr.script()[1] if m.hold == scr.DIP_S]) == 2 * len(scr.rows()) + len(scr.MIDDLE_X)
