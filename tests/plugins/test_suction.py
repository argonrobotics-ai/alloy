# SPDX-License-Identifier: Apache-2.0
"""Suction cups (alloy/plugins/library/suction.py): the harness laws A-SUC-1..4 and the lip-seal geometry (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

import alloy.plugins as ap
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.library.suction import Suction, SuctionCup
from alloy.plugins.npgeom import quat_rotate

pytestmark = pytest.mark.law

R, LIP_DROP, TOP = 0.02, 0.025, 0.04            # lip radius, mount origin to lip plane, box top height


def test_suction_quick_check_passes_every_law():
    report = ap.check("suction", quick=True)
    assert report.passed, report.summary()
    laws = {c.name for c in report.checks if c.group == "law"}
    assert {"A-SUC-1_holds_below", "A-SUC-1_lets_go_above", "A-SUC-2_evacuation", "A-SUC-3_shear_holds_below",
            "A-SUC-3_shear_slides_above", "A-SUC-4_seal_loss_drop_time", "A-SUC-4_vent_drop_time",
            "null_out_of_reach", "amount_balance"} <= laws


@pytest.mark.slow
@pytest.mark.parametrize("name", ["suction", "suction_cup"])
def test_suction_full_check(name):
    report = ap.check(name)
    assert report.passed, report.summary()


def _box_mesh(hx: float, hy: float, hz: float) -> newton.Mesh:
    """A closed box mesh, triangles wound outward."""
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], np.float32)
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))   # -x +x -y +y -z +z
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return newton.Mesh(v, np.array(tris, np.int32).ravel())


def _seal(tilt_deg: float = 0.0, half_x: float = 0.04, mesh: bool = False) -> float:
    """Seal fraction of a cup whose lip centre sits on a box top (half-width ``half_x`` in x), the cup tilted about y
    by ``tilt_deg`` around the lip centre; no gravity, no pressure difference (nothing moves). ``mesh``: the box is a
    triangle mesh."""
    th = math.radians(tilt_deg)
    q = (0.0, math.sin(0.5 * th), 0.0, math.cos(0.5 * th))
    mount = np.array([0.0, 0.0, TOP]) - quat_rotate(np.array(q), np.array([0.0, 0.0, -LIP_DROP]))

    def build(b: newton.ModelBuilder) -> None:
        SuctionCup(mount="fixed", stroke=0.0, lip_shape="box").build(b, body_xform(*mount, q=q), "suction")
        box = b.add_body(xform=body_xform(0.0, 0.0, 0.5 * TOP), label="box")
        cfg = newton.ModelBuilder.ShapeConfig(density=500.0)
        if mesh:
            b.add_shape_mesh(box, mesh=_box_mesh(half_x, 0.04, 0.5 * TOP), cfg=cfg)
        else:
            b.add_shape_box(box, hx=half_x, hy=0.04, hz=0.5 * TOP, cfg=cfg)

    scene = TestScene(build=build, worlds=2, periods=1, ground=False, gravity=(0.0, 0.0, 0.0))
    plug = Suction()
    sim = build_simulation(scene, [plug], "cpu")
    sim.step_period()
    seal = plug.read()["seal"]
    assert seal[0] == seal[1]
    assert np.all(plug.read()["target"] == (0 if seal[0] > 0 else -1))      # an index into targets, not a body id
    return float(seal[0])


@pytest.mark.parametrize("tilt, half_x, want", [
    (0.0, 0.04, 1.0),                    # flat on a wide face: all 16 lip points
    (0.0, 0.75 * R, 10.0 / 16.0),        # a face narrower than the lip: the 6 points at |cos φ| >= 0.924 hang off
    (math.degrees(math.asin(1.5e-3 / R)), 0.04, 11.0 / 16.0),   # tilted: 5 points rise > 1 mm (cos φ > 0.667)
    (40.0, 0.04, 0.0),                   # tilted past seal_max_angle_deg (30): no point seals
])
def test_seal_fraction_follows_the_lip_geometry(tilt, half_x, want):
    assert _seal(tilt, half_x) == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("tilt, half_x", [(0.0, 0.04), (0.0, 0.75 * R), (math.degrees(math.asin(1.5e-3 / R)), 0.04),
                                          (40.0, 0.04)])
def test_a_mesh_box_seals_like_the_primitive_box(tilt, half_x):
    assert _seal(tilt, half_x, mesh=True) == pytest.approx(_seal(tilt, half_x), abs=1.0 / 16.0 + 1e-6)


def test_config_is_validated():
    with pytest.raises(ap.PluginError, match="lip_points"):
        Suction(lip_points=2)
    with pytest.raises(ap.PluginError, match="targets"):
        Suction(targets=())
    with pytest.raises(ap.PluginError, match="mount"):
        SuctionCup(mount="wall")
