# SPDX-License-Identifier: Apache-2.0
"""The real round brush on the stores (CPU): the pointed tip's band is as wide as its press, the tip level on the
stroke's point; the metering constants are Gym-Anything's, its C mapped to lay its calibrated 50 µm on this tip; the
store meters the discrete law over its touching cells; strokes stop short of a line's ends by the paint's reach and
land a period short of their start; the scripted brush loads in the palette and paints the S inside the outline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import paint_brush as pb


@pytest.mark.parametrize("width", [2e-3, 4e-3, 6e-3])
@pytest.mark.parametrize("load", [pb.HELD * pb.CAPACITY, 0.5 * pb.CAPACITY])
def test_the_band_follows_the_press(width: float, load: float) -> None:
    assert pb.band_width(pb.press_for(width, load), load) == pytest.approx(width, rel=1e-12)
    assert pb.band_width(1.0, load) == pb.BAND_MAX                     # the widest band (the swatch card's bound)
    assert pb.press_for(width, load) == pytest.approx(width, rel=1e-12)     # Gym-Anything's #8: as wide as pressed


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_the_tip_is_level_on_the_point(yaw: float) -> None:
    point, t, press, z = np.array([0.01, -0.02]), np.array([math.cos(yaw), math.sin(yaw)]), 1e-3, 0.75
    pose = pb.tip_pose(point, t, 4e-3, press, z)
    assert np.allclose(pose.pos, (point[0], point[1], z - press), rtol=0.0, atol=1e-15) and pose.quat == pb.LEVEL
    assert np.allclose(pose.vel, pb.SPEED * np.array([t[0], t[1], 0.0]), rtol=0.0, atol=1e-15)


def test_the_metering_constants_are_gym_anythings_mapped_to_the_tip() -> None:
    """Gym-Anything's round #8: C 0.60 (params.py:155), kc 20 N/m on 21 tufts over π/4·25 mm², the acrylic's Cross
    viscosity at U/50 µm with no structure; C mapped so the tip's law lays the calibrated 50 µm."""
    assert (pb.METER_GA, pb.FILM_GA, pb.KC, pb.TUFTS, pb.METER_MAX) == (0.60, 50e-6, 20.0, 21, 150e-6)
    assert pb.STIFFNESS == pytest.approx(2.139042435155073e7, rel=1e-12)
    assert pb.metering_viscosity() == pytest.approx(0.3415456601719373, rel=1e-12)
    assert pb.metered_film(pb.METER_GA) == pytest.approx(33.595482351376146e-6, rel=1e-12)
    assert pb.METER == pytest.approx(1.329013820550863, rel=1e-12)
    assert pb.metered_film(pb.METER) == pytest.approx(50e-6, rel=1e-12)
    store = pb.brush_store("acrylic_fluid")
    assert (store.meter, store.meter_max, store.meter_viscosity) == (pb.METER, 150e-6, pb.metering_viscosity())


def test_on_the_cone_the_law_is_the_same_at_any_press() -> None:
    """The patch's area over its width and the pad's force over that width both scale with the press ``d``:
    ``L_c sqrt(C eta U / w)`` is :func:`metered_film` at any press (this model's cone; Gym-Anything's tufts differ)."""
    s, eta = pb.TIP_SLOPE, pb.metering_viscosity()
    for d in (1e-3, 3e-3, 6e-3):
        a = d / s
        lc, w = math.pi * a * a / (2.0 * a), pb.STIFFNESS * math.pi * d ** 3 / (3.0 * s * s) / (2.0 * a)
        assert lc * math.sqrt(pb.METER * eta * pb.SPEED / w) == pytest.approx(pb.metered_film(pb.METER), rel=1e-12)


def test_strokes_stop_short_of_the_line_by_the_paints_reach() -> None:
    from alloy.tasks import paint_metrics as pm

    pts, half = pm.s_curve(2000)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    first, *_, last = pb.stroke_points()
    k0, k1 = (int(np.argmin(np.linalg.norm(pts - p[0], axis=1))) for p in (first, last))
    spacing = float(np.diff(arc).max())                                # the template's points, 0.13 mm apart
    assert arc[k0] == pytest.approx(pb.end_inset(2.0 * half[0]), abs=1.5 * spacing)
    assert arc[-1] - arc[k1] == pytest.approx(pb.end_inset(2.0 * half[-1]), abs=1.5 * spacing)
    assert pb.end_inset(1.0) == 0.5 * pb.BAND_MAX


def _discrete_law(press: float, cell: float) -> float:
    """Gym-Anything's law on the cells a tip pressed ``press`` touches with its point on a cell corner: ``L_c`` their
    area over their width across the slide, ``w`` the pad's spring load over that width."""
    c = (np.arange(-8, 8) + 0.5) * cell
    x, y = np.meshgrid(c, c, indexing="ij")
    depth = press - pb.TIP_SLOPE * np.hypot(x, y)
    touch = depth > 0.0
    width = float(y[touch].max() - y[touch].min()) + cell
    lc, w = touch.sum() * cell * cell / width, pb.STIFFNESS * float(depth[touch].sum()) * cell * cell / width
    return lc * math.sqrt(pb.METER * pb.metering_viscosity() * pb.SPEED / w)


@pytest.mark.compile_heavy
def test_a_straight_stroke_meters_the_discrete_law() -> None:
    """A loaded tip slid at ``SPEED`` across dry paper pressed 4 mm, its point on the cells' corners: the store's h_m is
    the law on its 12 touching 1 mm cells (to float32 poses), and the four cells across the lane hold it to 1.2 %."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
    from alloy.tasks import paint_process as pp

    paint, z = "acrylic_fluid", 0.75
    flow = SurfaceFlow(charts=(PlaneChart("paper", cells=(60, 20)),), inertia=False, max_depth=1e-3,
                       tools=(pb.brush(paint, initial=pb.HELD),), tool_speed=pb.SPEED,
                       **pp.film_settings(paint, substrate="paper"))  # fmt: skip
    start = pb.tip_pose(np.array([-0.025, 0.0]), np.array([1.0, 0.0]), 4e-3, 0.0, z)
    sim = build_simulation(pb.brush_scene(lambda s: None, z, (0.06, 0.02), tuple(start.pos)), [flow], "cpu")
    body = list(sim.model.body_label).index("brush")
    for k in range(40):
        pb.place(sim, body, pb.stroke_pose(np.array([-0.025 + k * pb.SPEED * pb.DT, 0.0]), np.array([1.0, 0.0]),
                                           4e-3, 0.0, z))  # fmt: skip
        sim.step_period()
    law = _discrete_law(4e-3, 1e-3)
    assert law == pytest.approx(47.33889825536212e-6, rel=1e-12)
    assert float(np.asarray(flow.read()["tool_hm"]).ravel()[0]) == pytest.approx(law, rel=1e-5)
    h = flow.depth(0)[0][30]                                          # the column at x = 0.5 mm
    wet = h[h > 1e-6]
    assert len(wet) == 4 and np.abs(wet / law - 1.0).max() < 0.012


@pytest.mark.compile_heavy
def test_the_scripted_brush_paints_the_s() -> None:
    from alloy.tasks.paint_brush_stroke import run_brush

    from alloy.tasks.paint_materials import paint

    v = run_brush(settle=2.0, loaded=True)                  # the store starts at its held share: no dip, no hold
    assert v["dry_film"] == "stage"
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == set()
    assert v["checks"]["coverage"]["value"] >= 0.995 and v["checks"]["within_line"]["value"] == 0.0
    assert 0.0 < v["load_left"] < pb.HELD * pb.CAPACITY
    lo, hi = v["metered_um"]                                 # the law on whole cells saw-tooths about 50 µm
    assert 0.6 * pb.FILM_GA * 1e6 < lo and hi < 1.2 * pb.FILM_GA * 1e6
    dry = pb.FILM_GA * paint("acrylic_fluid").volume_solids * 1e6       # 50 µm wet at 0.65 solids: 32.5 µm dry
    assert v["dry_core"]["p50_um"] == pytest.approx(dry, rel=0.1)


def test_stroke_pose_is_the_tip_pose_at_its_press() -> None:
    p, t, w, load, z = np.array([0.01, 0.0]), np.array([1.0, 0.0]), 4e-3, pb.HELD * pb.CAPACITY, 0.75
    a, b = pb.stroke_pose(p, t, w, load, z, 0.3), pb.tip_pose(p, t, w, pb.press_for(w, load), z)
    assert np.array_equal(a.pos, b.pos) and a.quat == b.quat and a.yaw_rate == 0.0     # a round tip does not turn


def test_approach_and_lift_take_any_stroke_end() -> None:
    """By default the S's first and last points; any (point, tangent, width) otherwise: the approach lands the tip's
    point on the paper a period's travel short of the start, the lift rises 30 mm from the end."""
    well, z = (0.0, 0.1, 0.74), 0.75
    first = next(pb.stroke_points())
    assert [p.pos.tolist() for p in pb.approach_poses(well, z)] == [p.pos.tolist() for p in
                                                                    pb.approach_poses(well, z, first)]
    end = (np.array([0.02, 0.01]), np.array([0.0, 1.0]), 3e-3)
    at = pb.tip_pose(end[0], end[1], end[2], 0.0, z)
    landed = at.pos - (0.0, pb.SPEED * pb.DT, 0.0)                     # a period's travel short, along the stroke
    assert np.allclose(pb.approach_poses(well, z, end)[-1].pos, landed, rtol=0.0, atol=1e-12)
    assert np.allclose(pb.lift_poses(z, end)[-1].pos, at.pos + (0.0, 0.0, 0.03), rtol=0.0, atol=1e-12)
