# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's long-reach brush on the surface-flow stores (CPU, a small wall): it loads from the pail toward its
capacity, drains to its held share before it leaves the pail (its drips back into the pail), and lays the brushed
build GA's metering law gives; a brush carried over the radiator undrained drips on it; every volume booked
exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import radiator_brush as rb
from alloy.tasks import radiator_metrics as rm
from alloy.tasks import radiator_scene as rs
from alloy.tasks import radiator_script as scr
from alloy.tasks import radiator_task as rt

WALL = (0.20, 0.08)              # m
Z = 0.30                         # m, the small wall's centre height


def small_scene(top: bool = True):
    """A small wall (chart 0, the brush's) and, with ``top``, the radiator's top (chart 1, protected)."""
    from alloy.plugins.surface_flow import PlaneChart  # noqa: PLC0415

    box = rs.Box("wall", (0.0, 0.5 * rs.WALL_T, Z), (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
                 (0.5 * WALL[0], 0.5 * WALL[1], 0.5 * rs.WALL_T), "wall")
    chart = PlaneChart("wall", cells=(int(round(WALL[0] / rs.CELL)), int(round(WALL[1] / rs.CELL))))
    if not top:
        return [box], (chart,)
    return [box, rs.chart_boxes()[rs.RAD_TOP]], (chart, rs.charts()[rs.RAD_TOP])


def episode(moves: list, top: bool = True):
    """``moves`` from over the pail (the radiator's top charted with ``top``): per exchange, the brush's load [m³]
    and the wall's depth [nx, ny]; the final read and the cell areas."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    start = (np.asarray(scr.pail_tip(scr.ABOVE)), -90.0)
    tips, alphas = scr.poses(start, moves)
    tcps = rt.path(tips, alphas)
    boxes, charts = small_scene(top)
    flow = rt.flow_spec(charts=charts)
    sim = build_simulation(rt.scene(None, tcps[0], float(alphas[0]), boxes=boxes), [flow], "cpu")
    body = list(sim.model.body_label).index("brush")
    loads, depths = [], []
    for k in range(len(tcps)):
        n = min(k + 1, len(tcps) - 1)
        rt.place(sim, body, tcps[k], float(alphas[k]), (tcps[n] - tcps[k]) / scr.SUB, (alphas[n] - alphas[k]) / scr.SUB)
        loads.append(float(flow.read()["tool_load"][0].sum()))
        sim.step_period()
        depths.append(flow.depth(0)[0].copy())
    return np.asarray(loads + [float(flow.read()["tool_load"][0].sum())]), depths, flow.read(), flow.tables.area_np


def _first_pass(depths: list) -> int:
    """The first exchange the stroke lays paint (a drip from the lifted brush may land on the wall before it)."""
    return next(k for k, d in enumerate(depths) if d.max() > 0.5 * rb.BRUSHED_WET)


@pytest.fixture(scope="module")
def run():
    """A dip, the drain, then one stroke along the small wall (no radiator: its 9,000-cell top triples the cost)."""
    return episode(scr.load() + [scr.Move(tuple(rb.on_wall(-0.08, Z, scr.LIFT)), 0.0, scr.TRAVEL),
                                 scr.Move(tuple(rb.on_wall(-0.08, Z)), 0.0), scr.Move(tuple(rb.on_wall(0.08, Z)), 0.0)],
                   top=False)


def test_capacity_is_gym_anythings_wetted_pores():
    """Gym-Anything's wetted pore volume (PT/brush.py:69-76, :519-522), written out tuft by tuft from its formula: the
    20 x 5 tufts, 59.7 mm at the long side and 15° bevel shorter across the width; each its 0.8 plug share less its
    46.8 filaments' tip half twice (the taper's volume by quadrature); wetted over its dipped share. The script's dip
    puts the long side's tips IMMERSION into the pail's paint; the whole head holds 22.0 mL, a dip 5.6 mL."""
    from scipy.integrate import quad  # noqa: PLC0415

    ls = [rb.BRISTLE - (rb.WIDTH / 2 - ((i + 0.5) / 20 * rb.WIDTH - rb.WIDTH / 2)) * math.tan(math.radians(15.0))
          for i in range(20) for _ in range(5)]  # fmt: skip
    lm = sum(ls) / 100

    def d(s: float) -> float:
        return rb.D_ROOT + (rb.D_TIP - rb.D_ROOT) * s / lm

    tip_half = quad(lambda s: math.pi * d(s) ** 2 / 4, lm / 2, lm)[0]
    pores = [2 * max(0.8 * rb.WIDTH * rb.THICK / 100 * x / 2 - 46.8 * tip_half * x / lm, 1e-12) for x in ls]
    np.testing.assert_allclose(rb.tuft_pores(), pores, rtol=1e-12, atol=0)
    depth = [max(rb.IMMERSION - (max(ls) - x), 0.0) for x in ls]
    ga = sum(p * min(dp / x, 1.0) for p, dp, x in zip(pores, depth, ls))
    assert rb.capacity() == pytest.approx(ga, rel=1e-12) and rb.capacity() == pytest.approx(5.60e-6, rel=1e-3)
    assert rb.capacity(1.0) == pytest.approx(sum(pores), rel=1e-12) and sum(pores) == pytest.approx(22.0e-6, rel=1e-3)
    paint = rb.PAIL_PAINT / (math.pi * rb.PAIL_R**2)                    # 29.8 mm of paint in the pail
    assert rb.IMMERSION == pytest.approx(paint - scr.DIP, abs=5e-4)


def test_the_dip_fills_toward_capacity_and_the_drain_leaves_the_held_share(run):
    """In the paint for ``DIP_S``: load = cap·(1 - exp(-pickup·t)); lifted for ``DRAIN_S``: the excess over the held
    share drips at ``DRIP``, at most exp(-DRIP·DRAIN_S) of it left (8 nL), under what would mark one 4 mm protected
    cell 1 µm dry."""
    loads, depths, read, _ = run
    cap = rb.capacity()
    full = float(loads.max())
    assert full == pytest.approx(cap * (1.0 - math.exp(-rb.PICKUP * scr.DIP_S)), rel=0.02)
    first = _first_pass(depths)
    excess = loads[first] - rb.HELD * cap
    left = (full - rb.HELD * cap) * math.exp(-rb.DRIP * scr.DRAIN_S)
    assert 0.0 < excess <= left * (1.0 + 1e-9)
    assert left < rs.CELL**2 * rm.PAINTED_DFT / rt.paint(rt.PAINT).volume_solids
    laid = -float(read["to_tools_volume"][0])
    assert read["tool_dripped"][0].sum() == pytest.approx(full - loads[-1] - laid, rel=1e-12)
    np.testing.assert_allclose(read["store_ledger"], 0.0, rtol=0, atol=1e-18)
    np.testing.assert_allclose(read["tray_ledger"], rb.PAIL_PAINT, rtol=1e-12, atol=0)
    dripped = float(read["tool_dripped"][0].sum())                    # the drain's drips fell back into the pail;
    assert float(read["supply_staged"][0]) == pytest.approx(dripped, rel=1e-12)    # the last few onto the wall
    assert float(read["supply_to_tank"][0]) > 0.99 * dripped and float(read["supply_lost"][0]) < 1e-3 * dripped


def test_a_sliding_brush_meters_gas_film(run):
    """While it slides, the cells under the brush are driven to GA's ``h_m = min(L_c·sqrt(C·η·U/w), 150 µm)`` with
    the metering viscosity (0.18 Pa·s) and the 2 N press: the plug-in's ``tool_hm`` is the closed form for the patch
    in 4 mm cells (its width counts whole cells across: 56 mm, so 95 µm, not the 100 µm of the 50.8 mm brush; to
    5e-3), and the cells the brush passed wholly over hold it (to 3e-3: a second of drying)."""
    _, depths, read, _ = run
    hm = float(np.asarray(read["tool_hm"])[0, 0])
    zs = -0.5 * WALL[1] + (np.arange(int(round(WALL[1] / rs.CELL))) + 0.5) * rs.CELL
    wc = float(np.ptp(zs[np.abs(zs) < 0.5 * (rb.WIDTH + rs.CELL)])) + rs.CELL
    u = 0.16 / (math.ceil(0.16 / (scr.SPEED * scr.SUB) - 1e-9) * scr.SUB)
    closed = rb.WIDTH * rb.THICK / wc * math.sqrt(rb.METER * rb.METER_VISCOSITY * u * wc / rb.PRESS_FORCE)
    assert rb.meter_viscosity_supported() and hm == pytest.approx(closed, rel=5e-3)
    assert 80e-6 <= hm <= 115e-6                                        # the brushed build (Rhoplex SG-10M)
    end = depths[-1]
    x = -0.5 * WALL[0] + (np.arange(end.shape[0]) + 0.5) * rs.CELL
    z = -0.5 * WALL[1] + (np.arange(end.shape[1]) + 0.5) * rs.CELL
    inner = (x > -0.06) & (x < 0.05)                                    # wholly passed: the stroke ran -0.08..0.08
    rows = np.abs(z) < 0.5 * rb.WIDTH - rs.CELL                         # wholly under its width
    np.testing.assert_allclose(end[np.ix_(inner, rows)], hm, rtol=3e-3, atol=0)


def test_the_books_close_exactly(run):
    _, _, read, _ = run
    laid = -float(read["to_tools_volume"][0])
    assert laid > 0.0
    assert abs(float(read["ledger"][0])) <= 1e-12 * laid
    sup = float(read["species_supplied"][0, :, 0].sum())
    assert abs(float(read["species_ledger"][0, 0])) <= 1e-12 * sup


def test_an_undrained_brush_over_the_radiator_drips_on_it():
    """Lifted from the dip and carried straight over the radiator's top, the brush drips its load past the held share:
    on the way onto the floor (``lost``), then onto the radiator's top chart, which then fails ``protected`` (more
    than 0.5 cm² painted). What landed there is the film there plus what dried off it."""
    from alloy.tasks import radiator_metrics as rm  # noqa: PLC0415

    over = (0.0, -rs.RAD_GAP - 0.5 * rs.RAD_DEPTH, rs.RAD_HEIGHT + 0.10)
    _, _, read, areas = episode(scr.load()[:2] + [scr.Move(over, -90.0, scr.TRAVEL, hold=6.0)])
    n0 = int(np.prod(small_scene()[1][0].cells))
    landed = float(np.asarray(read["supplied"])[0, n0:].sum())
    held = float((np.asarray(read["h"])[0, n0:] * areas[n0:]).sum())
    gone = sum(float(np.asarray(read[k])[0, n0:].sum()) for k in ("evaporated", "absorbed", "spilled", "clipped"))
    assert landed > 0.2 * float(read["tool_dripped"][0].sum()) and float(read["supply_lost"][0]) > 0.0
    assert held + gone == pytest.approx(landed, rel=1e-9)
    assert not rm.protected([(np.asarray(read["dft"])[0, n0:], areas[n0:])], 0.0).ok

