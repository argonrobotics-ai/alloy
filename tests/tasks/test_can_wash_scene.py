# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can scene (alloy/tasks/can_wash_scene.py, can_wash_run.py): the can against Gym-Anything's numbers
(spec rinse_spec_more.md §2: 124 L, ~2.4 m² charted with the underside, 56.5 mg detergent, 354 mg debris), the
script's poses, a short scripted rinse and the overflow watch."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.can_wash_run import Rinse, pose, plan
from alloy.tasks.can_wash_scene import (CROP, DEBRIS_LOAD, DEBRIS_SHARE, FILM_H, WASH, CanWash, langmuir)
from alloy.tasks.rinse_flush_policy import NOZZLE

FULL = CanWash()


def _areas(task: CanWash) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from alloy.plugins.surface_flow.charts import revolution_geometry  # noqa: PLC0415

    return revolution_geometry(task.can.chart("can")).area, task.regions()


def test_can_holds_124_l_and_charts_their_area_but_the_underside():
    """Their 2.4 m² is 2 x lateral 1.06 + 2 x base 0.16; the chart ends at the foot's outer corner (an open edge), so
    it leaves out the underside (0.15 m²) and the inner floor's uncharted disc (r < r_min, 0.012 m²)."""
    assert FULL.can.cavity_volume() * 1e3 == pytest.approx(124.0, rel=0.01)
    can = FULL.can
    assert FULL.can.profile()[-1] == pytest.approx((float(can.radius(0.0)) + can.wall, -can.bottom))
    area, reg = _areas(FULL)
    underside = math.pi * (can.r_bot ** 2 - can.r_min ** 2)
    assert area.sum() == pytest.approx(2.4 - underside - math.pi * can.r_min ** 2, rel=0.01)
    assert sum(m.astype(int) for m in reg.values()).tolist() == [1] * area.size     # each cell in one region
    assert area[reg["inner"]].sum() == pytest.approx(1.18, abs=0.01)


def test_start_loads_match_their_report():
    """Film 30 µm at 0.75 g/L plus the walls' Langmuir load over the chart: 51.6 mg (their 56.5 mg counts foam and
    the underside's 3.2 mg too); debris on half the inner cells at 0.5e-3 x 2 x U(0.2, 1): 354 mg."""
    area, reg = _areas(FULL)
    soap = (FILM_H * WASH + float(langmuir(WASH))) * area.sum()
    assert soap * 1e6 == pytest.approx(51.6, abs=0.1) and float(langmuir(WASH)) == pytest.approx(5e-7 * 15 / 16)
    debris = DEBRIS_SHARE * DEBRIS_LOAD * 0.6 * area[reg["inner"]].sum()
    assert debris * 1e6 == pytest.approx(354.0, rel=0.01)


def _axis(p: tuple[float, ...]) -> np.ndarray:
    x, y, z, w = p[3:]
    return -np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])    # body -z in world


def test_pose_puts_the_nozzle_at_the_tip_aimed_at_the_wall():
    tip, aim = np.array([0.3, 0.2, 0.9]), np.array([0.6, 1.0, 0.1])
    p = pose(tip, aim)
    j = _axis(p)
    assert np.allclose(j, (aim - tip) / np.linalg.norm(aim - tip), atol=1e-12)
    assert np.allclose(np.asarray(p[:3]) + NOZZLE * j, tip, atol=1e-12)


@pytest.mark.parametrize("task", [FULL, CROP])
def test_plan_stays_clear_of_the_can(task: CanWash):
    """Outside, the nozzle stands off the wall; inside, it is over the rim. The jets always aim down."""
    rs = Rinse()
    poses = np.asarray(plan(task, rs))
    can = task.can
    j = np.stack([_axis(tuple(p)) for p in poses])
    tip = poses[:, :3] + NOZZLE * j
    r = np.hypot(tip[:, 0] - task.at[0], tip[:, 1] - task.at[1])
    z = tip[:, 2] - (can.bottom + 1e-4)
    out = r > can.r_top + 0.2
    over = z > can.depth + 0.5 * rs.over_rim
    assert (out | over).all() and out.any() and over.any() and (j[:, 2] < 0.0).all()
    n_out = sum(math.ceil(2 * math.pi * (float(can.radius(h)) + can.wall) / (rs.speed * 0.02)) for h in rs.outer)
    assert out.sum() == n_out


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_short_rinse_pools_cleans_where_it_hits_and_keeps_its_books():
    """CROP on the CPU, 1.5 s of spray (one floor ring, one inner and one outer height), then GA's settle: the books
    close (to rounding), no contact overflows, the cavity pools, the sprayed band loses its debris, the outside's
    soap runs off the foot, and the valve docks shut. Nothing tips the can (stage 1), so the cavity's detergent stays
    and emptied_upright fails."""
    from alloy.tasks.can_wash_run import run  # noqa: PLC0415

    res = run(CROP, Rinse(floor=(0.12,), inner=(0.5,), outer=(0.4,), speed=2.0, settle=4.0))
    w, start, end = res["worlds"][0], res["start"], res["end"][0]
    assert res["overflow"] == [False] and res["solver"] is not None             # MuJoCo Warp: overflow bits read
    assert 3.0 <= w["settle_s"] <= 4.0                    # the first quiet check after GA's 2 s, then each 1 s
    fault = w["gates"]["no_simulation_fault"]
    # the film's water closes to 1e-16; the tank's, kappa (P - P_ref), rounds at ~4e-15 m³ of its 0.2 L
    assert fault["ok"] and fault["value"]["water_rel"] < 1e-10 and fault["value"]["species_kg"] < 1e-15, fault
    assert sorted(w["failed"]) == ["emptied_upright", "soap_free", "visibly_clean"]
    hands = w["gates"]["water_off_and_hands_free"]                 # AB-gym-3: the hose off, no pads to read
    assert hands["ok"] is None and hands["value"]["hose_gpm"] == 0.0 and w["judged"] == [6, 8]
    assert w["not_judged"] == ["water_off_and_hands_free", "intact_and_not_dropped"]
    assert start["soap_kg"][0] * 1e6 == pytest.approx((FILM_H * WASH + float(langmuir(WASH))) * start["area_m2"] * 1e6)
    soap = res["soap_mg_regions"]
    assert soap["end"]["outer"][0] < 0.5 * soap["start"]["outer"][0]            # one pass: 24.2 -> 10.2 mg, off
                                                                                # the foot's open edge (CROP's 30 mm
                                                                                # cells leave the 6 mm foot empty)
    assert res["pool_L"][0] > 0.1 and end["free_ml"][0] > 100.0 and end["soap_kg"][0] < 0.8 * start["soap_kg"][0]
    (s0,), (s1,) = res["soil_mg"]
    assert s1 < 0.5 * s0


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_the_overflow_watch_fails_a_starved_run(monkeypatch: pytest.MonkeyPatch):
    """MuJoCo Warp drops what overflows its buffers with only a stdout line: starved to 16 constraint rows a world (it
    raises nconmax 4 to its minimum, 96), the can on the tile overflows, the watch sees it, and no_simulation_fault
    fails (as SC9's flush-floor test). WP8's OverflowWatch, in the run's plug-ins, keeps every substep's bits."""
    from alloy.tasks import can_wash_scene  # noqa: PLC0415
    from alloy.tasks.can_wash_run import run  # noqa: PLC0415

    monkeypatch.setattr(can_wash_scene, "CAPACITY", {"nconmax": 4, "njmax": 16})
    res = run(CROP, Rinse(floor=(), inner=(), outer=(0.4,), speed=8.0, settle=0.04))
    w = res["worlds"][0]
    assert res["overflow"] == [True] and res["watch"] and not w["gates"]["no_simulation_fault"]["ok"]
    assert w["gates"]["no_simulation_fault"]["value"]["finite"] is False and w["score"] is None


@pytest.mark.parametrize("task", [FULL, CROP])
def test_inside_jets_enter_the_mouth_whole(task: CanWash):
    """Every ray of the 6° cone (SC9's Hose) from each pose over the mouth crosses the rim's plane inside the rim,
    going down: no jet clears the rim toward the shelf. Gravity only bends them further in."""
    from alloy.tasks.rinse_flush_floor import Hose  # noqa: PLC0415

    rs, can, half = Rinse(), task.can, math.radians(Hose().cone_deg)
    poses = np.asarray(plan(task, rs))
    j = np.stack([_axis(tuple(p)) for p in poses])
    tip = poses[:, :3] + NOZZLE * j
    rim, z_rim = float(can.radius(can.depth)), can.bottom + 1e-4 + can.depth
    above = (tip[:, 2] > z_rim) & (np.hypot(tip[:, 0] - task.at[0], tip[:, 1] - task.at[1]) < rim)    # over the mouth
    assert above.sum() == sum(math.ceil(2 * math.pi * r / (rs.speed * 0.02))
                              for r in (*rs.floor, *(float(can.radius(z)) for z in rs.inner)))
    for t, d in zip(tip[above], j[above], strict=True):
        a = np.cross(d, [0.0, 0.0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        for phi in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            ray = math.cos(half) * d + math.sin(half) * (math.cos(phi) * a + math.sin(phi) * b)
            assert ray[2] < 0.0
            p = t + (z_rim - t[2]) / ray[2] * ray
            assert math.hypot(p[0] - task.at[0], p[1] - task.at[1]) < rim - 0.005
