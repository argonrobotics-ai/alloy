# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can's robot stage v1 (``can_wash_robot``): the band poses and the dock (fast), and a cut-down dry
run of the whole driver in its scene on the CPU (WP15: a dry run per driver)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks.can_wash_robot import bands, dock_tcp, walk_law
from alloy.tasks.can_wash_run import Rinse, can_z
from alloy.tasks.can_wash_table import Turntable
from alloy.tasks.can_wash_tip import COARSE, rotate
from alloy.tasks.can_wash_turn import OPEN, RELEASE_N

TASK = dataclasses.replace(COARSE, table=Turntable())


def test_each_band_aims_the_jets_where_stage_1_aimed_them():
    """Every band's tool pose puts the nozzle's axis (the tool's z) through the band's aim point at its radius: the
    inside bands across the can from the tip over it, the outside bands at the wall from ``standoff`` out."""
    from alloy.tasks.can_wash_arms import WAND  # noqa: PLC0415

    can, rs = TASK.can, Rinse()
    c = np.array([*TASK.at, can_z(TASK)])
    out = bands(TASK, rs)
    assert [b.kind for b in out] == ["inside"] * 6 + ["outside"] * 3
    heights = [0.0] * len(rs.floor) + list(rs.inner) + list(rs.outer)
    for b, z in zip(out, heights, strict=True):
        tip = b.tcp[:3] + WAND * rotate(b.tcp[3:], np.array([0.0, 0.0, 1.0]))
        j = rotate(b.tcp[3:], np.array([0.0, 0.0, 1.0]))
        k = (c[2] + z - tip[2]) / j[2]                   # where the axis reaches the band's height
        hit = tip + k * j
        assert math.hypot(*(hit[:2] - c[:2])) == pytest.approx(b.r, abs=1e-9), b.label
    assert out[0].r == rs.floor[0] and out[-1].r == pytest.approx(float(can.radius(rs.outer[-1])) + can.wall)


def test_the_dock_puts_the_valve_head_down_over_the_holster():
    tcp = dock_tcp(TASK)
    assert rotate(tcp[3:], np.array([0.0, 0.0, 1.0])) == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def rev(lip_n: float, aboard: float, offsets: list[float]) -> dict:
    return {"lip_force_n": lip_n, "aboard_kg": aboard, "offset_mm_strokes": offsets}


def test_the_walk_law_bounds_the_lip_on_both_routes_and_the_walk_by_route():
    """WP15's lip law (11-09): the lip's peak force under 0.3 of the can's and the water's weight in every band on both
    routes; the can's axis under 4 mm off the table's on MuJoCo C (the lip's clearance: it never touches), within 5.5
    mm on MuJoCo Warp (the clearance plus the lip's contact give: Warp's foot creeps 1.3-2.8 mm a stroke against 0.3
    on C, and the full row reached 4.80 mm with the lip at 0.121 of the load). A lip force not read fails."""
    m = TASK.can.mass
    c_row = [rev(0.0, 6.64, [0.52, 1.46]), rev(0.0, 0.07, [1.14, 0.35])]
    warp_row = [rev(8.09, 2.81, [1.51, 4.8]), rev(5.41, 0.07, [4.32, 3.21])]
    assert walk_law(c_row, m, "mujoco_c") == {"route": "mujoco_c", "lip_share": 0.0, "walk_mm": 1.46,
                                              "walk_bound_mm": 4.0, "ok": True}
    law = walk_law(warp_row, m, "mujoco_warp")
    assert law["ok"] and law["walk_mm"] == 4.8 and law["lip_share"] == pytest.approx(0.121, abs=1e-3)
    assert not walk_law(warp_row, m, "mujoco_c")["ok"]                          # 4.8 mm: past C's bound
    assert not walk_law([rev(0.0, 0.0, [5.6])], m, "mujoco_warp")["ok"]
    assert not walk_law([rev(0.31 * m * 9.81, 0.0, [1.0])], m, "mujoco_c")["ok"]
    assert not walk_law([rev(float("nan"), 0.0, [1.0])], m, "mujoco_c")["ok"]


@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
def test_a_cut_down_robot_run_turns_sprays_dumps_and_docks():
    """One rinse of the 0.30 m inside band, the dump, the 0.40 m outside band, the dock, a 5 s settle (CPU, the
    suite's MuJoCo Warp route, one world). Each band turns 360° within 0.5° in seven strokes or eight (a stroke the
    pinch slips turns short). No stroke nears tipping the can: tip ratio under 0.5 with the water aboard. The can
    slips under 3 mm on the table (at its foot) and 30 mm in the pads, its axis drifts under 5 mm and stays under 4 mm
    off the table's (WP15: the creep), and the table's lip bears under 0.3 of the can's and the water's weight
    (WP15). The plate's brake is released for every turn, and the can stays put in every gap between them (under
    3°). The UR's commands stay under its joint speed limits, and the FR3's plan within its limits. The valve's
    pulses give each band stage 1's dose per metre (0.95-1.0). The dump empties the pool (under 25 mL once the can
    stands again: the floor's film runs back) with nothing lost or caught by the table or the arms; the valve docks;
    the books close and no contact overflows. The cut-down rinse is not a clean can, so the score is not read. 775 s
    for this file, the turn's and the strokes' on a loaded 4-core host (11-09, the lip 4 mm clear): 7 strokes a band,
    pads 2.5 mm, the foot 2.2-2.4 mm, the axis up to 3.3 mm off, tip ratio 0.38 and 0.43, the lip 0 N, the pool
    16.4 mL after."""
    from alloy.tasks.can_wash_robot import run_robot  # noqa: PLC0415
    from alloy.tasks.can_wash_tip import Empty  # noqa: PLC0415

    em = Empty(rinses=1, rinse=Rinse(floor=(), inner=(0.30,), outer=()), outer=(0.40,), settle=5.0)
    out = run_robot(em=em)
    revs = out["plans"]["revolutions"]
    print(revs, out["dumps"], out["plans"]["fr3"], out["plans"]["ur_speed_share"], out["caught_ml"],
          out["idle_after_deg"], out["ur_end"])
    assert [r["band"] for r in revs] == ["inside z 0.30 r 0.24", "outside z 0.40"]
    for r in revs:                    # a stroke the pinch slips turns short; revolve_to then adds one (8 on Warp)
        assert 7 <= r["strokes"] <= 8 and abs(r["turned_deg"] - 360.0) < 0.5, r
        assert r["tip_ratio"] < 0.5 and r["idle_deg"] < 3.0 and 0.95 <= r["dose_share"] <= 1.0, r
        assert r["table_slip_mm"] < 3.0 and r["drift_mm"] < 5.0 and r["pad_slip_mm"] < 30.0, r
        assert max(r["offset_mm_strokes"]) < 4.0, r                                          # WP15: the creep
        assert r["lip_force_n"] < 0.3 * (TASK.can.mass + r["aboard_kg"]) * 9.81, r            # WP15: centres it
        assert r["brake_grip"] == 0.0 and max(abs(g) for g in r["idle_deg_gaps"]) < 3.0, r     # WP15: released
                                                                                     # for every turn, drift per gap
    (dump,) = out["dumps"]
    assert dump["pool_ml_before"][0] > 1000.0 and dump["pool_ml_after"][0] < 25.0, dump   # the floor's film runs
                                                                    # back while the can stands and is lowered
    assert all(dump["caught_ml"][k] < 1.0 for k in ("table", "ur", "fr3")), dump     # WP15: none on the plate, the
                                                                                    # lip or the arms in the dump
    assert out["lost_ml"] == [0.0] and out["overflow"] == [False]
    assert out["walk"]["route"] == "mujoco_warp" and out["walk"]["ok"], out["walk"]               # WP15's lip law
    fr3 = out["plans"]["fr3"]
    assert fr3["misses"] == 0 and fr3["steps_over_fr3_accel"] == 0 and fr3["steps_over_fr3_jerk"] == 0
    assert out["plans"]["ur_misses"] == 0 and out["plans"]["ur_speed_share"] < 1.0
    assert abs(out["idle_after_deg"]) < 3.0 and out["ur_end"] == {"width_m": OPEN, "force_n": RELEASE_N}
    gates = out["worlds"][0]["gates"]
    hands = gates["water_off_and_hands_free"]         # AB-gym-3: the hose is off, but the valve stays fixed in the
    assert gates["valve_parked"]["ok"] and hands["ok"] is False and hands["value"]["hose_gpm"] == 0.0   # FR3's hand
    assert hands["value"]["hands"] == "still gripping" and "water_off_and_hands_free" in out["worlds"][0]["failed"]
    assert gates["no_simulation_fault"]["ok"] and gates["contained"]["ok"]
