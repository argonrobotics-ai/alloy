# SPDX-License-Identifier: Apache-2.0
"""GA's fillet welding on Alloy (SCR-WP8-14): GA's section measures on synthetic slices, GA's judgements (fillet-2f,
its craters and progress, half-fillet) on synthetic rows, a short fillet on a cut-down coupon on the CPU (books, a bead
at the root, freeze times), and GA's fillet-2f on its own grid on CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.plugins.thermal.grid import graded
from alloy.tasks import weld_ga as wg
from alloy.tasks import weld_ga_criteria as gc
from alloy.tasks.weld_ga_measures import measure, section
from alloy.tasks.weld_ga_profile import Cap

T_HALF = 1765.15
EDGES = np.round(np.arange(-5e-3, 10e-3 + 1e-9, 2.5e-4), 9)


def _slice(weld, fused_depth: float = 5e-4):
    """A slice on 0.25 mm cells: plate below z = 0, bracket behind y = 0; ``weld(y, z)`` marks weld cell centres;
    the parent is fused ``fused_depth`` deep under / behind the weld's footprint."""
    c = 0.5 * (EDGES[1:] + EDGES[:-1])
    Y, Z = np.meshgrid(c, c, indexing="ij")
    cell = np.where(Z < 0, 1, np.where(Y < 0, 2, 0))
    w = (Y > 0) & (Z > 0) & weld(Y, Z)
    cell[w] = wg.WELD_CODE
    reach_y = Y[w].max() if w.any() else 0.0
    reach_z = Z[w].max() if w.any() else 0.0
    fused = ((cell == 1) & (Z > -fused_depth) & (Y <= reach_y)) | ((cell == 2) & (Y > -fused_depth) & (Z <= reach_z))
    return cell, np.where(fused | w, 1800.0, 300.0)


def _read(weld, fused_depth: float = 5e-4) -> dict:
    return section(*_slice(weld, fused_depth), EDGES, EDGES, wg.WELD_CODE, T_HALF)


def test_a_flat_fillet_reads_its_legs_size_and_throat_on_cells():
    m = _read(lambda y, z: y + z < 6.6e-3)
    assert m["leg_horizontal_m"] == pytest.approx(6.5e-3) and m["leg_vertical_m"] == pytest.approx(6.5e-3)
    assert 6.35e-3 <= m["size_m"] <= 6.6e-3 and m["throat_m"] == pytest.approx(m["size_m"] * np.sqrt(0.5), abs=2e-4)
    assert m["convexity_m"] < 2.5e-4 and m["root_fused"] and m["overlap_base_m"] == 0.0
    assert m["penetration_base_m"] == pytest.approx(5e-4) and m["penetration_bracket_m"] == pytest.approx(5e-4)


def test_a_round_bead_is_convex_and_undersize_and_a_cold_lap_reads():
    m = _read(lambda y, z: np.hypot(y, z) < 5.3e-3)          # a quarter disc of 22 mm²
    assert m["size_m"] < 6.35e-3 and m["convexity_m"] > 1e-3
    lap = _read(lambda y, z: y + z < 6.6e-3, fused_depth=0.0)
    assert lap["overlap_base_m"] > 6e-3 and lap["overlap_bracket_m"] > 6e-3 and not lap["root_fused"]


def test_a_capped_slice_reads_its_toes_on_the_surface():
    """A cap on toes 6.4 / 6.6 mm holding 26.3 mm² (GA's bead) filled by cell centre: the cells read the plate leg
    6.25 mm, the surface 6.4 mm (GA reads a slice with a pool profile on that profile)."""
    cap = Cap(6.4e-3, 6.6e-3, 26.3e-6)
    cell, peak = _slice(lambda y, z: cap.under(y, z), fused_depth=5e-4)
    on_cells = section(cell, peak, EDGES, EDGES, wg.WELD_CODE, T_HALF)
    on_cap = section(cell, peak, EDGES, EDGES, wg.WELD_CODE, T_HALF, cap)
    assert on_cells["size_m"] == pytest.approx(6.25e-3) and on_cap["geometry"] == "profile"
    assert on_cap["leg_horizontal_m"] == pytest.approx(6.4e-3) and on_cap["leg_vertical_m"] == pytest.approx(6.6e-3)
    assert on_cap["size_m"] == pytest.approx(6.4e-3) and on_cap["profile_missing_cells"] == 0
    assert on_cap["convexity_m"] == pytest.approx(on_cells["convexity_m"], abs=2.5e-4)


def _rows(size: float = 6.6e-3, metal=(5e-4, 0.1495), **over) -> list[dict]:
    """GA's rows over -2..152 mm at 0.5 mm: a fillet of ``size`` over ``metal`` laid by one pass along +x (frozen in
    that order), every section sound; ``over``: {key: f(x) -> value} overrides."""
    rows = []
    for x in np.arange(-2e-3, 0.152, 5e-4) + 2.5e-4:
        on = metal[0] <= x <= metal[1]
        r = dict(x_m=float(x), dx_m=5e-4, weld_area_m2=0.5 * size * size if on else 0.0, size_m=size if on else 0.0,
                 leg_vertical_m=size if on else 0.0, leg_horizontal_m=size if on else 0.0,
                 throat_m=size * np.sqrt(0.5) if on else 0.0, convexity_m=0.0, face_width_m=size * np.sqrt(2),
                 root_fused=on, penetration_base_m=1e-3 if on else 0.0, penetration_bracket_m=1e-3 if on else 0.0,
                 overlap_base_m=0.0, overlap_bracket_m=0.0, undercut_base_m=0.0, undercut_bracket_m=0.0,
                 porosity=0.0, liquid=False, freeze_t=float(5.0 + x / 0.005) if on else None)
        for k, f in over.items():
            r[k] = f(x) if on else r[k]
        rows.append(r)
    return rows


def test_fillet_2f_on_rows():
    ok = gc.judge_fillet(_rows())
    assert ok["success"] and ok["score"] == 1.0, {k: c for k, c in ok["checks"].items() if not c["ok"]}
    (crater,) = [c for c in ok["checks"]["crater_filled"]["craters"]]
    assert crater["stop_mm"] == [149.25]                      # the last section to freeze: the stop
    small = gc.judge_fillet(_rows(6.0e-3))
    assert not small["checks"]["leg_size"]["ok"]
    assert small["score"] == pytest.approx(0.9 * 6.0 / 6.35, abs=0.002)      # less the crater's deficit past 145 mm
    lap = gc.judge_fillet(_rows(overlap_base_m=lambda x: 1e-3))
    assert not lap["checks"]["overlap"]["ok"] and lap["score"] == 0.0
    cold = gc.judge_fillet(_rows(root_fused=lambda x: False))
    assert not cold["checks"]["fusion_sections"]["ok"] and cold["score"] == 0.0
    short = gc.judge_fillet(_rows(size_m=lambda x: 5e-3 if x > 0.146 else 6.6e-3))
    assert short["checks"]["leg_size"]["ok"] and not short["checks"]["crater_filled"]["ok"]


def test_progress_is_net_of_the_tacked_start_and_half_fillet_judges_its_half():
    tacks = _rows(metal=(8e-3, 23e-3))
    assert gc.judge_fillet(tacks, tacks)["progress"] == 0.0                  # standing still earns nothing
    assert gc.fillet_progress(tacks)["raw"] == pytest.approx(15 / 140, abs=0.01)
    half = gc.judge_half_fillet(_rows(metal=(5e-4, 0.0765)))
    assert half["success"], {k: c for k, c in half["checks"].items() if not c["ok"]}
    spill = gc.judge_half_fillet(_rows(metal=(5e-4, 0.110)))
    assert not spill["checks"]["left_half_unwelded"]["ok"]


@pytest.mark.timeout(900)
def test_a_short_fillet_on_the_cpu_lays_a_bead_at_the_root():
    """30 mm of fillet on a cut-down coupon at 1 mm cells (GA uses 0.25-0.5 mm): the wire is all deposited, nothing
    missed, the ledger closes, every crossed section holds a fillet at the root (size >= 3.95 mm, legs on both
    members) and has frozen in the pass's order. Fusion at 1 mm cells is not judged here, and the pool's flow is
    off: at 1 mm cells its dent leaves one section at 2 mm (GridThermal warns; GA's 0.25 mm cells resolve it)."""
    length = 0.03
    cfg = dict(x=graded(-0.005, length + 0.005, 1e-3, band=(0.0, length)),
               y=graded(-0.03, 0.03, 1e-3, band=(-0.01, 0.016)), z=graded(-0.01, 0.04, 1e-3, band=(-0.01, 0.016)),
               boxes=((-0.005, -0.03, -0.01, length + 0.005, 0.03, 0.0), (0.0, -0.008, 0.0, length, 0.0, 0.04)),
               arc_pressure=(0.0, 0.0))    # no pool flow: its dent (σ 1.79 mm) is not resolved by 1 mm cells
    grid = wg.run(wg.Pass(start=0.001, end=length - 0.003, cool=2.0), device="cpu", config=cfg, log=None)
    r = {k: float(v[0]) for k, v in grid.read().items() if np.ndim(v) == 1}
    assert r["missed_m3"] == 0.0 and r["deposited_m3"] >= 0.9 * r["fed_m3"]
    assert abs(r["residual_J"]) <= 1e-6 * (r["arc_J"] + abs(r["filler_J"]))
    rows = measure(grid, span=(0.004, length - 0.006))
    assert all(s["size_m"] >= 3.95e-3 and s["leg_horizontal_m"] > 0 and s["leg_vertical_m"] > 0 for s in rows)
    t = [s["freeze_t"] for s in rows]
    assert None not in t and not any(s["liquid"] for s in rows) and t[-1] > t[0] + 2.0, t


@pytest.mark.gpu
@pytest.mark.timeout(5400)
def test_fillet_2f_on_cuda():
    """GA's fillet-2f (the fixture's tacks, then aim_tacks) on GA's grid: the books close and GA's judgement is read
    (printed; docs/notes/heat-v3-design.md has the measured one)."""
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from alloy.tasks.weld_ga_tasks import run_task  # noqa: PLC0415

    out = run_task("fillet-2f", device="cuda:0", log=print)
    r = out["reads"]
    assert r["missed_m3"] == 0.0 and abs(r["residual_J"]) <= 1e-6 * (r["arc_J"] + abs(r["filler_J"]))
    print("[fillet-2f]", out["success"], round(out["score"], 3), {k: c["ok"] for k, c in out["checks"].items()})
