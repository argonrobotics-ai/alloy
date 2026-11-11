# SPDX-License-Identifier: Apache-2.0
"""Yield-stress pastes on the MPM paste route: the friction mapping, the sourced yield stresses and the short laws of
docs/notes/viscoplastic-design.md §5 on CPU (L0 spread, L3 slope). L1 slump and L2 pancake need finer voxels: they
run from alloy.solvers.mpm_rheology.laws on a GPU (results in the note's step 1 table; L1 below as a gpu test)."""

from __future__ import annotations

import math

import pytest

from alloy.cards.library import CardLibrary
from alloy.scene.emitters.deformable import mpm_values
from alloy.solvers.mpm_rheology import laws as L
from alloy.sourcing.tags import tag_of

LIB = CardLibrary(assets=None)


def test_paste_friction_is_zero_and_grains_keep_their_angle() -> None:
    for paste in ("honey", "sauce_ketchup", "dough_wheat"):
        assert mpm_values(LIB.material(paste))["friction"] == 0.0, paste      # not Newton's default 0.5
    sand = LIB.material("sand_silica_dry")
    assert mpm_values(sand)["friction"] == pytest.approx(math.tan(sand.plastic.friction_angle.value))


@pytest.mark.parametrize("name", ["sauce_ketchup", "mayonnaise", "dough_wheat"])
def test_pastes_that_hold_shape_carry_a_sourced_yield_stress(name: str) -> None:
    q = LIB.material(name).plastic.yield_stress
    assert q is not None and q.value > 0.0 and tag_of(q) == "DOC", name


def test_honey_stays_newtonian() -> None:
    m = LIB.material("honey")
    assert m.plastic is None or m.plastic.yield_stress is None
    assert mpm_values(m).get("yield_stress", 0.0) == 0.0


def test_formulas() -> None:
    assert L.pashias_slump(0.25) == pytest.approx(1.0 - 0.5 * (1.0 - math.log(0.5)))
    assert L.pancake_radius(150e-6, 1092.0, 20.0) == pytest.approx(0.0735, abs=2e-4)
    assert L.huppert_radius(150e-6, 6.0 / 1420.0, 1.0) == pytest.approx(0.0756, abs=2e-4)


@pytest.mark.compile_heavy
def test_l0_newtonian_spread_follows_huppert_on_cpu(tmp_path) -> None:
    row = L.spread(tmp_path, L.Paste(1420.0, 0.0, 6.0), size=(0.04, 0.04, 0.02), seconds=0.5, device="cpu")
    assert row["check_ok"]
    assert all(0.88 <= r["ratio"] <= 1.02 for r in row["rows"]), row


@pytest.mark.compile_heavy
@pytest.mark.parametrize("ratio, flows", [(1.6, True), (0.8, False)])
def test_l3_layer_flows_above_the_stop_thickness_and_stops_below_on_cpu(tmp_path, ratio: float, flows: bool) -> None:
    """Tilted gravity (30 deg) on a layer over the rough ground; h_stop = tau_y / (rho g sin theta) = 18.5 mm. At this
    5 mm voxel the threshold reads 1.2-1.4 h_stop (on an L4: 1.0-1.2 at 2.5 mm, below 1.1 at 1.25 mm), so the test
    keeps clear of it."""
    paste = L.Paste(1100.0, 100.0, 5.0)
    h_stop = paste.tau_y / (paste.rho * L.G * math.sin(math.radians(30.0)))
    row = L.slope(tmp_path, paste, thickness=ratio * h_stop, length=0.12, seconds=0.6, device="cpu")
    assert row["check_ok"]
    if flows:
        assert row["travel_mm"] > 5.0 and row["last_quarter_mm"] > 0.5, row
    else:
        assert row["travel_mm"] < 1.0 and abs(row["last_quarter_mm"]) < 0.2, row


@pytest.mark.gpu
def test_l1_cylinder_slump_near_pashias_on_gpu(tmp_path) -> None:
    """tau' = 0.2 at a 2.5 mm voxel (16 per height): s' 0.292 on an L4 vs Pashias 0.235; the 1-D formula reads
    0.02-0.06 low across tau' 0.1-0.3 at any voxel (note §5), hence a 0.07 band."""
    row = L.slump(tmp_path, L.Paste(1100.0, 86.0, 10.0), voxel=0.0025, device="cuda:0")
    assert row["check_ok"] and abs(row["slump"] - row["pashias"]) <= 0.07 and abs(row["still_mm"]) < 0.2, row
