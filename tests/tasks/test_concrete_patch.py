# SPDX-License-Identifier: Apache-2.0
"""task concrete_patch (Gym-Anything's patch-flush): the ported start state (CPU, fast) and the whole task, the
scripted strike-off into the bucket and floating against their seven criteria (CPU, slow), its strips (fast)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_patch as CP


def test_the_start_state_is_theirs():
    """layout.py:383-410: chipped 25-38 mm, overfilled 3-7 mm, the west void 6-9 mm below the slab and 18 mm wide,
    2 mm smears beside the north and east sides, nothing else on the slab."""
    f = CP.start_fields(seed=0)
    rec = f["recess"] > 0.5
    assert rec.sum() == 400 * 250                                              # 1 mm cells over the recess
    depth = -f["floor"][rec]
    assert 0.025 - 1e-9 <= depth.min() and depth.max() <= 0.038 + 1e-9
    nu = rec.shape[0]
    x = (np.arange(nu) + 0.5) * 0.001 - (CP.L / 2 + CP.MARGIN)
    void = rec & (x[:, None] < -CP.L / 2 + 0.018)
    top = f["top"]
    assert np.all(top[void] <= -0.006 + 1e-9) and np.all(top[void] >= -0.009 - 1e-9)
    over = top[rec & ~void]
    assert over.min() >= 0.003 - 1e-9 and over.max() <= 0.007 + 1e-9
    smear = f["smear"][~rec]
    assert 0.0004 - 1e-9 <= smear[smear > 0].min() and smear.max() <= 0.002 + 1e-9
    assert (f["smear_mask"] > 0.5).sum() == np.count_nonzero(smear)
    st = CP.start_state(CP.PatchFlush(cell=0.005, seed=0))                    # block-averaged to 5 mm
    assert st["recess"].sum() == 80 * 50
    assert np.all(st["base"] <= 0.0) and np.all(st["film"] >= 0.0)
    assert np.allclose((st["base"] + st["film"])[st["recess"]].mean(),
                       CP._block(f["top"], CP.PatchFlush(cell=0.005).cells)[st["recess"]].mean())
    other = CP.start_state(CP.PatchFlush(cell=0.005, seed=1))
    assert not np.allclose(other["film"], st["film"])                           # seeds vary the patch


def test_the_verdict_books_the_paste_that_left_the_slab_as_off_region():
    """grids() counts what left the slab region other than into the bucket as their off-region catch (off_m3):
    drained, spilled over the chart's edges, fallen on the floor (the supply's lost: a cone's share past the chart's
    edge, DC5), caught on a shape, or a full blade's overflow past the chart's edge (WP13's end bands)."""
    from types import SimpleNamespace as NS

    task = CP.PatchFlush(cell=0.01)
    C = task.cells[0] * task.cells[1]
    st = CP.start_state(task, 0)
    rd = {"drained_total_volume": np.array([4.6e-4, 0.0]), "spilled_volume": np.array([1e-6, 3e-8]),
          "supply_lost": np.array([2e-7, 0.0]), "supply_caught": np.array([0.0, 5e-9]),
          "tool_overflow_off": np.array([[0.0, 0.0], [1e-9, 2e-9]])}
    flow = NS(h=NS(numpy=lambda: np.zeros((2, C))), base=NS(numpy=lambda: np.zeros((2, C), np.float32)),
              read=lambda: rd)
    layer = NS(read=lambda: {k: np.zeros((2, C)) for k in ("a", "D", "tau_w", "worked")})
    g = CP.grids(task, flow, layer, [st, st])
    assert [x.off_m3 for x in g] == pytest.approx([4.612e-4, 3.8e-8], rel=1e-12, abs=0)


def _stick() -> float:
    """m³ the lifted blade keeps of 6 min repair mortar: 0.5·A·tau/(rho·g·6) (coupling.py:922-938)."""
    mat = CM.RepairMortar()
    tau = float(mat.tau_bulk(CP.START_AGE * CM.masco_factor(21.1)))
    return 0.5 * 4 * CP.TROWEL["half"][0] * CP.TROWEL["half"][1] * tau / (mat.rho * 9.81 * 6.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_strips_stay_under_the_stick_limit_and_each_row_fills_its_void_last(seed):
    """Per row, east to west, contiguous: every strip but the last takes at most STRIP_LOAD (or is one blade width
    wide), all under the lifted blade's stick limit; the last cuts the west
    void's deficit 1.5 times over before reaching it, and lifts what is left under the blade's stick limit; then the
    north smear. The trowel lets go over the bucket after every strip."""
    task = CP.PatchFlush(cell=0.01)
    st = CP.start_state(task, seed)
    out = CP.strips(task, st)
    half = CP.TROWEL["half"][1]
    stick = _stick()
    assert CP.STRIP_LOAD < stick
    for y in CP.ROWS:
        row = [(xa, xb) for yy, xa, xb in out if yy == y]
        x, vol, low = CP._row(task, st, y)
        assert row[0][0] + half == pytest.approx(CP.L / 2 + 0.035) and row[-1][1] == pytest.approx(CP.WEST_END + half)
        assert all(abs((a[1] - half) - (b[0] + half)) < 1e-9 for a, b in zip(row, row[1:]))     # contiguous
        taken = [vol[(x <= xa + half) & (x >= xb - half)].sum() for xa, xb in row]
        width = [xa - xb + 2 * half for xa, xb in row]
        assert min(width) >= 2 * half - 1e-9                                                     # a blade width or more
        assert all(v <= CP.STRIP_LOAD + 1e-12 or w_ < 2 * half + task.cell                       # over STRIP_LOAD only
                   for v, w_ in zip(taken[:-1], width[:-1], strict=True)), (taken, width)        # at the least width
        assert max(taken[:-1]) < stick
        assert taken[-1] >= 1.5 * low.sum() > 0.0                                             # 1.5 times over
        assert taken[-1] - low.sum() < stick
    assert [yy for yy, _, _ in out][-1] == CP.W / 2 + 0.0125
    keys = CP.trowel_program(task, st).keys
    assert sum(k.release > 0.0 for k in keys) == len(out)


@pytest.mark.slow
@pytest.mark.timeout(3000)
def test_the_scripted_strike_off_into_the_bucket_meets_their_seven_criteria():
    """The whole task at 10 mm cells on the CPU, seeds 0 and 1 (world 1 is the start whose west void a short last
    strip once left 1.2 mm low): the carried excess goes into the bucket (Alloy's sink; their task has none), nothing
    lands off-region, and the patch ends flush and floated: their seven criteria pass. The ledgers close and the tools
    stay under the paste's bearing capacity."""
    res = CP.run(CP.PatchFlush(cell=0.010), worlds=2)
    for v in res.verdicts:
        assert v["success"], {k: c for k, c in v["criteria"].items() if c["passed"] is False}
        assert v["judged"] == [7, 7] and v["grid"].startswith("10 mm cells, coarser")     # not their 1 mm grid
    assert res.off_region_l == [0.0, 0.0] and res.spilled_l == [0.0, 0.0]
    assert min(res.bucket_l) > 0.4 and res.bucket_l == pytest.approx(res.released_l, rel=1e-9)
    assert res.ledger_rel < 1e-9 and res.store_ledger_m3 < 1e-12
    assert res.max_pressure_pa < res.bearing_pa and res.cells_over_depth == 0
