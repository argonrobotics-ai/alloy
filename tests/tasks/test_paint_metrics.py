# SPDX-License-Identifier: Apache-2.0
"""Painting task layer on CPU: paint optics (Kubelka-Munk hide), the ported success checks on synthetic fields, the
paint cards and their bridge to the surface-flow film material."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import paint_metrics as pm
from alloy.tasks.paint_materials import PAINTS, dft, film_material, optics, tint_factor
from alloy.tasks.paint_optics import Optics, delta_e, km_reflectance, srgb8_to_linear

NAVAL, PRIMER, BEIGE = srgb8_to_linear((47, 61, 76)), srgb8_to_linear((110, 112, 114)), srgb8_to_linear((209, 199, 184))


def test_kubelka_munk_limits_and_monotone_hide() -> None:
    o = optics("latex_eggshell", "naval")
    assert np.allclose(o.reflectance(np.array(0.0), PRIMER), PRIMER)                  # no film: the substrate
    assert np.allclose(o.reflectance(np.array(1e4), PRIMER), NAVAL, atol=1e-6)         # opaque: the paint's colour
    de = o.delta_e(np.linspace(0.0, 200.0, 41), np.broadcast_to(PRIMER, (41, 3)))
    assert np.all(np.diff(de) <= 1e-9) and de[0] == pytest.approx(float(delta_e(PRIMER, NAVAL)))
    # a white base over beige hides at ~31 um DFT (Gym-Anything's white base ~33 um); naval over grey primer ~62 um,
    # so one 4-mil coat (43 um dry) does not hide it and a second does
    white = optics("latex_eggshell", "alabaster")
    assert white.hide_dft_um(BEIGE) == pytest.approx(30.8, abs=1.0)
    assert 43.0 < o.hide_dft_um(PRIMER) < 86.0


def test_tint_bases_by_luminance() -> None:
    assert tint_factor("alabaster") == 1.0 and tint_factor("naval") == pytest.approx(0.020)
    assert km_reflectance(np.zeros(3), 0.1, np.array(5.0), PRIMER).shape == (3,)


def test_hide_and_uniform_on_a_synthetic_strip() -> None:
    o = Optics(NAVAL, 0.187 * 0.020)
    dry = np.full((40, 60), 90e-6)                    # 90 um DFT everywhere: hidden
    mask = np.zeros_like(dry, dtype=bool)
    mask[5:35, 5:55] = True
    assert pm.hide(dry, mask, o, PRIMER).ok
    thin = dry.copy()
    thin[5:35, 5:20] = 20e-6                          # a third of the strip thin: not hidden, and not uniform
    assert not pm.hide(thin, mask, o, PRIMER).ok
    assert pm.uniform(dry, mask, o, PRIMER, cell=0.004).ok and not pm.uniform(thin, mask, o, PRIMER, cell=0.004).ok


@pytest.mark.parametrize("edge", ["first", "last"])
@pytest.mark.parametrize("rows", [48, 150])
def test_uniform_sees_a_thin_band_along_either_edge_of_the_strip(edge: str, rows: int) -> None:
    o = Optics(NAVAL, 0.187 * 0.020)
    mask = np.zeros((50, rows + 22), dtype=bool)
    mask[7:43, 5:5 + rows] = True                     # 12-cell patches (50 mm at 4 mm), off the grid origin; 150
    dry = np.full(mask.shape, 90e-6)                  # rows leave 6 past the last whole patch
    band = slice(5, 11) if edge == "first" else slice(rows - 1, rows + 5)
    dry[7:43, band] = 30e-6                           # a 24 mm band under the ceiling or over the baseboard
    assert pm.uniform(np.full(mask.shape, 90e-6), mask, o, PRIMER, cell=0.004).ok
    assert not pm.uniform(dry, mask, o, PRIMER, cell=0.004).ok


def test_runs_laps_protected_and_tray() -> None:
    assert pm.no_runs(100e-6, 0.2e-3).ok and not pm.no_runs(100e-6, 2e-3).ok      # settled: by the travel
    assert not pm.no_runs(300e-6, 0.0, settled=False).ok and pm.no_runs(300e-6, 0.5e-3).ok   # a thick coat that held
    lap = np.zeros((10, 10), np.int32)
    lap[0, :3] = 1
    assert pm.laps(lap, np.ones_like(lap, bool), area=4e-6 * 4e-6 * 1e6).ok        # 3 cells of 16 mm²
    prot = np.zeros((10, 10), bool)
    prot[:, 0] = True
    painted = np.zeros((10, 10))
    painted[:, 0] = 5e-6
    assert not pm.protected(painted, prot, area=16e-6).ok and pm.protected(painted * 0.1, prot, area=16e-6).ok
    # paint lost to a sink counts as its volume spread 100 um thick: 6 mL is 0.6 cm²
    assert pm.protected(painted * 0.1, prot, area=16e-6, lost_volume=4e-9).ok
    assert not pm.protected(painted * 0.1, prot, area=16e-6, lost_volume=6e-9).ok
    marred = np.zeros((10, 10), bool)
    marred[2, :3] = True                                       # 3 cells of 16 mm² touched by a link
    assert pm.untouched(marred, np.ones_like(marred), area=16e-6).ok
    marred[3:6, :] = True
    assert not pm.untouched(marred, np.ones_like(marred), area=16e-6).ok
    assert pm.in_tray(np.array([0.0, 0.0, 0.05]), (0.18, 0.14), released=True).ok
    assert not pm.in_tray(np.array([0.0, 0.0, 0.05]), (0.18, 0.14), released=False).ok


def test_a_verdict_counts_only_what_it_judges() -> None:
    """AB-gym-6: a criterion the port cannot fail is None, never True; one judged in part fails on that part and is
    None when it passes; success counts the judged checks, needs one of Gym-Anything's among them, and still fails on
    the port's own books; the verdict says k of n, why, and how the scene differs."""
    m, rest, high = np.ones((2, 2), bool), np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.5])
    none = pm.untouched(None, m, 1e-6)
    assert none.ok is None and none.note == pm.NO_MARKS and none.row()["ok"] is None
    assert pm.untouched(np.zeros_like(m), m, 1e-6).ok is True                    # tracked marks are judged
    held = pm.in_tray(rest, (0.18, 0.14), released=None)
    assert held.ok is None and held.note == pm.NO_RELEASE
    assert pm.in_tray(high, (0.18, 0.14), released=None).ok is False             # the judged part still fails
    hide, books = pm.Check("hide", 1.0, 0.98, True), pm.Check("ledger", 0.0, 1e-6, True)
    v = pm.verdict([hide, none, held, books], scene="smaller")
    assert v["succeeded"] and v["judged"] == "1 of 3" and list(v["not_judged"]) == ["untouched", "roller_in_tray"]
    assert v["summary"] == ("1 of Gym-Anything's 3 criteria judged; not judged: untouched, roller_in_tray. "
                            "Scene: smaller")
    assert v["checks"]["untouched"] == {"value": 0.0, "limit": pm.PROTECTED_AREA, "ok": None, "note": pm.NO_MARKS}
    assert not pm.verdict([hide, none, pm.Check("ledger", 1.0, 1e-6, False)])["succeeded"]     # the port's own check
    assert not pm.verdict([none, books])["succeeded"]                           # nothing of Gym-Anything's judged
    assert not pm.verdict([hide, pm.in_tray(high, (0.18, 0.14), released=None)])["succeeded"]
    other = hide.differs("other patches")
    assert other.ok is True and other.row()["note"] == "other patches" and "note" not in hide.row()
    assert pm.verdict([other])["judged"] == "1 of 1" and pm.verdict([other])["scene"] == ""
    part = pm.Check("protected", 0.0, 1.0, True).in_part("spill only")
    assert part.ok is None and pm.Check("protected", 2.0, 1.0, False).in_part("spill only").ok is False


def test_a_stroke_with_one_paint_judges_only_its_thickness() -> None:
    """AB-gym-6: with one paint the share is 1 by construction, so a covered core is None (thickness judged, share
    not) and an uncovered one still fails; with the share given the criterion is judged whole."""
    core, outside = np.ones((4, 4), bool), np.zeros((4, 4), bool)
    thick, thin = np.full((4, 4), 20e-6), np.full((4, 4), 1e-6)
    one = pm.stroke(thick, core, outside, 1e-6)[0]
    assert one.ok is None and one.value == 1.0 and one.note == pm.ONE_PAINT
    assert pm.stroke(thin, core, outside, 1e-6)[0].ok is False
    assert pm.stroke(thick, core, outside, 1e-6, share=np.ones((4, 4)))[0].ok is True
    assert pm.stroke(thick, core, outside, 1e-6, share=np.full((4, 4), 0.9))[0].ok is False


def test_settle_travel_is_the_furthest_cell_over_the_mask() -> None:
    start = np.zeros((4, 6))
    end = start.copy()
    end[1, 2], end[3, 5] = 0.4e-3, 2e-3                       # the far run is outside the mask
    mask = np.ones_like(start, bool)
    mask[3, 5] = False
    assert pm.settle_travel(start + 1e-3, end + 1e-3, mask) == pytest.approx(0.4e-3)
    assert pm.settle_travel(start, end, np.zeros_like(mask)) == 0.0


def test_roller_load_and_drip() -> None:
    cap = np.full(32, 3e-6)
    load = np.full(32, 1.5e-6)                                 # 48 mL in 32 bins: 62 g of latex
    checks = pm.roller_load(load, cap, density=1295.0)
    assert all(c.ok for c in checks)
    load[:10] = 0.0
    assert not pm.roller_load(load, cap, density=1295.0)[1].ok    # spread 22/32 < 0.8
    # capacity as a pore volume: a bin is loaded above 0.2 x the saturated share (0.154) of it, 92 nL here
    thin = np.full(32, 0.1e-6)
    assert pm.roller_load(thin, cap, density=1295.0, saturation=0.154)[1].ok
    assert not pm.roller_load(thin, cap, density=1295.0)[1].ok
    t = np.arange(0.0, 12.01, 0.5)
    assert pm.no_drip(np.zeros_like(t), t).ok
    d = np.zeros_like(t)
    d[-3:] = 1e-9
    assert not pm.no_drip(d, t).ok and not pm.no_drip(np.zeros(5), t[:5]).ok        # a drop; too short a hold
    assert pm.no_drip(d, t, drop=1e-9).ok and not pm.no_drip(d, t, drop=0.99e-9).ok   # less than one drop's volume
    t2 = np.arange(0.0, 26.0, 0.05) + 0.03                       # samples off the 10 s grid: the window opens on the
    assert pm.no_drip(np.zeros_like(t2), t2).value >= 10.0           # sample before it, so it spans the whole 10 s
    assert pm.held_above_tray(np.array([0.0, 0.0, 0.3]), (0.18, 0.14)).ok


def test_brush_template_and_stroke() -> None:
    cell = 0.5e-3
    x, y = np.meshgrid(np.arange(-0.12, 0.12, cell), np.arange(-0.05, 0.05, cell), indexing="ij")
    core, outside = pm.template_masks(np.stack([x, y], -1), n=400)
    assert core.any() and outside.any() and not (core & outside).any()
    perfect = np.where(core, 20e-6, 0.0)
    cov, line = pm.stroke(perfect, core, outside, area=cell * cell)
    assert cov.ok is None and cov.value == 1.0 and line.ok           # one paint: the thickness alone is not a pass
    blob = np.where(~outside | (np.abs(y) < 0.02), 20e-6, 0.0)       # paint well past the outline
    assert not pm.stroke(blob, core, outside, area=cell * cell)[1].ok
    share = np.where(x < 0.0, 1.0, 0.5)                              # the right half is mixed with another colour
    assert pm.stroke(perfect, core, outside, area=cell * cell, share=np.ones_like(x))[0].ok
    assert not pm.stroke(perfect, core, outside, area=cell * cell, share=share)[0].ok
    assert pm.stroke(perfect, core, outside, area=cell * cell, dft_min=15e-6)[0].value == 1.0   # a swatch's hide film
    assert pm.stroke(perfect, core, outside, area=cell * cell, dft_min=25e-6)[0].ok is False


def test_paint_cards_bridge_to_film_materials() -> None:
    latex = film_material("latex_eggshell")
    assert latex.density == 1295.0 and latex.viscosity == 7.25 and latex.yield_stress == 0.25
    assert film_material("acrylic_fluid").viscosity == pytest.approx(0.7)
    assert dft(np.array(PAINTS["latex_eggshell"].wet_film), "latex_eggshell") == pytest.approx(42.7e-6, rel=0.01)


def test_verdict_and_ledger() -> None:
    v = pm.verdict([pm.Check("a", 1.0, 1.0, True), pm.Check("b", 0.0, 1.0, False)])
    assert v["succeeded"] is False and v["checks"]["a"]["ok"] is True
    assert pm.ledger({"ledger": np.array([2.4e-7])}, set_volume=2.4e-7).ok
    assert not pm.ledger({"ledger": np.array([2.5e-7])}, set_volume=2.4e-7).ok
    # relative to the paint put in: 0.1 pL of drift on 1 mL supplied passes, 10 nL does not
    read = {"ledger": np.array([0.0, 1e-16]), "supplied_volume": np.array([0.0, 1e-6])}
    assert pm.ledger(read, world=1).ok
    read["ledger"][1] = 1e-11
    assert not pm.ledger(read, world=1).ok
    # paint a tool's store laid (to_tools < 0) counts as put in: 1e-21 m³ of drift on 70 nL laid passes
    laid = {"ledger": np.array([1.2e-21]), "to_tools_volume": np.array([-7e-8])}
    assert pm.ledger(laid).ok and not pm.ledger({"ledger": np.array([1.2e-21])}).ok


def test_chart_field_slices_chart_zero() -> None:
    read = {"dft": np.arange(2 * 10, dtype=float).reshape(2, 10)}       # 2 worlds, a 2 x 3 chart then 4 cells more
    assert np.array_equal(pm.chart_field(read, "dft", (2, 3), world=1), [[10, 11, 12], [13, 14, 15]])
    assert pm.chart_field(read, "lap", (2, 3)) is None
