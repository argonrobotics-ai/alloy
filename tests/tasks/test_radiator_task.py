# SPDX-License-Identifier: Apache-2.0
"""behind-radiator on the surface-flow film (CPU, slow): three lanes from the radiator's left end (a dip before each)
hide what they cover with no runs and nothing on the radiator or the trim once every wet cell has set, every volume
booked exactly; the same paint left on the wall too thick runs. The whole job is a row."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from alloy.tasks import radiator_metrics as rm
from alloy.tasks import radiator_scene as rs
from alloy.tasks import radiator_script as scr
from alloy.tasks import radiator_task as rt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_radiator_brush as tb  # noqa: E402  (its small wall)

pytestmark = [pytest.mark.slow]


def lanes_episode(settle: float = rt.SETTLE_MAX) -> tuple[dict, dict]:
    start = (np.asarray(scr.pail_tip(scr.ABOVE)), -90.0)
    rows = scr.rows()[:3]
    keep: dict = {}
    out = rt.run(moves=(start, scr.side(-1, rows)), settle=settle, keep=keep)
    x, z = keep["xz"][..., 0], keep["xz"][..., 1]
    keep["lanes"] = (x >= rs.BAND[0]) & (x <= -scr.X_IN - 0.007) & (z >= rs.BAND[2] + rs.CELL) & (z <= rows[-1])
    return out, keep


def _dry(keep: dict) -> np.ndarray:
    return np.asarray(keep["read"]["dft"])[0, : keep["lanes"].size].reshape(keep["lanes"].shape)


@pytest.mark.timeout(3600)                 # ~25 min on a loaded CPU host (1,446 s, 11-07)
def test_three_lanes_hide_what_they_cover_without_runs_or_damage():
    out, keep = lanes_episode()
    dry = _dry(keep)[keep["lanes"]]
    hidden = rt.optics(rt.PAINT, rt.COLOUR).hidden(1e6 * (rt.first_coat(keep["xz"])[keep["lanes"]] + dry),
                                                    rt.srgb8_to_linear(rt.COLOURS[rt.SUBSTRATE]))
    assert hidden.mean() >= rm.HIDE_FRAC
    c = out["checks"]
    assert out["runs"]["settled"] and out["settle_s"] < rt.SETTLE_MAX          # the coat set: runs read from it
    assert c["no_runs"]["ok"] and out["runs"]["flowed_in_max_um"] < 0.2 * rm.RUN_FLOW * 1e6
    assert c["protected"]["ok"] and c["no_marks"]["value"] == 0.0   # (film creeping over the bottom edge: baseboard)
    assert abs(out["books"]["store_ledger"]) <= 1e-12 * out["books"]["picked_mL"] * 1e-6
    assert out["books"]["tray_ledger"] == pytest.approx(rt.rb.PAIL_PAINT, rel=1e-12, abs=0)
    d = out["drips_mL"]               # the drain over the pail went back into it; the last few fell on the way
    assert d["dripped"] > 0.0 and d["staged"] == pytest.approx(d["dripped"], rel=1e-12, abs=0)
    assert d["sent"] == pytest.approx(d["staged"], rel=1e-12, abs=0) and d["to_tank"] > 0.99 * d["dripped"]
    assert d["on_charts"] + d["caught"] + d["lost"] < 0.01 * d["dripped"]
    assert not c["hide"]["ok"]                                 # three lanes are not the job
    assert c["ledger"]["ok"] and c["solids_ledger"]["ok"]


@pytest.mark.timeout(600)
def test_fresh_paint_left_too_thick_runs():
    """Fresh paint (its structure broken, as a brush leaves it) in three 40 x 40 mm patches on a small wall: in 80 s
    400 µm and 300 µm sag, their flow carrying more than 50 µm of dry paint into over 1 cm² below them (a run by
    the task's check); 200 µm (over the 157 µm the rebuilt yield holds) creeps under 10 µm and stands. The film's
    solids are those set, exactly. (The brush's metered film is at most 150 µm: runs behind a radiator come from
    drips and blobs.)"""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    boxes, charts = tb.small_scene()
    flow, box = rt.flow_spec(charts=charts), {}
    shape, phi0 = charts[0].cells, rt.paint(rt.PAINT).volume_solids

    def init(sim) -> None:  # noqa: ANN001
        xz = np.asarray(flow.cell_centres(0))[..., :2]
        x, top = xz[..., 0], xz[..., 1] > 0.0                           # the wall's upper half (centred chart)
        h = np.select([top & (np.abs(x - c) < 0.02) for c in (-0.06, 0.0, 0.06)], [400e-6, 300e-6, 200e-6], 0.0)
        flow.set_depth(h[None], chart=0)
        flow.set_species((h * phi0)[None], "solids", chart=0)
        lam = flow.fluid.lam.numpy()
        lam[0, : h.size] = np.where(h.ravel() > 0.0, 0.0, 1.0)
        flow.fluid.lam.assign(lam)
        box.update(h0=h, x=x)

    sim = build_simulation(rt.scene(init, np.array([1.0, -1.0, 1.0]), 0.0, boxes=boxes), [flow], "cpu")
    for _ in range(int(round(80.0 / scr.SUB))):
        sim.step_period()
    n0 = int(np.prod(shape))
    dft = np.asarray(flow.read()["dft"])[0, :n0].reshape(shape)
    flowed = rm.flowed_in(dft, box["h0"] * phi0, 0.0, 0.0, rs.CELL * rs.CELL)
    for lo, hi, runs in ((-0.09, -0.03, True), (-0.03, 0.03, True), (0.03, 0.09, False)):
        sel = (box["x"] > lo) & (box["x"] < hi)
        area = float(((flowed > rm.RUN_FLOW) & sel).sum()) * rs.CELL**2
        assert (area > rm.RUN_AREA) == runs and (runs or float(flowed[sel].max()) < 10e-6)
    assert float(dft.sum()) == pytest.approx(float(box["h0"].sum()) * phi0, rel=1e-12)
