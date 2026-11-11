# SPDX-License-Identifier: Apache-2.0
"""cut-in-trim (CPU).

Fast: the crops and bands match Gym-Anything's geometry on cell faces; the sash's store lays its 100 µm first coat
and the script reloads before the coat thins under what hides; the script lands and lifts along the wall's normal and
keeps the brush's head clear of the trim and the plate; the verdict on synthetic film fields passes and fails each
check exactly at its limit; a film-only run shows the trim's seal holds (and that without it paint reaches the trim).

Slow (not in the gate's fast lane): with a 0.4 m casing band the honest cut-in of both bands passes; on the casing
alone, a brush 3 mm over the casing's line fails only ``protected`` and one 8 mm off it fails only ``hide``. A lap
needs the first stroke's top past its wet edge (about 280 s at latex's drying rate): too long for a CPU test, so laps
are tested on synthetic fields here and by one ``--pause 400`` run in [internal note]."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import trim_sash as ts
from alloy.tasks import trim_verdict as tv
from alloy.tasks.trim_cut_in import Plan, casing_strokes, loads, make_flow, outlet_strokes, run, script, seal
from alloy.tasks.trim_layout import BAND, PAIL, Rect, TrimLayout

CELL_AREA = 0.002 * 0.002
SHORT = TrimLayout(casing_top=1.213)


def test_bands_and_trim_fall_on_cell_faces() -> None:
    """The casing band is 40 mm x its height, the ring the plate grown 40 mm less the plate, and the trim's footprint
    starts at the band's edge: no cell straddles a line. The first coat is Gym-Anything's ±15 % about 43 µm."""
    casing, outlet = (SHORT.masks(c, c.world_xz(tv.crop_centres(c))) for c in SHORT.crops())
    assert casing["target"].sum() * CELL_AREA == pytest.approx(BAND * (1.213 - 0.813), rel=1e-9)
    p = SHORT.plate()
    ring = (p.x1 - p.x0 + 2 * BAND) * (p.z1 - p.z0 + 2 * BAND) - (p.x1 - p.x0) * (p.z1 - p.z0)
    assert outlet["target"].sum() * CELL_AREA == pytest.approx(ring, rel=1e-9)
    assert outlet["protected"].sum() * CELL_AREA == pytest.approx((p.x1 - p.x0) * (p.z1 - p.z0), rel=1e-9)
    c = SHORT.crops()[0]
    xz = c.world_xz(tv.crop_centres(c))
    i = int(np.argmin(np.abs(xz[:, 0, 0] - (SHORT.casing_x0() - 0.001))))       # the band's last column
    j = int(np.argmin(np.abs(xz[0, :, 1] - 1.0)))
    assert casing["target"][i, j] and casing["protected"][i + 1, j]
    fc = SHORT.first_coat(xz)
    assert 0.85 * 43e-6 - 1e-12 <= fc.min() and fc.max() <= 1.15 * 43e-6 + 1e-12 and fc.std() > 1e-7
    with pytest.raises(ValueError, match="whole number"):
        TrimLayout(cell=0.0015)
    assert [c.kind for c in TrimLayout(jobs=("outlet",)).crops()] == ["outlet"]
    with pytest.raises(ValueError, match="jobs"):
        TrimLayout(jobs=("casing", "casing"))


def test_metering_store_lays_the_e7_film_and_reloads_in_time() -> None:
    """The metering store: η is Gym-Anything's latex Cross law at U/50 µm (0.18 Pa·s at 0.1 m/s); the pad's stiffness is
    the tufts' density; the press meters the E7 film (100 µm) and the law gives it back; that film hides over the
    thinnest first coat (61 µm wet); a load runs 0.8 of its free share: the full casing band takes two loads, the ring
    one."""
    rate = 0.1 / 50e-6                                        # GA's latex Cross law (params.py:36) at U / 50 um
    assert ts.meter_viscosity() == pytest.approx(0.1796 + (2.258 - 0.1796) / (1.0 + (rate / 0.03089) ** 0.676),
                                                 rel=1e-12)
    assert ts.TUFT_DENSITY == pytest.approx(100.0 * 144 / (0.0635 * 0.0159), rel=1e-12)
    assert ts.metered_film(ts.PRESS) == pytest.approx(ts.FIRST_LAY, rel=1e-12)
    assert ts.metered_film(0.5 * ts.PRESS) == pytest.approx(math.sqrt(2.0) * ts.FIRST_LAY, rel=1e-12)
    assert 80e-6 <= ts.FIRST_LAY <= 115e-6 and 2.0 < ts.TUFT_DENSITY * ts.PRESS * ts.FOOTPRINT < 5.0   # N
    need = tv.wet_to_hide(TrimLayout())
    assert need == pytest.approx(61.1e-6, abs=0.2e-6) and ts.FIRST_LAY > need
    run_ = ts.run_length()
    assert run_ == pytest.approx(0.8 * (ts.HELD - ts.RETAINED) * ts.CAPACITY / (ts.WIDTH * ts.FIRST_LAY), rel=1e-12)
    full = TrimLayout()
    for strokes, n in ((casing_strokes(full, Plan()), 2), (outlet_strokes(full, Plan()), 1)):
        groups = loads(strokes, run_)
        assert len(groups) == n
        assert all(sum(math.dist(a, b) for a, b in g) <= run_ + 1e-9 for g in groups)
    store = ts.sash_store("latex_eggshell")
    assert (store.meter, store.meter_max) == (0.075, 150e-6) and store.meter_viscosity == ts.meter_viscosity()


def _footprint(pose: ts.Pose) -> Rect:
    """The pressed head's rectangle on the wall (x, z): WIDTH across the stroke, PATCH along it."""
    across_x = np.allclose(pose.quat, ts.wall_quat(0.0))
    hx, hz = (ts.WIDTH, ts.PATCH) if across_x else (ts.PATCH, ts.WIDTH)
    x, z = float(pose.pos[0]), float(pose.pos[2])
    return Rect(x - 0.5 * hx, x + 0.5 * hx, z - 0.5 * hz, z + 0.5 * hz)


def _overlap(a: Rect, b: Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.z1, b.z1) - max(a.z0, b.z0))


def test_script_lands_lifts_along_the_normal_and_keeps_the_head_clear() -> None:
    """Every in-plane move of a brush against the wall is a stroke (landing and lifting change only y: the brush does
    not slide, so the metering store lays nothing there); every pose against the wall is pressed exactly ``PRESS``
    in (the press sets the metered film: WP15's M1); no pressed head overlaps the trim's or the plate's footprint
    (WP15: the casing stroke used to end with half the head inside the stool's horn)."""
    lay, plan = TrimLayout(), Plan()
    poses = list(script(lay, plan))
    solid = [*lay.trim(), lay.plate()]
    for tool in (0, 1):
        p = np.array([q[tool].pos for q in poses])
        near = p[:, 1] > lay.face_y - 0.002                      # against the wall
        prev = np.vstack([p[:1], p[:-1]])
        moved = np.abs(p[:, [0, 2]] - prev[:, [0, 2]]).max(axis=1) > 1e-12
        approach = np.abs(p[:, 1] - prev[:, 1]) > 1e-12
        assert near.any() and not (near & moved & approach).any()
        assert np.abs(p[near, 1] - (lay.face_y + ts.PRESS)).max() <= 1e-12
        for k in np.flatnonzero(near):
            box = _footprint(poses[k][tool])
            assert max(_overlap(box, r) for r in solid) == 0.0, (tool, k, box)
    a, _ = casing_strokes(lay, plan)[0]
    assert a[0] + 0.5 * ts.WIDTH == pytest.approx(lay.casing_x0() - plan.clearance, abs=1e-12)


def synthetic(layout: TrimLayout, new: float = 30e-6) -> dict:
    """A clean finish: ``new`` dry on every band cell, laid there (no flow), nothing on the trim, the books closed."""
    sl = tv.chart_slices(layout)
    n = sl[-1][0].stop
    dft, target = np.zeros((1, n)), np.zeros(n, bool)
    for (s, _), crop in zip(sl, layout.crops()):
        target[s] = layout.masks(crop, crop.world_xz(tv.crop_centres(crop)))["target"].ravel()
    dft[0, target] = new
    supplied = (dft * CELL_AREA)[..., None]
    return {"dft": dft, "h": dft / 0.42, "laps": np.zeros((1, n), np.int32), "species_supplied": supplied,
            "species_out": np.zeros_like(supplied), "species_ledger": np.zeros((1, 1)), "ledger": np.zeros(1),
            "store_ledger": np.zeros(1), "tray_ledger": np.full(1, PAIL["fill"]), "tool_picked": np.full((1, 2), 1e-5),
            "_target": target}


def fails(read: dict, layout: TrimLayout = SHORT, drips: float = 0.0) -> set[str]:
    v = tv.evaluate(layout, read, near_wall_drips=drips)
    return {k for k, c in v["checks"].items() if c["ok"] is False}


def _cells(mask: np.ndarray, count: int) -> np.ndarray:
    return np.flatnonzero(mask)[:count]


def test_the_verdict_says_what_it_judges() -> None:
    """AB-gym-6: ``untouched`` is None (no marks are tracked), never a pass on an empty array; runs and protected are
    judged by the port's own measure and region and say so; the verdict names the scene's differences."""
    from alloy.tasks import paint_metrics as pm  # noqa: PLC0415

    v = tv.evaluate(SHORT, synthetic(SHORT))
    assert v["succeeded"] and v["judged"] == "4 of 5" and v["not_judged"] == {"untouched": pm.NO_MARKS}
    assert v["checks"]["untouched"]["ok"] is None
    assert "smaller region" in v["checks"]["protected"]["note"] and "own measure" in v["checks"]["no_runs"]["note"]
    assert "two cropped wall charts" in v["scene"] and v["summary"].startswith("4 of Gym-Anything's 5 criteria judged")


def test_verdict_passes_and_fails_each_check_at_its_limit() -> None:
    """On synthetic fields: a clean 30 µm finish passes; each check fails one cell past its limit and passes at it
    (hide 95 %, protected 0.5 cm² of > 1 µm, near-wall drips by the 100 µm sink rule, runs 1 cm² of > 50 µm flowed,
    laps 1 cm², books 1e-12)."""
    assert fails(synthetic(SHORT)) == set()
    t = synthetic(SHORT)["_target"]
    for share, out in ((0.04, set()), (0.06, {"hide"})):
        r = synthetic(SHORT)
        r["dft"][0, _cells(t, int(share * t.sum()))] = 0.0
        r["species_supplied"][0, :, 0] = r["dft"][0] * CELL_AREA
        assert fails(r) == out
    prot = np.zeros_like(t)
    for (s, _), crop in zip(tv.chart_slices(SHORT), SHORT.crops()):
        prot[s] = SHORT.masks(crop, crop.world_xz(tv.crop_centres(crop)))["protected"].ravel()
    for count, out in ((12, set()), (13, {"protected"})):                  # 0.48 / 0.52 cm²
        r = synthetic(SHORT)
        r["dft"][0, _cells(prot, count)] = 2e-6
        r["species_supplied"][0, :, 0] = r["dft"][0] * CELL_AREA
        assert fails(r) == out
    assert fails(synthetic(SHORT), drips=0.49e-4 * 100e-6) == set()
    assert fails(synthetic(SHORT), drips=0.51e-4 * 100e-6) == {"protected"}
    for count, out in ((25, set()), (26, {"no_runs"})):                    # 1.00 / 1.04 cm² of 60 µm flowed in
        r = synthetic(SHORT)
        r["dft"][0, _cells(t, count)] = 90e-6
        assert fails(r) == out
    r = synthetic(SHORT)                     # laid 60, taken back 55 by the brush, 60 there: 55 flowed in
    r["dft"][0, _cells(t, 26)] = 60e-6
    r["species_supplied"][0, _cells(t, 26), 0] = 60e-6 * CELL_AREA
    r["species_out"][0, _cells(t, 26), 0] = 55e-6 * CELL_AREA
    assert fails(r) == {"no_runs"}
    for count, out in ((25, set()), (26, {"no_laps"})):
        r = synthetic(SHORT)
        r["laps"][0, _cells(t, count)] = 1
        assert fails(r) == out
    for drift, out in ((0.5e-12, set()), (2e-12, {"books"})):
        r = synthetic(SHORT)
        r["store_ledger"][0] = drift * 2e-5
        assert fails(r) == out


@pytest.mark.parametrize("sealed", [True, False])
def test_trim_seal_holds_the_film(sealed: bool) -> None:
    """A film-only crop: 300 µm of paint at the band's bottom corner, over the stool's horn and against the casing,
    sags for 1 s. Sealed, not a trace reaches the trim's footprint; unsealed, paint crosses onto it (the honest
    cut-in would then put paint on the trim: WP15's mutation)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.tasks.paint_scene import chart_scene  # noqa: PLC0415

    lay = TrimLayout(casing_top=0.913, jobs=("casing",))
    flow = make_flow(lay, tools=False)
    crop = lay.crops()[0]
    xz = crop.world_xz(tv.crop_centres(crop))
    m = lay.masks(crop, xz)
    corner = m["target"] & (xz[..., 0] > lay.casing_x0() - 0.024) & (xz[..., 1] < 0.833)

    def init(sim) -> None:  # noqa: ANN001
        if sealed:
            seal(flow, lay)
        flow.set_depth(np.where(corner, 3e-4, 0.0)[None], chart=0)
        flow.set_species(np.where(corner, 3e-4 * 0.42, 0.0)[None], "solids", chart=0)

    sim = build_simulation(chart_scene([crop.box(lay.face_y)], init, ts.DT), [flow], "cpu")
    for _ in range(50):
        sim.step_period()
    on_trim = float(flow.depth(0)[0][m["protected"]].max())
    if sealed:
        assert on_trim == 0.0 and float(flow.species("solids", 0)[0][m["protected"]].max()) == 0.0
    else:
        assert on_trim > 1e-6


def test_one_metered_stroke_lays_the_e7_film_under_the_whole_contact() -> None:
    """A loaded brush, driven through ``trim_sash.stroke`` (its own press), slides 60 mm up the casing band. The law
    meters ``metered_film(PRESS)`` x sqrt(63.5/64) (the engine's contact is 32 cells, 64 mm, wide); every cell it
    crossed holds that to 2e-3 (minus a trace of soak and evaporation), and so does the contact's back half behind its
    start (WP15's metering lays under the whole contact, at its pose after each slide), but the cell its trailing edge
    covers 0.75 after the first slide, which holds 0.75 of it."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    from alloy.tasks.trim_cut_in import body_name, brush_scene  # noqa: PLC0415

    lay = TrimLayout(casing_top=0.953, jobs=("casing",))
    flow = make_flow(lay, initial=ts.HELD)
    sim = build_simulation(brush_scene(lay, lambda sim: seal(flow, lay)), [flow], "cpu")
    body = list(sim.model.body_label).index(body_name("casing"))
    (x, z0), _ = casing_strokes(lay, Plan())[0]
    start = ts.on_wall(x, z0, ts.STANDOFF, lay.face_y)
    poses = ts.stroke(start, (x, z0), (x, z0 + 0.06), lay.face_y)
    for pose in [ts.Pose(start, ts.wall_quat(0.0), np.zeros(3)), *poses]:
        ts.place(sim, body, pose)
        sim.step_period()
    hm = float(flow.read()["tool_hm"][0, 0])
    assert hm == pytest.approx(ts.metered_film(ts.PRESS) * math.sqrt(ts.WIDTH / 0.064), rel=1e-4)
    crop = lay.crops()[0]
    xz = crop.world_xz(tv.crop_centres(crop))
    h = flow.depth(0)[0]
    inner = np.abs(xz[..., 0] - x) < 0.5 * ts.WIDTH - 0.004
    crossed = inner & (xz[..., 1] > z0 + 0.003) & (xz[..., 1] < z0 + 0.057)
    behind = inner & (xz[..., 1] > z0 - 0.5 * ts.PATCH + ts.SPEED * ts.DT + 0.003) & (xz[..., 1] < z0)
    first = z0 - 0.5 * ts.PATCH + ts.SPEED * ts.DT                  # the trailing edge after the first slide
    edge = inner & (np.abs(xz[..., 1] - (first + 0.0005)) < 1e-6)
    assert crossed.sum() > 100 and behind.sum() > 50 and edge.sum() > 10
    assert np.allclose(h[crossed], hm, rtol=2e-3, atol=0.0) and np.allclose(h[behind], hm, rtol=2e-3, atol=0.0)
    assert np.allclose(h[edge], 0.75 * hm, rtol=2e-3, atol=0.0)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(("jobs", "plan", "out"), [(("casing", "outlet"), Plan(), set()),
                                                   (("casing",), Plan(clearance=-3e-3), {"protected"}),
                                                   (("casing",), Plan(clearance=8e-3), {"hide"})])
def test_cut_in(jobs: tuple[str, ...], plan: Plan, out: set[str]) -> None:
    v = run(TrimLayout(casing_top=1.213, jobs=jobs), plan, settle=10.0)
    assert {k for k, c in v["checks"].items() if c["ok"] is False} == out, v
    assert v["checks"]["books"]["ok"] and v["checks"]["solids_ledger"]["ok"]
