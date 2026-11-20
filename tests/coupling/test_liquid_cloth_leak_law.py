# SPDX-License-Identifier: Apache-2.0
"""SPH liquid particles do not pass a cloth or film sheet (CPU, public API, one world).

A sheet is a closed surface for liquid whatever its mesh, so the number of particles that go THROUGH it (the meter:
``alloy.coupling.liquid_cloth_meter``) is zero; going round a free border is not a crossing and is counted apart
("edge"). Laws:
- meter: synthetic checks of the crossing meter itself (fast, no simulation);
- L1 one cell: a 2 x 2 x 4 particle jet (about 1 ml) hits ONE 40 mm cell (10 particle spacings) of a sheet lying on a
  static table, at four impact speeds, cotton_woven (route cloth) and film_ldpe (cloth_film): through == 0;
- L2 tank: the sheet is the floor of a tank (its border pinched between a box rim and a static frame, 1.5 mm above
  it) and holds a 60 ml pour: through == 0 and nothing under the sheet, for 1 s (fast) and 10 s (slow). No public
  API rig makes a free pouch: a sheet over an opening only bridges it, and its water runs off the sides.
- positive controls: with the liquid's cloth walls off the meter and the tank books count the leak and the tank
  check fails; a sheet narrower than the opening lets water under it and the check fails.
Before / after (ALLOY_VBD_FOREIGN_SELF_CONTACT=off is the old self-contact): red before and green after are L1
cotton at 1, 2 and 4 m/s (INVALID) and the tank (INVALID during the pour, 250 of 938 particles poured). L1 cotton at
0.5 m/s and the film rows are green either way: they guard the fix, they do not prove it.
Bounds: through == 0 and below == 0 come from physics (a sheet is a closed surface). The rig checks are set from
what was measured, and say so: L1 tracked >= 12 of 16, tank escaped <= 10 % (TANK_ESCAPED_MAX), books to 2
particles.
Not written: L3 (a fast sheet against resting water: the API has no way to drive a cloth) and L4 (jet load on a sheet);
Measured numbers are appended as JSON lines to $LAW_LEAK_OUT when it is set.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pytest

import alloy
from alloy.coupling import liquid_cloth_meter as meter

G = 9.81
SPACING = 4.0e-3                       # m, the default liquid spacing (SPH_SPACING)
LATCH_SPACINGS = 6                     # side first recorded within 24 mm: above the largest frame step (4 m/s x 5 ms)
TOL = 0.25 * SPACING                   # a far-side particle is a crossing once > 1 mm from the sheet
TABLE_TOP = 0.10
RIM_TOP = 0.10
MATERIALS = ["cotton_woven", "film_ldpe"]
TANK_GAP = 1.5e-3                      # m, sheet top to frame bottom (at 2.5 mm the stream leaves under the frame)
# The tank's rig check: at most 10 % of the pour outside the opening. What lies there is the pool's edge pushed under
# the frame's inner lip (all of it within 4 mm of the opening, none past the frame), and its count moves with the
# slosh: 43-81 of 938 particles over the hold after the pour and 45 at the end on three CPU kinds (AMD EPYC 9V45 and
# 9V74, Intel Xeon 8573C: the same bits), 50 on a hosted CI runner. The frame 0.5 mm higher leaves 102 outside
# (52 past the frame), 1 mm higher 673.
TANK_ESCAPED_MAX = 0.10


# ---------------------------------------------------------------------------------------------- the meter itself
def _grid(n: int = 3, size: float = 0.2, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-size / 2, size / 2, n)
    verts = np.array([[x, y, z] for y in xs for x in xs])
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            faces += [[a, b, c], [a, c, d]]
    return verts, np.array(faces)


def test_meter_side_distance_and_border():
    """Law of the meter: above/below a flat sheet read +1/-1 at the true distance; a point off the free edge reads
    border, a point over the middle does not. Bound: 1e-12 m (exact geometry)."""
    v, f = _grid()
    pts = np.array([[0.03, 0.02, 0.01], [0.03, 0.02, -0.02], [0.15, 0.0, 0.01], [0.0, 0.0, 0.005]])
    r = meter.side(pts, v, f)
    assert list(r.side) == [1, -1, 1, 1]
    assert np.allclose(r.dist, [0.01, 0.02, math.hypot(0.05, 0.01), 0.005], atol=1e-12)
    assert list(r.border) == [False, False, True, False]
    assert meter.winding_consistent(f)
    assert not meter.winding_consistent(np.vstack([f, f[:1]]))
    assert meter.border_flags(f)[1].sum() == 8


def _path(points: list[tuple[float, float, float]]) -> np.ndarray:
    return np.array(points, dtype=float)[:, None, :]


def test_meter_counts_through_edge_and_kept():
    """Law of the meter: a particle that goes down through the sheet is `through`; one that goes round the free
    edge and under is `edge` (not through, even when it ends under the middle); one that stays above is `kept`;
    one that never comes within the latch distance is untracked. Bound: exact counts."""
    v, f = _grid()
    down = _path([(0.02, 0.02, 0.02), (0.02, 0.02, 0.005), (0.02, 0.02, -0.01), (0.02, 0.02, -0.03)])
    around = _path([(0.05, 0.0, 0.02), (0.05, 0.0, 0.008), (0.15, 0.0, 0.004), (0.15, 0.0, -0.01),
                    (0.05, 0.0, -0.01), (0.0, 0.0, -0.03)])
    above = _path([(0.0, 0.0, 0.02)] * 6)
    far = _path([(0.0, 0.0, 0.5)] * 6)
    n_f = 6
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], n_f - len(a), axis=0)], axis=0)  # noqa: E731
    pts = np.concatenate([pad(down), pad(around), pad(above), pad(far)], axis=1)
    c = meter.crossings(pts, v, f, None, latch=0.024, tol=0.001)
    assert (c.through, c.edge, c.kept, c.tracked) == (1, 1, 1, 3)
    assert list(c.through_ids) == [0]


def test_meter_counts_through_after_going_round_the_edge():
    """Law of the meter: a particle that went round the free edge and later comes back up
    through the sheet's interior is `through`, not `edge`. Bound: exact counts."""
    v, f = _grid()
    back = _path([(0.05, 0.0, 0.02), (0.05, 0.0, 0.008), (0.15, 0.0, 0.004), (0.15, 0.0, -0.01),
                  (0.05, 0.0, -0.01), (0.0, 0.0, -0.005), (0.0, 0.0, 0.01), (0.0, 0.0, 0.02)])
    c = meter.crossings(back, v, f, None, latch=0.024, tol=0.001)
    assert (c.through, c.edge, c.kept) == (1, 0, 0)


def test_meter_ignores_inactive_and_small_flips():
    """Law of the meter: an inactive particle is not read; a flip closer than `tol` is not yet a crossing.
    Bound: exact counts."""
    v, f = _grid()
    pts = np.concatenate([_path([(0, 0, 0.01), (0, 0, 0.005), (0, 0, -0.0005)] * 2),
                          _path([(0, 0, 0.01), (0, 0, 0.005), (0, 0, -0.01)] * 2)], axis=1)
    act = np.ones(pts.shape[:2], bool)
    act[:, 1] = False
    c = meter.crossings(pts, v, f, act, latch=0.024, tol=0.001)
    assert (c.through, c.edge, c.kept) == (0, 0, 1)


# ------------------------------------------------------------------------------------------------- the sim rigs
@dataclass
class Run:
    """Frames of one run: particles, sheet, active flags, plus faces, spacing, health and wall times."""

    pts: np.ndarray
    sheet: np.ndarray
    active: np.ndarray
    faces: np.ndarray
    volume: float
    ok: bool
    routes: dict
    wall_s: float
    t0_frame: int            # first recorded frame after the emitters start


def _release_after_settle(sim: alloy.Simulation, seconds: float) -> None:
    """Keep the pour waiting while the sheet settles (the public API starts a pour at t = 0)."""
    from alloy.api._liquid import liquid_solvers      # private: no public switch for a pour's start time yet
    sol = liquid_solvers(sim.engine)[0]
    n = len(sol.emitters.table.world)
    for e in range(n):
        sol.set_emitter_enabled(e, False)
    sim.settle(seconds)
    for e in range(n):
        sol.set_emitter_enabled(e, True)


def _record(sim: alloy.Simulation, seconds: float, every: int, dt: float) -> tuple[list, list, list]:
    pts, sheet, act = [], [], []
    for i in range(int(round(seconds / dt / every))):
        sim.step(every)
        s = sim.state(["sheet", "water"], world=0)
        pts.append(s["water.points"].astype(np.float32))
        sheet.append(s["sheet.points"].astype(np.float32))
        act.append(s["water.active"].astype(bool))
    return pts, sheet, act


def _finish(sim: alloy.Simulation, rec: tuple, t_start: float) -> Run:
    pts, sheet, act = rec
    return Run(np.stack(pts), np.stack(sheet), np.stack(act), np.asarray(sim.faces("sheet")),
               float(sim.particle_volume("water")), bool(sim.check()["ok"]),
               {k: v["route"] for k, v in sim.routes().items()}, time.perf_counter() - t_start, 0)


def run_drop(material: str, impact: float, hold: float = 0.2, dt: float = 0.005) -> tuple[Run, float]:
    """A ~1 ml jet onto the centre cell of a 0.2 m sheet (6 x 6 vertices, 40 mm cells) lying on a static table.
    Returns the run and the nominal impact speed sqrt(v0^2 + 2 g h)."""
    t_start = time.perf_counter()
    h = 0.02
    v0 = math.sqrt(max(impact ** 2 - 2 * G * h, 0.05 ** 2))
    rate = v0 * (2 * SPACING) ** 2                                  # a 2 x 2 particle nozzle
    scene = alloy.Scene(dt=dt)
    scene.add_ground()
    scene.add_box("table", size=(0.24, 0.24, TABLE_TOP), pos=(0, 0, TABLE_TOP / 2), static=True)
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, TABLE_TOP + 0.0025), resolution=6, material=material,
                    thickness=2e-3)
    scene.add_pour("water", 1e-6, pos=(0.006, -0.004, TABLE_TOP + 0.003 + h + 0.016), rate=rate, speed=v0)
    sim = alloy.build(scene, device="cpu")
    _release_after_settle(sim, 0.3)
    run = _finish(sim, _record(sim, hold, 1, dt), t_start)
    return run, math.sqrt(v0 ** 2 + 2 * G * h)


def run_rim(ml: float, seconds: float, dt: float, settle: float = 0.3) -> Run:
    """The rim rig: a 0.2 m cotton_woven sheet (11 x 11 vertices, 20 mm cells) resting on the rim of an open
    box (16 cm opening, 30 mm walls), a pour from 5 cm above it, every control period recorded."""
    t_start = time.perf_counter()
    rim = 0.10 + 0.03
    scene = alloy.Scene(dt=dt)
    scene.add_ground()
    scene.add_open_box("frame", inner=(0.16, 0.16, 0.10), wall=0.03, pos=(0, 0, 0))
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, rim + 0.0025), resolution=11, material="cotton_woven",
                    thickness=2e-3)
    scene.add_pour("water", ml * 1e-6, pos=(0, 0, rim + 0.05), speed=0.5)
    sim = alloy.build(scene, device="cpu")
    _release_after_settle(sim, settle)
    return _finish(sim, _record(sim, seconds, 1, dt), t_start)


def run_tank(material: str, ml: float, hold: float, dt: float = 0.005, every: int = 4, width_y: float = 0.20) -> Run:
    """The cloth-floor tank: a 0.20 m sheet (14 x 14 vertices, ~15 mm cells)
    lies on the rim of an open box (opening 0.10 m, 40 mm walls, rim top 0.12 m) under a static frame of four bars
    (opening 0.10 m, 0.06 m wide and high) whose bottom is TANK_GAP above the sheet's top face. The frame holds the
    sheet's border down, so water poured inside the frame can leave only through the sheet or under the frame.
    A free sheet over an opening only bridges it, so no API rig makes a pouch."""
    t_start = time.perf_counter()
    wall, depth = 0.04, 0.08
    rim = depth + wall                                  # add_open_box: the cavity floor stands `wall` above z = 0
    fh, fw, out = 0.06, 0.06, 0.22
    zc = rim + 0.002 + TANK_GAP + fh / 2
    o = out / 2 - fw / 2
    scene = alloy.Scene(dt=dt)
    scene.add_ground()
    scene.add_open_box("box", inner=(0.10, 0.10, depth), wall=wall, pos=(0, 0, 0))
    scene.add_cloth("sheet", size=(0.20, width_y), pos=(0, 0, rim + 0.0025), resolution=14, material=material,
                    thickness=2e-3)
    scene.add_box("bar_n", size=(out, fw, fh), pos=(0, o, zc), static=True)
    scene.add_box("bar_s", size=(out, fw, fh), pos=(0, -o, zc), static=True)
    scene.add_box("bar_e", size=(fw, 0.10, fh), pos=(o, 0, zc), static=True)
    scene.add_box("bar_w", size=(fw, 0.10, fh), pos=(-o, 0, zc), static=True)
    scene.add_pour("water", ml * 1e-6, pos=(0, 0, rim + 0.07), speed=0.5)
    sim = alloy.build(scene, device="cpu")
    _release_after_settle(sim, 0.75)
    pour_s = ml * 1e-6 / 2e-4                           # add_pour's default rate, 2e-4 m^3/s
    return _finish(sim, _record(sim, pour_s + hold, every, dt), t_start)


def _count(run: Run) -> meter.CrossingCounts:
    return meter.crossings(run.pts, run.sheet, run.faces, run.active, latch=LATCH_SPACINGS * SPACING, tol=TOL)


def _log(tag: str, **kw) -> None:
    line = json.dumps({"law": tag, **kw}, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("LAWLEAK", line)
    path = os.environ.get("LAW_LEAK_OUT")
    if path:
        with open(path, "a") as fh:
            fh.write(line + "\n")


# --------------------------------------------------------------------------------------------------------- L1
@pytest.mark.law
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("impact", [0.5, 1.0, 2.0, 4.0])
def test_l1_one_cell_jet_does_not_pass_the_sheet(material, impact):
    """L1: a ~1 ml jet (2 x 2 particles, 4 layers) falls on ONE 40 mm cell (10 particle spacings) of a sheet lying on
    a static table; no particle crosses the sheet (a sheet is a closed surface whatever its mesh, so the bound is
    zero: 'through' == 0, in whole particles). Impact speeds sqrt(v0^2 + 2 g h) from 0.5 m/s (the jet starts 2 cm up,
    so the lowest is ~0.63) to 4 m/s. The build must be healthy (sim.check ok) and the jet must reach the sheet."""
    run, v_imp = run_drop(material, impact, hold=0.2 if impact >= 2.0 else 0.35)
    c = _count(run)
    _log("L1", material=material, impact_nominal=impact, impact_v=round(v_imp, 3), through=c.through, edge=c.edge,
         kept=c.kept, tracked=c.tracked, active=int(run.active[-1].sum()), ok=run.ok, wall_s=round(run.wall_s),
         first_through_frame=c.first_through_frame, routes=run.routes)
    assert run.active[-1].sum() >= 12, "the jet did not come out"
    assert c.tracked >= 12, "the jet never reached the sheet: the rig is off"
    assert run.ok, "the world went INVALID and froze (sim.check): a frozen world cannot show whether liquid leaks"
    assert c.through == 0, f"{c.through} of {c.tracked} particles went through the {material} sheet at {v_imp:.2f} m/s"


# --------------------------------------------------------------------------------------------------------- L2
def _tank_counts(run: Run) -> dict:
    """End-of-run books of the tank: poured, through (meter), held over the sheet / below it inside the opening,
    escaped outside the opening (under the frame)."""
    c = _count(run)
    a, w, sh = run.active[-1], run.pts[-1], run.sheet[-1]
    opening = a & (np.abs(w[:, 0]) < 0.05) & (np.abs(w[:, 1]) < 0.05)
    r = meter.side(w[opening], sh, run.faces)
    up = int(meter.side(np.array([[0.0, 0.0, sh[:, 2].max() + 0.05]]), sh, run.faces).side[0])  # the water's side
    return {"poured": int(a.sum()), "slots": int(run.pts.shape[1]), "through": c.through, "edge": c.edge,
            "held": int((r.side == up).sum()), "below": int((r.side == -up).sum()),
            "escaped": int((a & ((np.abs(w[:, 0]) > 0.05) | (np.abs(w[:, 1]) > 0.05))).sum()), "ok": run.ok}


def _check_tank(material: str, run: Run, ml: float, tag: str) -> None:
    """Law bounds from physics: through == 0 and below == 0 (a sheet is a closed surface). Rig checks, set at or near
    the measured rows and said so: escaped <= 10 % (TANK_ESCAPED_MAX), books close to 2 particles (meter noise)."""
    k = _tank_counts(run)
    _log(tag, material=material, ml=ml, wall_s=round(run.wall_s), routes=run.routes, **k)
    poured, escaped = k["poured"], k["escaped"]
    assert poured == k["slots"], f"the pour did not come out ({poured} of {k['slots']} particles)"
    assert run.ok, "the world went INVALID and froze (sim.check): the counts below would not be a result"
    assert k["through"] == 0, f"{k['through']} of {poured} particles went through the {material} sheet"
    assert k["below"] == 0, f"{k['below']} particles ended under the sheet inside the opening"
    assert escaped <= TANK_ESCAPED_MAX * poured, \
        f"{escaped} of {poured} outside the opening (bound {TANK_ESCAPED_MAX:.0%}): the frame does not hold the sheet"
    assert k["held"] + escaped >= poured - 2, f"held {k['held']} + escaped {escaped} < poured {poured}: unaccounted"


# ------------------------------------------------------------------------------------------- positive controls
@pytest.mark.law
@pytest.mark.timeout(7200)
def test_control_meter_and_books_count_a_leak(monkeypatch):
    """Positive control: the leak laws can see a leak. With the liquid's cloth walls turned
    off (SPH builds no soft faces: monkeypatched ``soft_walls.soft_topology``), the tank's pour falls through the
    sheet. The meter must count crossings through its interior (through >= 10 % of what reached the sheet), the
    books must count water under it (below > 0), and the tank law's check must fail."""
    from alloy.solvers.sph import soft_walls
    monkeypatch.setattr(soft_walls, "soft_topology", lambda *args, **kwargs: None)
    run = run_tank("cotton_woven", 60.0, hold=0.3)
    k = _tank_counts(run)
    c = _count(run)
    _log("control_contact_off", tracked=c.tracked, wall_s=round(run.wall_s), **k)
    assert c.tracked >= 100, "the pour never reached the sheet: the rig is off"
    assert k["through"] >= 0.1 * c.tracked, f"{k['through']} of {c.tracked} counted through: the meter is blind"
    assert k["below"] > 0, "no particle counted under the sheet: the books are blind"
    with pytest.raises(AssertionError):
        _check_tank("cotton_woven", run, 60.0, "control_contact_off_check")


@pytest.mark.law
@pytest.mark.timeout(7200)
def test_control_tank_counts_water_under_the_sheet():
    """Positive control: the tank's books see a leak. A sheet 6 cm wide in y leaves 2 cm gaps to the box walls, so
    the pour runs off its long edges into the box: below > 0 and the tank law's check fails."""
    run = run_tank("cotton_woven", 60.0, hold=0.5, width_y=0.06)
    k = _tank_counts(run)
    _log("control_tank", wall_s=round(run.wall_s), **k)
    assert k["poured"] == k["slots"] and run.ok, "the rig is off"
    assert k["below"] >= 0.1 * k["poured"], f"only {k['below']} of {k['poured']} counted under the sheet: blind books"
    with pytest.raises(AssertionError):
        _check_tank("cotton_woven", run, 60.0, "control_tank_check")


@pytest.mark.law
@pytest.mark.timeout(7200)
def test_l2_tank_fast():
    """L2 (fast): 60 ml poured into the cloth-floor tank (cotton_woven), held 1 s after the pour. Law: no particle
    goes through the sheet (through == 0, bound 0) or ends under it inside the opening (below == 0). Rig checks: the
    world stays OK, at most 10 % lies outside the opening, and the books close to 2 particles (meter noise)."""
    _check_tank("cotton_woven", run_tank("cotton_woven", 60.0, hold=1.0), 60.0, "L2fast")


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(14400)
def test_l2_tank_10s():
    """L2: the same tank held 10 simulated s after the pour, the "pouch holds N ml for 10 s" law (measured: 894
    of 938 particles held, 0 through, 0 below, 44 under the frame; the old detection goes INVALID during the pour)."""
    _check_tank("cotton_woven", run_tank("cotton_woven", 60.0, hold=10.0), 60.0, "L2")
