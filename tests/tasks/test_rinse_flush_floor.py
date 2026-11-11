# SPDX-License-Identifier: Apache-2.0
"""flush-floor (CPU): the hose against Gym-Anything's reel, the bay's slope, shelf and start soil, the scripted spiral
and valve poses, Gym-Anything's log reduction, the verdict's gates and the settle's quiet test on synthetic reads, and
the scripted flush on a bay cropped round the drain: the soil goes, the scraps end on the drain, every gate passes with
the books closed."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks import rinse_flush_checks as fc
from alloy.tasks import rinse_flush_floor as ff
from alloy.tasks import rinse_flush_policy as fp


def test_the_hose_delivers_gym_anythings_flow_from_60_psi() -> None:
    h = ff.Hose()
    a_eff = 1.0 / math.sqrt(1.0 / h.exit_area ** 2 + 1.0 / h.hose_cd_area ** 2)      # orifices in series
    assert a_eff * math.sqrt(2.0 * h.mains / ff.RHO) == pytest.approx(3.3 * ff.GPM, rel=1e-12)
    assert h.flow / h.exit_area == pytest.approx(24.0, rel=1e-12)
    assert ff.RHO * h.flow * h.speed == pytest.approx(5.1, rel=0.03)                 # their 5.1 N
    noz = h.nozzle()
    assert (noz.pattern, noz.rays, noz.body) == ("cone", 9, ff.VALVE)
    assert math.degrees(noz.angle) == pytest.approx(2.0 * 6.0, rel=1e-12)       # their ring jets tilt 6° off the axis
    assert noz.pos == (0.0, 0.0, -0.104)                                         # their nozzle face
    assert [e["name"] for e in h.network()["edges"]].index(ff.EDGE) == fc.Columns().outlet


def test_the_valve_docks_at_their_holster_centre() -> None:
    bay = ff.Bay()
    assert fp.parked(bay)[:3] == bay.holster == (1.08, 0.10, 0.95 + 0.105)      # GA's z0 + 0.105, the body's origin


def test_the_bay_slopes_to_its_drain_and_carries_their_soil() -> None:
    bay = ff.Bay()
    c, base = bay.centres(), bay.base()
    r = np.hypot(c[..., 0] - bay.drain[0], c[..., 1] - bay.drain[1])
    assert base[r <= bay.drain_r].max() == 0.0
    assert np.allclose(base[r > bay.drain_r], (r[r > bay.drain_r] - bay.drain_r) / 96.0, rtol=1e-12, atol=0.0)
    soil = bay.soil(np.random.default_rng(1))
    (x0, x1), (y0, y1) = bay.region
    region = (c[..., 0] >= x0) & (c[..., 0] <= x1) & (c[..., 1] >= y0) & (c[..., 1] <= y1) & (r > bay.drain_r)
    assert not soil[~region].any() and not soil[r <= bay.drain_r].any()
    assert abs((soil[region] > 0).mean() - 0.85) < 0.02
    on = soil[soil > 0]
    assert 0.4e-3 <= on.min() and on.max() <= 2.0e-3                                 # 40-200 ug/cm2
    assert np.array_equal(bay.wash(soil) > 0, soil > 0) and bay.wash(soil).max() == 0.6e-3    # on the soiled cells
    posts = np.array([(x, y) for _, x, y in bay.posts()])                            # GA's x0 + 0.015, ... (bay frame)
    assert np.allclose(np.sort(posts[:, 0]), [-0.835, -0.835, -0.165, -0.165], rtol=0.0, atol=1e-12)
    assert np.allclose(np.sort(posts[:, 1]), [0.065, 0.065, 0.535, 0.535], rtol=0.0, atol=1e-12)


def test_the_spirals_close_on_the_drain_at_the_aim_speed() -> None:
    bay, fl = ff.Bay(), fp.Flush()
    aims = fp.spiral(bay, fl)                                       # one pass; the run repeats it fl.passes times
    r = np.hypot(aims[:, 0] - bay.drain[0], aims[:, 1] - bay.drain[1])
    assert np.all(np.diff(r) < 0) and r[-1] <= fl.r_end + 1e-3 and r[0] > 0.89
    step = np.linalg.norm(np.diff(aims, axis=0), axis=1)
    assert np.allclose(step[r[1:] > 0.1], fl.speed * fp.CONTROL_DT, rtol=0.01)       # chords: the arc near the drain


def test_the_sweep_keeps_the_valve_and_its_jets_off_the_can() -> None:
    """Near Gym-Anything's garbage can the jets lean less; where nothing clears it they are off; on, the valve keeps
    ``CLEAR`` off it; every soiled cell has a jet aimed within a third of the pitch over the two passes."""
    bay, fl = ff.Bay(), fp.Flush()
    aims = np.concatenate([fp.spiral(bay, fl), fp.spiral(bay, fl, 0.5 * fl.pitch)])        # both passes
    res = [fp.aim_pose(bay, tuple(a), fl.height, fl.lean) for a in aims]
    poses, ok = np.array([p for p, _ in res]), np.array([o for _, o in res])
    cx, cy, r, _ = bay.can
    assert 0 < (~ok).sum() < 0.1 * len(ok)
    assert (np.hypot(poses[ok, 0] - cx, poses[ok, 1] - cy) - r).min() >= fp.CLEAR
    c, soil = bay.centres(), bay.soil(np.random.default_rng(0)) > 0
    near = np.hypot(aims[ok, :1] - c[..., 0][soil][None], aims[ok, 1:] - c[..., 1][soil][None]).min(axis=0)
    assert near.max() < 0.35 * fl.pitch                             # the second pass runs between the first's turns


@pytest.mark.parametrize("aim", [(0.3, 0.35), (1.7, 0.4), (1.0, 0.9)])
def test_the_valve_aims_its_jets_at_the_point_leaning_to_the_drain(aim: tuple[float, float]) -> None:
    from scipy.spatial.transform import Rotation

    bay, fl = ff.Bay(), fp.Flush()
    pose = fp.valve_pose(bay, aim, fl.height, fl.lean)
    j = Rotation.from_quat(pose[3:]).apply([0.0, 0.0, -1.0])
    tip = np.asarray(pose[:3]) + fp.NOZZLE * j
    hit = tip + (-tip[2] / j[2]) * j
    assert np.allclose(hit[:2], aim, rtol=0.0, atol=1e-12) and tip[2] == pytest.approx(fl.height, rel=1e-12)
    over = aim == bay.drain
    assert math.degrees(math.acos(-j[2])) == pytest.approx(0.0 if over else 30.0, abs=1e-9)
    if not over:
        to = np.asarray(bay.drain) - np.asarray(aim)
        assert np.dot(j[:2], to) > 0.0 and abs(j[0] * to[1] - j[1] * to[0]) < 1e-12


@pytest.mark.parametrize("j", [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.6, 0.0, -0.8), (0.3, -0.4, math.sqrt(0.75))])
def test_quat_down_to_turns_the_valves_axis_onto_a_direction(j: tuple[float, float, float]) -> None:
    from scipy.spatial.transform import Rotation

    q = fp.quat_down_to(np.asarray(j))
    assert math.hypot(*q) == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(Rotation.from_quat(q).apply([0.0, 0.0, -1.0]), j, rtol=0.0, atol=1e-12)


def _read(bay: ff.Bay, start: np.ndarray, soil: np.ndarray, susp: np.ndarray, shelf_m3: float = 0.0,
          spilled_m3: float = 0.0, lost_kg: float = 0.0, caught_m3: tuple[float, ...] = (0.0, 0.0)) -> dict:
    """A one-world read with closed books: the soil left plus what came off is the start (less ``lost_kg``), all of
    what came off went into the film, the hose shut; ``caught_m3`` per occluder slot (in the cross ledger too)."""
    n0, m = soil.size, bay.shelf_cells()
    C = n0 + len(bay.levels) * m[0] * m[1]
    sent, area = 1e-3, bay.cell * bay.cell
    s0, s1 = float(start.sum()) * area, float(soil.sum()) * area

    def cells(first: np.ndarray, rest: float = 0.0) -> np.ndarray:
        out = np.full((1, C), rest / max(C - n0, 1))
        out[0, :n0] = first.ravel()
        return out

    one = np.array
    return {"soil": cells(soil), "species": cells(susp)[..., None], "supplied": cells(np.zeros_like(soil), shelf_m3),
            "spilled": cells(np.full(soil.shape, spilled_m3 / n0)), "supply_discharge": np.zeros((1, 2)),
            "supplied_volume": one([sent]), "pending_volume": np.zeros(1), "ledger": np.zeros(1),
            "supply_swept": np.zeros(1), "supply_to_tank": np.zeros(1),
            "supply_lost": np.zeros(1), "supply_overspray": np.zeros(1), "supply_withdrawn": np.zeros(1),
            "supply_mist": np.zeros(1), "supply_caught": one([sum(caught_m3)]), "supply_lumped": np.zeros(1),
            "supply_caught_by": one([caught_m3]), "supply_sent": one([sent + sum(caught_m3)]),
            "species_ledger": np.zeros((1, 1)), "drained_total_volume": one([sent]),
            "soil_start_mass": one([s0]), "soil_mass": one([s1]), "soil_removed_mass": one([s0 - s1 - lost_kg]),
            "species_supplied": cells(np.zeros_like(soil))[..., None] + (s0 - s1 - lost_kg) / C,
            "species_mass": one([[float(susp.sum()) * area]])}


def _end(bay: ff.Bay, start: np.ndarray, off: float = 0.0, pre: np.ndarray | None = None, pre_shelf_g: float = 0.0,
         scrap_off: float | None = None) -> ff.End:
    scrap = np.array([bay.drain]) + (off if scrap_off is None else np.array([scrap_off, 0.0]))
    return ff.End(np.asarray(bay.holster) + off, scrap, start * 0.0 if pre is None else pre, 0.0, pre_shelf_g)


def test_the_verdict_gates_on_a_synthetic_floor() -> None:
    bay = ff.CROP
    start = bay.soil(np.random.default_rng(0))
    zero = np.zeros_like(start)
    clean = ff.verdict(bay, start, _read(bay, start, zero, zero), _end(bay, start), 0)
    assert clean["success"] and clean["score"] == 1.0 and clean["terms"]["a"] == 1.0
    for lost in (2e-12, 0.0):                                       # 2 ng gone from the chain fails; none passes
        r = _read(bay, start, zero, zero, lost_kg=lost)
        assert ff.verdict(bay, start, r, _end(bay, start), 0)["success"] is (lost == 0.0)
    landed = _read(bay, start, zero, zero)                          # drops put 1 ug of the film's soil back (WP9)
    landed["species_supplied"] = landed["species_supplied"] + 1e-9 / landed["species_supplied"].shape[1]
    assert ff.verdict(bay, start, landed, _end(bay, start), 0)["failed"] == ["no_simulation_fault"]
    landed[fc.LANDED] = np.array([[1e-9]])
    assert ff.verdict(bay, start, landed, _end(bay, start), 0)["success"]
    landed["drops_species_shed"] = np.array([[1.5e-9]])              # the drops' own identity (WP9): shed = landed
    landed["drops_species_in_flight"] = np.array([[0.5e-9]])         # + in flight; 1 ng lost in flight fails it
    assert ff.verdict(bay, start, landed, _end(bay, start), 0)["success"]
    landed["drops_species_in_flight"] = np.array([[0.0]])
    assert ff.verdict(bay, start, landed, _end(bay, start), 0)["failed"] == ["no_simulation_fault"]
    leak = ff.verdict(bay, start, _read(bay, start, zero, zero, lost_kg=1e-9), _end(bay, start), 0)
    assert leak["failed"] == ["no_simulation_fault"] and leak["score"] is None
    for key, bad in (("species_ledger", np.array([[2e-12]])), ("ledger", np.array([1e-9]))):    # soil in water; water
        r = _read(bay, start, zero, zero)
        r[key] = bad
        assert ff.verdict(bay, start, r, _end(bay, start), 0)["failed"] == ["no_simulation_fault"]
    dirty = zero.copy()
    dirty[5, 5] = 1.1 * ff.FLUSHED                                  # one cell over 10 ug/cm2 of suspended soil
    v = ff.verdict(bay, start, _read(bay, start, zero, dirty, 3e-6, 60e-6), _end(bay, start, off=0.2, pre_shelf_g=3.0),
                   0)
    assert set(v["failed"]) == {"floor_flushed", "contained", "floor_water", "valve_parked", "scraps_on_drain"}
    assert v["gates"]["contained"]["value"] == pytest.approx(3.0 * 0.998, rel=1e-9)  # 3 mL of water on the shelf
    t = v["terms"]
    assert t["k"] == 0.0 and t["shelf"] == 1.0 and t["a"] == 1.0                      # 3 g: over twice the limit
    assert v["score"] == pytest.approx(0.9 * max(0.0, (t["a"] + t["a"] * t["k"]) / 2.0 - t["shelf"]), abs=0.0)
    assert not v["success_ga"] and ff.verdict(bay, start, _read(bay, start, zero, zero, spilled_m3=60e-6),
                                              _end(bay, start), 0)["success_ga"]       # floor_water is Alloy's


def test_the_verdict_reads_each_gate_where_gym_anything_does() -> None:
    """DC5's surviving mutations: a scrap just past the drain, the cross ledger off by 1e-8, the valve's edge (not the
    hose's) flowing, the shelf's catch per occluder slot, and a failing world's k term."""
    bay = ff.CROP
    start = bay.soil(np.random.default_rng(0))
    zero = np.zeros_like(start)
    clean = _read(bay, start, zero, zero)

    def gate(read: dict, end: ff.End | None = None, cols: fc.Columns | None = None) -> dict:
        return ff.verdict(bay, start, read, end or _end(bay, start), 0, cols)["gates"]

    for off, ok in ((bay.drain_r - 0.005, True), (bay.drain_r + 0.005, False)):
        assert gate(clean, _end(bay, start, scrap_off=off))["scraps_on_drain"]["ok"] is ok
    for rel, ok in ((1e-10, True), (1e-8, False)):              # the film's ledger closed, DC5's cross ledger not
        r = dict(clean, supply_lost=clean["supply_sent"] * rel)
        assert gate(r)["no_simulation_fault"]["ok"] is ok
    q = 1e-3 * ff.GPM                                           # 10x the outlet-off limit
    for flow, ok in (([0.0, -q], False), ([-q, 0.0], None)):    # edges: hose, valve; the hands part is not judged
        assert gate(dict(clean, supply_discharge=np.array([flow])))["water_off_and_hands_free"]["ok"] is ok
    caught = _read(bay, start, zero, zero, caught_m3=(5e-6, 2e-6))                    # slot 1: a shelf board's edge
    assert gate(caught)["contained"]["ok"]
    assert gate(caught, cols=fc.Columns(shelf=(1,)))["contained"]["value"] == pytest.approx(2.0 * 0.998, rel=1e-12)
    assert gate(caught, cols=fc.Columns(shelf=((1,),)))["contained"]["value"] == pytest.approx(2.0 * 0.998, rel=1e-12)
    assert gate(caught, cols=fc.Columns(shelf=((0,),)))["contained"]["value"] == pytest.approx(5.0 * 0.998, rel=1e-12)
    dirty = zero.copy()
    dirty[start > ff.FLUSHED] = ff.FLUSHED * 2.0                # fails floor_flushed; parked and off: k = 1
    halfway = np.where(start > ff.FLUSHED, np.sqrt(start * ff.FLUSHED), 0.0)
    v = ff.verdict(bay, start, _read(bay, start, dirty, zero), _end(bay, start, pre=halfway), 0)
    assert v["failed"] == ["floor_flushed"] and v["terms"]["k"] == 1.0
    assert v["progress"] == pytest.approx(v["terms"]["a"], rel=1e-12) and v["terms"]["a"] == pytest.approx(0.5)
    assert v["score"] == pytest.approx(0.9 * 0.5, rel=1e-12)


def test_the_settle_stops_when_a_world_is_quiet() -> None:
    """Gym-Anything's quiet (``task.py:817-848``): no outlet, no item faster than 2 mm/s, under 1 g/s change in the
    film or in what left it."""
    prev = np.zeros((5, 2))
    now = np.array([[0.5e-3, 0.5e-3], [2e-3, 0.0], [0.0, -2e-3], [0.0, 0.0], [0.0, 0.0]])
    q_off = np.array([0.0, 0.0, 0.0, 0.0, 1e-3])
    speed = np.array([1e-3, 0.0, 0.0, 3e-3, 0.0])
    assert fc.quiet(prev, now, q_off, speed).tolist() == [True, False, False, False, False]
    read = {"film_volume": np.array([1e-3]), "drained_total_volume": np.array([2e-3]),
            "spilled_volume": np.array([0.5e-3])}
    assert np.allclose(fc.quiet_books(read), [[0.998, 2.495]], rtol=1e-12, atol=0.0)


def test_the_score_reads_the_floor_before_the_settle() -> None:
    """Gym-Anything's reduction term ``a`` (``scoring.py:77-87``) is taken before the settle, over the cells dirty at
    the start, and is 0 when none was dirty."""
    bay = ff.CROP
    start = bay.soil(np.random.default_rng(0))
    zero = np.zeros_like(start)
    halfway = np.where(start > ff.FLUSHED, np.sqrt(start * ff.FLUSHED), 0.0)            # half the log reduction
    v = ff.verdict(bay, start, _read(bay, start, zero, zero), _end(bay, start, pre=halfway), 0)
    assert v["terms"]["a"] == pytest.approx(0.5, rel=1e-12) and v["success"]
    assert ff.verdict(bay, zero, _read(bay, zero, zero, zero), _end(bay, zero), 0)["terms"]["a"] == 0.0


#: a 0.3 m crop round the drain with one scrap. Its spiral starts past the crop's edge, which would put the valve
#: over the shelf's place beside the full bay, so the shelf stands 0.6 m further west here.
SMALL = dataclasses.replace(ff.CROP, size=(0.3, 0.3), drain=(0.15, 0.15), region=((0.05, 0.25), (0.05, 0.25)),
                            holster=(0.28, 0.03, 1.055), debris=((0.26, 0.08),), shelf=((-1.45, -0.75), (0.05, 0.55)))


def test_the_overflow_watch_catches_dropped_contacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """MuJoCo Warp drops contacts and constraints past its buffers with only a stdout line: starved to 2 contacts a
    world, the scrap on the floor overflows and the watch sees it, every substep's through WP8's plug-in (attached at
    install); at the task's 64 it does not."""
    from alloy.plugins.harness.scene import build_simulation
    from alloy.tasks.paint_supply import Watched
    from alloy.tasks.rinse_hand import OverflowWatch

    seen = {}
    for ncon in (ff.NCONMAX, 2):
        monkeypatch.setattr(ff, "NCONMAX", ncon)
        flow = ff.make_flow(SMALL, ff.Hose())
        sim = Watched(build_simulation(ff.build_scene(SMALL, ff.Hose(), None, 1), [flow, OverflowWatch()], "cpu"))
        assert sim.watch.plugin is not None and sim.watch.plugin.bits is not None      # every substep counted
        for _ in range(3):
            sim.step_period()
        seen[ncon] = int(sim.watch.worlds(1)[0])
    assert seen[64] == 0 and seen[2] != 0
    occ = flow.supply.occluder_shapes
    slots = fc.shelf_slots(sim.model, occ)
    assert len(slots) == 1                                          # one world: its own row of the occluder table
    shelf = [sim.model.shape_label[occ[0][j]].split("/")[-1] for j in slots[0]]
    assert sorted(shelf) == [*(f"shelf_{k}" for k in range(3)), *(f"shelf_post_{k}" for k in range(4))]


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_scripted_flush_on_a_cropped_bay() -> None:
    res = ff.run(SMALL, fp.Flush(speed=0.25, settle=5.0))
    w = res["worlds"][0]
    nf = w["gates"]["no_simulation_fault"]["value"]
    assert nf["water_rel"] < 1e-12 and nf["soil_kg"] <= 1e-15 and abs(nf["species_kg"]) <= 1e-15
    assert nf["overflow"] == 0                                      # MuJoCo Warp dropped no contact or constraint
    start, adhered, suspended = w["soil_mg"]
    assert adhered == 0.0 and suspended < 0.05 * start
    assert w["gates"]["scraps_on_drain"]["value"] <= fp.Flush().chase_done + 0.01
    assert fc.SETTLE_MIN + fc.SETTLE_STEP <= w["settle_s"] <= 5.0          # quiet at the earliest check, or capped
    assert w["success"] and w["success_ga"], w["failed"]


def test_not_judged_hands_gate_and_success_ga_on_the_judged_gates() -> None:
    """water_off_and_hands_free has no hands part (None while the outlet is off), floor_water and scraps_on_drain are
    the port's own gates (not counted in ``judged``), success_ga is every judged Gym-Anything gate."""
    bay = ff.CROP
    start = bay.soil(np.random.default_rng(0))
    zero = np.zeros_like(start)
    clean = _read(bay, start, zero, zero)
    v = ff.verdict(bay, start, clean, _end(bay, start), 0)
    assert v["gates"]["water_off_and_hands_free"]["ok"] is None
    assert v["success"] and v["success_ga"] and v["judged"] == [4, 6]
    assert v["summary"].startswith("4 of 6 Gym-Anything criteria judged")
    assert v["not_judged"] == ["water_off_and_hands_free", "intact_and_not_dropped"]
    assert v["gates"]["intact_and_not_dropped"]["ok"] is None
    q = 1e-3 * ff.GPM
    open_ = ff.verdict(bay, start, dict(clean, supply_discharge=np.array([[0.0, -q]])), _end(bay, start), 0)
    assert open_["failed"] == ["water_off_and_hands_free"] and not open_["success"] and not open_["success_ga"]
    wet = ff.verdict(bay, start, _read(bay, start, zero, zero, spilled_m3=60e-6), _end(bay, start), 0)
    assert wet["failed"] == ["floor_water"] and not wet["success"] and wet["success_ga"]
