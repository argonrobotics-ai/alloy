# SPDX-License-Identifier: Apache-2.0
"""task rinse_plates: Gym-Anything's pre-rinse-plates, the second surface-flow proof for rinsing.

"Pre-rinse six tomato-soiled plates with the spray valve until clean; plates in rack; valve hung up"
(``task_set/catalogue.py:34, 55-63``). The scene (``rinse_plates_scene.Layout``) is the peg rack on the pre-rinse
sink with six 9 in plates whose faces carry dried tomato paste (their ``water/system.py:380-388``: 0.20 kg/m² x
(1 + 0.25 N(0, 1)) inside a ragged well edge, at least 0.02 kg/m²), and the 9-jet spray valve on its balancer.
Each plate's face is a surface-flow chart; WP9's soil stage (``surface_flow`` ``process_specs.Soil``: soaking, jet
impact, jet-sheet and film shear, Wilson's cleaning front; Gym-Anything's constants) removes the paste into the film's
``soil`` species, which runs off to the sink; the valve's jets come from
DC5's supply (mains -> valve -> nine 0.6 mm orifices at 28 m/s; the plates' runoff drains into the sink tank).

The policy (a privileged scripted hand, ``rinse_hand.PoseHand``; no robot): for each plate, lift it out of its slot
along its lean, hold it upright at the spray station, raster the valve 0.20 m in front of its face at 0.30 m/s in
rows 2 cm apart (their calibration raster) until every world's plate reads clean (at most ``max_passes``), put it
back and let go; finally hang the valve at its park point (the balancer holds it) and let go.

Verdict (``task.py:734-742`` and the common gates ``task.py:677-692``): ``visibly_clean`` (every plate's worst 1 cm
soil gauge <= 1 µg/cm², the gauge as ``task.py:74-88``), ``valve_parked`` (<= 0.15 m from its park point),
``contained`` (floor water from the supply's ``lost`` ledger; the clean-table gauge is not judged), ``placed``
(``task.py:540-551``: in the rack, face normal within 30° of horizontal), ``water_off_and_hands_free`` (the
valve closed; the hands part is not judged),
``intact_and_not_dropped`` (no plate below 0.25 m; no plate in the rack faster than 1 m/s after a release, each
world's peak over every period of the release holds and the settle), ``no_simulation_fault`` (the film's water and
soil-species ledgers and the stage's soil ledger closed). Score ``task_set/scoring.py:103-111``.
Criteria with a part this port cannot read (the plate's centre disc, the clean table, contacts while held, the hands)
are None unless a judged part fails; ``Verdict.summary()`` says how many are judged (audit AB-gym-3).

The valve hangs on a balancer (``balancer``: a soft hold at the park point that carries its weight); while the hand
holds the valve the balancer lets go, so the weight is carried once. The valve sprays during raster rows only (closed
on the returns), and each plate's time to clean is read at the end of every row.

Limit: a revolution chart has no pole cell; the plate's chart starts at ``Plate.r_min`` (10.4 mm at 2 mm cells), so
the centre disc (3.4 of 247 cm²) carries no film and no soil and is not gauged.

Run: ``python -m alloy.tasks.rinse_plates [--plates N] [--worlds W] [--device cuda:0] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import SurfaceFlow
from alloy.plugins.surface_flow.process_specs import Soil
from alloy.tasks.rinse_hand import OverflowWatch, PoseHand, mjw_solver, overflow_watch, period_overflow
from alloy.tasks.rinse_plates_scene import RACK_Z, Layout, Pose, Step, move, raster, row_steps
from alloy.tasks.rinse_score import (FLOOR_WATER_ML, OUTLET_OFF_GPM, Verdict, finish, hands_gate, partly,
                                     reduction)
from alloy.tasks.rinse_sink import GPM

log = logging.getLogger("alloy.tasks.rinse_plates")

VRL = 1.0e-5                          # kg/m², the visible-residue limit 1 µg/cm² (their params.py:158)
GAUGE = 0.01                          # m, gauge bin (task.py:74-88)
PARK_M = 0.15
LEAN_MAX = math.radians(30.0)         # rack_lean_deg (params.py:254)
DROP_Z = 0.25                         # m, the dropped line (task.py:391-393)


@dataclass(frozen=True)
class PreRinse:
    """Task parameters. ``seed`` scales each world's soil load by 1 ± 0.3 (worlds >= 1, as their seeds > 0)."""

    layout: Layout = field(default_factory=Layout)
    plates: int = 6
    seed: int = 0
    temperature: float = 20.0         # °C of the spray (their t_soak reference)
    speed: float = 0.30               # m/s, raster
    spacing: float = 0.02             # m, raster rows
    half: float = 0.10                # m, raster half width
    max_passes: int = 5
    clean_margin: float = 0.5         # a pass ends the plate when its worst gauge is below margin x VRL
    pace: float = 1.0                 # scales the hand's move and hold times (tests run faster moves)
    settle: float = 2.0               # s, the verdict's settle
    control_dt: float = 0.02
    dt: float = 1.25e-3

    def plugins(self) -> tuple[SurfaceFlow, PoseHand, PoseHand]:
        """The film with WP9's soil stage, the hand and the valve's balancer."""
        lay, v = self.layout, self.layout.valve
        names = [f"plate{k}" for k in range(self.plates)]
        soil = Soil(initial=0.0, temperature=self.temperature)        # the jet's share: the supply's impact_share
        flow = SurfaceFlow(charts=tuple(lay.plate.chart(n) for n in names), material="water",
                           nozzles=v.nozzles("valve"), drains=v.drains(self.plates), network=v.network(),
                           species=("soil",), process={"soil": soil})
        hand = PoseHand(bodies=(*names, "valve"))
        balancer = PoseHand(bodies=("valve",), kp=30.0, kd=8.0, kr=0.5, dr=0.02, label="balancer")
        return flow, hand, balancer

    def scene(self, worlds: int) -> TestScene:
        return TestScene(build=lambda b: self.layout.build(b, self.plates), worlds=worlds, ground=False,
                         control_dt=self.control_dt, dt=self.dt)


def start_soil(task: PreRinse, flow: SurfaceFlow, worlds: int) -> np.ndarray:
    """``[W, C]`` paste load [kg/m²] per their ``water/system.py:380-388`` (a ragged edge of 6 % at 5 lobes)."""
    rng = np.random.default_rng(task.seed)
    load = np.zeros((worlds, flow.tables.cell_count))
    scale = np.where(np.arange(worlds) == 0, 1.0, rng.uniform(0.7, 1.3, worlds))
    for k in range(task.plates):
        xyz = flow.cell_centres(k).reshape(-1, 3)
        r, th = np.hypot(xyz[:, 0], xyz[:, 1]), np.arctan2(xyz[:, 1], xyz[:, 0])
        cells = flow.tables.cells_of(k)
        for w in range(worlds):
            edge = task.layout.plate.well_radius * (1.0 + 0.06 * np.sin(5.0 * th + rng.uniform(0.0, 2.0 * np.pi)))
            m = 0.20 * scale[w] * (1.0 + 0.25 * np.clip(rng.standard_normal(r.size), -2.0, 2.0))
            load[w, cells] = np.where(r <= edge, np.maximum(m, 0.02), 0.0)
    return load


def adhered(flow: SurfaceFlow) -> np.ndarray:
    """``[W, C]`` the soil stuck to the charts [kg/m²] (their gauge counts adhered soil only, ``task.py:495``)."""
    return flow.process.soil.numpy()


def gauge_bins(flow: SurfaceFlow, soil: np.ndarray, chart: int) -> tuple[np.ndarray, np.ndarray]:
    """``chart``'s 1 cm gauges: cells binned by floor(p/1 cm) after a 5 mm shift along their normal (body frame), each
    bin's area-weighted mean (their ``task.py:74-88``). Returns the gauges [W, bins] (kg/m²) and the bins' areas."""
    t = flow.tables
    cells = t.cells_of(chart)
    p = t.cpos.numpy()[cells] + 0.5 * GAUGE * t.cnrm.numpy()[cells]
    key = np.floor(p / GAUGE).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    area = t.area_np[cells]
    num = np.zeros((soil.shape[0], inv.max() + 1))
    den = np.bincount(inv.ravel(), weights=area, minlength=inv.max() + 1)
    for w in range(soil.shape[0]):
        num[w] = np.bincount(inv.ravel(), weights=soil[w, cells] * area, minlength=inv.max() + 1)
    return num / np.maximum(den, 1e-30), den


def gauges(flow: SurfaceFlow, soil: np.ndarray, chart: int) -> np.ndarray:
    """``[W]`` worst 1 cm gauge [kg/m²] of ``chart`` (``gauge_bins``)."""
    return gauge_bins(flow, soil, chart)[0].max(axis=1)


class Runner:
    """Steps the simulation while the hand follows moves; one world plan, every world in lockstep."""

    def __init__(self, task: PreRinse, sim: Any, flow: SurfaceFlow, hand: PoseHand, balancer: PoseHand) -> None:
        self.task, self.sim, self.flow, self.hand, self.balancer = task, sim, flow, hand, balancer
        n = len(hand.config.bodies)
        q = sim.state.body_q.numpy()[hand.held.numpy()]
        self.pos, self.quat = q[..., :3].copy(), q[..., 3:].copy()
        self.vel = np.zeros_like(self.pos)
        self.grip = np.zeros(self.pos.shape[:2], np.float32)
        self.valve = n - 1
        self.t = 0.0
        self.peak_speed = np.zeros(self.pos.shape[0])     # [W] fastest plate in the rack after a release
        self.valve_err = np.zeros(self.pos.shape[0])      # [W] the valve's largest error off its path at row ends [m]
        self.solver, self.watch = mjw_solver(sim), overflow_watch(sim)
        self.overflowed = np.zeros(self.pos.shape[0], bool)                       # [W] contacts dropped

    def pose(self, i: int) -> Pose:
        return Pose(self.pos[0, i].astype(np.float64), self.quat[0, i].astype(np.float64))

    def step(self, moves: dict[int, Step]) -> None:
        self.vel[:] = 0.0
        for i, s in moves.items():
            self.pos[:, i], self.quat[:, i], self.vel[:, i] = s.pose.pos, s.pose.quat, s.vel
        for name, value in (("target_pos", self.pos), ("target_quat", self.quat), ("target_vel", self.vel),
                            ("grip", self.grip)):
            self.hand.set_input(name, value.astype(np.float32))
        self.sim.step_period()
        self.t += self.task.control_dt
        over = period_overflow(self.watch, self.solver)
        if over is not None:
            self.overflowed |= over

    def go(self, i: int, target: Pose, seconds: float) -> None:
        for s in move(self.pose(i), target, seconds, self.task.control_dt):
            self.step({i: s})

    def hold(self, seconds: float, watch: bool = False) -> None:
        """Hold every target; ``watch`` tracks each world's fastest plate that is not held, every period."""
        plates = self.hand.held.numpy()[:, :self.task.plates]
        for _ in range(max(int(round(seconds / self.task.control_dt)), 1)):
            self.step({})
            if watch:
                v = np.linalg.norm(self.sim.state.body_qd.numpy()[plates, :3], axis=2)      # [W, plates]
                free = self.grip[:, :self.task.plates] <= 0.0
                self.peak_speed = np.maximum(self.peak_speed, np.where(free, v, 0.0).max(axis=1))

    def track_valve(self) -> None:
        """The valve origin's distance from its target across the raster (y, z): rows run along x, where the targets
        step once per period (a ±speed·dt/2 sawtooth that is not an error)."""
        b = self.hand.held.numpy()[:, self.valve]
        off = self.sim.state.body_q.numpy()[b, 1:3] - self.pos[:, self.valve, 1:3]
        self.valve_err = np.maximum(self.valve_err, np.linalg.norm(off, axis=1))

    def take(self, i: int) -> None:
        """Grip body ``i``; the balancer lets go of the valve while the hand holds it (its weight is carried once)."""
        self.grip[:, i] = 1.0
        if i == self.valve:
            self.balancer.set_input("grip", 0.0)

    def release(self, i: int) -> None:
        self.grip[:, i] = 0.0
        if i == self.valve:
            self.balancer.set_input("grip", 1.0)
        self.hold(0.5 * self.task.pace, watch=True)


def spray_pass(run: Runner, k: int, clean_s: np.ndarray, sprayed: float) -> float:
    """One raster over plate ``k`` with the valve open; at each row's end, worlds whose plate first reads clean get
    the spray time so far in ``clean_s`` [W]. Returns the spray time after the pass."""
    task = run.task
    row = row_steps(task.speed, task.half, task.control_dt)
    run.flow.set_input("valve", 1.0)
    for i, s in enumerate(raster(task.layout, task.speed, task.spacing, task.half, task.control_dt)):
        run.step({run.valve: s})
        sprayed += task.control_dt
        if (i + 1) % row == 0:
            run.track_valve()
            fresh = np.isnan(clean_s) & (gauges(run.flow, adhered(run.flow), k) <= VRL)
            clean_s[fresh] = sprayed
    run.flow.set_input("valve", 0.0)
    return sprayed


def rinse_plate(run: Runner, k: int, clean_s: np.ndarray) -> int:
    """Take plate ``k`` to the station, raster until clean, put it back; returns the passes sprayed and fills
    ``clean_s`` [W] with each world's spray time to clean (NaN: not clean)."""
    task, lay, p = run.task, run.task.layout, run.task.pace
    run.take(k)
    run.go(k, lay.lifted(k), 1.2 * p)
    run.go(k, lay.above_station(k), 1.2 * p)
    run.go(k, lay.at_station(), 0.8 * p)
    first = next(raster(lay, task.speed, task.spacing, task.half, task.control_dt)).pose
    run.take(run.valve)
    run.go(run.valve, first, 1.0 * p)
    passes, sprayed = 0, 0.0
    while passes < task.max_passes:
        sprayed = spray_pass(run, k, clean_s, sprayed)
        passes += 1
        if gauges(run.flow, adhered(run.flow), k).max() <= task.clean_margin * VRL:
            break
        run.go(run.valve, first, 0.5 * p)                          # valve closed on the return
    run.hold(0.5 * p)                                               # let the face drain into the sink
    run.go(k, lay.above_station(k), 0.8 * p)
    run.go(k, lay.lifted(k), 1.2 * p)
    run.go(k, lay.slot(k), 1.2 * p)
    run.release(k)
    return passes


@dataclass
class PlatesResult:
    worlds: int
    plates: int
    success: list[bool]
    score: list[float | None]
    failed: list[list[str]]
    gauge_ug_cm2: list[list[float]]          # [W][plate] worst gauge at the end
    start_gauge_ug_cm2: list[list[float]]
    passes: list[int]
    clean_after_s: list[list[float | None]]  # [W][plate] spray time when the worst gauge first read <= VRL (per row)
    release_speed: list[float]               # [W] fastest plate in the rack after a release [m/s]
    valve_track_mm: list[float]              # [W] the valve's largest error off its path (y, z) at row ends
    valve_park_m: list[float]
    plate_lean_deg: list[list[float]]
    soil_ledger_max: float
    film_ledger_max: float
    supply: dict[str, float]
    seconds: float
    wall_s: float
    removed_by: dict[str, float] = field(default_factory=dict)
    judged: list[list[int]] = field(default_factory=list)      # per world: [judged, all] Gym-Anything criteria
    not_judged: list[list[str]] = field(default_factory=list)  # per world: the criteria this port does not judge
    summary: list[str] = field(default_factory=list)           # per world: "k of n Gym-Anything criteria judged ..."


def placed(task: PreRinse, q: np.ndarray) -> tuple[bool, float]:
    """In the rack (rack frame: |x| < 0.24, |y| < 0.12, 0 < z < 0.20; task.py:540-551) and the face normal within 30°
    of horizontal; returns (placed, lean from vertical [deg])."""
    p, quat = q[:3], q[3:]
    x, y, z, w = quat
    n = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])   # body z in world
    ok = abs(p[0]) < 0.24 and abs(p[1]) < 0.12 and 0.0 < p[2] - RACK_Z < 0.20 and abs(n[2]) < math.sin(LEAN_MAX)
    return bool(ok), float(math.degrees(math.asin(min(1.0, abs(n[2])))))


@dataclass
class EndState:
    """One world at the verdict: plate poses ``q`` [plates, 7], the valve's distance from its park point, closed
    ledgers, the valve's state, the hand and the world's fastest plate in the rack after a release."""

    q: np.ndarray
    park_m: float
    books: bool
    valve_open: bool
    held: bool
    release_speed: float
    floor_ml: float | None = None     # water that reached the floor [mL], None when the run does not measure it


def verdict(task: PreRinse, g0: np.ndarray, g: np.ndarray, end: EndState) -> Verdict:
    """One world's gates (task.py:734-742, 677-692) and score (task_set/scoring.py:103-111); ``g0``, ``g`` its
    plates' worst gauges at the start and the end [kg/m²]. Criteria with a part this port cannot read are None unless
    a judged part fails (module docstring of ``rinse_score``)."""
    places = [placed(task, end.q[k]) for k in range(task.plates)]
    released = not end.valve_open and not end.held
    dropped = any(end.q[k][2] < DROP_Z for k in range(task.plates)) or end.release_speed > 1.0
    floor_ok = None if end.floor_ml is None else end.floor_ml <= FLOOR_WATER_ML
    v = Verdict(criteria={
        "visibly_clean": (partly(bool(np.all(g <= VRL)), None),
                          {"gauge_ug_cm2": [round(x * 1e5, 4) for x in g],
                           "centre_disc": "not judged: the plate's centre disc (3.4 of 247 cm2) is not charted"}),
        "valve_parked": (end.park_m <= PARK_M, end.park_m),
        "contained": (partly(floor_ok, None),
                      {"floor_mL": end.floor_ml,
                       "floor_read": "supply 'lost' ledger only (jets and drips that hit no chart or pool)",
                       "clean_table_gauge": "not judged: no clean-table chart in this port (Gym-Anything: "
                                            "<= 1 ug/cm2)"}),
        "placed": (all(p for p, _ in places), [round(a, 2) for _, a in places]),
        "water_off_and_hands_free": hands_gate(not end.valve_open, end.held, valve_open=end.valve_open),
        "intact_and_not_dropped": (partly(not dropped, None),
                                   {"release_speed": end.release_speed,
                                    "contacts_while_held": "not judged: only the released plates' body speed and "
                                                           "the drop height are read (Gym-Anything: every "
                                                           "contact's normal speed, held or not)"}),
        "no_simulation_fault": (end.books, "film and soil ledgers; no contact overflow"),
    })
    credit = [reduction(float(g0[k]), float(g[k]), VRL) for k in range(task.plates)]
    c = float(np.mean(credit))
    k = float(v.criteria["valve_parked"][0]) * float(released)
    units = sum(not p for p, _ in places) / task.plates
    v.terms = {"cleaned": c, "parked_released": k, "unplaced_units": units}
    return finish(v, (c + c * k) / 2.0 - units, end.books)


def run(task: PreRinse | None = None, *, worlds: int = 1, device: str = "cpu",
        graph: bool | None = None) -> PlatesResult:
    """Run the task in ``worlds`` worlds; ``graph`` (default: CUDA only) captures each control period."""
    task = task or PreRinse()
    graph = device.startswith("cuda") if graph is None else graph
    wall = time.perf_counter()
    flow, hand, balancer = task.plugins()
    sim = build_simulation(task.scene(worlds), [flow, hand, balancer, OverflowWatch()], device, graph=graph)
    flow.process.set_soil(start_soil(task, flow, worlds))
    balancer.set_input("grip", 1.0)
    flow.set_input("valve", 0.0)
    run_ = Runner(task, sim, flow, hand, balancer)
    run_.hold(1.0 * task.pace)                                        # plates settle in the rack
    g0 = np.stack([gauges(flow, adhered(flow), k) for k in range(task.plates)], axis=1)
    clean_s = np.full((worlds, task.plates), np.nan)
    passes = [rinse_plate(run_, k, clean_s[:, k]) for k in range(task.plates)]
    run_.take(run_.valve)
    run_.go(run_.valve, task.layout.valve_park(), 1.5 * task.pace)
    run_.release(run_.valve)
    run_.hold(task.settle, watch=True)                                # the verdict's settle
    return _result(task, sim, flow, run_, (g0, clean_s), passes, wall)


def ledgers(flow: SurfaceFlow) -> tuple[np.ndarray, np.ndarray]:
    """Per world: the film's water ledger relative to the water supplied, and the larger of the stage's soil ledger
    (adhered + removed = start) and the film's soil-species ledger, relative to the start load."""
    fr, sr = flow.read(), flow.process.read()
    water = np.abs(fr["ledger"]) / np.maximum(fr["supplied_volume"], 1e-30)
    start = np.maximum(sr["soil_start_mass"], 1e-30)
    stage = np.abs(sr["soil_mass"] + sr["soil_removed_mass"] - sr["soil_start_mass"]) / start
    carried = np.abs(fr["species_ledger"][:, 0]) / start
    return water, np.maximum(stage, carried)


def _result(task: PreRinse, sim: Any, flow: SurfaceFlow, run_: Runner,
            gauge0: tuple[np.ndarray, np.ndarray], passes: list[int], wall: float) -> PlatesResult:
    g0, clean_s = gauge0
    worlds = g0.shape[0]
    g = np.stack([gauges(flow, adhered(flow), k) for k in range(task.plates)], axis=1)
    fr, sr = flow.read(), flow.process.read()
    film_rel, soil_rel = ledgers(flow)
    books = (film_rel < 1e-9) & (soil_rel < 1e-9) & np.isfinite(fr["h"]).all(1) & ~run_.overflowed
    bq = sim.state.body_q.numpy()
    ids = run_.hand.held.numpy()
    park = task.layout.valve_park().pos
    outlet = np.abs(np.asarray(flow.supply.read()["discharge"])[:, 0]) > OUTLET_OFF_GPM * GPM   # the valve's edge
    lost_ml = np.ravel(flow.supply.read()["lost"]) * 1e6              # jet and drips that missed every chart and pool
    vs = []
    for w in range(worlds):
        end = EndState(q=bq[ids[w, :task.plates]], park_m=float(np.linalg.norm(bq[ids[w, -1], :3] - park)),
                       books=bool(books[w]), valve_open=bool(outlet[w]), held=bool(run_.grip[w].any()),
                       release_speed=float(run_.peak_speed[w]), floor_ml=float(lost_ml[w]))
        vs.append(verdict(task, g0[w], g[w], end))
    sup = {k[len("supply_"):]: float(v[0]) for k, v in fr.items() if k.startswith("supply_") and np.ndim(v) == 1}
    res = PlatesResult(worlds=worlds, plates=task.plates, success=[v.success for v in vs],
                       score=[v.score for v in vs], failed=[v.failed() for v in vs],
                       judged=[list(v.judged) for v in vs], not_judged=[v.not_judged() for v in vs],
                       summary=[v.summary() for v in vs],
                       gauge_ug_cm2=(g * 1e5).round(5).tolist(), start_gauge_ug_cm2=(g0 * 1e5).round(1).tolist(),
                       passes=passes, release_speed=run_.peak_speed.round(4).tolist(),
                       valve_track_mm=(run_.valve_err * 1e3).round(3).tolist(),
                       clean_after_s=[[None if np.isnan(x) else round(float(x), 2) for x in row] for row in clean_s],
                       valve_park_m=[float(v.criteria["valve_parked"][1]) for v in vs],
                       plate_lean_deg=[v.criteria["placed"][1] for v in vs],
                       soil_ledger_max=float(soil_rel.max()), film_ledger_max=float(film_rel.max()), supply=sup,
                       seconds=run_.t, wall_s=time.perf_counter() - wall,
                       removed_by=dict(zip(("impact", "shear", "front", "wipe"),
                                           sr["soil_removed_by_mass"][0].tolist())))
    log.info("rinse_plates: %d/%d worlds pass, %.1f s simulated in %.1f s wall", sum(res.success), worlds,
             res.seconds, res.wall_s)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plates", type=int, default=6)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run(PreRinse(plates=args.plates, seed=args.seed), worlds=args.worlds, device=args.device)
    text = json.dumps(asdict(res), indent=1)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(text)
    return 0 if all(res.success) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EndState", "PreRinse", "PlatesResult", "gauge_bins", "gauges", "run", "start_soil", "verdict"]
