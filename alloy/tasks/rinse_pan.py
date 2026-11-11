# SPDX-License-Identifier: Apache-2.0
"""task rinse_pan: Gym-Anything's scrub-rinse-pan.

"Scrub the egg off the pan, rinse it, sanitize it for 30 s, store it mouth down on the shelf; sponge back on the
ledge" (``task_set/catalogue.py:56``; ``task.py:714-733``). The scene is ``rinse_pan_scene.Station``: the pot sink as
pools, the 10 in pan in the wash full of wash water, the sponge (a ceiling ``Applicator``) on the ledge, the spray
valve parked over the rinse bowl, the shelf. The egg is WP9's soil stage with their egg constants
(``params.py:152-157``: σ_dry 200 kPa, σ_wet 300 Pa, t_soak 40 s at 20 °C, at the wash's 43 °C; Archard k_abr 1.2e-5 x 4
for the scour face, σ_ref 1 kPa), pre-soaked as theirs (``system.py:390-400``). Soap adsorbs on the aluminium (Γ_max
3e-7 kg/m²). The pan's
``PanTracker`` (their sanitizer clock) is fed every period (``rinse_sense``). Policy: ``rinse_pan_policy``; the rinse
bowl's dunks repeat while a pan is not visibly clean (``rinse``): a heavy egg load does not enter the sanitizer soiled.
Verdict: ``rinse_pan_verdict``; ``no_simulation_fault`` closes ``rinse_books`` (film water, species with the walls'
load, the supply's cross ledger), the soil's own sum, and no contact overflow in any substep.

Where it differs from theirs (no gate reads these unless said):
- settling: a fixed ``settle`` (2 s) with the trackers sampling; theirs settles at least 2 s, then until quiet (up to
  30 s) with the tracker frozen (``task.py:849-882``). A pan draining mouth down only loses film and soap with more
  time, so this reading is the stricter one;
- sampling: the tracker reads every control period (20 ms), theirs every 10 ms (``task.py:133``);
- worlds 1..W-1 vary only the egg load (x U(0.7, 1.3) and its per-cell spread). Theirs also vary the soak (U(20, 240)
  s), c_wash (x U(0.67, 1.67), which scales surf_ref), T_wash (43-49 °C) and the pan's and sponge's poses
  (``layout.py:433-472``): they are not their seeds;
- the sponge: one rigid 111 x 66 x 40 mm block of 80 g with the pad's face 5 mm under it, no pores. Theirs is
  111 x 66 x 17 mm, two bodies on a compression slide, 8 g, its pores 75 % full of wash solution (``layout.py:463``,
  ``ware.py:440-454``). Its ledge ceiling (55 mm) admits ours lying flat or on its side; on its short end ours stands
  55.5 mm and fails, theirs 49.5 mm and passes;
- no wash foam (WP15's ruling G7; theirs carries 2 cm, ``layout.py:440``);
- the spray valve hangs at its park over the rinse bowl's centre on a soft six-axis pose spring (30 N/m, 8 N·s/m),
  theirs on a hose balancer parked at the wash/rinse partition; it is not a hand;
- the pour rolls the pan about its handle's axis (no term reads how a pan is poured);
- the wash film a poured pan carries: 20.3 mg of soap on the faces 9 s in, theirs measured 13.5-13.7 mg
  (``task.py:58-62``); the surplus goes to the rinse bowl;
- solids: the soil's own sum closes (soil + removed = start); the removed soil is not traced into the film's species
  and the tanks as their whole-system solids ledger does, and the tanks keep their own books (FluidNetwork's laws);
- shed film: the pan's flow has no drop stage, so film that runs off the chart's open edge (the outer bottom's) leaves
  the books as ``spilled``; theirs drips as particles into the pool below (``water/engine.py:1807-1813``). The wash
  pour sheds about 120-170 mL and 90-125 mg of soap that never reach the wash bowl, each dump a few mL (0.04-3.3 mL,
  0.03-0.1 mg) that never reach the rinse bowl (DC5's probes on the coarse chart). The pan loses that film either
  way, so no gate moves.

Run: ``python -m alloy.tasks.rinse_pan [--worlds W] [--device cuda:0] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from collections.abc import Callable
from typing import Any

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import SurfaceFlow
from alloy.plugins.surface_flow.adsorption import Adsorption
from alloy.plugins.surface_flow.process_specs import Soil
from alloy.tasks.rinse_books import REL, Books
from alloy.tasks.rinse_hand import OverflowWatch, PoseHand
from alloy.tasks.rinse_pan_policy import TOOL_KP, PanRunner, lift_and_pour, scrub_floor, spray_band, to_shelf
from alloy.tasks.rinse_tracker import (VRL_KG_M2, PanTracker, clean_term, rinse_strength_kg_m3, rinsed_term,
                                       soap_limit_kg)
from alloy.tasks.rinse_pan_scene import SOAP, SPECIES, TEMPS, Station
from alloy.tasks.rinse_wash_station import CONTACTS
from alloy.tasks.rinse_pan_verdict import PanEnd, verdict
from alloy.tasks.rinse_plates import gauge_bins, gauges
from alloy.tasks.rinse_plates_scene import Pose
from alloy.tasks.rinse_score import OUTLET_OFF_GPM, Verdict
from alloy.tasks.rinse_sense import VesselSense, face_soap_kg, free_water_ml
from alloy.tasks.rinse_sink import GPM

log = logging.getLogger("alloy.tasks.rinse_pan")

# their egg: inner cells below 12 mm over the pan's foot, 4 mm under its inner floor (system.py:390-400,
# layout.py:471-472, ware.py:421-422): 8 mm over the inner floor here
EGG = dict(load=0.08, spread=0.3, floor=0.01, z_max=0.008, soaked_s=120.0)
H_BREAK = 6.0e-5                      # m, aluminium (params.py:106)
GAMMA_M, K_ADS, K_DES = 3.0e-7, 20.0, 0.05     # aluminium: kg/m², m³/kg, 1/s (params.py:106, 123-129)
V_WITHDRAW = 18.0e-6                  # m³, the water a poured pan takes away (task.py:183)
CLEAN_MARGIN = 0.5                    # of the visible-residue limit: the look before the sanitizer (``rinse``) asks for
                                      # this much clear of it, as the tracker's entry check reads the worst of the carry


@dataclass(frozen=True)
class PanTask:
    station: Station = field(default_factory=lambda: Station(weir=("wash", "rinse", "sanitize")))
    seed: int = 0
    press: float = 20.0               # N, the sponge hand's spring
    passes: int = 8                   # scrub passes per strip
    rinse_s: float = 18.0
    rinse_dunks: int = 2              # the walls hold their wash equilibrium and shed 5 %/s under the rinse water
                                      # (k_d): 2 x 18 s, each pour swapping the cavity's water
    extra_rinses: int = 2             # dunks more, one at a time, while a pan is not visibly clean (WP5's 100k pass,
                                      # 11-09: world 6, the heaviest egg of 16, entered the sanitizer soiled)
    sanitize_s: float = 34.0          # the clock needs 30 s under; the soak runs in 2 s swirls
    pace: float = 1.0
    settle: float = 2.0
    control_dt: float = 0.02
    dt: float = 2.5e-3

    def soil(self) -> Soil:
        return Soil(initial=0.0, charts=(0,), sigma_dry=2.0e5, sigma_wet=300.0, t_soak=40.0,
                    temperature=TEMPS["wash"], k_abr=4.0 * 1.2e-5)

    def plugins(self) -> tuple[SurfaceFlow, PoseHand, PoseHand, PoseHand]:
        st = self.station
        flow = SurfaceFlow(charts=st.charts(), material="water", network=st.network(), nozzles=st.nozzles(),
                           species=SPECIES, tools=st.tools(), process={"soil": self.soil()},
                           adsorption=Adsorption("soap", K=K_ADS, k_d=K_DES, capacity=GAMMA_M))
        pan = PoseHand(bodies=("pan",))
        tools = PoseHand(bodies=("sponge", "valve"), kp=TOOL_KP, kd=3.0, kr=0.3, dr=0.01, label="tool_hand")
        balancer = PoseHand(bodies=("valve",), kp=30.0, kd=8.0, kr=0.5, dr=0.02, label="balancer")   # as rinse_plates'
        return flow, pan, tools, balancer

    def scene(self, worlds: int) -> TestScene:
        return TestScene(build=self.station.build, worlds=worlds, ground=True, control_dt=self.control_dt,
                         dt=self.dt, solver_kwargs=CONTACTS)


def egg_band(inner: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``[cells]`` the egg's cells: the inner ones under ``EGG["z_max"]`` over the inner floor (``z``, the pan's
    frame)."""
    return np.asarray(inner, bool) & (np.asarray(z, np.float64) < EGG["z_max"])


def start_egg(task: PanTask, flow: SurfaceFlow, sense: VesselSense, worlds: int) -> None:
    """Egg on the inner cells below 8 mm: 0.08 kg/m² x (1 + 0.3 N(0, 1)) clipped at ±2 σ, at least 0.01; soaked
    120 s at the wash's temperature (their system.py:390-400; world 0 nominal, others scaled 0.7-1.3)."""
    rng = np.random.default_rng(task.seed)
    on = egg_band(sense.cells.inner, sense.cells.local[:, 2])
    load = np.zeros((worlds, flow.tables.cell_count))
    cells = flow.tables.cells_of(0)
    for w in range(worlds):
        scale = 1.0 if w == 0 else rng.uniform(0.7, 1.3)
        m = EGG["load"] * scale * (1.0 + EGG["spread"] * np.clip(rng.standard_normal(on.size), -2.0, 2.0))
        load[w, cells] = np.where(on, np.maximum(m, EGG["floor"]), 0.0)
    flow.process.set_soil(load, hyd=1.0 - math.exp(-EGG["soaked_s"] / task.soil().t_soak_at))


def soap_load(flow: SurfaceFlow, w: int) -> np.ndarray:
    """``[cells]`` the pan's soap per cell [kg/m²], film and walls (its 1 cm gauges; the total is
    ``rinse_sense.face_soap_kg``)."""
    cells = flow.tables.cells_of(0)
    return flow.sp.mass.numpy()[w, cells, 0] + flow.adsorption.load.numpy()[w, cells, 0]


def outlets_running(flow: SurfaceFlow, st: Station) -> Callable[[], np.ndarray]:
    """``[W]`` whether an outlet runs over 1e-4 gpm in each world (their outlet test, ``task.py:598-602``): the faucet,
    or the spray valve by its jets' total discharge. The water-off gate and the tracker's 5 mL rule read it."""
    names = [e.name for e in st.edges()]
    faucet, jets = names.index("faucet"), [k for k, n in enumerate(names) if n.startswith("jet")]
    off = OUTLET_OFF_GPM * GPM

    def running() -> np.ndarray:
        q = np.abs(flow.supply.read()["discharge"])
        return (q[:, faucet] > off) | (q[:, jets].sum(axis=1) > off)
    return running


def visibly_dirty(flow: SurfaceFlow) -> Callable[[], bool]:
    """Whether any world's pan still shows soil: its worst 1 cm gauge over ``CLEAN_MARGIN`` of the visible-residue
    limit (the gauge the tracker's sanitizer entry reads)."""
    return lambda: bool((gauges(flow, flow.process.soil.numpy(), 0) > CLEAN_MARGIN * VRL_KG_M2).any())


def rinse(task: PanTask, run: PanRunner, dirty: Callable[[], bool] | None = None) -> int:
    """The rinse bowl's dunks, each poured out: ``rinse_dunks``, then up to ``extra_rinses`` more while ``dirty()``
    (a look at the pan before the sanitizer: their tracker resets the clock of a pan that enters it soiled). Every
    world takes the same dunks. Returns the dunks made."""
    st, p, n = task.station, task.pace, 0
    while n < task.rinse_dunks or (dirty is not None and n < task.rinse_dunks + task.extra_rinses and dirty()):
        run.dunk(0, "rinse", st.pan.depth + 0.03, task.rinse_s, p, pour=False)
        lift_and_pour(run, st, "rinse", p)
        n += 1
    if n > task.rinse_dunks:
        log.info("rinse_pan: %d rinse dunks (%d more while a pan was not visibly clean)", n, n - task.rinse_dunks)
    return n


def policy(task: PanTask, run: PanRunner, dirty: Callable[[], bool] | None = None) -> None:
    st, p = task.station, task.pace
    run.grip[:, 0] = 1.0
    scrub_floor(run, st, task.press, task.passes)
    lift_and_pour(run, st, "wash", p)
    over = st.bowl_centre(1) + [0.0, 0.0, 0.254 + 0.25]
    run.go(0, Pose(over, run.pose(0).quat), 2.0 * p)
    spray_band(run, st)
    rinse(task, run, dirty)
    run.dunk(0, "sanitize", st.pan.depth + 0.03, task.sanitize_s, p, pour=False)
    lift_and_pour(run, st, "sanitize", p)
    to_shelf(run, st, p)


@dataclass
class PanResult:
    worlds: int
    success: list[bool]
    score: list[float | None]
    progress: list[float | None]
    failed: list[list[str]]
    gauge_ug_cm2: list[float]
    start_gauge_ug_cm2: list[float]
    egg_removed_mg: list[dict[str, float]]
    soap_ug: list[float]
    sanitized_s: list[float]
    free_water_ml: list[float]
    ledgers: dict[str, float]
    seconds: float
    wall_s: float
    criteria: list[dict] = field(default_factory=list)
    judged: list[list[int]] = field(default_factory=list)      # per world: [judged, all] Gym-Anything criteria
    not_judged: list[list[str]] = field(default_factory=list)  # per world: the criteria this port does not judge
    summary: list[str] = field(default_factory=list)           # per world: "k of n Gym-Anything criteria judged ..."


def setup(task: PanTask, worlds: int, device: str) -> tuple[Any, SurfaceFlow, PanRunner]:
    flow, pan, tools, balancer = task.plugins()
    sim = build_simulation(task.scene(worlds), [flow, pan, tools, balancer, OverflowWatch()], device,
                           graph=device.startswith("cuda"))
    balancer.set_input("grip", 1.0)                               # the valve hangs at its park (it fell into the rinse
    #                                                               bowl unheld, and the pan's dunk later threw it)
    st = task.station
    for e in st.closed():
        flow.set_input(e, 0.0)
    area = float(flow.tables.area_np[flow.tables.cells_of(0)].sum())
    sense = VesselSense(flow, st, 0, pan.held.numpy()[:, 0], rinse_strength_kg_m3(soap_limit_kg(area)))
    start_egg(task, flow, sense, worlds)
    c = SOAP                                                       # soaked 120 s in the wash: walls at equilibrium
    flow.set_adsorbed(GAMMA_M * K_ADS * c / (1.0 + K_ADS * c), "soap", chart=0)
    trackers = [[PanTracker(soap_limit=soap_limit_kg(area), dt=task.control_dt) for _ in range(worlds)]]
    run = PanRunner(sim, pan, st, [sense], trackers, task.control_dt, outlets=outlets_running(flow, st), tools=tools,
                    balancer=balancer)
    return sim, flow, run


def run(task: PanTask | None = None, *, worlds: int = 1, device: str = "cpu") -> PanResult:
    task = task or PanTask()
    wall = time.perf_counter()
    sim, flow, run_ = setup(task, worlds, device)
    g0 = gauge_bins(flow, flow.process.soil.numpy(), 0)
    books = Books.start(flow)                                      # after the walls' and the egg's start loads
    policy(task, run_, visibly_dirty(flow))
    run_.hold(task.settle)
    return _result(task, flow, run_, g0, books, wall)


@dataclass
class Readout:
    """One world's readings at the verdict (SI; poses x, y, z, qx, qy, qz, qw)."""

    soil_gauge: float                 # worst 1 cm adhered-soil gauge [kg/m²]
    faces: float                      # soap on the faces, film and walls [kg]
    soap_gauge: float                 # worst 1 cm soap gauge [kg/m²]
    area: float                       # the pan chart's area [m²]
    cav_v: float                      # the cavity's water [m³]
    cav_soap: float                   # its soap [kg]
    bowl_conc: float | None           # the soap strength of the bowl holding the pan [kg/m³]; None: not in one
    sanitized_s: float
    into_sanitizer: float
    free_water_ml: float
    pan_pose: np.ndarray
    sponge_pose: np.ndarray
    outlet_running: bool
    held: bool


def rinse_soap(r: Readout) -> float:
    """progress-v1's S [kg]: the faces plus the water a poured pan takes, the cavity's strength x min(its water,
    18 mL), or the bowl's strength x 18 mL when that is more while the pan is in a bowl (``task.py:349-358``)."""
    water = r.cav_soap / r.cav_v * min(r.cav_v, V_WITHDRAW) if r.cav_v > 1e-12 else 0.0
    if r.bowl_conc is not None:
        water = max(water, r.bowl_conc * V_WITHDRAW)
    return r.faces + water


def compose(st: Station, r: Readout, books: bool) -> tuple[PanEnd, float]:
    """The verdict's ``PanEnd`` and S from one world's readings: the pan's soap is its faces' and its cavity's, the
    sponge is read at its scour face, the tilt is the pan's axis from up."""
    x, y = float(r.pan_pose[3]), float(r.pan_pose[4])
    end = PanEnd(gauge=r.soil_gauge, soap=r.faces + r.cav_soap, soap_gauge=r.soap_gauge, area=r.area,
                 sanitized_s=r.sanitized_s, into_sanitizer=r.into_sanitizer, free_water_ml=r.free_water_ml,
                 pan_pos=np.asarray(r.pan_pose[:3], np.float64),
                 tilt_deg=math.degrees(math.acos(float(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))),
                 sponge_pos=st.scour_face(r.sponge_pose), outlet_running=r.outlet_running, held=r.held, books=books)
    return end, rinse_soap(r)


def pan_rinsed(end: PanEnd, s: float) -> float:
    """progress-v1's R for this pan: S against its own soap limit (``soap_limit_kg`` of its area: 16.4 µg; GA's
    16.5 µg is their pan's)."""
    return rinsed_term(s, soap_limit=soap_limit_kg(end.area))


def judge(st: Station, r: Readout, books: bool, clean: float) -> tuple[Verdict, PanEnd]:
    """One world's verdict from its readings: the nine gates, and progress-v1 with C = ``clean`` and this pan's R."""
    end, s = compose(st, r, books)
    return verdict(st.sink, end, clean, pan_rinsed(end, s)), end


def books_ok(closed: np.ndarray, soil_rel: np.ndarray, overflowed: np.ndarray) -> np.ndarray:
    """``[W]`` no_simulation_fault: every ``rinse_books`` book and the soil's own within ``REL``, and no substep's
    contacts overflowed."""
    return np.asarray(closed, bool) & (np.asarray(soil_rel) < REL) & ~np.asarray(overflowed, bool)


def _read(task: PanTask, flow: SurfaceFlow, run_: PanRunner, w: int) -> Readout:
    """World ``w``'s readings from the simulation."""
    st, sense, tr = task.station, run_.senses[0], run_.trackers[0][w]
    bq = run_.sim.state.body_q.numpy()
    q = bq[run_.hand.held.numpy()[w, 0]]
    net = flow.supply.read()
    sp = flow.supply.fluid.species.numpy()
    cav_v = float(net["amount"][w][sense.cav_node])
    pool = sense.sample(w, run_.t, False, bq, net).pool
    conc = None
    if pool is not None:
        k = [n.name for n in st.nodes()].index(pool)
        v = float(net["amount"][w][k])
        conc = float(sp[w, k, 0]) / v if v > 1e-12 else 0.0
    outlet = np.broadcast_to(np.asarray(run_.outlets(), bool), (len(run_.grip),))[w]
    return Readout(soil_gauge=float(gauge_bins(flow, flow.process.soil.numpy()[w:w + 1], 0)[0].max()),
                   faces=face_soap_kg(flow, sense.cells.area, 0, w),
                   soap_gauge=float(gauge_bins(flow, _full(flow, soap_load(flow, w)), 0)[0][0].max()),
                   area=float(sense.cells.area.sum()), cav_v=cav_v, cav_soap=float(sp[w, sense.cav_node, 0]),
                   bowl_conc=conc, sanitized_s=tr.sanitized_s, into_sanitizer=tr.into_sanitizer_kg,
                   free_water_ml=free_water_ml(flow, sense.cells, 0, w, q, H_BREAK, cav_v), pan_pose=q,
                   sponge_pose=bq[run_.tools.held.numpy()[w, 0]], outlet_running=bool(outlet),
                   held=bool(run_.grip[w].any() or run_.tool_grip[w].any()))


def _full(flow: SurfaceFlow, per: np.ndarray) -> np.ndarray:
    out = np.zeros((1, flow.tables.cell_count))
    out[0, flow.tables.cells_of(0)] = per
    return out


def _result(task: PanTask, flow: SurfaceFlow, run_: PanRunner, g0: tuple, books: Books, wall: float) -> PanResult:
    pr = flow.process.read()
    resid = books.residuals(flow)
    closed, _ = books.closed(flow)
    soil = np.abs(pr["soil_mass"] + pr["soil_removed_mass"] - pr["soil_start_mass"]) / pr["soil_start_mass"]
    ok = books_ok(closed, soil, run_.overflowed)
    worlds = len(ok)
    g1 = gauge_bins(flow, flow.process.soil.numpy(), 0)
    vs, ends = [], []
    for w in range(worlds):
        clean = clean_term(g0[0][w].tolist(), g1[0][w].tolist(), g0[1].tolist())
        v, end = judge(task.station, _read(task, flow, run_, w), bool(ok[w]), clean)
        vs.append(v)
        ends.append(end)
    names = ("impact", "shear", "front", "wipe")
    res = PanResult(worlds=worlds, success=[v.success for v in vs], score=[v.score for v in vs],
                    progress=[v.progress for v in vs], failed=[v.failed() for v in vs],
                    judged=[list(v.judged) for v in vs], not_judged=[v.not_judged() for v in vs],
                    summary=[v.summary() for v in vs],
                    gauge_ug_cm2=[round(e.gauge * 1e5, 4) for e in ends],
                    start_gauge_ug_cm2=[round(float(g0[0][w].max()) * 1e5, 1) for w in range(worlds)],
                    egg_removed_mg=[dict(zip(names, (pr["soil_removed_by_mass"][w] * 1e6).round(3).tolist()))
                                    for w in range(worlds)],
                    soap_ug=[round(e.soap * 1e9, 3) for e in ends], sanitized_s=[e.sanitized_s for e in ends],
                    free_water_ml=[round(e.free_water_ml, 3) for e in ends],
                    ledgers={**{k: float(v.max()) for k, v in resid.items()}, "soil": float(soil.max())},
                    seconds=run_.t, wall_s=time.perf_counter() - wall,
                    criteria=[{k: str(c) for k, c in v.criteria.items()} for v in vs])
    log.info("rinse_pan: %d/%d worlds pass, %.1f s simulated in %.1f s wall", sum(res.success), worlds, res.seconds,
             res.wall_s)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run(PanTask(), worlds=args.worlds, device=args.device)
    text = json.dumps(asdict(res), indent=1)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(text)
    return 0 if all(res.success) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLEAN_MARGIN", "EGG", "PanResult", "PanTask", "Readout", "books_ok", "compose", "egg_band", "judge",
           "outlets_running", "pan_rinsed", "policy", "rinse", "rinse_soap", "run", "soap_load", "start_egg",
           "visibly_dirty"]
