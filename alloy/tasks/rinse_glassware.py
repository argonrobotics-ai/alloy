# SPDX-License-Identifier: Apache-2.0
"""task rinse_glassware: Gym-Anything's rinse-glassware, the third surface-flow proof for rinsing.

"Rinse six soapy glasses, sanitize each for 30 s, set them mouth down in the rack" (``task_set/catalogue.py:15-20``).
The scene (``rinse_glass_scene.GlassStation``) is the pot sink as pools with six bar glasses on the soiled drainboard,
each wet with 25 µm of wash water, and the glass rack on the clean board. Every glass's inside is a cavity pool that
floods over its rim when dunked and spills over its rim when tipped (DC5); each glass has a ``PanTracker`` (their v1
glass tracker shares the pan's rules: ``task_set/sanitizer.py``) fed every control period (``rinse_sense``).

The policy (a privileged scripted hand, ``rinse_dunk.DunkRunner``; no robot): for each glass, lift it off the board,
dunk it mouth up in the rinse bowl (its inside floods, its soap film joins the bowl), swirl, lift and pour, twice
(``rinse_dunks``: the first pour leaves the inside wet with its cavity's soapy water, about 4 µg on a mixing glass,
which with its walls' 5.7 µg put it over the sanitizer's 10 µg carry-in limit; a second dunk in the same rinse bowl
dilutes that film to the bowl's own strength, which rises a little with each glass: DC5's note); dunk it the same way
in the sanitizer for ``sanitize_s``; pour; turn it mouth down over its rack compartment and set it down.
Verdict and score: ``rinse_glass_verdict``, on their v1 contract (``task_set``: evaluator, sanitizer, scoring); their
default for rinse-glassware is the legacy one (``task_set/catalogue.py:84-85``, its clock ``task.py:404-422``).

Differences from theirs, disclosed:
- settling: a fixed ``settle`` (2 s) with the trackers sampling; theirs settles at least 2 s, then until quiet (up to
  30 s) with the trackers frozen (``task.py:817-882``, ``task_set/evaluator.py:130-143``);
- floor water: what missed every chart and pool (the supply's ``lost``). Film spilled off a glass's open edge (its
  foot) is reported (``spilled_ml``) but not counted: the hand carries glasses only over the bowls, the boards and the
  rack, so it lands there, not on the floor;
- impact: the fastest body-centre speed of a released glass, every control period (theirs: the normal speed at each
  new contact that is not a finger pad, every tick, held or not, ``task.py:91-128``);
- the body origin is the inner floor's centre, 12 mm above their foot: the rack's x, y and the floor line (z < 0.25)
  read the origin (at most 4 mm at 20° tilt against the 25 mm compartment); the rack's z uses the foot.

Run: ``python -m alloy.tasks.rinse_glassware [--glasses N] [--worlds W] [--device cuda:0] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import SurfaceFlow
from alloy.tasks.rinse_dunk import DunkRunner
from alloy.tasks.rinse_glass_scene import RACK, GlassStation
from alloy.tasks.rinse_glass_verdict import GlassEnd, verdict
from alloy.tasks.rinse_books import Books
from alloy.tasks.rinse_hand import OverflowWatch, PoseHand
from alloy.tasks.rinse_score import OUTLET_OFF_GPM
from alloy.tasks.rinse_sink import GPM
from alloy.tasks.rinse_tracker import PanTracker
from alloy.tasks.rinse_plates import gauges
from alloy.tasks.rinse_sense import VesselSense, face_soap_kg
from alloy.tasks.rinse_wash_station import CONTACTS, SPECIES

log = logging.getLogger("alloy.tasks.rinse_glassware")

SOAP_LIMIT = 10.0e-9                  # kg per glass (params.py:251)
C_RINSE = SOAP_LIMIT / 2.0e-6         # kg/m³: 10 µg in 2 mL (sanitizer.py:25-26)
H_RET = 50.0e-6                       # m: glass clings nothing above 50 µm (h_break 0, task.py:507-523)
CELLS = (11, 12, 13, 16, 17, 18)      # rack compartments used, nearest the sink first


@dataclass(frozen=True)
class Glassware:
    """Task parameters; ``station.weir`` names the bowls a dunked glass exchanges with (DC5's tuple from 11-08)."""

    station: GlassStation = field(default_factory=lambda: GlassStation(weir=("wash", "rinse", "sanitize")))
    rinse_s: float = 4.0
    rinse_dunks: int = 2
    sanitize_s: float = 34.0          # the clock needs 30 s under; the soak runs in 2 s swirls (31 s gave 30.02 s)
    pace: float = 1.0
    settle: float = 2.0
    control_dt: float = 0.02
    dt: float = 2.5e-3

    def plugins(self) -> tuple[SurfaceFlow, PoseHand]:
        st = self.station
        flow = SurfaceFlow(charts=st.charts(), material="water", network=st.network(), nozzles=st.nozzles(),
                           species=SPECIES, adsorption=st.adsorption())
        return flow, PoseHand(bodies=tuple(i.name for i in st.items()))

    def scene(self, worlds: int) -> TestScene:
        return TestScene(build=self.station.build, worlds=worlds, ground=True, control_dt=self.control_dt,
                         dt=self.dt, solver_kwargs=CONTACTS)


def soap_ug(flow: SurfaceFlow, sense: VesselSense, w: int) -> float:
    """Soap on a glass [µg]: its faces' film and walls (adsorbed) plus its cavity's water (their item_surfactant,
    ``task.py:440-452``)."""
    faces = face_soap_kg(flow, sense.cells.area, sense.k, w)
    return (faces + float(flow.supply.fluid.species.numpy()[w, sense.cav_node, 0])) * 1e9


def free_water_ml(flow: SurfaceFlow, sense: VesselSense, w: int, q: np.ndarray) -> float:
    """Cavity water plus the inner film above what clings (50 µm on glass), as their free_film (task.py:507-523)."""
    h = flow.h.numpy()[w, flow.tables.cells_of(sense.k)]
    film = float((np.maximum(h - H_RET, 0.0) * sense.cells.area)[sense.cells.inner].sum())
    return (film + float(flow.supply.read()["amount"][w][sense.cav_node])) * 1e6


def rinse_glass(run: DunkRunner, task: Glassware, k: int) -> None:
    """Glass ``k``: ``rinse_dunks`` rinse dunks, a sanitizer dunk, mouth down into its compartment."""
    st, g = task.station, task.station.items()[k].vessel
    run.grip[:, k] = 1.0
    depth = g.depth + 0.017                                       # the rim 17 mm under the surface
    for _ in range(task.rinse_dunks):
        run.dunk(k, "rinse", depth, task.rinse_s, task.pace)
    run.dunk(k, "sanitize", depth, task.sanitize_s, task.pace)
    rest = st.rack_centre()[2] + RACK["floor"] + g.height - g.bottom
    run.place_inverted(k, st.cell_centres()[CELLS[k]], rest, task.pace)


@dataclass
class GlassResult:
    worlds: int
    glasses: int
    success: list[bool]
    score: list[float | None]
    failed: list[list[str]]
    soap_ug: list[list[float]]
    sanitized_s: list[list[float]]
    free_water_ml: list[list[float]]
    soap_gauge_mg_m2: list[list[float]]
    ledger_max: float
    seconds: float
    wall_s: float
    criteria: list[dict] = field(default_factory=list)
    spilled_ml: list[float] = field(default_factory=list)   # film off the glasses' open edges (module docstring)
    judged: list[list[int]] = field(default_factory=list)      # per world: [judged, all] Gym-Anything criteria
    not_judged: list[list[str]] = field(default_factory=list)  # per world: the criteria this port does not judge
    summary: list[str] = field(default_factory=list)           # per world: "k of n Gym-Anything criteria judged ..."


def end_of(task: Glassware, run: DunkRunner, k: int, w: int, soap0: float) -> GlassEnd:
    st, sense, tr = task.station, run.senses[k], run.trackers[k][w]
    g, f = st.items()[k].vessel, sense.flow
    q = run.sim.state.body_q.numpy()[run.hand.held.numpy()[w, k]]
    x, y, z, qw = q[3:]
    axis = np.array([2 * (x * z + qw * y), 2 * (y * z - qw * x), 1 - 2 * (x * x + y * y)])
    soap = (f.sp.mass.numpy() + f.adsorption.load.numpy())[w:w + 1, :, 0]   # film + adsorbed (task.py:489)
    base = st.rack_centre()
    return GlassEnd(soap_ug0=soap0, soap_ug=soap_ug(f, sense, w), soap_gauge=float(gauges(f, soap, k)[0]),
                    sanitized_s=tr.sanitized_s, into_ug=tr.into_sanitizer_kg * 1e9,
                    free_water_ml=free_water_ml(f, sense, w, q), rack_xyz=q[:3] - base,
                    foot_up=float(q[2] - g.bottom * axis[2] - base[2]), height=g.height,
                    tilt_deg=float(math.degrees(math.acos(np.clip(axis[2], -1.0, 1.0)))),
                    peak_impact=float(run.peak_speed[w, k]), z=float(q[2]))


def run(task: Glassware | None = None, *, worlds: int = 1, device: str = "cpu") -> GlassResult:
    task = task or Glassware()
    wall = time.perf_counter()
    flow, hand = task.plugins()
    sim = build_simulation(task.scene(worlds), [flow, hand, OverflowWatch()], device,
                           graph=device.startswith("cuda"))
    st, n = task.station, len(task.station.items())
    for e in st.closed():
        flow.set_input(e, 0.0)
    st.wet(flow)
    books = Books.start(flow)
    senses = [VesselSense(flow, st, k, hand.held.numpy()[:, k], C_RINSE) for k in range(n)]
    trackers = [[PanTracker(soap_limit=SOAP_LIMIT, dt=task.control_dt) for _ in range(worlds)] for _ in range(n)]
    run_ = DunkRunner(sim, hand, st, senses, trackers, task.control_dt, outlets=outlet_running(flow, st))
    soap0 = [[soap_ug(flow, s, w) for s in senses] for w in range(worlds)]
    run_.sample()                                                 # their start sample (task_set/evaluator.py:31-33)
    for k in range(n):
        rinse_glass(run_, task, k)
    run_.hold(task.settle)
    return _result(task, run_, flow, soap0, books, worlds, wall)


def outlet_running(flow: SurfaceFlow, st: GlassStation) -> Callable[[], np.ndarray]:
    """``[W]`` whether each world's faucet runs over 1e-4 gpm (their outlet test, ``task.py:601``): the tracker's 5 mL
    outlet rule (``sanitizer.py:123-134``) reads it. The scripted policy never opens it."""
    k = [e.name for e in st.edges()].index("faucet")
    return lambda: np.abs(flow.supply.read()["discharge"][:, k]) > OUTLET_OFF_GPM * GPM


def _result(task: Glassware, run_: DunkRunner, flow: SurfaceFlow, soap0: list, books: Books, worlds: int,
            wall: float) -> GlassResult:
    n = len(task.station.items())
    closed, worst = books.closed(flow)
    sup = flow.supply.read()
    lost = np.ravel(sup["lost"]) * 1e6                                    # mL that missed every chart and pool
    gripping = bool(run_.grip.any())
    outlet = np.broadcast_to(np.asarray(outlet_running(flow, task.station)(), bool), (worlds,))
    vs, ends = [], []
    for w in range(worlds):
        e = [end_of(task, run_, k, w, soap0[w][k]) for k in range(n)]
        ends.append(e)
        cells = task.station.cell_centres() - task.station.rack_centre()[:2]
        vs.append(verdict(e, cells, RACK["floor"], float(lost[w]), not outlet[w], gripping,
                          bool(closed[w]) and not run_.overflowed[w]))
    res = GlassResult(worlds=worlds, glasses=n, success=[v.success for v in vs], score=[v.score for v in vs],
                      failed=[v.failed() for v in vs], judged=[list(v.judged) for v in vs],
                      summary=[v.summary() for v in vs], not_judged=[v.not_judged() for v in vs],
                      soap_ug=[[round(g.soap_ug, 3) for g in e] for e in ends],
                      sanitized_s=[[round(g.sanitized_s, 2) for g in e] for e in ends],
                      free_water_ml=[[round(g.free_water_ml, 3) for g in e] for e in ends],
                      soap_gauge_mg_m2=[[round(g.soap_gauge * 1e6, 4) for g in e] for e in ends],
                      ledger_max=worst, seconds=run_.t, wall_s=time.perf_counter() - wall,
                      criteria=[{k: str(c) for k, c in v.criteria.items()} for v in vs],
                      spilled_ml=[round(float(x) * 1e6, 3) for x in flow.read()["spilled_volume"]])
    log.info("rinse_glassware: %d/%d worlds pass, %.1f s simulated in %.1f s wall", sum(res.success), worlds,
             res.seconds, res.wall_s)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glasses", type=int, default=6)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    task = Glassware(station=GlassStation(count=args.glasses, weir=("wash", "rinse", "sanitize")))
    res = run(task, worlds=args.worlds, device=args.device)
    text = json.dumps(asdict(res), indent=1)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(text)
    return 0 if all(res.success) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Glassware", "GlassResult", "rinse_glass", "run"]
