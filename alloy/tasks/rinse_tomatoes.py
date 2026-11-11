# SPDX-License-Identifier: Apache-2.0
"""task rinse_tomatoes: Gym-Anything's rinse-tomatoes, a surface-flow rinsing task (WP8).

"Rinse the cherry tomatoes in the colander under the faucet, drain them, and set the colander in the clean bowl"
(``task_set/catalogue.py``). The scene (``rinse_tomatoes_scene.PrepSink``) is the prep sink with its drain open, the
colander of 15 dusty tomatoes on the drainboard, and the work table with the cutting board and the clean bowl. Each
tomato's film is one lump (WP9's lumped films): the faucet's stream on a tomato (DC5's LUMP hits) erodes its dust by
the soil laws (jet impact and the sheet's shear), the removed dust rides in its water, and what it sheds goes into the
colander's cavity (it lies inside the colander), which drains through its holes (DC5's drips) into the sink and its
open drain. Tomatoes float in what the colander holds (DC5's floats).

Start dust (``system.py:403-412``, ``params.py:149-151``, ``layout.py:504``): on each tomato 60 % of its 2 mm cells
carry 0.5e-3 x load x U(0.3, 1) x 2 kg/m² (load 1 at seed 0). A lump starts at its cells' mean (about 0.39 g/m², 14.4
mg for 15 tomatoes); the score's start gauge is its worst 1 cm bin (``sphere_bins``: about 2.3x the mean, as theirs).
Dust: σ 4 Pa dry, 0.5 Pa wet, soaking in 2 s at 20 °C; the removal constants are WP9's (Gym-Anything's). Policy:
``rinse_tomatoes_policy``; verdict: ``rinse_tomatoes_verdict``, on their v1 contract (``task_set``: evaluator and
scoring; their default for a bare rinse-tomatoes is the legacy one, ``task_set/catalogue.py:84-85``).

Differences from theirs, disclosed: a tomato's gauge (``rinse_tomatoes_verdict``); settling is a fixed 2 s after the
release plus the policy's 2 s, every gate and term read once at the end (theirs settles at least 2 s, then until quiet
up to 30 s, reads the work terms before it and the gates after, ``task.py:849-880``, ``task_set/evaluator.py:110-133``);
their colander's cavity base sits about 2 mm inside this one's (their floor fillet, ``ware.py:160``).

Run: ``python -m alloy.tasks.rinse_tomatoes [--worlds W] [--device cuda:0] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow import SurfaceFlow
from alloy.plugins.surface_flow.drops import DropsConfig
from alloy.plugins.surface_flow.lumped_specs import Lumped
from alloy.plugins.surface_flow.process_specs import Soil
from alloy.tasks.rinse_books import REL, Books
from alloy.tasks.rinse_hand import OverflowWatch, PoseHand
from alloy.tasks.rinse_score import OUTLET_OFF_GPM
from alloy.tasks.rinse_sink import GPM
from alloy.tasks.rinse_plates import gauge_bins
from alloy.tasks.rinse_sense import VesselCells, free_water_ml
from alloy.tasks.rinse_tomatoes_policy import ColanderRunner, aim, flush_and_drain, to_bowl, to_faucet
from alloy.tasks.rinse_tomatoes_scene import SKIN_HOLD, STEEL, TOMATO_R, TOMATOES, PrepSink
from alloy.tasks.rinse_tomatoes_verdict import Frustum, TomatoesEnd, verdict
from alloy.tasks.rinse_wash_station import CONTACTS

log = logging.getLogger("alloy.tasks.rinse_tomatoes")

DUST = dict(sigma_dry=4.0, sigma_wet=0.5, t_soak=2.0)     # params.py:149-151 ("estimate")
BOARD_DIRT = 2.0                      # their board reads soil + suspended + tracer: the dust its water carries, twice
DUST_CELL = 2.0e-3                    # m, their tomato chart's cells
BIN = 0.01                            # m, their gauge's bin


@dataclass(frozen=True)
class Tomatoes:
    """Task parameters. ``clean_to`` [kg/m²]: the aim moves on from a tomato under this load (0.1 µg/cm²)."""

    scene_: PrepSink = field(default_factory=PrepSink)
    seed: int = 0
    temperature: float = 20.0         # °C, the faucet's mix
    clean_to: float = 1.0e-6
    spray_max: float = 60.0
    flush_s: float = 3.0
    drain_s: float = 4.0
    tilt_s: float = 3.0
    pace: float = 1.0
    settle: float = 2.0
    control_dt: float = 0.02
    dt: float = 2.5e-3

    def plugins(self) -> tuple[SurfaceFlow, PoseHand]:
        st = self.scene_
        soil = Soil(initial=0.0, temperature=self.temperature, **DUST)
        lumps = tuple(Lumped(body=f"tomato{k}", hold=SKIN_HOLD, soil=soil) for k in range(TOMATOES))
        flow = SurfaceFlow(charts=st.charts(), material="water", network=st.network(), nozzles=st.nozzles(),
                           drains=st.drains(), species=("soil",), lumped=lumps, drops=DropsConfig())
        return flow, PoseHand(bodies=("colander",))

    def scene(self, worlds: int) -> TestScene:
        return TestScene(build=self.scene_.build, worlds=worlds, ground=True, control_dt=self.control_dt,
                         dt=self.dt, solver_kwargs=CONTACTS)


def start_dust(task: Tomatoes, worlds: int, area: float) -> tuple[np.ndarray, np.ndarray]:
    """Each tomato's start dust [kg/m²] from their recipe on 2 mm cells (module docstring; world 0 at load 1, others
    at 1 ± 0.3): ``[W, 15]`` its mean (the lump's load) and its worst 1 cm bin (25 cells; the score's start gauge)."""
    rng = np.random.default_rng(task.seed)
    n = max(int(round(area / DUST_CELL ** 2)), 1)
    load = np.where(np.arange(worlds) == 0, 1.0, rng.uniform(0.7, 1.3, worlds))
    cells = (rng.random((worlds, TOMATOES, n)) < 0.6) * rng.uniform(0.3, 1.0, (worlds, TOMATOES, n))
    cells *= 0.5e-3 * 2.0 * load[:, None, None]
    onehot = sphere_bins(n)
    bins = cells @ onehot / onehot.sum(axis=0)
    return cells.mean(axis=2), bins.max(axis=2)


def sphere_bins(n: int) -> np.ndarray:
    """``[n, bins]`` one-hot: ``n`` equal cells spread over a tomato's sphere (a Fibonacci lattice) in their gauge's
    1 cm bins, cells shifted 5 mm along their normal, in the body frame (``task.py:74-88``): edge bins hold few
    cells."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    r = np.sqrt(1.0 - z * z)
    p = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1) * (TOMATO_R + 0.5 * BIN)
    _, inv = np.unique(np.floor(p / BIN).astype(np.int64), axis=0, return_inverse=True)
    return np.eye(int(inv.max()) + 1)[inv.ravel()]


@dataclass
class TomatoesResult:
    worlds: int
    success: list[bool]
    score: list[float | None]
    failed: list[list[str]]
    dust_mg0: list[float]
    dust_mg: list[float]
    worst_ug_cm2: list[float]
    spray_s: float
    free_water_ml: list[float]
    floor_ml: list[float]
    ledger_max: float
    seconds: float
    wall_s: float
    criteria: list[dict] = field(default_factory=list)
    water_ml: dict = field(default_factory=dict)      # world 0: where the water went (``water_book``)
    books: dict = field(default_factory=dict)         # each book's largest relative residual over the worlds
    judged: list[list[int]] = field(default_factory=list)      # per world: [judged, all] Gym-Anything criteria
    not_judged: list[list[str]] = field(default_factory=list)  # per world: the criteria this port does not judge
    summary: list[str] = field(default_factory=list)           # per world: "k of n Gym-Anything criteria judged ..."


def setup(task: Tomatoes, worlds: int, device: str) -> tuple[Any, SurfaceFlow, ColanderRunner, tuple]:
    """The simulation, its film, the runner (watching the tomatoes, then the colander, then the bowl), and the start
    dust (``start_dust``: means and worst bins, [W, 15] each)."""
    flow, hand = task.plugins()
    sim = build_simulation(task.scene(worlds), [flow, hand, OverflowWatch()], device,
                           graph=device.startswith("cuda"))
    for e in task.scene_.closed():
        flow.set_input(e, 0.0)
    labels = list(sim.model.body_label)
    per = len(labels) // worlds
    first = np.arange(worlds)[:, None] * per
    tomatoes = first + np.array([labels.index(f"tomato{k}") for k in range(TOMATOES)])[None, :]
    watch = np.concatenate([tomatoes, first + labels.index("colander"), first + labels.index("bowl")], axis=1)
    load = start_dust(task, worlds, float(flow.lumped.lump_area.numpy()[0]))
    flow.lumped.set_soil(load[0])
    run_ = ColanderRunner(sim, hand, flow, task.control_dt, watch)
    return sim, flow, run_, load


def run(task: Tomatoes | None = None, *, worlds: int = 1, device: str = "cpu") -> TomatoesResult:
    task = task or Tomatoes()
    wall = time.perf_counter()
    sim, flow, run_, load0 = setup(task, worlds, device)
    books = Books.start(flow)
    run_.hold(0.5)
    to_faucet(run_, task.scene_, task.pace)
    spray = aim(run_, run_.watch[:, :TOMATOES], task.clean_to, task.spray_max)
    flush_and_drain(run_, task.flush_s, task.drain_s, task.tilt_s, task.pace)
    to_bowl(run_, task.scene_, task.pace)
    run_.hold(task.settle)
    return _result(task, sim, flow, run_, (load0, books), spray, wall)


def _frame(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Points ``p`` [..., 3] (world) in the frame of pose ``q`` (x, y, z, qx, qy, qz, qw)."""
    u, w = -q[3:6], q[6]
    v = np.asarray(p, np.float64) - q[:3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _books(flow: SurfaceFlow, run_: ColanderRunner, books: Books, start_kg: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per world: ``rinse_books`` (the film's water and soil, the supply's cross ledger) and the lumps' water (to the
    water supplied) and soil (to the start load) closed to 1e-9, and no contact overflow; the largest residual."""
    res = books.residuals(flow)
    fr = flow.read()
    res["lump_water"] = np.abs(fr["lump_ledger"]).max(axis=1) / np.maximum(fr["supplied_volume"], 1e-30)
    res["lump_soil"] = np.abs(fr["lump_species_ledger"][..., 0]).max(axis=1) / np.maximum(start_kg, 1e-30)
    worst = np.max(np.stack(list(res.values())), axis=0)
    ok = (worst < REL) & np.isfinite(fr["h"]).all(axis=1) & ~run_.overflowed
    return ok, {k: float(v.max()) for k, v in res.items()}


def _end(task: Tomatoes, sim: Any, flow: SurfaceFlow, run_: ColanderRunner, w: int, books: bool,
         worst0: np.ndarray) -> TomatoesEnd:
    st, bq = task.scene_, sim.state.body_q.numpy()
    c, b = st.colander, st.bowl
    qc, qb = bq[run_.watch[w, TOMATOES]].astype(np.float64), bq[run_.watch[w, TOMATOES + 1]].astype(np.float64)
    col = Frustum(0.0, c.depth, float(c.radius(0.0)), float(c.radius(c.depth)))
    bowl = Frustum(0.0, b.depth, float(b.radius(0.0)), float(b.radius(b.depth)))
    base = qc[:3] + _rot(qc, np.array([0.0, 0.0, -(c.foot + c.bottom)]))      # their origin: the foot's underside
    net, soil = flow.supply.read(), flow.read()["lump_soil"][w]
    cav = float(net["amount"][w][[n.name for n in st.nodes()].index("colander_cavity")])
    faucet = float(net["discharge"][w][[e.name for e in st.edges()].index("faucet")])
    return TomatoesEnd(
        gauge0=worst0[w], gauge=tomato_gauge(soil, flow.lumped.lump_area.numpy()),
        kept=col.holds(_frame(qc, bq[run_.watch[w, :TOMATOES], :3])),
        placed=bool(bowl.holds(_frame(qb, base))), free_water_ml=free_water_ml(
            flow, VesselCells.of_vessel(flow, c, 0), 0, w, qc, STEEL["h_break"], cav),
        board_gauge=board_gauge(flow, w), bowl_gauge=soil_gauge(flow, 1, w),
        floor_ml=floor_ml(flow, w),
        lowest_z=float(bq[run_.watch[w, :TOMATOES + 1], 2].min()),
        faucet_open=abs(faucet) > OUTLET_OFF_GPM * GPM, held=bool(run_.grip[w].any()), books=books)


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[3:6], q[6]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def soil_gauge(flow: Any, chart: int, w: int) -> float:
    """World ``w``'s worst 1 cm gauge of the soil in chart ``chart``'s film [kg/m²] (the film's species are per area;
    their contamination gauge, ``task.py:618-642``; the board reads twice it: ``rinse_tomatoes_verdict``)."""
    return float(gauge_bins(flow, flow.sp.mass.numpy()[w:w + 1, :, 0], chart)[0].max())


def tomato_gauge(load: np.ndarray, area: np.ndarray) -> np.ndarray:
    """A tomato's conservative gauge [kg/m²]: its remaining dust (``load`` kg/m² over ``area`` m²) over 1 cm², which no
    1 cm² bin of theirs can exceed (``rinse_tomatoes_verdict``)."""
    return np.asarray(load) * np.asarray(area) / BIN ** 2


def board_gauge(flow: Any, w: int) -> float:
    """The cutting board's gauge [kg/m²]: their soil + suspended + tracer, twice the dust its water carries."""
    return BOARD_DIRT * soil_gauge(flow, 2, w)


def floor_ml(flow: SurfaceFlow, w: int) -> float:
    """Water on the floor [mL]: stream and drips that hit nothing or the ground, and drops that fell past the floor."""
    s = flow.supply.read()
    ground = [i for i, n in enumerate(flow.supply.occluders) if "ground" in n.lower()]
    caught = float(np.asarray(s["caught_by"])[w, ground].sum()) if ground else 0.0
    floored = float(flow.drops.read()["floored_volume"][w]) if flow.drops is not None else 0.0
    return (float(np.ravel(s["lost"])[w]) + caught + floored) * 1e6


def water_book(flow: SurfaceFlow, st: PrepSink, w: int) -> dict[str, Any]:
    """World ``w``'s water [mL]: the supply's ledger columns, the drop stage's books, what each shape caught, and
    each tank's water (non-zero entries)."""
    s = flow.supply.read()
    from alloy.plugins.surface_flow.supply import LEDGER  # noqa: PLC0415

    book = {"supply": {k: float(np.ravel(s[k])[w]) * 1e6 for k in LEDGER},
            "caught_by": dict(zip(flow.supply.occluders, (np.asarray(s["caught_by"])[w] * 1e6).tolist())),
            "tanks": dict(zip((n.name for n in st.nodes()), (np.asarray(s["amount"])[w] * 1e6).tolist()))}
    if flow.drops is not None:
        book["drops"] = {k: float(np.ravel(v)[w]) * 1e6 for k, v in flow.drops.read().items()
                         if k.endswith("_volume")}
    return {g: {k: round(v, 3) for k, v in d.items() if abs(v) > 1e-6} for g, d in book.items()}


def _result(task: Tomatoes, sim: Any, flow: SurfaceFlow, run_: ColanderRunner, start: tuple[np.ndarray, Books],
            spray: float, wall: float) -> TomatoesResult:
    (load0, worst0), books0 = start
    area = flow.lumped.lump_area.numpy()
    start_kg = (load0 * area[None, :]).sum(axis=1)
    books, rel = _books(flow, run_, books0, start_kg)
    W = len(load0)
    ends = [_end(task, sim, flow, run_, w, bool(books[w]), worst0) for w in range(W)]
    left = flow.read()["lump_soil"] * area[None, :]                       # [W, 15] kg of dust still on each
    vs = [verdict(e) for e in ends]
    res = TomatoesResult(
        worlds=W, success=[v.success for v in vs], score=[v.score for v in vs], failed=[v.failed() for v in vs],
        judged=[list(v.judged) for v in vs], not_judged=[v.not_judged() for v in vs],
        summary=[v.summary() for v in vs],
        dust_mg0=[round(float(k) * 1e6, 4) for k in start_kg],
        dust_mg=[round(abs(float(x)) * 1e6, 6) for x in left.sum(axis=1)],
        worst_ug_cm2=[round(float(e.gauge.max()) * 1e5, 4) for e in ends], spray_s=round(spray, 2),
        free_water_ml=[round(e.free_water_ml, 3) for e in ends], floor_ml=[round(e.floor_ml, 3) for e in ends],
        ledger_max=max(rel.values()), books=rel, seconds=run_.t, wall_s=time.perf_counter() - wall,
        criteria=[{k: str(c) for k, c in v.criteria.items()} for v in vs], water_ml=water_book(flow, task.scene_, 0))
    log.info("rinse_tomatoes: %d/%d worlds pass, %.1f s simulated (%.1f s spraying) in %.1f s wall",
             sum(res.success), W, res.seconds, spray, res.wall_s)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    res = run(Tomatoes(), worlds=args.worlds, device=args.device)
    text = json.dumps(asdict(res), indent=1)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(text)
    return 0 if all(res.success) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Tomatoes", "TomatoesResult", "board_gauge", "floor_ml", "run", "setup", "soil_gauge", "sphere_bins",
           "start_dust", "tomato_gauge"]
