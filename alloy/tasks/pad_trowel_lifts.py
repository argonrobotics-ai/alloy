# SPDX-License-Identifier: Apache-2.0
"""pad-trowel's margin lifts (WP15's rulings, 2026-11-08), and the run: between the steel trowel's lane passes, the
margin trowel lifts the paste the trowel planed and carries it off the pad, into the tub.

Why: the whole pad's 16-world L4 row (reconcile 2026-11-08T0000Z) failed their ride rule in every world (170 N against
the 20 N hold). Held 0.25 mm under the pad's mean surface (95 % of the core within the 0.3 mm bearing band), the
trowel planes about 279 cm³ and the roll fills about 22 cm³ of lows. The other ~258 cm³ (on seed 0's start) has to go
somewhere: the side spill's beads (capped 1 mm over the paste ahead) and, mostly, the lane ends' stops and end spill
bands, where the bands' caps ratcheted about 1 mm a pass and the cross runs met 100-171 N.
What (v2, after the v1 L4 check; [internal note] notes/wp4-trowel/l4an/):
- After lane pass ``FROM_PASS`` and every lane pass after it that another pass follows, the trowel waits lifted at
  its last stop while the margin trowel runs the strips ``pad_trowel_strips`` lays out: the lane lines over each
  end's stop, its roll toe and its end spill bands; after the last lane pass also the bead lines over the side beads
  the first cross pass would meet. Each line is pad-finish's two runs. No lift follows the final pass, and no strip
  meets a core cell the steel trowel does not work again at its level (rule 1: the knife leaves its own texture).
- The blade is pad_finish_margin's: WP15's scraping store at their 0.15 L carry cap (``concrete_tub.store``). Its
  release goes into the tub network's pool (``concrete_tub``), and its books are what the film gave the tools
  (``to_tools``), what it let go, and its load left.
- Each line is planned as ``S_MAX`` stage slots of its two runs (``pad_trowel_stages``; WP15 2026-11-08 ~21:30Z). At
  the lift each world's surface is read: the paste's depth over the level along both runs sets S = ceil(d / 1.6 mm)
  stages stepping down to the level, each first run set down ``CLEAR`` over the highest paste under its start
  (``pad_finish_margin.touch``); the unused slots hang clear over the paste. The knife's keys are rewritten in place
  (``ToolCarriage.set_heights``). The plan's times, runs and ride hooks do not change, and it ends inside their
  trowel window (210 min paste age). A line's second run sets down at the plane on paste the first cut and slides
  through the first's set-down wedge (a loaded store fills the lows under a blade coming down); since WP15's ruling 2
  (2026-11-09) the stages are predicted with that fill (``pad_trowel_stages.knife_fill``), and the row books the blade's
  load at each read and lists every run that beats its prediction by more than ``UNDER_N``, with the blade's sole.
  v5 (ruling 3): every run is set down over what the runs before it leave there, that fill included, not the read.
- Their ride rule watches every margin run too (``RideWatch``, one hook per run), on the run as rewritten. A world
  fails when a lift carries more than the blade keeps lifted (a gob, ``tool_dropped``), or when the paste books do
  not close.
Disclosed:
- GA scripts none. Their report puts the margin trowel on the pad's cart ("the darby, floats, trowels, edger, margin
  trowel and the pocket penetrometer at the pad"; CARRY_CAP margin_trowel 0.00015). The lifts are Alloy's policy
  with their tool, as pad-finish's margin pass is.
- The scripted policy reads the paste at each lift to set the margin trowel down (privileged sight, like the
  carriage's).
- The last lane round's end spill leaves a 0.8-1 mm ridge rectangle over |x| < 0.22. Nothing re-trowels there, so it
  is not knifed. GA's criteria read it only through the straightedge (their own report keeps its excess piles as a
  stated limitation).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np

from alloy.plugins.errors import PluginError
from alloy.tasks import concrete_metrics as M
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_trowel as T
from alloy.tasks import pad_forms as PF
from alloy.tasks import pad_trowel_stages as ST
from alloy.tasks import pad_trowel_strips as S
from alloy.tasks import pad_window as PW
from alloy.tasks.concrete_pad_start import FORM_H, pad_fields
from alloy.tasks.concrete_patch import CARRY_CAP
from alloy.tasks.concrete_ports import PAD_TROWEL
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram
from alloy.tasks.concrete_verdict import counted
from alloy.tasks.pad_finish_metrics import remaining_bleed

FROM_PASS = 1                              # the first lane pass the lifts follow (v1: 2, and 2.4-2.7 mm toes by then)
SPEED = 0.2                                # m/s along a strip (the steel trowel's)
PASTE_TOL = 1e-9                           # the paste books' residuals, relative to the start's film (pad-finish's)
UNDER_N = 2.0                              # N: a margin run beating its prediction by more is listed (WP15's v4 book)
MARGIN = 1                                 # the margin trowel's index in the carriage's programs (the trowel's 0)
EDGE = S.EDGE
WINDOW_MIN = 210.0                         # their trowel window: the plan ends before this paste age [min] (WP15)
SetDown = ST.SetDown

Run = T.Run
log = logging.getLogger("alloy.tasks")


@dataclass(frozen=True)
class Plan:
    trowel: T.Plan
    margin: ToolProgram | None
    end: float
    runs: tuple[tuple[float, RD.SoleRun], ...]     # both tools' runs in time order
    tools: dict[int, str]                          # id(run) -> "trowel" | "margin"
    lifts: tuple[tuple[int, float, float], ...]    # (pass, the strips' start, their end)
    downs: tuple[SetDown, ...] = ()                # the lines (their stage slots), in time order
    margin_worlds: tuple[ToolProgram, ...] | None = None   # per world, set-downs planned up front (``touches``)
    touches: tuple[tuple[ST.LineRead, ...], ...] | None = None   # per world, each line's read


def lift_runs(task: T.PadTrowel, f: dict[str, Any], k: int, later: list[np.ndarray]) -> list[Run]:
    """The lift after pass k: none before ``FROM_PASS``, after a cross pass, or after the final pass; else the lane
    lines, and after the last lane pass the bead lines too (``pad_trowel_strips``)."""
    final = len(later) - 1
    if k < FROM_PASS or k >= task.passes or k >= final:
        return []
    runs = S.lane_lines(task, f, later[k])
    return runs + (S.bead_lines(task, f, later[k]) if k == task.passes - 1 else [])


def plan(task: T.PadTrowel, f: dict[str, Any], t0: float = 0.0,
         touches: list[list[ST.LineRead]] | None = None) -> Plan:
    """The trowel's passes (``pad_trowel.plan``) with, when ``task.lifts``, the margin trowel's lifts (``lift_runs``),
    the trowel waiting at its stop till the last strip is lifted. ``touches``: per world, every line's read
    (``pad_trowel_stages.LineRead``) planned up front (per-world programs, no rewrite at the lift; the same times and
    runs)."""
    if not task.lifts:
        p = T.plan(task, f, t0)
        return Plan(p, None, p.end, p.runs, {id(r): "trowel" for _, r in p.runs}, ())
    knife = ST.Knife(task, f, [Stroke(t0, TUB.home(f), mode=HEIGHT)], t0)
    later = S.steel_later(task, f)

    def after(k: int, level: float, t: float) -> float:
        runs = lift_runs(task, f, k, later)
        return t if not runs else knife.lift(k, runs, level, t + 0.3) + 0.3

    tp = T.plan(task, f, t0, after_pass=after)
    end = MG.fly(knife.keys, knife.t, TUB.home(f))
    runs = sorted([*tp.runs, *knife.timed], key=lambda e: e[0])
    tools = {id(r): "trowel" for _, r in tp.runs} | {id(r): "margin" for _, r in knife.timed}
    margin, worlds = ToolProgram(MG.BODY, tuple(knife.keys)), None
    if touches is not None:
        worlds = tuple(ST.with_reads(knife.keys, knife.downs, rs) for rs in touches)
    p = Plan(tp, margin, max(tp.end, end + 0.5), tuple(runs), tools, tuple(knife.lifts), tuple(knife.downs), worlds,
             None if touches is None else tuple(tuple(u) for u in touches))
    age = (float(f["age"]) + p.end) / 60.0
    log.info("pad_trowel: the plan ends at %.1f s, paste age %.1f min (their trowel window: %.0f)", p.end, age,
             WINDOW_MIN)
    if age >= WINDOW_MIN:
        raise PluginError(f"pad_trowel: the plan ends at paste age {age:.1f} min, past their trowel window "
                          f"({WINDOW_MIN:.0f} min)")
    if S.knife_last(f, p.runs, p.tools).any():
        raise PluginError("pad_trowel: a lift's strip meets a core cell no later trowel pass works at its level "
                          "(rule 1: the margin trowel leaves its own texture)")
    return p


class LiftWatch(T.RideWatch):
    """Their ride rule on both tools' runs (``RideWatch``), each log entry naming its tool, on each run as world w's
    tool makes it (``run_for``). With ``carriage``: at each SetDown's flight start, the height read on each world's
    surface (``pad_finish_margin.touch``) and written into the carriage's keys in place. Also the margin trowel's load
    before every run (its peak: a lift's carry), each margin run's need per world, the most each lift leaves in its
    strips per world, and world 0's surface at each lift's first strip and at the run after its last (``maps``).
    The blade's load at each line's read (``carry``), and with ``sole`` (per world, the blade's sole now [m over the
    form top]) its sole where each predicted margin run needs most (``sole_w`` in the log; WP15's ruling 2)."""

    def __init__(self, task: T.PadTrowel, fs: list[dict[str, Any]], flow: Any, p: Plan, carriage: Any = None,
                 sole: Callable[[], np.ndarray] | None = None) -> None:
        super().__init__(task, fs, flow, p.runs)
        self.p, self.carriage, self.tools, self.load = p, carriage, p.tools, np.zeros(len(fs))
        self.sole, self.live = sole, None                        # live: the hooks being stepped (samples join them)
        self.carry: dict[int, list[float]] = {}                  # id(line) -> the blade's load at its read [m³]
        self.down_t = {} if p.margin is None else {id(sr.sole): p.margin.keys[sr.keys[1]].t
                                                     for d in p.downs for slot in d.slots for sr in slot}
        self.reads: dict[int, list[ST.LineRead]] = {}    # id(line) -> its read per world
        self.single: dict[int, list[float]] = {}         # id(line) -> per world, its need as one pair at the level
        self.stage: dict[int, list[tuple[float, float]]] = {}   # id(planned sole) -> (level, touch) per world
        self.pred: dict[int, list[float | None]] = {}           # id(planned sole) -> its predicted need per world
        if p.touches is not None:
            for j, d in enumerate(p.downs):
                self._record(d, [ws[j] for ws in p.touches])
        firsts = {s for _, s, _ in p.lifts}
        self.snap = {t: "before" for t, _ in p.runs if any(abs(t - s) < 1e-9 for s in firsts)}
        self.after_lift: dict[float, int] = {}
        for i, (_, _, e) in enumerate(p.lifts):
            nxt = next((t for t, _ in p.runs if t > e + 1e-9), None)
            if nxt is not None:
                self.snap.setdefault(nxt, "after")
                self.after_lift[nxt] = i
        self.maps: list[tuple[float, str, np.ndarray]] = []
        self.left: list[list[float]] = [[] for _ in p.lifts]  # per lift: per world, the most left in its strips [m]

    def hooks(self) -> dict[int, Callable[[], None]]:
        hooks = self.live = super().hooks()
        if self.carriage is None:
            return hooks
        for d in self.p.downs:
            i = int(d.t / PR.PERIOD)
            hooks[i] = partial(_chain, partial(self.set_down, d), hooks.get(i))
        return hooks

    def margin_load(self) -> np.ndarray:
        """Per world, the margin blade's load now [m³] (``tool_load``, its last column)."""
        load = np.asarray(self.flow.read()["tool_load"])
        return load.reshape(load.shape[0], -1)[:, -1] if load.ndim > 1 else load

    def set_down(self, d: SetDown) -> None:
        """Read each world's surface now and set the line's stages there (``pad_trowel_stages.read_line``), in place."""
        h = PR.surface(self.flow, self.fs[0]["h"].shape)
        reads = [ST.read_line(d, h[w], f, T.tau_b(self.task, f["age"] + d.t)) for w, f in enumerate(self.fs)]
        for w, r in enumerate(reads):
            self.carriage.set_heights(MARGIN, w, ST.line_z(d, r)[0])
        self._record(d, reads)
        self.single[id(d)] = [self._single(d, h[w], f) for w, f in enumerate(self.fs)]
        self.carry[id(d)] = self.margin_load().tolist()

    def _single(self, d: SetDown, h: np.ndarray, f: dict[str, Any]) -> float:
        """The line's need [N] on this surface as one pair at its level (the v2 plan): the staging law's reference."""
        hh, worst, tau = np.array(h, dtype=float), 0.0, T.tau_b(self.task, f["age"] + d.t)
        for i, sr in enumerate(d.slots[-1]):
            up = MG.touch(sr.run, d.level, hh, f) if i == 0 else 0.0
            sole = dataclasses.replace(sr.sole, level=d.level, touch=up)
            worst = max(worst, RD.run_need(sole, hh, f["x"], f["y"], f["paste_mask"], tau, float(f["cell"]),
                                           h_out=hh).need_n)
        return worst

    def _record(self, d: SetDown, reads: list[ST.LineRead]) -> None:
        self.reads[id(d)] = reads
        for w, r in enumerate(reads):
            for k, v in ST.line_z(d, r)[1].items():
                self.stage.setdefault(k, [v] * len(reads))[w] = v
            for slot, pr in zip(d.slots, r.predicted or (None,) * len(d.slots), strict=True):
                for sr, need in zip(slot, pr or (None, None), strict=True):
                    self.pred.setdefault(id(sr.sole), [None] * len(reads))[w] = need

    def run_for(self, run: RD.SoleRun, world: int) -> RD.SoleRun:
        st = self.stage.get(id(run))
        return run if st is None else dataclasses.replace(run, level=st[world][0], touch=st[world][1])

    def check(self, t: float, run: RD.SoleRun) -> None:
        if self.flow.stores is not None:
            self.load = np.maximum(self.load, self.margin_load())
        if t in self.after_lift:
            self._left(self.after_lift[t])
        super().check(t, run)
        tool = self.tools.get(id(run), "trowel")
        self.log[-1]["tool"] = tool
        if tool == "margin":
            self.log[-1]["need_w"] = [r.need_n for r in self.last]
            if id(run) in self.stage:
                self.log[-1]["level_w"] = [lv for lv, _ in self.stage[id(run)]]
                self.log[-1]["touch_w"] = [u for _, u in self.stage[id(run)]]
            if id(run) in self.pred:
                self.log[-1]["pred_w"] = list(self.pred[id(run)])
                if any(v is not None for v in self.pred[id(run)]):
                    self._sample(self.log[-1], self.down_t.get(id(run)))
        if t in self.snap:
            h = PR.surface(self.flow, self.fs[0]["h"].shape)[0]
            self.maps.append((t, self.snap[t], (1e3 * h).astype(np.float16)))

    def _sample(self, e: dict, down: float | None) -> None:
        """Read the blade's sole into ``e`` when it reaches the run's worst point (set down, then at ``SPEED``)."""
        if self.sole is None or self.live is None or down is None:
            return
        i = int((down + e["at_m"] / ST.SPEED) / PR.PERIOD)

        def read() -> None:
            e["sole_w"] = np.asarray(self.sole()).tolist()
        self.live[i] = partial(_chain, read, self.live.get(i))

    def _left(self, i: int) -> None:
        """Lift i's strips (its margin runs' footprints, paste cells): the highest surface per world [m]."""
        _, s, e = self.p.lifts[i]
        f0 = self.fs[0]
        X, Y = np.meshgrid(f0["x"], f0["y"], indexing="ij")
        under = np.zeros(X.shape, bool)
        for t, r in self.p.runs:
            if self.tools.get(id(r)) == "margin" and s - 1e-9 <= t <= e:
                under |= S.footprint(r, X, Y)
        under &= np.asarray(f0["paste_mask"]) > 0.5
        h = PR.surface(self.flow, f0["h"].shape)
        self.left[i] = [float(h[w][under].max()) if under.any() else math.nan for w in range(len(self.fs))]


def _chain(first: Callable[[], None], then: Callable[[], None] | None) -> None:
    first()
    if then is not None:
        then()


@dataclass
class TrowelResult:
    verdicts: list[dict] | None            # their pad-trowel verdict per world (the whole pad, with dwc and bl)
    windows: list[dict] | None             # a corner window's scores per world
    success: int
    worlds: int
    material_s: float                      # material clock used (their advisory limit 90 min)
    sim_s: float
    wall_s: float
    ledger_rel: float
    faces_at_cap: int
    cells_over_depth: int
    ride_need_n: list[float]               # per world: the largest force their plough needs over a run (the sim's)
    pressed_m3: list[float]                # per world: what a held sole cut at a set-down (theirs: nothing)
    ride_planned_n: float                  # the plan's largest need on the start (refused over the hold)
    ride_worst: dict | None = None         # the run with the largest need on the sim's surface (all worlds)
    ride_log: list[dict] | None = None     # every run's tool, need and set-down cut on the sim's surface
    books: dict[str, list[float]] | None = None    # per world: the margin trowel's paste books (lifts)
    books_ok: list[bool] | None = None
    lifts: list[list[float]] | None = None         # (pass, strips start, end) per lift
    set_downs: list[dict] | None = None            # per line: when, where, its depth and stages run per world
    strips_left_m: list[list[float]] | None = None # per lift: per world, the highest paste left in its strips [m]
    middle_ridge_m: list[float] | None = None      # per world: the highest core paste at the end over |x| < 0.2235 m
    end_age_min: float | None = None               # the plan's end, paste age [min] (their window: WINDOW_MIN)
    ride_underpredict: dict | None = None          # the margin run whose need most beats its prediction (v4 book)
    ride_underpredicted: list[dict] | None = None  # every margin run beating its prediction by > UNDER_N, the blade's
                                                   # sole there (WP15's ruling 2: the row notes name them)
    forms_clearance: list[dict] | None = None      # per world: the forms proxy (pad_forms; >= 0 m: no touch)
    not_applicable: tuple[str, ...] = ("arms_clear",)   # no robot: a privileged carriage holds the tools
    ride_hold_n: float = T.HOLD
    cell_m: float = 0.002
    not_measured: tuple[str, ...] = PR.NOT_MEASURED


def _books(flow: Any, v0: np.ndarray, load: np.ndarray) -> tuple[dict[str, list[float]], np.ndarray]:
    """The margin trowel's books per world [m³]: what the film gave it, what it let go over the tub, its load left,
    gobs dropped, its peak carry; the film's and the store's ledgers (relative to the start's film). Closed: both
    ledgers within ``PASTE_TOL``, nothing dropped, nothing left on the blade, no carry over the cap."""
    r = flow.read()
    if flow.stores is None:
        return {}, np.ones(len(v0), bool)
    rel = np.maximum(np.asarray(v0, float), 1e-12)
    taken = np.asarray(r["to_tools_volume"], float)
    released = np.asarray(r["tool_released"], float).sum(axis=1)
    left, dropped = np.asarray(r["tool_load"], float).sum(axis=1), np.asarray(r["tool_dropped"], float).sum(axis=1)
    film = np.abs(np.asarray(r["ledger"], float) - v0) / rel
    store = np.abs(np.asarray(r["store_ledger"], float)) / rel
    ok = (film <= PASTE_TOL) & (store <= PASTE_TOL) & (dropped <= 0.0) & (left <= PASTE_TOL * rel) & (load <= CARRY_CAP)
    books = {"taken_m3": taken, "released_m3": released, "load_left_m3": left, "dropped_m3": dropped,
             "peak_carry_m3": load, "film_ledger_rel": film, "store_ledger_rel": store}
    return {k: np.asarray(v, float).tolist() for k, v in books.items()}, ok


@dataclass
class Sim:
    """A built and stepped pad-trowel run (``simulate``)."""

    task: T.PadTrowel
    fs: list[dict[str, Any]]
    p: Plan
    sim: Any
    carriage: Any
    flow: Any
    layer: Any
    watch: LiftWatch
    bodies: tuple
    v0: np.ndarray
    periods: int
    cap: int
    over: int
    wall: float


def simulate(task: T.PadTrowel, *, worlds: int = 1, device: str = "cpu", touches: list[list[float]] | None = None,
             until: float | None = None) -> Sim:
    """Build pad-trowel (world w: seed + w) and step it to the plan's end (or ``until`` [s]). The lifts' set-downs are
    read and rewritten at each lift, or planned up front from ``touches`` (per world, per SetDown)."""
    from alloy.plugins.harness.scene import TestScene, build_simulation            # noqa: PLC0415

    fs = [T.start(task, task.seed + w) for w in range(worlds)]
    p = plan(task, fs[0], t0=(T.WAIT_AT + 5) * PR.PERIOD, touches=touches)
    carriage, flow, layer = T.plugins(task, fs[0], p.trowel.program, p.margin, p.margin_worlds)
    bodies = (T.trowel_body(task),) + ((MG.body(fs[0]),) if p.margin is not None else ())
    sim = build_simulation(TestScene(build=PR.build_scene(fs[0], bodies), ground=False, worlds=worlds,
                                     dt=task.exchange_dt), [carriage, flow, layer], device)
    PR.set_start(flow, layer, fs, T.split(task))
    v0 = flow.read()["film_volume"]
    watch = LiftWatch(task, fs, flow, p, carriage if touches is None and p.downs else None,
                      sole=lambda: sim.state.body_q.numpy()[carriage.tool.numpy()[:, MARGIN], 2] - MG.HZ - FORM_H)
    t0 = time.perf_counter()
    periods = int(math.ceil((p.end if until is None else min(until, p.end)) / PR.PERIOD))
    cap, over = PR.step_watch(sim, flow, periods, hooks=watch.hooks())
    return Sim(task, fs, p, sim, carriage, flow, layer, watch, bodies, v0, periods, cap, over,
               time.perf_counter() - t0)


def run(task: T.PadTrowel | None = None, *, worlds: int = 1, device: str = "cpu", npz: str | None = None,
        maps: str | None = None) -> TrowelResult:
    """Their pad-trowel, scripted: the trowel's passes over the lanes and the cross lanes, the margin lifts between
    (``task.lifts``), both tools back on their carts; the verdict per world (world w: seed + w), or a window's
    scores; a world whose run their trowel would ride, or whose books do not close, fails. ``maps``: an npz of world
    0's surface [mm] at each lift (``LiftWatch.maps``)."""
    task = task or T.task()
    planned = max(T.ride_need(task, T.start(task, task.seed + w)) for w in range(worlds))
    if planned > T.HOLD:
        raise PluginError(f"pad_trowel: a run cuts what needs {planned:.1f} N to plough, over the {T.HOLD} N push "
                          f"holding the trowel (coupling.py:644-674); raise level0 or lower the step")
    r = simulate(task, worlds=worlds, device=device)
    fs, flow, layer, watch, p = r.fs, r.flow, r.layer, r.watch, r.p
    books, books_ok = _books(flow, r.v0, watch.load)
    age = PR.layer_fields(layer, fs[0]["h"].shape)["age"]
    ends = [_with_bleed(g, age[w], task.temperature) for w, g in enumerate(PR.grids(flow, layer, fs))]
    off = PR.tools_back(r.sim, r.carriage, r.bodies)
    pads = [M.PadScalars(tools_off=int(o), tools_total=len(r.bodies)) for o in off]
    rides, verdicts, windows = watch.ok() & books_ok, None, None
    halves = [T.TROWEL["half"], (MG.HA, MG.HC, MG.HZ)][:r.carriage._tools]       # programs: trowel, then margin
    forms = PF.clearance(r.carriage._table, r.carriage.n.numpy(), r.carriage._tools, halves, fs[0])
    if PR.whole(fs[0]):
        if not PW.has_water_states(ends[0]):
            raise PluginError("pad_trowel: the paste layer publishes no dwc/bl yet (WP6's concrete_paste); their "
                              "verdict reads both")
        starts = [_with_bleed(PR.start_grid(f), np.full(f["h"].shape, f["age"]), task.temperature) for f in fs]
        vs = [M.pad_trowel(g, q, s) for g, q, s in zip(ends, pads, starts, strict=True)]
        verdicts = [_proxied(PR.verdict_dict(v), c) for v, c in zip(vs, forms, strict=True)]
        success = sum(d["success"] and ok for d, ok in zip(verdicts, rides, strict=True))
    else:
        whole = [pad_fields(T.TASK, task.seed + w, task.temperature, task.air) for w in range(worlds)]
        reset = T.window_reset(task, fs[0], whole[0])
        ws = [PW.window_score(g, f, s, reset, q) for g, f, s, q in zip(ends, fs, whole, pads, strict=True)]
        windows = [_asdict(w) for w in ws]
        success = sum(PW.window_passed(w, int(o)) and ok for w, o, ok in zip(ws, off, rides, strict=True))
    _save(npz, maps, ends, fs, age, task, watch)
    sim_s = r.periods * PR.PERIOD
    downs = [dict(t=round(d.t, 2), start=d.runs[0][0], end=d.runs[1][1], level=d.level,
                  depth_w=[r.depth for r in watch.reads.get(id(d), [])], load_w=watch.carry.get(id(d)),
                  stages_w=[r.count for r in watch.reads.get(id(d), [])], single_need_w=watch.single.get(id(d)))
             for d in p.downs]
    unders = _unders(watch.log)
    return TrowelResult(verdicts=verdicts, windows=windows, success=int(success), worlds=worlds, material_s=sim_s,
                        sim_s=sim_s, wall_s=r.wall, ledger_rel=PR.ledger_rel(flow, r.v0), faces_at_cap=r.cap,
                        cells_over_depth=r.over, ride_need_n=watch.need.tolist(), pressed_m3=watch.pressed.tolist(),
                        ride_planned_n=planned, ride_worst=max(watch.log, key=lambda e: e["need_n"], default=None),
                        ride_log=watch.log, books=books or None, books_ok=books_ok.tolist(),
                        lifts=[list(x) for x in p.lifts] or None, set_downs=downs or None,
                        strips_left_m=watch.left or None, middle_ridge_m=_middle(ends, fs[0]),
                        end_age_min=(float(fs[0]["age"]) + p.end) / 60.0, ride_underpredict=(unders or [None])[0],
                        ride_underpredicted=[u for u in unders if u["excess_n"] > UNDER_N],
                        forms_clearance=[dataclasses.asdict(c) for c in forms],
                        cell_m=float(fs[0]["cell"]))


ARMS_NA = "not applicable (privileged carriage, no robot)"
FORMS_PROXY = ("proxy: the lowest sole clearance [m] over the form tops along every tool's commanded path (pad_forms); "
               "the forms are cells, not bodies (WP6's known gap), so this is not a force measurement")


def _proxied(d: dict, c: PF.Clearance) -> dict:
    """A whole-pad verdict with forms_undisturbed by its proxy (``c``: >= 0 m passes) and arms_clear not applicable
    (not counted as a pass, WP15's ruling); success is every other criterion and the proxy."""
    crit = d["criteria"]
    crit["forms_undisturbed"] = {"value": c.min_m if math.isfinite(c.min_m) else None, "threshold": 0.0,
                                 "passed": c.passed, "measured": "proxy", "note": FORMS_PROXY}
    crit["arms_clear"] = {"value": None, "threshold": None, "passed": None, "measured": False, "note": ARMS_NA}
    ok = c.passed and all(v["passed"] for k, v in crit.items() if k not in ("arms_clear", "forms_undisturbed"))
    if d["success"] and not ok:
        d["score_before_proxy"], d["score"] = d["score"], None
    d["success"] = bool(ok)
    return counted(d, PAD_TROWEL)                                      # the count with the proxy in (AB-gym-7)


def _unders(log: list[dict]) -> list[dict]:
    """Every predicted margin run per world, the most beaten prediction first (WP15's v4 book): its time, world,
    need, prediction, the excess [N], its level, the blade's sole at its worst point (None: not sampled) and there."""
    out = []
    for e in log:
        for w, (need, pred) in enumerate(zip(e.get("need_w", ()), e.get("pred_w", ()), strict=False)):
            if pred is not None:
                sole = e.get("sole_w")
                out.append(dict(t=e["t"], world=w, need_n=need, pred_n=pred, excess_n=need - pred,
                                level=e["level_w"][w], sole=None if sole is None else sole[w], at_m=e["at_m"]))
    return sorted(out, key=lambda u: -u["excess_n"])


def _middle(ends: list[M.PasteGrid], f: dict[str, Any]) -> list[float]:
    """The highest core paste [m, from the form top] over |x| < 0.2235 m (no cross lane reaches there): the last lane
    round's ridge rectangle, disclosed (pad_trowel)."""
    mid = S.core(f) & (np.abs(f["x"])[:, None] < 0.2235)
    return [float(g.h[mid].max()) if mid.any() else math.nan for g in ends]


def _with_bleed(g: M.PasteGrid, age: np.ndarray, temperature: float) -> M.PasteGrid:
    """Their committed water reads the bleed still to rise at each cell's age (T:75-85; TSP:114-121)."""
    g.bleed_left = remaining_bleed(age, temperature)
    return g


def _asdict(w: Any) -> dict:
    from dataclasses import asdict                                                 # noqa: PLC0415
    return asdict(w)


def _save(npz: str | None, maps: str | None, ends: list[M.PasteGrid], fs: list[dict[str, Any]], age: np.ndarray,
          task: T.PadTrowel, watch: LiftWatch) -> None:
    if npz:
        np.savez_compressed(npz, **{k: np.stack([getattr(g, k) for g in ends]) for k in PW.FIELDS
                                    if getattr(ends[0], k) is not None},
                            h0=np.stack([f["h"] - FORM_H for f in fs]), x=fs[0]["x"], y=fs[0]["y"], age=age,
                            paste=fs[0]["paste_mask"], covered=T.covered(task, fs[0]))
    if maps and watch.maps:
        np.savez_compressed(maps, t=np.array([m[0] for m in watch.maps]), when=np.array([m[1] for m in watch.maps]),
                            h_mm=np.stack([m[2] for m in watch.maps]), x=fs[0]["x"], y=fs[0]["y"],
                            h0_mm=(1e3 * (fs[0]["h"] - FORM_H)).astype(np.float16), paste=fs[0]["paste_mask"])


__all__ = ["EDGE", "FROM_PASS", "LiftWatch", "MARGIN", "PASTE_TOL", "Plan", "SPEED", "SetDown", "Sim",
           "TrowelResult", "UNDER_N", "heights", "lift_runs", "plan", "run", "simulate"]
