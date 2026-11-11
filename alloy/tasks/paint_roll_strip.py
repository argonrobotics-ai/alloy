# SPDX-License-Identifier: Apache-2.0
"""roll-strip: roll the marked strip to full hide, joining the wet strip on its left, with no runs, laps or paint on the
ceiling or the baseboard (Gym-Anything's painting-tasks-v1 roll-strip, scaled to a fixed FR3:
docs/notes/soft-3-painting.md).

Two rollers:
- the real one, :func:`run_roller`: the cover on the plug-in's stores (:mod:`alloy.tasks.paint_roller`), on a
  kinematic script until a robot holds it. It loads in the tray (load-roller's sub-verdict), rolls the lanes and goes
  back into the tray;
- the stand-in, :func:`run_standin`: each pass lays a known coat through ``set_source`` under a 9 in band that moves
  down the strip (``standin_passes``), to test the checks against known outcomes.

The film then settles and :func:`evaluate` judges it with :mod:`alloy.tasks.paint_metrics`.

What the verdict judges (audit AB-gym-6; :func:`scene` and each check's note say the same in the result). Of
Gym-Anything's seven roll-strip criteria four are judged: ``hide``, ``no_laps``, ``no_runs`` (by their older rule:
no cell's film travels over 1 mm in the settle) and ``uniform`` (their patches plus two edge rows). Three are not:
``untouched`` (no marks are tracked), ``protected`` (only paint spilled over the bottom edge is read: this wall has
nothing else to protect) and ``roller_in_tray`` (its position is read, its release is not: the roller is never let
go). The scene is smaller and easier than theirs: a 1.0 m strip (theirs 2.23 m), a uniform first coat (theirs
±15 % patchy), a 60 s settle (theirs until the film sets, up to 1800 s), a fixed FR3 (theirs mobile), one start.
load-roller's four criteria are all judged, at the end of the first hold of a scripted load; ``no_drip_10s`` allows
one drop.

Example::

    from alloy.tasks.paint_roll_strip import run_roller
    print(run_roller()["succeeded"])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_process as pp
from alloy.tasks.paint_materials import COLOURS, aged_yield, film_material, optics, paint
from alloy.tasks.paint_optics import srgb8_to_linear
from alloy.tasks.paint_scene import WallLayout, chart_scene, wall_box

ROLLER_WIDTH = 0.2286           # m, a 9 in cover
SPEED = 0.25                    # m/s down the wall
CONTROL_DT = 0.05               # s: a creeping paint film steps at seconds
CONTACT = 2 * SPEED * CONTROL_DT    # m, the band a pass lays under; two periods long, so each cell dwells exactly 2
SETTLE = 60.0                   # s of settling before the verdict: the window until drying sets the film (a 4-mil
                                # coat of the latex card creeps ~0.3 mm in it; the run limit is 1 mm)
GA_STRIP = (0.6, 2.228)         # m: Gym-Anything's strip (layout.py STRIP: x -0.75..-0.15, z 0.15..2.378)
OLD_RUNS = ("judged by another measure: Gym-Anything's older rule (no cell's film travels over 1 mm during the "
            "settle), over the port's settle (60 s by default); painting-tasks-v1 asks that film which flowed over "
            "50 µm covers at most 1 cm2, after settling until the film sets (up to 1800 s)")
EDGE_PATCHES = "judged with Gym-Anything's 50 mm patches plus a row and a column flush with the strip's far edges"
SPILL_ONLY = ("only paint spilled over the chart's bottom edge is judged; this wall has no ceiling, trim, outlet "
              "plate or radiator to paint by mistake; Gym-Anything: paint on those <= 0.5 cm2")
LOAD_SCENE = ("the roll-strip port's run, judged once at the end of the first hold above the tray, on a scripted "
              "load; the roller is fixed in a fixed FR3's hand (Gym-Anything: a mobile FR3 holds it); one start")


def scene(layout: WallLayout) -> str:
    """How this port's scene differs from Gym-Anything's roll-strip (the verdict's ``scene``)."""
    s = layout.strip
    return (f"smaller and easier than Gym-Anything's: a {s.x1 - s.x0:.2f} x {s.y1 - s.y0:.2f} m strip on a "
            f"{layout.width:.2f} x {layout.height:.3f} m chart at {1e3 * layout.cell:.0f} mm cells (theirs: "
            f"{GA_STRIP[0]:.1f} x {GA_STRIP[1]:.2f} m on the whole wall at 2 mm); a uniform "
            f"{1e6 * layout.first_coat_dft:.0f} µm first coat (theirs: ±15 % patchy); a settle of {SETTLE:.0f} s by "
            "default before the verdict (theirs: until the film sets, up to 1800 s); a fixed FR3 on a pedestal or a "
            "scripted roller (theirs: a mobile FR3); one start, the seed changes nothing")


@dataclass(frozen=True)
class Pass:
    x0: float                   # m, the lane's left edge (chart frame)
    x1: float
    coat: float                 # m wet, laid once over the lane


def standin_passes(layout: WallLayout, coat: float, overlap: float = 0.0) -> list[Pass]:
    """Lanes as wide as the roller, side by side over the strip, the last one ending at its right tape line. With
    ``overlap`` [m], each lane after the first starts that far back over the lane before it."""
    s = layout.strip
    n = int(np.ceil((s.x1 - s.x0) / ROLLER_WIDTH - 1e-9))
    edges = np.linspace(s.x0, s.x1, n + 1)
    return [Pass(float(a) - (overlap if k else 0.0), float(b), coat)
            for k, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]


def pass_rates(layout: WallLayout, centres: np.ndarray, p: Pass) -> list[np.ndarray]:
    """Per control period, the source [m/s] of one pass: a CONTACT-tall band moving down the strip at SPEED, from
    just above its top line to past its bottom line, so that every cell of the lane dwells CONTACT/SPEED under it and
    gets exactly ``p.coat``."""
    s = layout.strip
    x, y = centres[..., 0], centres[..., 1]
    lane = (x >= p.x0) & (x < p.x1)
    rate = p.coat * SPEED / CONTACT
    step = SPEED * CONTROL_DT
    out = []
    for k in range(int(np.ceil((s.y1 - s.y0 + CONTACT) / step - 1e-9))):
        lo = s.y1 - step * (k + 1)
        band = lane & (y >= max(lo, s.y0)) & (y < min(lo + CONTACT, s.y1))
        out.append(np.where(band, rate, 0.0).astype(np.float32))
    return out


@dataclass
class RunRecord:
    """What a roll-strip run set and measured besides the film's ``read()``: the film and solids set directly (the
    prepaint) [m³], the per-cell ``travel`` [nx, ny] when the settle began and ended, and the wet cells a robot link
    marred (None: no marks were tracked, and ``untouched`` is then not judged)."""

    set_volume: float
    set_solids: float
    travel_start: np.ndarray
    travel_end: np.ndarray
    marred: np.ndarray | None = None


def evaluate(layout: WallLayout, centres: np.ndarray, h_end: np.ndarray, read: dict, run: RunRecord,
             extra: tuple[pm.Check, ...] = ()) -> dict:
    """The roll-strip verdict of one world from the depth map [nx, ny] after the settle and the film's ``read()``
    then, with the ``extra`` checks of a real roller's run (the tray, its books). The dry film and laps are the
    process stage's (``dft``, ``laps``). Without the stage the dry film falls back to ``h·φ0`` (the verdict's
    ``dry_film`` says which), and the lap and solids checks fail: they cannot be measured. ``untouched`` is judged
    only from ``run.marred``, and ``protected`` only in part (the module's docstring)."""
    m = layout.masks(centres)
    area = layout.cell * layout.cell
    opt = optics(layout.paint, layout.colour)
    r_g = srgb8_to_linear(COLOURS[layout.substrate])
    new, source = pp.dry_film(read, h_end, layout.paint)
    lap = pm.chart_field(read, "laps", h_end.shape)
    lost = float(np.asarray(read.get("spilled_volume", 0.0)).reshape(-1)[0])     # over the bottom edge: baseboard
    travel = pm.settle_travel(run.travel_start, run.travel_end, m["strip"])
    no_laps = pm.Check("no_laps", float("inf"), pm.LAP_AREA, False) if lap is None else pm.laps(lap, m["strip"], area)
    checks = [pm.hide(layout.first_coat_dft + new, m["strip"], opt, r_g),
              pm.no_runs(float(h_end[m["strip"]].max()), travel).differs(OLD_RUNS),
              no_laps,
              pm.uniform(layout.first_coat_dft + new, m["strip"], opt, r_g, layout.cell).differs(EDGE_PATCHES),
              pm.protected(new, np.zeros_like(m["strip"]), area, lost_volume=lost).in_part(SPILL_ONLY),
              pm.untouched(run.marred, m["strip"], area),
              pm.ledger(read, set_volume=run.set_volume),
              pm.solids_ledger(read, set_solids=run.set_solids), *extra]  # fmt: skip
    return {**pm.verdict(checks, scene=scene(layout)), "dry_film": source,
            "dry_strip": pm.dry_summary(new, m["strip"])}


def _travel(flow, shape: tuple[int, int]) -> np.ndarray:  # noqa: ANN001
    t = pm.chart_field(flow.read(), "travel", shape)
    if t is None:
        raise RuntimeError("roll-strip's run check needs the surface-flow film's per-cell travel (read()['travel'])")
    return t.copy()


def run_standin(layout: WallLayout | None = None, coat: float | None = None, device: str = "cpu",
                lane_pause: float = 0.0, overlap: float = 0.0) -> dict:
    """The stand-in roll-strip on a wall-only scene (the plug-in harness: a static wall and a free dummy body).
    ``lane_pause`` [s] waits between lanes and ``overlap`` [m] rolls each lane back over the one before: a lane
    started after the last one's wet edge has passed leaves a lap mark."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415

    layout = layout or WallLayout()
    coat = paint(layout.paint).wet_film if coat is None else coat
    flow = SurfaceFlow(charts=(PlaneChart("wall", cells=layout.cells, edges=layout.edges),), inertia=False,
                       max_depth=4e-4, per_cell_rheology=True,
                       **pp.film_settings(layout.paint, substrate="drywall_primed"))  # fmt: skip
    carried = pp.source_species(layout.paint)
    box: dict = {}

    def init(sim) -> None:  # noqa: ANN001
        c = layout.chart_centres(flow.cell_centres(0))
        h0, age = layout.prepaint(c)
        h0, solids = pp.aged_film(layout.paint, h0, age)                 # dried since it was laid
        flow.set_depth(h0[None], chart=0)
        flow.set_species(solids[None], pp.SOLIDS, chart=0)
        # the prepainted bands have rebuilt their yield stress; the new coat stays fresh through the settle (the
        # paint's own recovery would stiffen it within ~20 s: the run check errs toward a run)
        flow.set_input("yield_stress", aged_yield(layout.paint, age).reshape(1, -1).astype(np.float32))
        pause = [np.zeros_like(h0)] * int(round(lane_pause / CONTROL_DT))
        lanes = [pass_rates(layout, c, p) for p in standin_passes(layout, coat, overlap)]
        area = flow.tables.area_np.reshape(h0.shape)       # the chart's cell areas (float32 half extents)
        box.update(centres=c, set_volume=float((h0 * area).sum()), set_solids=float((solids * area).sum()),
                   rates=[r for k, lane in enumerate(lanes) for r in (pause if k else []) + lane])  # fmt: skip

    sim = build_simulation(chart_scene([wall_box(layout)], init, CONTROL_DT), [flow], device)
    rates = box["rates"]
    for r in rates:                                         # the passes: one source map per control period
        flow.set_source(r[None], chart=0, species=carried if r.any() else None)
        sim.step_period()
    flow.set_source(0.0)
    sim.step_period()
    start = _travel(flow, layout.cells)
    for _ in range(int(round(SETTLE / CONTROL_DT))):
        sim.step_period()
    run = RunRecord(box["set_volume"], box["set_solids"], start, _travel(flow, layout.cells))
    return evaluate(layout, box["centres"], flow.depth(0)[0], flow.read(), run)


class _Yield:
    """The film's yield-stress map from each cell's age since it was laid or rolled: ``aged_yield`` with its card read
    once."""

    def __init__(self, name: str) -> None:
        self.fresh = float(aged_yield(name, np.zeros(1))[0])
        self.rest = float(aged_yield(name, np.full(1, 1e12))[0])
        self.t_r = paint(name).recovery

    def __call__(self, age: np.ndarray) -> np.ndarray:
        rebuilt = 1.0 - np.exp(-np.clip(age, 0.0, None) / self.t_r)
        return (self.fresh + (self.rest - self.fresh) * rebuilt).reshape(1, -1).astype(np.float32)


DROP = 14.2e-9  # m³: one drop (Gym-Anything's drip drop, drips.py:24)


@dataclass
class _LoadLog:
    """The store over the first hold above the tray (load-roller's moment): dripped volume per control period, and
    the bins' load, the cover's place and the time at its end."""

    times: list = field(default_factory=list)
    dripped: list = field(default_factory=list)
    end: dict | None = None
    held: bool = False
    _last: dict | None = None

    def sample(self, t: float, holding: bool, flow, pos: np.ndarray) -> None:  # noqa: ANN001 - SurfaceFlow
        if self.end is not None or not (holding or self.held):
            return
        r = flow.read()
        if holding:
            self.held = True
            self.times.append(t)
            self.dripped.append(float(r["tool_dripped"][0].sum()))
            self.end = None
            self._last = {"load": r["tool_load"][0].copy(), "pos": pos}
        else:
            self.end = self._last                       # the hold is over: its last sample is the verdict's

    def checks(self, layout: WallLayout) -> list[pm.Check]:
        from alloy.tasks import paint_roller as pr  # noqa: PLC0415

        if self.end is None:
            return [pm.Check("load", 0.0, pm.LOAD_G, False)]
        loc, half = pr.tray_frame(layout, self.end["pos"])
        density, load = film_material(layout.paint).density, self.end["load"]
        capacity = np.full_like(load, pr.CAPACITY / load.size)          # the store's capacity is its saturated load
        return [*pm.roller_load(load, capacity, density), pm.held_above_tray(loc, half),
                pm.no_drip(np.asarray(self.dripped), np.asarray(self.times), DROP)]


def _tool_books(read: dict, tray0: float) -> pm.Check:
    """The cover's store (load + dripped - picked - film taken: 0 for a dry cover) and the tray (volume + picked) close,
    relative to what the tray held."""
    store = abs(float(read["store_ledger"][0]))
    tray = abs(float(read["tray_ledger"][0]) - tray0)
    drift = max(store, tray) / tray0
    return pm.Check("tool_books", drift, 1e-9, drift <= 1e-9)


def run_roller(layout: WallLayout | None = None, device: str = "cpu", passes: int = 3, settle: float = SETTLE,
               driver=None) -> dict:  # noqa: ANN001
    """roll-strip with the real roller (the stores): per lane, load the dry cover in the tray's well and hold it above
    the tray, then roll the lane ``passes`` times; at the end the cover goes back into the tray and the film settles
    ``settle`` s. ``driver`` moves the cover (default :class:`alloy.tasks.paint_roller.Kinematic`: put at each pose of
    the script): its ``scene(init)``, ``bind(sim)``, ``steps``, ``apply(sim, step)``, ``pose(step)`` (the cover's
    planned pose, None in the air), ``cover_pos(sim)`` and ``report()``; optionally ``rolling(sim, step)``, the pose the
    cover rolls with where it is (default: the planned pose). The film's yield stress follows each cell's
    age since it was laid or rolled. The verdict adds
    the tray and the tools' books to roll-strip's checks, and ``load_roller``, the load-roller sub-verdict at the end
    of the first hold."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.tasks import paint_roller as pr  # noqa: PLC0415

    layout = layout or WallLayout()
    driver = driver or pr.Kinematic(layout, passes)
    tray = pr.tray_well(layout)
    flow = SurfaceFlow(charts=(PlaneChart("wall", cells=layout.cells, edges=layout.edges),), inertia=False,
                       max_depth=4e-4, per_cell_rheology=True, tools=(pr.roller(layout.paint, body=driver.cover),),
                       trays=(tray,), tool_speed=pr.SPEED,          # the fastest the cover touches the film
                       **pp.film_settings(layout.paint, substrate="drywall_primed"))  # fmt: skip
    box: dict = {}

    def init(sim) -> None:  # noqa: ANN001
        c = layout.chart_centres(flow.cell_centres(0))
        h0, age = layout.prepaint(c)
        h0, solids = pp.aged_film(layout.paint, h0, age)                 # dried since it was laid
        flow.set_depth(h0[None], chart=0)
        flow.set_species(solids[None], pp.SOLIDS, chart=0)
        area = flow.tables.area_np.reshape(h0.shape)       # the chart's cell areas (float32 half extents)
        box.update(centres=c, age=age, set_volume=float((h0 * area).sum()), set_solids=float((solids * area).sum()))

    sim = build_simulation(driver.scene(init), [flow], device)
    driver.bind(sim)
    c, age, yld, log, k = box["centres"], box["age"], _Yield(layout.paint), _LoadLog(), 0
    for k, step in enumerate(driver.steps):
        driver.apply(sim, step)
        pose = driver.pose(step)                        # the cover's planned pose (None: in the air, a transfer)
        age = np.where(age >= 0.0, age + pr.SUB, age)
        rolling = getattr(driver, "rolling", None)      # where the cover is (a robot's), else where it is put
        at = pose if rolling is None else rolling(sim, step)
        if at is not None:
            age[pr.rolled(at, layout, c)] = 0.0
        flow.set_input("yield_stress", yld(age))
        sim.step_period()
        getattr(driver, "observe", lambda *a: None)(sim, step)
        if k % 4 == 0 and pose is not None:            # the load log, once a control period
            log.sample(k * pr.SUB, pose.roll is None and not pose.vel.any() and k > 0, flow, driver.cover_pos(sim))
    rest = driver.cover_pos(sim)
    start_travel = _travel(flow, layout.cells)
    for j in range(int(round(settle / pr.SUB))):
        if j % 4 == 0:
            flow.set_input("yield_stress", yld(np.where(age >= 0.0, age + (j + 1) * pr.SUB, age)))
        sim.step_period()
    read = flow.read()
    loc, half = pr.tray_frame(layout, rest)
    run = RunRecord(box["set_volume"], box["set_solids"], start_travel, _travel(flow, layout.cells))
    v = evaluate(layout, c, flow.depth(0)[0], read, run, extra=(pm.in_tray(loc, half, released=None),
                                                                 _tool_books(read, tray.volume)))  # fmt: skip
    return {**v, "load_roller": pm.verdict(log.checks(layout), scene=LOAD_SCENE), "express": pr.express(),
            "sim_s": (k + 1) * pr.SUB + settle, **driver.report()}


__all__ = ["CONTACT", "DROP", "Pass", "ROLLER_WIDTH", "RunRecord", "SETTLE", "SPEED", "evaluate", "pass_rates",
           "run_roller", "run_standin", "scene", "standin_passes"]
