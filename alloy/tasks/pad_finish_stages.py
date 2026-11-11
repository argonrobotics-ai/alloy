# SPDX-License-Identifier: Apache-2.0
"""pad-finish's stages (:mod:`alloy.tasks.pad_finish`): each its owner's tools, program and chart split on its own
simulation, reused without edits, started from the chain's fields and run to a rest point:
- **darby** (WP6's pad-darby, :mod:`alloy.tasks.concrete_pad` + :mod:`alloy.tasks.concrete_tub`): the margin trowel
  fills each world's low from the tub, the darby 4626D takes the high spots and the ridges down, the stone under; on
  their chart, the pad plus ``GROUND`` of ground beyond each end form (its roll falls off there: off the pad, booked);
- **edge** (WP4's pad-edge, :mod:`alloy.tasks.pad_edge`): the edger 9.5 mm round all four forms, its lip's mark
  (``pad_lip``) kept as ``compact``;
- **trowel** (WP4's pad-trowel, :mod:`alloy.tasks.pad_trowel`): the held PB64D's lane and cross passes under their ride
  rule (``RideWatch``, their 20 N assumed push).

The darby leaves the pad as a plane ``ref`` under the form tops (WP6's last passes run 1.5 mm under, ``late_below``).
The edger's and the trowel's heights are WP4's relative to the form-top plane, so here they are taken relative to
``ref`` (the edger's keys lowered by it, the trowel's levels shifted by it): a finisher sets the tools on the slab.

After its program a stage settles (the tools still) and waits (their wait command, ``pad_run.material_wait``: the
layer's clock alone, refused unless the tools rest and the film rests by ``film_rest``) until the next stage is
ready, on its own chart where the film came to rest; the next stage then holds ``WAIT_AT`` periods (0.2 s, their
RESEAT_S) on the cart before its program.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from alloy.tasks import concrete_pad as CP
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_edge as PE
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_finish_stage as ST
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_trowel as PT
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolProgram
from alloy.tasks.pad_finish_handoff import Records
from alloy.tasks.pad_finish_watch import (FILM_SETTLE_S, PERIOD, SETTLE_S, STILL, WAIT_CHUNK, Ready, Waited, Watch,
                                          age_at_least, all_of, bleed_over, settle, sheen_gone, snapshot, wait_for)

log = logging.getLogger("alloy.tasks.pad_finish_stages")
WAIT_AT = 10                               # periods on the cart before a program (0.2 s: their RESEAT_S, env.py:23)
START = (WAIT_AT + 5) * PERIOD + SETTLE_S  # s: a program's first key (WP4's 0.3 s, plus room for the start's watch)


@dataclass
class StageEnd:
    """A stage's end at its rest point, on the pad's cells [W, nx, ny]: the surface, the layer's start states and age,
    the records and books so far, its stepped and waited seconds, tools left off the cart, other plug-ins' fields
    (``extra``), the hand-off's at-rest sizes, and a rest-point refusal if any."""

    surface: np.ndarray
    states: dict[str, np.ndarray]
    records: Records
    books: dict[str, np.ndarray]
    stepped_s: float
    waited_s: float
    tools_off: np.ndarray
    rest: dict[str, float]
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    rest_error: str | None = None
    pre_wait: dict[str, np.ndarray] = field(default_factory=dict)   # the surface and states before the end's wait
    start_wait: dict[str, np.ndarray] = field(default_factory=dict)  # before_* / after_* the start's wait
    diag: dict[str, Any] = field(default_factory=dict)


def _build(scene: Any, plugins: list[Any], worlds: int, dt: float, device: str) -> Any:
    from alloy.plugins.harness.scene import TestScene, build_simulation  # noqa: PLC0415

    return build_simulation(TestScene(build=scene, ground=False, worlds=worlds, dt=dt), plugins, device)


def _step(sim: Any, flow: Any, periods: int, name: str, hooks: dict[int, Any] | None = None) -> tuple[int, int]:
    t0 = time.perf_counter()
    cap = over = 0
    for i in range(periods):
        if hooks and i in hooks:
            hooks[i]()
        sim.step_period()
        if i % 25 == 0:
            r = flow.read()
            cap, over = max(cap, int(r["faces_at_cap"].max())), max(over, int(r["cells_over_depth"].max()))
        if i % 500 == 0:
            log.info("%s: period %d of %d (%.0f s wall)", name, i, periods, time.perf_counter() - t0)
    return cap, over


def _open(sim: Any, flow: Any, layer: Any, carriage: Any, ready: Ready | None, limit: float,
          crop: Callable[[np.ndarray], np.ndarray]) -> tuple[Watch, Waited, dict[str, np.ndarray]]:
    """A stage's start on the cart (WP15, DC5): 0.2 s watched with the tools clear (their RESEAT_S), then their wait
    until ``ready`` on this stage's chart, within ``limit`` (the budget left); the fields before and after it. The
    program starts ``SETTLE_S`` later than WP4's (``START``), so a watch that steps on never meets a moving tool."""
    watch = Watch()
    watch.after(sim, flow, carriage, layer)
    if ready is None:
        return watch, Waited(), {}
    before = snapshot(flow, layer, crop)
    waited = wait_for(sim, flow, layer, carriage, ready, limit, crop, watch)
    if not waited.seconds:
        return watch, waited, {}
    after = snapshot(flow, layer, crop)
    return watch, waited, {**{f"before_{k}": v for k, v in before.items()},
                           **{f"after_{k}": v for k, v in after.items()}}


def _finish(sim: Any, flow: Any, layer: Any, carriage: Any, records: Records | None, books: dict | None,
            crop: Callable[[np.ndarray], np.ndarray], watch: Watch, waits: tuple[Waited, ...] = (),
            **kw: Any) -> StageEnd:
    """The hand-off at the rest point (``check_rest`` on the watch's last check; with the refusals of the stage's
    ``waits``: kept, not raised, so the run's fields survive); this stage's records, on the pad's cells, folded into a
    copy of ``records``."""
    errs = [w.refusal for w in waits if w.refusal]
    try:
        ST.check_rest(flow, carriage, watch.last)
    except ValueError as e:
        errs.append(str(e))
    c = ST.carry(flow, layer, carriage, books=books)
    c.rest.update(watch.summary())
    rec = Records({k: v.copy() for k, v in (records.fields if records else {}).items()})
    rec.add({k: crop(v) for k, v in c.records.fields.items()})
    return StageEnd(surface=crop(ST.end_surface(c)), states={k: crop(v) for k, v in c.layer.items()}, records=rec,
                    books=c.books, rest=c.rest, rest_error="; ".join(errs) or None, **kw)


# --- the darby (WP6) ---------------------------------------------------------------------------------------------

def _crop_g(a: np.ndarray, shape_g: tuple[int, int], n: int) -> np.ndarray:
    """[W, C] on the grounded chart -> [W, nx, ny] on the pad's cells (``n`` ground columns off each end)."""
    return np.asarray(a).reshape(a.shape[0], *shape_g)[:, n:shape_g[0] - n, :]


def _darby_chart(task: CP.PadDarby) -> Callable[[dict[str, Any]], dict[str, np.ndarray]]:
    def rule(f: dict[str, Any]) -> dict[str, np.ndarray]:
        base, film = CP.split(task, f)
        return {"base": base, "film": film, "paste": np.asarray(f["paste_mask"])}
    return rule


def darby(fs: list[dict[str, Any]], ready: Ready | None = None, limit: float = 0.0, device: str = "cpu",
          task: CP.PadDarby | None = None) -> StageEnd:
    """WP6's tub fill and darby on the chain's start ``fs`` (one per world, the pad's 2 mm cells), then their wait
    until ``ready``. The first stage: records and books start here; the film left on the ground is off the pad."""
    task = task or CP.PadDarby(factor=1)
    if task.factor != 1:
        raise ValueError("pad-finish runs on their 2 mm cells (factor 1)")
    fsg = [CP.with_ground(f, CP.GROUND) for f in fs]
    n, shape_g = int(round(CP.GROUND / float(fs[0]["cell"]))), fsg[0]["h"].shape
    carriage, flow, layer, duration = CP.plugins(task, fsg)
    sim = _build(CP.build_scene(fsg[0]), [carriage, flow, layer], len(fs), 2.0e-3, device)
    ST.require_placed(flow, layer, carriage)
    ST.start_stage(flow, layer, fsg, _darby_chart(task))
    periods, t0 = int(math.ceil(duration / PERIOD)), time.perf_counter()
    cap, over = _step(sim, flow, periods, "darby")
    periods += settle(sim, layer, carriage)
    watch = Watch()
    watch.after(sim, flow, carriage, layer)
    homes = [np.asarray([CP.darby_home(f), TUB.home(f)]) for f in fsg]
    pos = sim.state.body_q.numpy()[carriage.tool.numpy()][..., :3]
    off = np.array([int((np.linalg.norm(p - h, axis=1) > PR.ON_CART).sum()) for p, h in zip(pos, homes, strict=True)])
    pre = snapshot(flow, layer, lambda a: _crop_g(a, shape_g, n))
    waited = wait_for(sim, flow, layer, carriage, ready, limit - (periods + watch.periods) * PERIOD,
                      lambda a: _crop_g(a, shape_g, n), watch)
    film = flow.h.numpy().astype(np.float64).reshape(len(fs), *shape_g).copy()
    film[:, n:shape_g[0] - n, :] = 0.0                                       # the film on the ground cells only
    read = flow.read()
    books = {"off_pad": film.reshape(len(fs), -1) @ flow.tables.area_np.astype(np.float64)}
    return _finish(sim, flow, layer, carriage, None, books, lambda a: _crop_g(a, shape_g, n), watch,
                   waits=(waited,), stepped_s=(periods + watch.periods) * PERIOD, waited_s=waited.seconds,
                   tools_off=off, pre_wait=pre,
                   diag={"wall_s": time.perf_counter() - t0, "faces_at_cap": cap, "cells_over_depth": over,
                         "tools_total": 2, "tub_left_m3": np.asarray(read["tray_volume"]).sum(axis=-1).tolist(),
                         "store_ledger": np.asarray(read["store_ledger"]).tolist(),
                         "tray_ledger": np.asarray(read["tray_ledger"]).tolist()})


def plane(surface: np.ndarray, fs: list[dict[str, Any]], band: float = 0.08) -> float:
    """The darby's end plane [m, relative to the form-top plane]: the median surface over every world's core (over
    ``band`` in from the forms)."""
    from alloy.tasks.concrete_pad_start import FORM_H  # noqa: PLC0415

    core = [np.asarray(f["paste_mask"], bool) & (np.asarray(f["d_edge"]) > band) for f in fs]
    return float(np.median(np.concatenate([s[m] for s, m in zip(surface, core, strict=True)]))) - FORM_H


# --- the edger (WP4) ---------------------------------------------------------------------------------------------

def lowered(prog: ToolProgram, dz: float, cart: tuple[float, float, float]) -> ToolProgram:
    """``prog`` with every key off the cart moved ``dz`` in height (WP4's heights are from the form-top plane)."""
    keys = tuple(k if tuple(k.pos) == tuple(cart) else replace(k, pos=(k.pos[0], k.pos[1], k.pos[2] + dz))
                 for k in prog.keys)
    return ToolProgram(prog.body, keys)


def edge(fs: list[dict[str, Any]], ref: float, records: Records, books: dict, ready: Ready | None = None,
         limit: float = 0.0, device: str = "cpu", task: PE.PadEdge | None = None,
         pre_ready: Ready | None = None) -> StageEnd:
    """WP4's edger round the forms on ``fs``, its sole on the plane ``ref``: their wait until ``pre_ready`` (the
    edger's time), the program, then their wait until ``ready`` (the next stage's). The lip's mark goes on as
    ``extra["compact"]`` (with the start's)."""
    task = task or PE.PadEdge()
    prog, end = PE.program(task, fs[0], t0=START)
    carriage, flow, layer, lip = PE.plugins(task, fs[0], lowered(prog, ref, task.cart))
    sim = _build(PE.build_scene(task, fs[0]), [carriage, flow, layer, lip], len(fs), task.exchange_dt, device)
    ST.require_placed(flow, layer, carriage, (lip, ("lip",)))      # the lip's mark is carried as compact
    ST.start_stage(flow, layer, fs, PE.chart_fields)
    shape = fs[0]["h"].shape

    def crop(a: np.ndarray) -> np.ndarray:
        return np.asarray(a).reshape(a.shape[0], *shape)
    t0 = time.perf_counter()
    watch, start_waited, start_wait = _open(sim, flow, layer, carriage, pre_ready, limit, crop)
    ST.check_opening(watch.periods * PERIOD, START)
    periods = int(math.ceil(end / PERIOD)) - watch.periods
    cap, over = _step(sim, flow, periods, "edge")
    periods += settle(sim, layer, carriage)
    watch.after(sim, flow, carriage, layer)
    off = PR.tools_back(sim, carriage, (PE.edger_body(task),))
    pre = snapshot(flow, layer, crop)
    left = limit - start_waited.seconds - (periods + watch.periods) * PERIOD
    waited = wait_for(sim, flow, layer, carriage, ready, left, crop, watch)
    compact = np.maximum(np.stack([np.asarray(f["compact"], np.float64) for f in fs]), crop(lip.marks()))
    return _finish(sim, flow, layer, carriage, records, books, crop, watch, waits=(start_waited, waited),
                   stepped_s=(periods + watch.periods) * PERIOD, waited_s=start_waited.seconds + waited.seconds,
                   tools_off=off, pre_wait=pre, start_wait=start_wait, extra={"compact": compact},
                   diag={"wall_s": time.perf_counter() - t0, "faces_at_cap": cap, "cells_over_depth": over,
                         "tools_total": 1, "start_waited_s": start_waited.seconds, "limit_s": limit,
                         "ledger": np.asarray(flow.read()["ledger"]).tolist()})


# --- their wait, as a stage --------------------------------------------------------------------------------------

def _wait_plugins(task: PT.PadTrowel, f: dict[str, Any]) -> tuple[Any, Any, Any]:
    """WP4's trowel pieces (``pad_trowel.plugins``) with its program parked on the cart and the layer writing the bulk
    yield into the film (``film_yield="bulk"``): a wait asks whether the paste would move, and the darby leaves bulk
    paste as film there (a ridge up to 13 mm inside the forms), which the trowel's thin-film surface yield would let
    flow at 10 min."""
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.tasks.concrete_paste import ConcretePaste, SoleWork  # noqa: PLC0415
    from alloy.tasks.concrete_tools import ToolCarriage  # noqa: PLC0415

    nx, ny = f["h"].shape
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(nx, ny)),), material=PR.surface_film(f, task.temperature),
                       inertia=False, max_depth=0.025, u_cap=PR.U_CAP, tools=(PT.trowel_tool(task),),
                       tool_speed=max(task.speed, 0.1), per_cell_rheology=True)
    carriage = ToolCarriage(programs=(ToolProgram("trowel", (Stroke(0.0, task.cart, mode=HEIGHT),)),),
                            omega=task.omega, omega_rot=task.omega)
    layer = ConcretePaste(material="concrete", temperature=task.temperature, air=task.air, film_yield="bulk",
                          tools=(SoleWork("trowel", *PT.TROWEL["work"], half=PT.TROWEL["half"][:2]),), flow=flow,
                          carriage=carriage)
    return carriage, flow, layer


def wait(fs: list[dict[str, Any]], ref: float, records: Records, books: dict, ready: Ready, limit: float,
         device: str = "cpu") -> StageEnd:
    """Their wait as a stage of its own, between two tools' stages: on the trowel's chart (``_wait_plugins``: the
    paste above the plane's film, yielding as the bulk; the forms' smear as film, their tb) with the trowel parked on
    its cart, the 0.2 s rest watch, then their wait until ``ready``, and the hand-off. A tool's own chart may not rest
    at the age the wait starts (WP4's edger chart gives 16 mm by the forms the surface yield, which flows at 10 min)."""
    task = trowel_task(ref)
    carriage, flow, layer = _wait_plugins(task, fs[0])
    sim = _build(PR.build_scene(fs[0], (PT.trowel_body(task),)), [carriage, flow, layer], len(fs), task.exchange_dt,
                 device)
    ST.require_placed(flow, layer, carriage)
    ST.start_stage(flow, layer, fs, PT.split(task))
    shape = fs[0]["h"].shape

    def crop(a: np.ndarray) -> np.ndarray:
        return np.asarray(a).reshape(a.shape[0], *shape)
    t0 = time.perf_counter()
    watch, waited, start_wait = _open(sim, flow, layer, carriage, ready, limit, crop)
    return _finish(sim, flow, layer, carriage, records, books, crop, watch, waits=(waited,),
                   stepped_s=watch.periods * PERIOD, waited_s=waited.seconds, tools_off=np.zeros(len(fs), int),
                   start_wait=start_wait, diag={"wall_s": time.perf_counter() - t0, "tools_total": 0,
                                                "start_waited_s": waited.seconds, "limit_s": limit})


# --- the steel trowel (WP4) --------------------------------------------------------------------------------------

FLOOR = 0.0                                # m: the trowel's lowest level above the darby's plane (no planing)
RIM = 0.010                                # m: the trowel's blade ends and reversals from the forms (WP4's 30 mm)


def trowel_task(ref: float, floor: float = FLOOR, **kw: Any) -> PT.PadTrowel:
    """WP4's trowel with its levels from the plane ``ref`` instead of the form-top plane: their first level
    (``LEVEL0`` over it), stepped down to ``floor`` over it. Their levels go 0.25 mm under to plane pad-trowel's
    0.3 mm roughness; the darby's plane is flat to about 0.03 mm, and the paste a pass planes off ends in the 10-30 mm
    rim, where pad-finish reads the arris's top (11.5-21.5 mm in, T:413-451): so the trowel works it, at the plane.
    Its rim is ``RIM``, not their 30 mm: pad-finish reads smooth-hard from 11.5 mm in, and the edger's one pass leaves
    the 11.5-30 mm strip at D 0.3 and texture 0.45 mm (the L4 run); at the plane the trowel cuts almost nothing, so
    there is little end spill to keep off the arcs."""
    return PT.PadTrowel(level0=ref + PT.LEVEL0, level_min=ref + floor, **{"rim": RIM, **kw})


@dataclass
class Ridden:
    """A tool held at its levels under their ride rule: the trowel's task (its chart split, the rule's temperature),
    its plug-ins (carriage, film, layer), body, timed runs and program end, and the plan's largest need."""

    name: str
    task: PT.PadTrowel
    plugins: tuple[Any, Any, Any]
    body: PR.ToolBody
    runs: tuple
    end: float
    planned_n: float


def _ride(fs: list[dict[str, Any]], records: Records, books: dict, held: Ridden, device: str,
          pre_ready: Ready | None, limit: float) -> StageEnd:
    """Their wait until ``pre_ready``, then ``held``'s program on ``fs`` under their ride rule (``RideWatch``, at the
    paste's age when each run starts, the wait included); back to the cart, the rest watch, the hand-off."""
    carriage, flow, layer = held.plugins
    sim = _build(PR.build_scene(fs[0], (held.body,)), [carriage, flow, layer], len(fs), held.task.exchange_dt, device)
    ST.require_placed(flow, layer, carriage)
    ST.start_stage(flow, layer, fs, PT.split(held.task))
    shape = fs[0]["h"].shape

    def crop(a: np.ndarray) -> np.ndarray:
        return np.asarray(a).reshape(a.shape[0], *shape)
    t0 = time.perf_counter()
    watch, start_waited, start_wait = _open(sim, flow, layer, carriage, pre_ready, limit, crop)
    aged = [dict(f, age=float(f["age"]) + start_waited.seconds) for f in fs]
    ride = PT.RideWatch(held.task, aged, flow, held.runs)
    ST.check_opening(watch.periods * PERIOD, START)
    periods = int(math.ceil(held.end / PERIOD)) - watch.periods
    hooks = ST.shifted_hooks(ride.hooks(), watch.periods)
    cap, over = _step(sim, flow, periods, held.name, hooks=hooks)
    periods += settle(sim, layer, carriage)
    watch.after(sim, flow, carriage, layer)
    off = PR.tools_back(sim, carriage, (held.body,))
    return _finish(sim, flow, layer, carriage, records, books, crop, watch, waits=(start_waited,),
                   stepped_s=(periods + watch.periods) * PERIOD, waited_s=start_waited.seconds, tools_off=off,
                   start_wait=start_wait,
                   diag={"wall_s": time.perf_counter() - t0, "faces_at_cap": cap, "cells_over_depth": over,
                         "tools_total": 1, "ride_ok": ride.ok().tolist(), "ride_need_n": ride.need.tolist(),
                         "pressed_m3": ride.pressed.tolist(), "ride_planned_n": held.planned_n, "hold_n": PT.HOLD,
                         "start_waited_s": start_waited.seconds, "limit_s": limit, "ride_log": ride.log,
                         "ledger": np.asarray(flow.read()["ledger"]).tolist()})


def trowel(fs: list[dict[str, Any]], ref: float, records: Records, books: dict, device: str = "cpu",
           task: PT.PadTrowel | None = None, pre_ready: Ready | None = None, limit: float = 0.0) -> StageEnd:
    """Their wait until ``pre_ready``, then WP4's lane and cross passes on ``fs`` at levels from ``ref``, under their
    ride rule; back to the cart. The last stage: no wait after."""
    task = task or trowel_task(ref)
    p = PT.plan(task, fs[0], t0=START)
    held = Ridden("trowel", task, PT.plugins(task, fs[0], p.program), PT.trowel_body(task), p.runs, p.end,
                  max(PT.ride_need(task, f) for f in fs))
    end = _ride(fs, records, books, held, device, pre_ready, limit)
    end.diag["levels"] = PT.levels(task)
    return end


# --- the margin trowel (pad_finish_margin) -----------------------------------------------------------------------

def margin(fs: list[dict[str, Any]], ref: float, records: Records, books: dict, device: str = "cpu",
           pre_ready: Ready | None = None, limit: float = 0.0) -> StageEnd:
    """Their wait until ``pre_ready``, then the margin trowel's pass over the edger's bead (``pad_finish_margin``:
    planned on every world's start, one program), under their ride rule, on the steel trowel's chart; back to the
    cart."""
    task = trowel_task(ref)
    p = MG.plan(fs, ref, t0=START, temperature=task.temperature)
    held = Ridden("margin", task, MG.plugins(fs[0], p.program, task.temperature, task.air), MG.body(fs[0]), p.runs,
                  p.end, p.need_n)
    end = _ride(fs, records, books, held, device, pre_ready, limit)
    read = held.plugins[1].read()
    end.diag.update(tools_total=1, planned_cut_m3=p.cut_m3, planned_pressed_m3=p.pressed_m3,
                    store_ledger=np.asarray(read["store_ledger"]).tolist(),
                    taken_m3=np.asarray(read["to_tools_volume"]).tolist(),
                    released_m3=np.asarray(read["tool_released"]).sum(axis=1).tolist(),
                    load_left_m3=np.asarray(read["tool_load"]).sum(axis=1).tolist())
    return end


__all__ = ["FILM_SETTLE_S", "FLOOR", "PERIOD", "RIM", "Ready", "Ridden", "SETTLE_S", "START", "STILL", "StageEnd",
           "WAIT_AT", "WAIT_CHUNK", "Waited", "Watch", "age_at_least", "all_of", "bleed_over", "darby", "edge",
           "lowered", "margin", "plane", "settle", "sheen_gone", "snapshot", "trowel", "trowel_task", "wait",
           "wait_for"]
