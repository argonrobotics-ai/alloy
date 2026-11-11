# SPDX-License-Identifier: Apache-2.0
"""task pad_trowel: Gym-Anything's pad-trowel, a surface-flow concrete proof.

"Steel-trowel the pad smooth and hard. Tools back on the cart when you are done." (``task.py:25``). Their pad 165 min
after strike-off at 21 C, floated: darbied flat (about 0.3 mm rough at 8 mm), edged on all four forms, texture
0.25 mm, densification 0.10, a 3 mm surface layer, bleeding over (``concrete_pad_start.pad_fields("pad-trowel")``).
Success is their pad-trowel criteria (``concrete_metrics.pad_trowel``, T:371-396, progress-v1 TSP:81-131): on 95 % of
the core (over 30 mm from the forms) D >= 0.8 and texture <= 0.05 mm, flat, the edge arcs kept, no blister, trapped
water, trowel burn, worked bleed water or dusting.

Model (as WP6's darby, ``concrete_pad``): one ``surface_flow`` plane chart at their 2 mm cells. A surface-flow tool
rides on the chart's base and pushes the film from under it (``applicators.py:13``), and a force-driven trowel on the
plastic bulk (forming) bulldozed the probe lane (surface moved -18/+47 mm in one pass). So the trowel is held at a
level: the paste above ``GAP`` under its lowest level is film, the rest fixed base; each pass cuts what stands above the
level into the roll (their plough, WP13's roll cap with their 1 mm lift), the roll fills the cells within the bearing
band below the sole and the excess spills in their 20 mm band beyond the blade's ends. Sole work is WP6's
``concrete_paste`` (their per-pass laws): a cell is worked while the sole is within 0.3 mm of it and sliding.

Their ride rule (``coupling.py:644-674``, ``pad_ride``): a sliding sole ploughs only while PLOUGH·tau_b over the frontal
area it meets stays within its hold. Before every run the run is replayed on the sim's own surface, a row at a time,
at the paste's tau_b then (``RideWatch``); a world whose run needs more than ``HOLD``, or whose held trowel would cut
at its set-down (theirs, pressed and not sliding, cuts nothing at 0.5 kPa), fails. ``ride_need`` replays the plan on
the start before the run and refuses a plan over the hold.

The scripted policy (a privileged carriage, ``concrete_tools``; no robot, as their fixtures):
1. ``passes`` passes over lanes along the pad's length (``lanes``), the blade's long side across them, its ends ``RIM``
   in from the long forms and its face reversing ``RIM`` in from the end forms, each lane the other way from its last
   pass, at 0.2 m/s; the level ``LEVEL0`` above the form-top plane first, then ``STEP`` lower each pass down to
   ``LEVEL_MIN`` and held there (``levels``). The end spill lands in the next lane (along the motion: a row meets a
   20 mm strip of it, never its length) or in the rim between the arcs and the core; the passes at the lowest level
   carry the strips out to the rim (about halving them each pass).
2. ``cross_passes`` passes of a cross lane along each end form (the blade along it): the cells within a blade's width
   of a lane's reversal pass under only part of the sole there (their per-pass work: the bearing patch's extent along
   the motion); here they get whole passes. They run last, when the strips they cross are gone, each pass from the
   lane's middle out to either stop (``outward``): no run sets down by the pile at a stop, which the long forms'
   spill strips and the last run's roll build there.
3. Every run sets down ``TOUCH`` above its level and comes down to it over the first ``RAMP`` of travel (no cut while
   pressed). A lane run comes up ``TOUCH`` again over its last ``RISE``: at its stop lies the roll its lane's last run
   the same way left there, and a sole held at its level meets that pile's toe along the blade's whole length in one
   row (above the hold). The rising sole leaves the roll's toe under it, cut to the rise (up to ``TOUCH`` high); the
   cross lanes plane it along its length. Nothing planes after a cross run, so it comes up only ``CROSS_LIFT`` over
   its last ``CROSS_RISE``: its stop's pile slumps into its last row alone (0.7 mm deep), and under 1.2 mm is left
   in the core's edge rows. Back to the cart.
Deviations, disclosed: held at a level, the trowel loads the paste with nothing, so the laws that read the pressure
(embedding, sealing, texture's yield ratio) run at their floor; D and texture do not depend on it here (y = m_i >
p/(N_c·tau_s)), and with no bleed left at 165 min the trapped water, blisters and dusting stay 0 whatever the sealing.
``HOLD`` is an assumed push of 20 N: their hold is the larger of the applied push, the tool's own force and its weight
(``coupling.py:673``) with no fixed finisher load; on its weight alone (4.4 N) their trowel would ride over the first
pass's cut. Flat sole, not their finisher's 2° (a pitched sole leaves its wedge on lift-off, pad-edge). The paste
planed off (about 0.25 mm over the core) ends as a ridge in the 10-30 mm rim, which no criterion reads (their radius
lines read 1.5-9.5 mm at no shift). A run's set-down and a lane run's rise keep its first and last 18.5 mm of travel
above their bearing band. The cross lanes work the lanes' ends, but there the core's cells by the long forms pass
under a cross run only in its last few mm, rising: 218 cells (0.09 % of the core, four strips of about 22 x 8 mm)
get under half a pass. Not judged as theirs (``concrete_ports.PAD_TROWEL``): ``arms_clear`` (no robot: None) and
``forms_undisturbed`` (no forms as bodies: a clearance proxy, ``pad_forms``); ``tools_back`` is the port's tools' end
poses (theirs: three tools in the cart's footprint).

A corner window is scored as part of their whole pad (``pad_window``): the cells its runs pass less than the whole
pad's would (beyond their reach toward the open edges, by their stops there) are reset to the start
(``window_reset``); every other cell, the forms' arcs, rim and core edge included, is scored as the run left it.

pad-trowel's own runs (``task()``, the CLI) add the margin lifts (``pad_trowel_lifts``, WP15's rulings 2026-11-08). On
the whole pad the paste planed off (about 258 cm³ net) stacked at the lanes' stops past the hold. So after each lane
pass from the second on that another pass follows, the margin trowel lifts what the stops left into the tub, while
the trowel waits; after the last lane pass also the side beads the first cross pass would meet. The steel trowel
works every core cell the margin trowel touches again at its level. The lowest level is -0.20 mm there
(``TASK_LEVEL_MIN``). ``PadTrowel()``, as pad-finish's trowel builds it, has neither. Disclosed:
- GA scripts none; their report puts the margin trowel on the pad's cart. The lifts are Alloy's policy with their tool.
- The scripted policy reads the paste at each lift to set the margin trowel down (privileged sight, like the carriage).
- The last lane round's end spill leaves a 0.8-1 mm ridge rectangle over |x| < 0.22 (0.8 mm above the mean plane on
  the v1 L4 check). No pass re-trowels there, so it is not knifed. Their criteria read it only through the
  straightedge; their own report keeps its excess piles as a stated limitation. The row gives its height.
- Piles stay in the 20 mm by the long forms at the lane ends (outside the core), where no pass after the last lane
  lift trowels at its level.

Run: ``python -m alloy.tasks.pad_trowel --window 0.42 0.42`` (a corner, CPU) or the whole pad (``--worlds 16 --device
cuda:0``); ``--maps`` saves world 0's surface at each lift, ``--no-lifts`` runs the trowel alone.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import numpy as np

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_finish_margin as MG
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_window as PW
from alloy.tasks.concrete_pad_start import FORM_H, INTERIOR
from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolCarriage, ToolProgram

TASK = "pad-trowel"
# steel trowel PB64D (params.py:143; tools.py:89-101): blade 356 x 101.6 x 1.5 mm (collision 2 mm), 0.45 kg; sole
# work: texture 0.03 mm, densification 1.0, sealing 1.0. Body x across the blade (the motion), y along it.
TROWEL = dict(half=(0.0508, 0.178, 0.001), mass=0.45, work=(3.0e-5, 1.0, 1.0))
ROLL_LIFT = 0.001                          # m: their trowel's roll lift (params.py:136)
CART = (0.0, 0.80, 0.30)                   # the trowel's place on the cart, clear of the pad (pad frame)
RIM = 0.030                                # m: blade ends and reversing face this far in from the forms: the 20 mm end
                                           # spill lies in 10-30 mm, off the core (> 30 mm) and the arcs' lines
LEVEL0, STEP, LEVEL_MIN = 1.0e-4, 5.0e-5, -2.5e-4   # m above the form-top plane: first level, drop per pass, lowest
PASSES, CROSS_PASSES = 14, 8               # lane passes (8 lowering, 6 at the lowest) and cross-lane passes
GAP = 5.0e-4                               # m: film from this far under the lowest level up (fixed base below)
LIFT = 0.02                                # m: the trowel's clearance moving between lanes
TOUCH, RAMP = 0.004, 0.02                  # m: set down this far above a run's level, down to it over this travel
RISE = 0.02                                # m: up by TOUCH again over a lane run's last this much (off its stop's pile)
CROSS_RISE, CROSS_LIFT = 0.006, 0.0012     # m: a cross run comes up this much over its last this much (RAMP's slope)
HOLD = 20.0                                # N: the assumed push holding the trowel at its level
WAIT_AT = 10                               # periods on the cart before the first pass
LIMIT_S = 90 * 60.0                        # s of material clock the task allows (T:49), advisory
WINDOW_END = 0.02                          # m: a window's lanes stop this short of its open edges (the end spill)
CREDIT_TOL = 0.5                           # passes: a window cell worked this much less than the whole pad's is reset

Run = tuple[tuple[float, float], tuple[float, float], float]     # the trowel centre's start and end, its yaw


@dataclass
class PadTrowel:
    """Settings: an optional window from the south-west inner corner, seed, site, the levels and the passes."""

    seed: int = 0
    temperature: float = 21.1
    air: str = "still"
    window: tuple[float, float] | None = None   # m east and north of the inner corner; None: the whole pad
    speed: float = 0.2                          # m/s along a lane
    air_speed: float = 0.6                      # m/s in the air
    passes: int = PASSES
    cross_passes: int = CROSS_PASSES
    level0: float = LEVEL0
    step: float = STEP
    level_min: float = LEVEL_MIN
    touch: float = TOUCH
    ramp: float = RAMP
    rise: float = RISE
    cross_rise: float = CROSS_RISE
    cross_lift: float = CROSS_LIFT
    cart: tuple[float, float, float] = CART
    exchange_dt: float = PR.EXCHANGE_DT
    omega: float = PR.CARRIAGE_OMEGA
    roll_lift: float | None = ROLL_LIFT
    rim: float = RIM
    lifts: bool = False                         # the margin trowel lifts the planed excess between passes
                                                # (pad_trowel_lifts; pad-trowel's own runs, never pad-finish's)


TASK_LEVEL_MIN = -2.0e-4                   # m: pad-trowel's lowest level with the lifts (WP15's ruling C, 11-08)


def task(**kw: Any) -> PadTrowel:
    """pad-trowel's own settings: the margin lifts on, the lowest level ``TASK_LEVEL_MIN`` (``kw`` overrides)."""
    return PadTrowel(**{"level_min": TASK_LEVEL_MIN, "lifts": True, **kw})


def start(task: PadTrowel, seed: int) -> dict[str, Any]:
    return PR.window(TASK, seed, task.temperature, task.air, task.window)


def trowel_body(task: PadTrowel) -> PR.ToolBody:
    return PR.ToolBody("trowel", TROWEL["half"], TROWEL["mass"], task.cart)


def levels(task: PadTrowel) -> list[float]:
    """Each lane pass's sole level [m above the form-top plane]: ``level0``, then ``step`` lower, to ``level_min``."""
    return [max(task.level0 - k * task.step, task.level_min) for k in range(task.passes)]


def split(task: PadTrowel):
    """base (fixed), film and paste mask on the chart: inside the pad the paste above ``GAP`` under the lowest level
    is film (what a pass can cut or the roll fill), the forms carry their smear as film."""
    cut = FORM_H + task.level_min - GAP

    def chart(f: dict[str, Any]) -> dict[str, np.ndarray]:
        inside, top = f["paste_mask"], f["b"] + f["tb"] + f["ts"]
        base = np.where(inside, np.minimum(top, cut), f["b"])
        return {"base": base, "film": np.where(inside, top - base, f["smear"]), "paste": inside}
    return chart


# --- the runs -----------------------------------------------------------------------------------------------------

def lanes(task: PadTrowel, f: dict[str, Any]) -> list[Run]:
    """The trowel's runs, (start, end, yaw) of its centre, the lanes first:
    - lanes along x (yaw 0, the blade's long side along y), the blade from ``rim`` in from the south form to ``rim`` in
      from the north one, as few as cover it, the face reversing ``rim`` in from the west and east forms;
    - a cross lane along each end form (yaw pi/2, the blade along x from ``rim`` in from it), as far as the lanes'
      blades reach.
    A window keeps the runs whose blade fits it, stopping ``WINDOW_END`` short of its open edges; its cross lane ends
    ``WINDOW_END`` short of where its lane's blade does (beyond lies that lane's end spill, which a whole pad's next
    lane carries on and a window's never does)."""
    hx, hy = INTERIOR[0] / 2, INTERIOR[1] / 2
    a, b = TROWEL["half"][0], TROWEL["half"][1]
    xe = min(hx - task.rim, float(f["x"][-1]) - WINDOW_END)          # how far east and north the blade may reach
    yn = min(hy - task.rim, float(f["y"][-1]) - WINDOW_END)
    ys, yt = -hy + task.rim + b, hy - task.rim - b
    n = max(1, math.ceil((yt - ys) / (2 * b)) + 1)
    out: list[Run] = []
    x0, x1 = -hx + task.rim + a, xe - a
    for k in range(n):
        y = ys + k * (yt - ys) / (n - 1) if n > 1 else ys
        if y + b <= yn + 1e-9:
            out.append(((x0, y), (x1, y), 0.0))
    y0, y1 = -hy + task.rim + a, max(r[0][1] for r in out) + b - a
    y1 -= WINDOW_END if y1 < hy - task.rim - a - 1e-9 else 0.0
    for x in (-hx + task.rim + b, hx - task.rim - b):
        if x + b <= xe + 1e-9:
            out.append(((x, y0), (x, y1), math.pi / 2))
    return out


def ordered(runs: list[Run], k: int) -> list[Run]:
    """Pass k over ``runs``: run j forward when k + j is even (each run the other way from its last pass), the runs'
    order reversed every other pass."""
    seq = [(p0, p1, yaw) if (k + j) % 2 == 0 else (p1, p0, yaw) for j, (p0, p1, yaw) in enumerate(runs)]
    return seq if k % 2 == 0 else seq[::-1]


def outward(runs: list[Run], k: int) -> list[Run]:
    """Pass k over the cross lanes ``runs`` (along y): each from the pad's middle line (y = 0, within the lane: a
    window's lane starts at its reach) out to either stop, its start's stop first on even passes."""
    out: list[Run] = []
    for p0, p1, yaw in runs:
        mid = (p0[0], min(max(0.0, min(p0[1], p1[1])), max(p0[1], p1[1])))
        pair = [(mid, p, yaw) for p in (p0, p1) if math.dist(mid, p) > 1e-9]
        out += pair if k % 2 == 0 else pair[::-1]
    return out


def passes(task: PadTrowel, f: dict[str, Any]) -> list[tuple[float, list[Run]]]:
    """(level, runs in order) per pass: the lane passes at ``levels``, then the cross-lane passes at the lowest."""
    runs = lanes(task, f)
    xl, cl = [r for r in runs if r[2] == 0.0], [r for r in runs if r[2] != 0.0]
    return ([(lv, ordered(xl, k)) for k, lv in enumerate(levels(task))]
            + [(task.level_min, outward(cl, k)) for k in range(task.cross_passes if cl else 0)])


def rise(task: PadTrowel, run: Run) -> tuple[float, float]:
    """(travel, height) a run comes up over at its stop: a lane run ``touch`` over ``rise``, a cross run only
    ``cross_lift`` over ``cross_rise``. A rising sole leaves the roll it carries under it, cut to the rise, and
    nothing after the cross lanes planes theirs."""
    return (task.rise, task.touch) if run[2] == 0.0 else (task.cross_rise, task.cross_lift)


def sole_run(task: PadTrowel, run: Run, level: float) -> RD.SoleRun:
    (p0, p1, yaw), (ha, hc) = run, TROWEL["half"][:2]
    return RD.SoleRun(p0, p1, yaw, ha, hc, level, task.touch, task.ramp, *rise(task, run))


@dataclass(frozen=True)
class Plan:
    program: ToolProgram
    end: float                                     # s: the program's end
    runs: tuple[tuple[float, RD.SoleRun], ...]     # each run's sole, from the time it hangs over its start


def run_keys(task: PadTrowel, run: Run, z: float, t: float) -> tuple[list[Stroke], float]:
    """One run's keys from ``t`` (hanging ``LIFT`` over its start): set down ``touch`` above ``z``, down to it over
    ``ramp`` of travel, at it to its ``rise`` short of the end, up by its lift at the end, lifted. Returns the keys
    and the time it is lifted."""
    a, b, yaw = run
    length, (r, lift) = math.dist(a, b), rise(task, run)

    def at(s: float) -> tuple[float, float]:
        return a[0] + s / length * (b[0] - a[0]), a[1] + s / length * (b[1] - a[1])

    up, down = (task.touch if task.ramp > 0.0 else 0.0), (lift if r > 0.0 else 0.0)
    path = [(0.0, z + up), (task.ramp, z), (length - r, z), (length, z + down)]
    path = [p for i, p in enumerate(path) if i == 0 or p[0] > path[i - 1][0]]          # a zero ramp or rise: no key
    keys = [Stroke(t, (*a, z + LIFT), yaw=yaw, mode=HEIGHT)]
    t += 0.3
    for i, (s, zk) in enumerate(path):
        t += (s - path[i - 1][0]) / task.speed if i else 0.0
        keys.append(Stroke(t, (*at(s), zk), yaw=yaw, mode=HEIGHT))
    t += 0.3
    keys.append(Stroke(t, (*b, z + LIFT), yaw=yaw, mode=HEIGHT))
    return keys, t


def plan(task: PadTrowel, f: dict[str, Any], t0: float = 0.0,
         after_pass: Callable[[int, float, float], float] | None = None) -> Plan:
    """From the cart at ``t0``: every pass (``passes``), each run as ``run_keys``; then back to the cart.
    ``after_pass(k, level, t)``, when given, is called once pass k ends (lifted at its last stop at ``t``) and returns
    when the trowel goes on: it waits there till then (pad-trowel's margin lifts). None: no wait, the same keys."""
    hz = TROWEL["half"][2]
    keys, timed = [Stroke(t0, task.cart, mode=HEIGHT)], []
    t, pos = t0, task.cart[:2]
    for k, (level, runs) in enumerate(passes(task, f)):
        z = FORM_H + level + hz
        for run in runs:
            t += math.dist(pos, run[0]) / task.air_speed + 0.3
            timed.append((t, sole_run(task, run, level)))
            ks, t = run_keys(task, run, z, t)
            keys += ks
            pos = run[1]
        go = t if after_pass is None else after_pass(k, level, t)
        if go > t:
            keys.append(Stroke(go, keys[-1].pos, yaw=keys[-1].yaw, mode=HEIGHT))
            t = go
    t += math.dist(pos, task.cart[:2]) / task.air_speed + 0.3
    keys.append(Stroke(t, task.cart, mode=HEIGHT))
    return Plan(ToolProgram("trowel", tuple(keys)), t + 0.5, tuple(timed))


def program(task: PadTrowel, f: dict[str, Any], t0: float = 0.0) -> tuple[ToolProgram, float]:
    p = plan(task, f, t0)
    return p.program, p.end


def footprint(run: Run, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cells the blade sweeps over a run."""
    (xa, ya), (xb, yb), yaw = run
    a, b = TROWEL["half"][:2]
    if abs(math.sin(yaw)) < 0.5:                                       # along x, the blade across it
        return (np.abs(Y - ya) <= b) & (X >= min(xa, xb) - a) & (X <= max(xa, xb) + a)
    return (np.abs(X - xa) <= b) & (Y >= min(ya, yb) - a) & (Y <= max(ya, yb) + a)


def covered(task: PadTrowel, f: dict[str, Any]) -> np.ndarray:
    """Cells the runs' blade sweeps."""
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    out = np.zeros(X.shape, bool)
    for run in lanes(task, f):
        out |= footprint(run, X, Y)
    return out


def credit(task: PadTrowel, f: dict[str, Any]) -> np.ndarray:
    """Their passes per cell over the plan, by geometry (``pad_ride.pass_credit``)."""
    return RD.pass_credit([r for _, r in plan(task, f).runs], f["x"], f["y"])


def window_reset(task: PadTrowel, f: dict[str, Any], whole_start: dict[str, Any]) -> np.ndarray:
    """A window's cells its runs pass less than the whole pad's would (by ``CREDIT_TOL``) and less than a lane's
    passes: beyond its lanes' reach toward its open edges (their reversal's roll and end spill land there) and by its
    runs' stops there (the whole pad's runs pass those cells mid-lane). Reset to the start in its score. Along the
    real forms the window's runs are the whole pad's, so nothing there is reset."""
    whole = credit(task, whole_start)[PW.window_slices(f, whole_start)]
    return credit(task, f) + CREDIT_TOL < np.minimum(whole, task.passes)


# --- their ride rule ------------------------------------------------------------------------------------------------

def tau_b(task: PadTrowel, age: float) -> float:
    return float(CM.Concrete().tau_bulk(float(age), task.temperature))


def ride_need(task: PadTrowel, f: dict[str, Any]) -> float:
    """The largest force [N] their plough needs over the plan (``pad_ride.run_need``, one row at a time) on the start,
    each run leaving it cut to its sole (no roll, no spill), at the paste's tau_b when the run starts."""
    h = (f["b"] + f["tb"] + f["ts"] - FORM_H).astype(float)
    worst = 0.0
    for t, run in plan(task, f).runs:
        r = RD.run_need(run, h, f["x"], f["y"], f["paste_mask"], tau_b(task, f["age"] + t), float(f["cell"]), h_out=h)
        worst = max(worst, r.need_n)
    return worst


class RideWatch:
    """Their ride rule on the sim's own surface before each run, per world: the largest need [N] and the volume the
    held trowel would cut at a set-down [m³]."""

    def __init__(self, task: PadTrowel, fs: list[dict[str, Any]], flow: SurfaceFlow,
                 runs: tuple[tuple[float, RD.SoleRun], ...]) -> None:
        self.task, self.fs, self.flow, self.runs = task, fs, flow, runs
        self.need = np.zeros(len(fs))
        self.pressed = np.zeros(len(fs))
        self.log: list[dict] = []                  # per run: when, where, the largest need and set-down cut
        self.last: list[RD.RunNeed] = []           # the last run's need per world

    def hooks(self) -> dict[int, Any]:
        hooks = {int(t / PR.PERIOD): partial(self.check, t, run) for t, run in self.runs}
        if len(hooks) != len(self.runs):
            raise PluginError("pad_trowel: two runs start in one control period; the ride watch checks one per period")
        return hooks

    def check(self, t: float, run: RD.SoleRun) -> None:
        f0 = self.fs[0]
        h = PR.surface(self.flow, f0["h"].shape)
        tau = tau_b(self.task, f0["age"] + t)
        rs = [RD.run_need(self.run_for(run, w), h[w], f["x"], f["y"], f["paste_mask"], tau, float(f["cell"]))
              for w, f in enumerate(self.fs)]
        self.last = rs                             # world by world (pad-trowel's lifts report the margin's)
        for w, r in enumerate(rs):
            self.need[w], self.pressed[w] = max(self.need[w], r.need_n), max(self.pressed[w], r.pressed_m3)
        top = max(rs, key=lambda r: r.need_n)
        self.log.append(dict(t=round(t, 2), start=run.start, end=run.end, yaw=run.yaw, level=run.level,
                             need_n=top.need_n, at_m=top.at_m, depth_m=top.depth_m, rows=top.rows,
                             pressed_m3=max(r.pressed_m3 for r in rs)))

    def run_for(self, run: RD.SoleRun, world: int) -> RD.SoleRun:
        """The run as world ``world``'s tool makes it (pad-trowel's lifts set a set-down's height per world)."""
        del world
        return run

    def ok(self) -> np.ndarray:
        return (self.need <= HOLD) & (self.pressed <= 0.0)


# --- plug-ins and the verdict -------------------------------------------------------------------------------------

def trowel_tool(task: PadTrowel) -> Applicator:
    hx, hy, hz = TROWEL["half"]
    return Applicator(body="trowel", shape="rect", size=(hx, hy), center=(0.0, 0.0, -hz), stiffness=PR.PAD_STIFFNESS,
                      damping=0.0, implicit=True, **({} if task.roll_lift is None else {"roll_lift": task.roll_lift}))


def plugins(task: PadTrowel, f: dict[str, Any], prog: ToolProgram, margin: ToolProgram | None = None,
            margin_worlds: tuple[ToolProgram, ...] | None = None):
    """The film with the trowel, its carriage and the paste layer. With ``margin`` (pad-trowel's lifts) the margin
    trowel too: pad_finish_margin's blade and sole work on its program, the tub's network catching its release.
    ``margin_worlds``: per world, the margin trowel's own program (its set-down heights; the trowel's is shared)."""
    nx, ny = f["h"].shape
    tools, works = (trowel_tool(task),), (SoleWork("trowel", *TROWEL["work"], half=TROWEL["half"][:2]),)
    progs, extra = (prog,), {}
    if margin is not None:
        tools, works, progs, extra = (*tools, MG.blade()), (*works, MG.sole_work()), (prog, margin), {
            "network": TUB.network(f)}
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(nx, ny)),), material=PR.surface_film(f, task.temperature),
                       inertia=False, max_depth=0.025, u_cap=PR.U_CAP, tools=tools,
                       tool_speed=max(task.speed, 0.1), per_cell_rheology=True, **extra)
    per_world = {} if margin_worlds is None else {"world_programs": tuple((prog, m) for m in margin_worlds)}
    carriage = ToolCarriage(programs=progs, omega=task.omega, omega_rot=task.omega,
                            **({} if margin is None else {"flow": flow}), **per_world)
    layer = ConcretePaste(material="concrete", temperature=task.temperature, air=task.air, film_yield="surface",
                          tools=works, flow=flow, carriage=carriage)
    return carriage, flow, layer


def run(task: PadTrowel | None = None, **kw: Any) -> Any:
    """Their pad-trowel, scripted (``pad_trowel_lifts.run``; ``task`` None: pad-trowel's own settings, ``task()``)."""
    from alloy.tasks import pad_trowel_lifts as LF                             # noqa: PLC0415 (it imports this)
    return LF.run(task, **kw)


def main(argv: list[str] | None = None) -> int:
    import argparse                                                                # noqa: PLC0415
    import json                                                                    # noqa: PLC0415
    import logging                                                                 # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, nargs=2, default=None, help="m east and north of the SW inner corner")
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="end fields and age per world, the start's heights")
    ap.add_argument("--maps", default=None, help="world 0's surface at each margin lift (pad_trowel_lifts)")
    ap.add_argument("--no-lifts", action="store_true", help="the trowel alone (before WP15's 11-08 ruling)")
    ap.add_argument("--level-min", type=float, default=None, help="m: the lowest level (default: the task's)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kw = {"lifts": False, "level_min": LEVEL_MIN} if args.no_lifts else {}
    kw.update({} if args.level_min is None else {"level_min": args.level_min})
    res = run(task(seed=args.seed, window=tuple(args.window) if args.window else None, passes=args.passes, **kw),
              worlds=args.worlds, device=args.device, npz=args.npz, maps=args.maps)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(out)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
