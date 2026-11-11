# SPDX-License-Identifier: Apache-2.0
"""pad-finish's margin pass (:func:`alloy.tasks.pad_finish_stages.margin`): the margin trowel 52D cuts the edger's bead
off before the steel trowel.

WP4's edger leaves its cut (the arris and the lip's slot, about 0.12 L round the pad) on its open side, in their 20 mm
spill band beyond the blade (``coupling.py:54``): a bead 2-4 mm over the plane 79-98 mm from the forms, and piles up
to 12.5 mm in the corners, where two sides' beads and the runs' ends meet (the 2 mm L4 run, seed 0). Left there, the
steel trowel held at the plane meets it along its whole blade: their ride rule needs 187 N at 165 min against the 20 N
hold (their trowel would ride over it), and the trowel's roll carries the bead into the rim (flatness, the arris and
the lip fail).

A finisher cuts it off with the margin trowel right after edging, while it is soft (tau_b about 22 kPa at 132 min,
61 kPa at 165): their pad-finish puts the margin trowel and the tub on its cart (``layout.py:159-161``). The blade is
the darby stage's carrying knife (``concrete_tub.store``: WP15's scraping store at their 0.15 L carry cap, the knife
fill). The scripted policy reads the stage's start surface, as a mason reads the bead:
1. per side, two runs at the plane ``ref`` (the steel trowel's floor) out to the corners: the first from the middle,
   the second from ``2 HA + BEYOND`` past it on the first's way, back over the middle; the blade's length along the
   bead, its inner edge ``IN`` from the form (clear of the edger's 77.7 mm track) and its 50.8 mm across the bead;
2. a run sets down at the plane where nothing under the blade stands over it, else ``CLEAR`` over the highest and
   comes down over ``RAMP`` of travel (their set-down: a pressed sole cuts nothing);
3. a loaded knife fills every low under it from its carry up to its sole, so a blade coming down onto paste leaves a
   wedge behind its trailing end (0.8 mm in the CPU smoke), and a pass over the plane would spread what it cut as a
   slab at its level (set-downs planned without this pressed 0.5-2 mL into such paste). The plan lays the most the
   knife can (``lay``: every cell it covers up to the sole's height when it last covers it), so later set-downs clear
   it and later runs cut it: the second run, set down at the plane on paste the first left there, slides through
   the first's set-down wedge. One pass at the plane takes the corner piles within their hold (15 N on the L4 run's
   edge end at the margin's age; levels 4 mm apart needed 30 N, cutting the slabs they laid);
4. after each side, over the tub, letting go of the carry (``release``; the supply's tub pool catches it); back to the
   cart at the end.
Their ride rule is planned on the start (``plan``: each run cut to its sole, ``pad_ride.run_need``, with ``lay``) and
watched on the sim's surface before every run (``pad_trowel.RideWatch``: tau_b at the paste's age then, their 20 N
assumed push). Disclosed: GA's pad-finish has no scripted solution; this pass is Alloy's policy with their tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import pad_ride as RD
from alloy.tasks import pad_run as PR
from alloy.tasks.concrete_pad_start import FORM_H
from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolCarriage, ToolProgram

HA, HC, HZ = TUB.TROWEL["half"]            # m: the blade's half length (along the run), half width, half thickness
IN = 0.078                                 # m: the blade's inner edge from the form, on a cell edge (2 mm cells
                                           # from the form face; the edger's flat and lip end at 77.7)
RAMP = 0.02                                # m of travel a run comes down over from its set-down
CLEAR = 0.001                              # m: set down this far over what stands under the blade
BEYOND = 0.005                             # m: a side's second run starts this far past the first's set-down
LIFT = 0.02                                # m: lifted this far over the plane between runs
SPEED, AIR_SPEED = 0.1, 0.5                # m/s: along the bead; in the air
BODY = "mtrowel"

Run = tuple[tuple[float, float], tuple[float, float], float]     # the blade centre's start and end, its yaw


@dataclass(frozen=True)
class Plan:
    program: ToolProgram
    end: float                                     # s: the program's end
    runs: tuple[tuple[float, RD.SoleRun], ...]     # each run's sole, from the time it hangs over its start
    need_n: float                                  # the plan's largest ride need on the start
    pressed_m3: float                              # what its set-downs would cut on the start (kept at zero)
    cut_m3: tuple[float, ...]                      # per world: what the plan cuts on its start, less what it lays
    left: np.ndarray                               # per world: the start's surface as the plan leaves it [W, nx, ny]


def surface(f: dict[str, Any]) -> np.ndarray:
    """Their surface [m, from the form-top plane] (b + tb + ts)."""
    return (np.asarray(f["b"], float) + np.asarray(f["tb"], float) + np.asarray(f["ts"], float)) - FORM_H


def sides(f: dict[str, Any]) -> list[list[Run]]:
    """Per side (south, east, north, west): the first run from its middle out to one corner, the second from
    ``2 HA + BEYOND`` past the middle on the first's way (paste the first left at the plane) back over the middle out
    to the other corner; the blade's inner edge ``IN`` from the form, its ends ``IN`` from the side forms."""
    x0, x1, y0, y1 = PR.paste_faces(f)
    xm, ym, off = 0.5 * (x0 + x1), 0.5 * (y0 + y1), IN + HC
    xa, xb, ya, yb = x0 + IN + HA, x1 - IN - HA, y0 + IN + HA, y1 - IN - HA
    return [two_runs(y0 + off, xm, xa, xb, 0.0), two_runs(x1 - off, ym, ya, yb, math.pi / 2),
            two_runs(y1 - off, xm, xa, xb, 0.0), two_runs(x0 + off, ym, ya, yb, math.pi / 2)]


def two_runs(fixed: float, mid: float, lo: float, hi: float, yaw: float) -> list[Run]:
    """One line's two runs (``sides``; pad-trowel's lifts): the first from ``mid`` out to ``hi``, the second from
    ``2 HA + BEYOND`` past ``mid`` on the first's way (paste the first left at the plane) back over ``mid`` out to
    ``lo``, so it slides through the first's set-down wedge. Along x (yaw 0, at y = ``fixed``) or along y (yaw pi/2,
    at x = ``fixed``); the blade centre's extent along the line is [lo, hi]."""
    def pt(s: float) -> tuple[float, float]:
        return (s, fixed) if yaw == 0.0 else (fixed, s)
    return [(pt(mid), pt(hi), yaw), (pt(min(mid + 2.0 * HA + BEYOND, hi)), pt(lo), yaw)]


def under_start(run: Run, h: np.ndarray, f: dict[str, Any]) -> float:
    """The highest paste cell [m] under the blade at the run's start (-inf when none)."""
    (a, b, yaw), m = run, np.asarray(f["paste_mask"]) > 0.5
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    c, s = math.cos(yaw), math.sin(yaw)
    along, across = (X - a[0]) * c + (Y - a[1]) * s, (Y - a[1]) * c - (X - a[0]) * s
    under = m & (np.abs(along) <= HA + 1e-9) & (np.abs(across) <= HC + 1e-9)
    return float(h[under].max()) if under.any() else -np.inf


def touch(run: Run, level: float, h: np.ndarray, f: dict[str, Any]) -> float:
    """How far over ``level`` to set the blade down at the run's start: ``CLEAR`` over the highest cell under it, or
    not at all (down at its level) when none stands over the level."""
    hi = under_start(run, h, f)
    return hi - level + CLEAR if hi > level else 0.0


def run_keys(run: Run, level: float, up: float, t: float, speed: float = SPEED) -> tuple[list[Stroke], float]:
    """One run at ``level``: hanging ``LIFT`` over its set-down, set down ``up`` over the level, down to it over
    ``RAMP``, along to its end at ``speed``, lifted. Returns the keys and the time it is lifted."""
    (a, b, yaw), z = run, FORM_H + level + HZ
    length = math.dist(a, b)
    ramp = min(RAMP, length)
    mid = (a[0] + ramp / length * (b[0] - a[0]), a[1] + ramp / length * (b[1] - a[1]))
    keys = [Stroke(t, (*a, z + up + LIFT), yaw=yaw, mode=HEIGHT)]
    t += 0.3
    keys.append(Stroke(t, (*a, z + up), yaw=yaw, mode=HEIGHT))
    t += ramp / speed
    keys.append(Stroke(t, (*mid, z), yaw=yaw, mode=HEIGHT))
    if length - ramp > 1e-9:
        t += (length - ramp) / speed
        keys.append(Stroke(t, (*b, z), yaw=yaw, mode=HEIGHT))
    t += 0.3
    keys.append(Stroke(t, (*b, z + LIFT), yaw=yaw, mode=HEIGHT))
    return keys, t


def fly(keys: list[Stroke], t: float, p: tuple[float, float, float], release: float = 0.0,
        yaw: float = 0.0) -> float:
    """Through the air to ``p``, turning to ``yaw`` on the way (a run's heading: turned before it sets down)."""
    t += math.dist(keys[-1].pos, p) / AIR_SPEED + 0.3
    keys.append(Stroke(t, p, yaw=yaw, mode=HEIGHT, release=release))
    return t


def empty(keys: list[Stroke], t: float, f: dict[str, Any]) -> float:
    """Over the tub, let go of the carry (``concrete_tub``'s flight and release), back up."""
    tx, ty, _ = TUB.tub(f).center
    t = fly(keys, t, (tx, ty, FORM_H + TUB.AIR_Z))
    t = fly(keys, t, (tx, ty, FORM_H + TUB.DROP_H + HZ), release=1.0)
    return fly(keys, t + TUB.RELEASE_S, (tx, ty, FORM_H + TUB.AIR_Z))


def tau_b(age: float, temperature: float) -> float:
    return float(CM.Concrete().tau_bulk(float(age), temperature))


def last_under(sole: RD.SoleRun, f: dict[str, Any]) -> np.ndarray:
    """Per cell [nx, ny]: the sole's height the last time the run's blade covers it (its level, or on the ramp the
    height it is coming down from when the cell leaves under its trailing end; a rising blade lays nothing), -inf
    for cells it never covers."""
    X, Y = np.meshgrid(f["x"], f["y"], indexing="ij")
    (ax, ay), (bx, by) = sole.start, sole.end
    length = math.dist(sole.start, sole.end)
    ux, uy = (bx - ax) / length, (by - ay) / length
    along, across = (X - ax) * ux + (Y - ay) * uy, np.abs((Y - ay) * ux - (X - ax) * uy)
    under = (along >= -HA - 1e-9) & (along <= length + HA + 1e-9) & (across <= HC + 1e-9)
    last = np.clip(along + HA, 0.0, length)                      # the travel at which the cell leaves (or the end)
    z = sole.level + (sole.touch * np.clip(1.0 - last / sole.ramp, 0.0, 1.0) if sole.ramp > 0.0 else 0.0)
    return np.where(under, z, -np.inf)


def lay(sole: RD.SoleRun, h: np.ndarray, f: dict[str, Any]) -> float:
    """The most a loaded knife can lay on ``h`` over a run (in place): it fills every low under it from its carry up
    to its sole (``stores_fill``), so a cell it covers is left at least at the sole's height when it last covers it
    (``last_under``). Returns that volume [m³]."""
    up = np.where(np.asarray(f["paste_mask"]) > 0.5, np.maximum(last_under(sole, f) - h, 0.0), 0.0)
    h += up
    return float(up.sum()) * float(f["cell"]) ** 2


@dataclass
class _Planner:
    """The plan as it is built: the keys and timed runs so far, each world's surface as their rule leaves it with the
    most the knife can lay under each run (``lay``: later set-downs clear it, later runs cut it back), the worst need
    and set-down cut, and the net cut per world (what their rule cuts less what is laid)."""

    fs: list[dict[str, Any]]
    hs: list[np.ndarray]
    temperature: float
    keys: list[Stroke]
    t: float
    timed: list[tuple[float, RD.SoleRun]] = field(default_factory=list)
    need: float = 0.0
    pressed: float = 0.0
    cut: np.ndarray | None = None

    def run(self, run: Run, level: float) -> None:
        """One run at ``level``: flown to over its set-down, timed for the ride watch, replayed on every world."""
        up = touch(run, level, np.max(self.hs, axis=0), self.fs[0])
        self.t = fly(self.keys, self.t, (*run[0], FORM_H + level + HZ + up + LIFT), yaw=run[2])
        sole = RD.SoleRun(run[0], run[1], run[2], HA, HC, level, up, min(RAMP, math.dist(run[0], run[1])))
        self.timed.append((self.t, sole))
        net = []
        for h, f in zip(self.hs, self.fs, strict=True):
            r = RD.run_need(sole, h, f["x"], f["y"], np.asarray(f["paste_mask"]) > 0.5,
                            tau_b(float(f["age"]) + self.t, self.temperature), float(f["cell"]), h_out=h)
            self.need, self.pressed = max(self.need, r.need_n), max(self.pressed, r.pressed_m3)
            net.append(r.cut_m3 - lay(sole, h, f))
        self.cut = np.asarray(net) + (0.0 if self.cut is None else self.cut)
        ks, self.t = run_keys(run, level, up, self.t)
        self.keys += ks[1:]


def plan(fs: list[dict[str, Any]], ref: float, t0: float = 0.0, temperature: float = 21.1) -> Plan:
    """From the cart at ``t0``: each side's two runs at the plane ``ref``, its carry let go over the tub; back to the
    cart. One program for every world: the set-downs read on the highest of the worlds' surfaces as the plan leaves
    them, each run replayed on each world (the plan's need and set-down cut: the worst)."""
    f0 = fs[0]
    p = _Planner(fs, [surface(f) for f in fs], temperature, [Stroke(t0, TUB.home(f0), mode=HEIGHT)], t0)
    for side in sides(f0):
        for run in side:
            p.run(run, ref)
        p.t = empty(p.keys, p.t, f0)
    t = fly(p.keys, p.t, TUB.home(f0))
    return Plan(ToolProgram(BODY, tuple(p.keys)), t + 0.5, tuple(p.timed), p.need, p.pressed,
                tuple(np.asarray(p.cut).tolist()), np.stack(p.hs))


def body(f: dict[str, Any]) -> PR.ToolBody:
    return PR.ToolBody(BODY, TUB.TROWEL["half"], TUB.TROWEL["mass"], TUB.home(f))


def blade() -> Applicator:
    """The margin trowel's blade: ``concrete_tub.store`` (WP15's scraping store at their carry cap) on a flat pad."""
    return Applicator(body=BODY, shape="rect", size=(HA, HC), center=(0.0, 0.0, -HZ), stiffness=PR.PAD_STIFFNESS,
                      damping=0.0, implicit=True, store=TUB.store())


def sole_work() -> SoleWork:
    """What the blade's sole does to the paste under it (their margin trowel's per-pass work)."""
    return SoleWork(BODY, *TUB.TROWEL["work"], half=(HA, HC))


def plugins(f: dict[str, Any], prog: ToolProgram, temperature: float = 21.1, air: str = "still"):
    """The blade (``blade``), the film on the plane's chart with the tub's supply, the carriage driving the release,
    and the layer working under the blade, writing the bulk yield into the film (the bead is the edger's bulk paste,
    as the waits' film)."""
    nx, ny = f["h"].shape
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(nx, ny)),), material=PR.surface_film(f, temperature),
                       inertia=False, max_depth=0.025, u_cap=PR.U_CAP, tools=(blade(),), tool_speed=SPEED,
                       per_cell_rheology=True, network=TUB.network(f))
    carriage = ToolCarriage(programs=(prog,), omega=PR.CARRIAGE_OMEGA, omega_rot=PR.CARRIAGE_OMEGA, flow=flow)
    layer = ConcretePaste(material="concrete", temperature=temperature, air=air, film_yield="bulk",
                          tools=(sole_work(),), flow=flow, carriage=carriage)
    return carriage, flow, layer


def band(f: dict[str, Any]) -> np.ndarray:
    """The cells the blade passes over (``IN`` to ``IN`` + its width from a form) [nx, ny]."""
    d = np.asarray(f["d_edge"])
    return (np.asarray(f["paste_mask"]) > 0.5) & (d >= IN) & (d <= IN + 2.0 * HC)


__all__ = ["BEYOND", "BODY", "CLEAR", "IN", "LIFT", "Plan", "RAMP", "SPEED", "band", "blade", "body", "empty", "fly",
           "last_under", "lay", "plan", "plugins", "run_keys", "sides", "sole_work", "surface", "tau_b", "touch",
           "two_runs", "under_start"]
