# SPDX-License-Identifier: Apache-2.0
"""task concrete_pad: Gym-Anything's pad-darby, a surface-flow concrete proof.

"Darby it: take out the high and low spots and get the stone down before the bleed water comes up. There is concrete
in the tub for low spots. Tools back on the cart." (``task.py:20-22``; spec
``[internal note]`` §1). A 1.219 x 0.914 m pad inside 2x4 forms, 89 mm of
concrete 5 min old, struck but not flat: ridges, a low corner (-5 mm), a high band, noise, coarse aggregate exposed
(E about 0.3), a 0.8 mm cream layer, smear on two form tops. The start is theirs cell for cell
(``concrete_pad_start``, ``layout.py:225-345``), block-averaged to a coarser grid for a light run.

Model: one ``surface_flow`` plane chart on the subgrade under the pad, its forms and 150 mm of ground beyond each end
form (Alloy's, where a pushed roll falls off as on a site). The darby never loads the concrete past its bearing
capacity (held by the carriage, it presses the paste with at most the carriage's push: under 30 N on the paste, under
0.4 kPa over its 0.078 m² sole, against 5.14·tau_b, about 8.4 kPa), so the concrete below the sole's lowest level
is the chart's fixed ``base`` and the paste above it is the film, at the bulk's yield: the high spots, the 0.8 mm
cream and the form-top smear (once: their ``tb`` on a form cell is that smear). The tools have no rigid contacts;
their pads are implicit. A privileged scripted carriage (``concrete_tools``; no robot: their probes would label it
assisted) runs two tools, each world on its own program read from its start surface (``world_programs``):
- the margin trowel fills the world's low corner from the tub (``concrete_tub``, WP15's ruling): their carrying
  knife on a surface-flow store at their carry cap, the tub a Tray at their place (and a supply pool for what the
  blade sheds). Each trip it dips, lifts (the blade keeps their stick limit, ~38 mL of 5 min concrete), flies over the
  low and lets go: the gob falls through DC5's supply and lands as their cone. The gobs go cell by cell to the
  deepest pockets near the low, 1.5 times its deficit in loads (9 trips): the pad holds 0.14 mm less paste than its
  form plane on average, so a darby alone runs dry before the low corner, and pockets its roll passes over stay;
- then the darby 4626D (WP13's roll cap, their 3 mm roll lift) makes ``passes`` passes back and forth over the pad's
  length, the last toward the low (``heading``), each from over one end form to past the other: high spots and the
  gob piles are cut to the sole and their paste rolls ahead into the lows and off the far form onto the ground (their
  plough, ``coupling.py:513-642``); both ends get passes that arrive with a full roll. The sole runs ``below_plane``
  under the form-top plane (their straightedge's reference, ``task.py:253``), and ``late_below`` (1.5 mm) on the last
  ``late_passes`` while the blade is clear of the forms (the stone in the ridges' troughs, about 1 mm low, is worked
  only by a sole that reaches it); then ``end_strokes`` short strokes at that level over each end band, where a blade
  held inside the forms covers a cell only over its distance in. Every pass embeds the stone
  (``coupling.py:1024-1025``); held by the carriage, the darby loads the paste with nothing, so the layer embeds at
  the law's floor (0.1 kPa).
Both end on the cart. The process layer (``concrete_paste``) ages and bleeds the concrete and keeps E, S and the
wet-work record. Verdict: their criteria (``concrete_metrics.pad_darby``). Disclosed: the pad ends 1.5 mm under the
form plane (the late passes' level; their flat_plane allows 3 mm), which a following trowel stage must hold to. The
verdict judges 5 of their 8 criteria (audit AB-gym-7; ``concrete_ports.PAD_DARBY``): with no robot (a privileged
carriage; their task interface is robot-only) ``arms_clear`` is not judged, nor is ``forms_undisturbed`` (the forms
are cells, not bodies, and no force on them is measured: a known gap); ``tools_back`` is judged in part (the darby
and the margin trowel at their rests; theirs: three tools in the cart's footprint). Also disclosed (SCR-WP6-5,
open): the film runs surface flow's default u_cap
(1.5 m/s). Late in the program (2 mm grid) up to about 450 faces on the sole's leading edge sit at it, where the
roll slumps into the edge cell, and the hand-off's straightedge depends on the cap: 0.516 mm at 1.5, 0.789 at 3.0
(SC9's probe). The plane, flat_plane, stone_down and the books do not move. WP15 bounds the flux under a sole by its
gap in surface flow (after 11-14).

Run: ``python -m alloy.tasks.concrete_pad [--worlds W] [--device cuda:0] [--factor 5] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import FilmMaterial, PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.tasks import concrete_materials as CM
from alloy.tasks import concrete_metrics as MX
from alloy.tasks import concrete_tub as TUB
from alloy.tasks import concrete_verdict as CV
from alloy.tasks.concrete_pad_start import FORM_H, blocked, pad_fields
from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
from alloy.tasks.concrete_ports import PAD_DARBY
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolCarriage, ToolProgram, add_tool_body, lane

log = logging.getLogger("alloy.tasks.concrete_pad")

FORM_TOP = FORM_H                           # m: the forms' top, the pad's plane (params.py:30-32; task.py:253)
INTERIOR_HALF = (0.6095, 0.457)             # m (params.py:30-31)
SURFACE_VISCOSITY = 5.0                     # Pa·s: the surface layer (design note §7, estimate; NISTIR 6094 range)
# darby 4626D, magnesium blade 914 x 85 x 8 mm, 1.30 kg; sole work: densification, sealing, texture (tools.py:104-115)
DARBY = dict(half=(0.457, 0.0425, 0.004), mass=1.30, work=(4.0e-4, 0.2, 0.05))
DARBY_FROM_PAD = (1.15, -1.07)              # its place on their cart from the pad's centre (layout.py:148-150)
EDGE_IN = INTERIOR_HALF[0] - DARBY["half"][1]   # m from the centre: the blade clear of the east and west forms
RAMP = 0.01                                 # m of travel to go down to (or up from) a late pass's level
END_RUN = 0.10                              # m: an end stroke's travel, from its outer limit inward
END_CLEAR = 0.005                           # m: an end stroke's blade edge stops this short of the end form's face
OVERRUN = 0.03                              # m: a pass turns with the darby's centre this far past the inner face
GROUND = 0.15                               # m of ground modelled beyond each end form, where the roll falls
RETURN_LIFT = 0.10                          # m: the return flight clears the roll piled on the far form (cm tall)
ROLL_LIFT = 0.003                           # m: a float's roll lift (params.py:136): caps the darby's roll at
                                            # 0.5·(rise + lift)²·L, the rest spills at its ends (coupling.py:597-634)


@dataclass
class PadDarby:
    """Settings: grid (their 2 mm cells x ``factor``), seed, site temperature, and the scripted passes."""

    factor: int = 5                          # 1: theirs (2 mm); 5: 10 mm, a light CPU run
    seed: int = 0
    temperature: float = 21.1
    passes: int = 8                          # passes back and forth, the last toward the low spot (``heading``)
    speed: float = 0.1                       # m/s along a pass
    below_plane: float = 0.0                 # m: the sole held this far under the form-top plane
    late_below: float = 0.0015               # m: ... and this far, between the forms' inner faces, on the last
    late_passes: int = 5                     # ``late_passes`` (the stone in the ridges' troughs, ~1 mm low, is
                                             # worked only by a sole that reaches it; over the forms: the plane)
    end_strokes: int = 6                     # short strokes there and back at the late level over each end band
    stiffness: float = 1.0e5                 # Pa/m: the tools' pads (implicit)
    air_speed: float = 0.5                   # m/s for the returns in the air
    tub: bool = True                         # fill the low spot from the tub first
    tub_fill: float = 1.5                    # trowel loads for this times the low's deficit (the darby strikes the
                                             # excess off; GA's cones cover only part of the low's deep cells)


def start(task: PadDarby, seed: int) -> dict[str, Any]:
    f = pad_fields("pad-darby", seed, task.temperature)
    return with_ground(blocked(f, task.factor) if task.factor > 1 else f, GROUND)


def with_ground(f: dict[str, Any], width: float) -> dict[str, Any]:
    """Their region plus ``width`` of ground beyond the west and the east form: paste a darby pushes over an end form
    falls off it there, outside the pad, as it does on a site."""
    cell = float(f["cell"])
    n = int(round(width / cell))
    out = dict(f)
    out["x"] = np.concatenate([f["x"][0] - cell * np.arange(n, 0, -1), f["x"], f["x"][-1] + cell * np.arange(1, n + 1)])
    for key, v in f.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            fill = np.zeros((n, v.shape[1]), v.dtype)
            if key == "d_edge":
                fill = fill - 1.0
            out[key] = np.concatenate([fill, v, fill], axis=0)
    return out


def build_scene(f: dict[str, Any]):
    cx, cy, sx, sy = _extent(f)

    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(cx, cy, -0.025), hx=sx / 2, hy=sy / 2, hz=0.025, label="pad")
        add_tool_body(b, "darby", DARBY["half"], DARBY["mass"], darby_home(f), collide=False)
        add_tool_body(b, "mtrowel", TUB.TROWEL["half"], TUB.TROWEL["mass"], TUB.home(f), collide=False)
    return build


def _extent(f: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, cell = f["x"], f["y"], float(f["cell"])
    return 0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1]), len(x) * cell, len(y) * cell


def darby_home(f: dict[str, Any]) -> tuple[float, float, float]:
    """The darby's place on the cart, as theirs from the pad's centre."""
    ix, iy = TUB.interior(f)
    return (ix + DARBY_FROM_PAD[0], iy + DARBY_FROM_PAD[1], TUB.CART_TOP + DARBY["half"][2])


def heading(f: dict[str, Any]) -> float:
    """+1: the passes run east to west, -1: west to east; toward the world's low spot (``concrete_tub.low_spot``), so
    the roll reaches it full, after the high spots it cut, and leaves the pad over the form beyond it (a mason works
    toward the low)."""
    return 1.0 if TUB.low_spot(f)[0] <= TUB.interior(f)[0] else -1.0


def program(task: PadDarby, f: dict[str, Any], t0: float = 0.0) -> tuple[ToolProgram, float]:
    """From the cart at ``t0``, ``passes`` passes back and forth, the last toward the low spot (``heading``), so each
    end of the pad gets passes that arrive with a full roll (an empty roll leaves the troughs at its start end
    unfilled, unworked, their stone showing). A pass (east to west below) sets down over the east form at the form
    plane (``OVERRUN`` past the pad's inner face, so the darby's whole width crosses every core cell), crosses the pad
    and the west form and lifts over the ground beyond, pushing the roll off the form there; then it returns high in
    the air (a low return would drag the pile's top along). A late pass goes ``late_below`` under the plane only while
    the blade is clear of both forms, ramping over RAMP. Back to the cart at the end."""
    ix, iy = TUB.interior(f)
    toward = heading(f)
    across = math.pi / 2                                    # the 914 mm blade across the pad's width (body x along y)
    x_start = INTERIOR_HALF[0] + OVERRUN
    x_off = INTERIOR_HALF[0] + 0.038 + DARBY["half"][1] + 0.03   # wholly past the far form, over the ground
    zb = DARBY["half"][2]
    home = darby_home(f)
    keys = [Stroke(0.0, home, yaw=across, mode=HEIGHT)]
    if t0 > 0.0:
        keys.append(Stroke(t0, home, yaw=across, mode=HEIGHT))
    t = t0
    for n in range(task.passes):
        s = toward if (task.passes - 1 - n) % 2 == 0 else -toward
        z0, z1 = FORM_TOP - task.below_plane + zb, FORM_TOP - level(task, n) + zb
        t = lane(keys, t, (ix + s * x_start, iy), (ix + s * EDGE_IN, iy), z0, across, 0.0, task.speed,
                 task.air_speed, mode=HEIGHT, lift=RETURN_LIFT)
        keys.pop()                                          # stay down: on along the pass, at its level inside
        t -= 0.3
        for x, z in ((EDGE_IN - RAMP, z1), (-EDGE_IN + RAMP, z1), (-EDGE_IN, z0), (-x_off, z0)):
            t += abs(keys[-1].pos[0] - (ix + s * x)) / task.speed
            keys.append(Stroke(t, (ix + s * x, iy, z), yaw=across, mode=HEIGHT))
        t += 0.3
        keys.append(Stroke(t, (ix - s * x_off, iy, z0 + RETURN_LIFT), yaw=across, mode=HEIGHT))
    for e in (-toward, toward):                             # the near end first: the last pass lifted there
        t = _end_strokes(task, keys, t, (ix, iy), e)
    t += math.dist(keys[-1].pos, home) / task.air_speed + 0.5
    keys.append(Stroke(t, home, yaw=across, mode=HEIGHT))
    return ToolProgram("darby", tuple(keys)), t + 1.0


def _end_strokes(task: PadDarby, keys: list[Stroke], t: float, centre: tuple[float, float], e: float) -> float:
    """``end_strokes`` short strokes at the late level over the end band on side ``e`` (+1 east): set down END_RUN
    inside the outer limit, slide out to it (the blade's edge END_CLEAR short of the form's inner face) and back,
    then lift.
    A cell d in from that form lies under a blade held inside the forms over only d of a pass's travel, so the last
    passes leave the end bands' stone (30-80 mm in) part-worked; the surface there is already at the sole's level,
    so these strokes cut nothing and only work it, as a mason works the ends."""
    if task.end_strokes <= 0:
        return t
    ix, iy = centre
    across = math.pi / 2
    z1 = FORM_TOP - task.late_below + DARBY["half"][2]
    inner, outer = ix + e * (EDGE_IN - END_CLEAR - END_RUN), ix + e * (EDGE_IN - END_CLEAR)
    t += math.dist(keys[-1].pos[:2], (inner, iy)) / task.air_speed + 0.3
    keys.append(Stroke(t, (inner, iy, z1 + RETURN_LIFT), yaw=across, mode=HEIGHT))
    t += 0.6
    keys.append(Stroke(t, (inner, iy, z1), yaw=across, mode=HEIGHT))
    for _ in range(task.end_strokes):
        for x in (outer, inner):
            t += END_RUN / task.speed
            keys.append(Stroke(t, (x, iy, z1), yaw=across, mode=HEIGHT))
    t += 0.3
    keys.append(Stroke(t, (inner, iy, z1 + RETURN_LIFT), yaw=across, mode=HEIGHT))
    return t


def level(task: PadDarby, n: int) -> float:
    """How far under the form-top plane pass ``n`` holds the sole between the forms [m]."""
    return task.late_below if n >= task.passes - task.late_passes else task.below_plane


def gob_volume(task: PadDarby, f: dict[str, Any]) -> float:
    """m³ a lifted margin trowel keeps of this start's concrete (their stick limit at its age)."""
    mat = CM.Concrete()
    return TUB.gob(float(mat.tau_bulk(float(f["age"]), task.temperature)), mat.rho)


def trips(task: PadDarby, f: dict[str, Any]) -> int:
    """The trowel's trips from the tub to this start's low spot (none without the tub)."""
    return TUB.loads(task.tub_fill * TUB.low_spot(f)[2], gob_volume(task, f)) if task.tub else 0


def world_programs(task: PadDarby, fs: list[dict[str, Any]]) -> tuple[tuple[ToolProgram, ToolProgram], ...]:
    """Per world: the darby's passes after the trowel's tub fill of that world's low spot (or none)."""
    sets = []
    for f in fs:
        n = trips(task, f)
        trowel = TUB.trowel_program(f, TUB.drops(f, n, gob_volume(task, f)))
        darby, _ = program(task, f, t0=trowel.duration + 0.5 if n else 0.0)
        sets.append((darby, trowel))
    return tuple(sets)


def plugins(task: PadDarby, fs: list[dict[str, Any]]) -> tuple[ToolCarriage, SurfaceFlow, ConcretePaste, float]:
    f = fs[0]
    mat = CM.Concrete()
    tau_b = float(mat.tau_bulk(float(f["age"]), task.temperature))
    paste = FilmMaterial("concrete_paste", density=mat.rho, viscosity=SURFACE_VISCOSITY, yield_stress=tau_b,
                         surface_tension=0.0)
    darby = Applicator(body="darby", shape="rect", size=DARBY["half"][:2], center=(0.0, 0.0, -DARBY["half"][2]),
                       stiffness=task.stiffness, damping=0.0, implicit=True, roll_lift=ROLL_LIFT)
    trowel = Applicator(body="mtrowel", shape="rect", size=TUB.TROWEL["half"][:2],
                        center=(0.0, 0.0, -TUB.TROWEL["half"][2]), stiffness=task.stiffness, damping=0.0,
                        implicit=True, store=TUB.store())
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=f["b"].shape),), material=paste, inertia=False,
                       max_depth=0.12, tools=(darby, trowel), tool_speed=0.2, per_cell_rheology=True,
                       trays=(TUB.tub(f),), network=TUB.network(f))
    sets = world_programs(task, fs)
    carriage = ToolCarriage(programs=sets[0], world_programs=sets, flow=flow)
    layer = ConcretePaste(material="concrete", temperature=task.temperature, film_yield="bulk",
                          tools=(SoleWork("darby", *DARBY["work"], half=DARBY["half"][:2]),
                                 SoleWork("mtrowel", *TUB.TROWEL["work"], half=TUB.TROWEL["half"][:2])),
                          flow=flow, carriage=carriage)
    return carriage, flow, layer, carriage.duration + 1.0


@dataclass
class PadResult:
    verdicts: list[dict]
    success: int
    worlds: int
    sim_s: float
    wall_s: float
    moved_l: list[float]
    released_l: list[float]               # let go over the low
    landed_l: list[float]                 # gobs landed on the pad (the film's supplied)
    shed_l: list[float]                   # shed back into the tub by the lifted blade
    lost_l: list[float]                   # fell past the pad and the tub
    trips: list[int]
    ledger_rel: float
    store_ledger_m3: float
    tub_ledger_m3: float
    max_load_n: float                       # the carriage's vertical push (servo transients in the air included): an
                                            # upper bound on a tool's load on the paste, the bearing check
    max_pad_pa: float                       # the film's tool_pressure readout, sampled. Not a load: at grazing contact
                                            # it piles the solve's gravity give on cells um proud (WP15's PRESS-1)
    cells_over_depth: int
    faces_at_cap: int
    cell_m: float


def split(task: PadDarby, f: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(base, film) [nx, ny]: the concrete below the darby's lowest sole level is the base, the rest of the paste the
    film (with the cream and the form-top smear); the forms are base at their top. Their ``tb`` on a form cell is that
    smear (h = b + tb + ts everywhere, layout.py:263-269), so the forms' base is ``b`` alone and the smear is film
    once (SC9's find: base + film stood a smear above h on the form tops). The split is exact as the film holds it:
    the base as float32 (the film's ``base`` input), then the film the rest of their surface in float64 (WP15's nit),
    so a hand-off that splits a surface the same way meets this start cell for cell."""
    paste = np.asarray(f["paste_mask"]) > 0.5
    bulk = np.asarray(f["b"], np.float64) + np.where(paste, np.asarray(f["tb"], np.float64), 0.0)
    cut = FORM_TOP - max(level(task, n) for n in range(task.passes))
    base = np.where(paste, np.minimum(bulk, cut), bulk).astype(np.float32).astype(np.float64)
    return base, np.maximum(np.asarray(f["h"], np.float64) - base, 0.0)


def _set_start(task: PadDarby, flow: SurfaceFlow, layer: ConcretePaste, fs: list[dict[str, Any]]) -> None:
    flat = lambda key, f: np.asarray(f[key], np.float64).reshape(-1)          # noqa: E731
    parts = [split(task, f) for f in fs]
    flow.set_input("base", np.stack([b.reshape(-1) for b, _ in parts]).astype(np.float32))
    flow.set_depth(np.stack([h.reshape(-1) for _, h in parts]))
    layer.set_start(paste=np.stack([flat("paste_mask", f) for f in fs]).astype(np.int32), age=float(fs[0]["age"]),
                    **{k: np.stack([flat(k, f) for f in fs]) for k in ("E", "a", "D", "S", "w", "q")})


def grids(flow: SurfaceFlow, layer: ConcretePaste, fs: list[dict[str, Any]], end: bool = True) -> list[MX.PasteGrid]:
    """The verdict's grids (heights relative to the form-top plane), at the end or from the start fields."""
    n = fs[0]["b"].shape
    if end:
        surf = (flow.base.numpy().astype(np.float64) + flow.h.numpy()).reshape(-1, *n)
        film = flow.h.numpy().reshape(-1, *n)
        r = {k: v.reshape(-1, *n) for k, v in layer.read().items() if v.ndim == 2 and v.shape[1] == n[0] * n[1]}
    out = []
    for w, f in enumerate(fs):
        mask = np.asarray(f["paste_mask"]) > 0.5
        if end:
            g = MX.PasteGrid(h=surf[w] - FORM_TOP, t=film[w], cell=float(f["cell"]), paste_mask=mask, E=r["E"][w],
                             S=r["S"][w], wet_worked=r["wet_worked"][w], D=r["D"][w], a=r["a"][w],
                             d_edge=np.asarray(f["d_edge"]))
        else:
            g = MX.PasteGrid(h=np.asarray(f["h"]) - FORM_TOP, t=np.asarray(f["ts"]), cell=float(f["cell"]),
                             paste_mask=mask, E=np.asarray(f["E"]), S=np.asarray(f["S"]),
                             wet_worked=np.zeros(n, np.int32), D=np.asarray(f["D"]), a=np.asarray(f["a"]),
                             d_edge=np.asarray(f["d_edge"]))
        out.append(g)
    return out


def _tools_off(sim: Any, carriage: ToolCarriage, fs: list[dict[str, Any]]) -> list[int]:
    """Per world: tools not back at their cart places (1 cm)."""
    pos = sim.state.body_q.numpy()[carriage.tool.numpy()][..., :3]               # [W, T, 3]
    return [int((np.linalg.norm(p - np.asarray([darby_home(f), TUB.home(f)]), axis=1) > 0.01).sum())
            for p, f in zip(pos, fs, strict=True)]


def run(task: PadDarby | None = None, *, worlds: int = 1, device: str = "cpu", npz: str | None = None) -> PadResult:
    task = task or PadDarby()
    fs = [start(task, task.seed + w) for w in range(worlds)]                # world w: seed + w
    carriage, flow, layer, duration = plugins(task, fs)
    sim = build_simulation(TestScene(build=build_scene(fs[0]), ground=False, worlds=worlds, dt=2.0e-3),
                           [carriage, flow, layer], device)
    _set_start(task, flow, layer, fs)
    v0 = flow.read()["film_volume"]
    periods = int(math.ceil(duration / 0.02))
    max_load, max_pad, over, cap, t0 = 0.0, 0.0, 0, 0, time.perf_counter()
    for i in range(periods):
        sim.step_period()
        if i % 500 == 0:
            log.info("period %d of %d (%.0f s sim, %.0f s wall)", i, periods, i * 0.02, time.perf_counter() - t0)
        if i % 25 == 0:
            hr = flow.read()
            over, cap = max(over, int(hr["cells_over_depth"].max())), max(cap, int(hr["faces_at_cap"].max()))
            max_load = max(max_load, float(carriage.read()["load"].max()))
            max_pad = max(max_pad, float(hr["tool_pressure"].max()))
    wall = time.perf_counter() - t0
    read = flow.read()
    area = flow.tables.area_np
    moved = [float((np.abs(flow.h.numpy()[w] - fl) @ area) * 1e3 / 2)
             for w, fl in enumerate(np.stack([split(task, f)[1].reshape(-1) for f in fs]))]
    pad = [MX.PadScalars(forms_peak_n=0.0, tools_off=off, tools_total=2, arms_over=0)
           for off in _tools_off(sim, carriage, fs)]
    ends, starts = grids(flow, layer, fs), grids(flow, layer, fs, end=False)
    verdicts = [MX.pad_darby(g, p, s) for g, p, s in zip(ends, pad, starts, strict=True)]
    if npz:
        np.savez_compressed(npz, h=np.stack([g.h for g in ends]).astype(np.float32),
                            E=np.stack([g.E for g in ends]).astype(np.float32),
                            h0=np.stack([g.h for g in starts]).astype(np.float32), mask=ends[0].paste_mask,
                            cell=float(fs[0]["cell"]))
    out = [CV.verdict_dict(v, PAD_DARBY) for v in verdicts]
    return PadResult(verdicts=out, success=sum(d["success"] for d in out), worlds=worlds, sim_s=periods * 0.02,
                     wall_s=wall, moved_l=moved,
                     released_l=(read["tool_released"].sum(axis=1) * 1e3).tolist(),
                     landed_l=(read["supplied_volume"] * 1e3).tolist(),
                     shed_l=(read["supply_to_tank"] * 1e3).tolist(), lost_l=(read["supply_lost"] * 1e3).tolist(),
                     trips=[trips(task, f) for f in fs],
                     ledger_rel=float((np.abs(read["ledger"] - v0) / np.maximum(v0, 1e-12)).max()),
                     store_ledger_m3=float(np.abs(read["store_ledger"]).max()),
                     tub_ledger_m3=float(np.abs(read["tray_ledger"] - TUB.TUB_VOLUME).max()),
                     max_load_n=max_load, max_pad_pa=max_pad, cells_over_depth=over, faces_at_cap=cap,
                     cell_m=float(fs[0]["cell"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--factor", type=int, default=5, help="their 2 mm cells x factor (1, 2 or 5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="final heights and exposure per world, and the start")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(PadDarby(factor=args.factor, seed=args.seed, passes=args.passes), worlds=args.worlds,
              device=args.device, npz=args.npz)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(out)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
