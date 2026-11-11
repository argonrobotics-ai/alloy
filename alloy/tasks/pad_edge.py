# SPDX-License-Identifier: Apache-2.0
"""task pad_edge: Gym-Anything's pad-edge, a surface-flow concrete proof.

"Edge the pad. Tools back on the cart when you are done." (``task.py:23``). Their pad (1.219 x 0.914 m inside 2x4
forms, 89 mm of concrete) 60 min after strike-off at 21 C: darbied flat, a bleed film standing, square sharp edges
against the forms (``layout.py:271``; ``concrete_pad_start.pad_fields("pad-edge")``). Success is their pad-edge
criteria (``pad_metrics.pad_edge``, T:327-395): the 3/8 in arc of the edger 136D on 95 % of the perimeter lines, its
lip's mark, no gouge over 2 mm, the field 150 mm in unchanged and no edge cell worked under bleed water (CIP 14).

Model (as WP6's pad): one ``surface_flow`` plane chart over the pad's subgrade and its forms, at their 2 mm cells
(the arc needs them). Within ``DEEP_BAND`` (82 mm) of the forms the paste is film over a fixed base ``DEEP`` (16 mm)
below the form top: the edger's cut (the arris, 9.5 mm, and the lip's slot, 12.7 mm) is a plough on film, and a gouge
anywhere in their 80 mm gouge band can go deeper than their 2 mm limit. Elsewhere the film is the 2 mm surface layer
over the bulk. The concrete's yield (``concrete_paste``, the only yield writer) holds the film as a plug at rest. The
edger is WP15's ``shape="edger"`` (their Sole "edger", paste.py:557-566) with WP13's roll cap (their 3 mm lift; the
rise from its flat, no spill on the lip's side). Its lip marks the cells it runs down while it works them
(``pad_lip``, their ``compact``).

The scripted policy (a privileged carriage, ``concrete_tools``; no robot: not their task's actor, see below):
1. Wait while the edge band's sheen stands (w > 20 um, until about 135 min in still air): the material clock alone,
   in 1 s steps, the film unstepped (``material_wait``; their wait command, ``env.py:88-111``, ``world.py:322-338``),
   refused while a tool is down or any face of the film could flow (``rest_ratio``, their ``_bound``): exact for a
   paste standing as a plug.
2. Each form counter-clockwise: set down at one end, lip on the form, the sole pitched leading end up with its low
   (trailing) end held at the form-top plane, run to the far end at 0.1 m/s, rock to the other pitch and run back a
   blade length so the low end passes the last stretch too, lift mid-side. The whole pad's loop closes with the first
   form's first 0.4 m again (the last run's overflow lands on its arc in that corner).
3. Back to the cart.
Deviations, disclosed: the pitch is 0.3°, not their finisher's 2°: under a pitched sole the roll fills the wedge
below its raised end and is left there on lift-off, up to L·sin(pitch) (5.3 mm at 2°, over the 1.5 mm arc
tolerance; 0.8 mm at 0.3°).
Not tested on the CPU: the whole pad's run-end corners and the loop's closing re-cut (the 64-world L4 row).

What the verdict judges (audit AB-gym-7; ``concrete_ports.PAD_EDGE``): 5 of Gym-Anything's 8 criteria, the five read
from the paste. Not judged (``passed`` None, never a pass): ``forms_undisturbed`` (the forms are cells, not bodies:
no force is measured; WP6's known gap) and ``arms_clear`` (no robot). ``tools_back`` is judged in part: the edger
alone, by a point test at its rest on the cart (GA: its three tools inside the cart's footprint), so a missing edger
fails and its return is not counted. The scene differs from Gym-Anything's: a privileged carriage moves the edger
(their task interface is robot-only; their carriage is an ASSISTED probe fixture, ``probes/common.py:1-4``), the
scene has the edger alone, and every world starts at one temperature and bleed (theirs vary by seed). A window is
not the whole pad. Each verdict says so (``judged``, ``not_judged``, ``scene_differs``, ``summary``).

Measured (CPU, a 0.37 x 0.37 m south-west corner): see tests/tasks/test_pad_edge.py's corner law.

Run: ``python -m alloy.tasks.pad_edge --window 0.37 0.37`` (a corner, CPU) or the whole pad (``--worlds 64 --device
cuda:0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.tasks import concrete_materials as CM
from alloy.tasks import pad_run as PR
from alloy.tasks.concrete_metrics import PadScalars, PasteGrid
from alloy.tasks.concrete_pad_start import FORM_H
from alloy.tasks.concrete_paste import ConcretePaste, SoleWork
from alloy.tasks.concrete_tools import HEIGHT, Stroke, ToolCarriage, ToolProgram
from alloy.tasks.pad_lip import EdgerLip
from alloy.tasks.pad_run import (CARRIAGE_OMEGA, EXCHANGE_DT, NOT_MEASURED, PAD_STIFFNESS, U_CAP,  # noqa: F401
                                 material_wait, rest_ratio, tools_clear)  # noqa: F401

TASK = "pad-edge"
DEEP, DEEP_BAND = 0.016, 0.082             # m: film over the fixed base within this of the forms: the edger's
                                           # cut, and its gouge band (80 mm, T:308) can be cut deeper than 2 mm
# edger 136D (params.py:145-146; tools.py:121-133): blade 152 x 76.2 x 2 mm, 0.25 kg, r 9.5 mm, lip 12.7 x 1.5 mm;
# sole work: texture 0.05 mm, densification 0.6, sealing 0.2
EDGER = dict(half=(0.076, 0.0381, 0.001), mass=0.25, radius=0.0095, lip=0.0127, lip_w=0.0015,
             work=(5.0e-5, 0.6, 0.2))
CART = (0.0, 0.80, 0.30)                   # the edger's place on the cart, clear of the pad (pad frame)
ROCK_S = 0.3                               # s: rocking from one pitch to the other at a run's far end
PITCH = math.radians(0.3)                  # leading end up: the low end defines the cut; the wedge under the
                                           # raised end holds the roll, left behind on lift-off: 0.8 mm at most
ROLL_LIFT = 0.003                          # m: their Sole's default lift, which the edger keeps (paste.py:534)
RETURN_EXTRA = 0.02                        # m: a run's return goes back a blade length and this much
CLOSE = 0.40                               # m: the first form re-cut at the end of the whole pad's loop
# the carriage's servos (pad_run.CARRIAGE_OMEGA): 15.6 kN/m on the 0.25 kg edger, 0.05 mm under the paste's 0.4 N


@dataclass
class PadEdge:
    """Settings: grid (their 2 mm cells; the arc needs them), an optional window from the south-west inner corner,
    seed, site, and the scripted strokes."""

    seed: int = 0
    temperature: float = 21.1
    air: str = "still"
    window: tuple[float, float] | None = None   # m east and north of the inner corner; None: the whole pad
    speed: float = 0.1                          # m/s along a form
    air_speed: float = 0.3                      # m/s in the air
    sheen: float = CM.SHEEN_W                    # m: wait until the edge band's bleed film is below this
    cart: tuple[float, float, float] = CART      # the edger's rest (pad frame), off the work
    exchange_dt: float = EXCHANGE_DT             # s: target exchange step
    omega: float = CARRIAGE_OMEGA                # rad/s: the carriage's servos (stable: omega·dt, 1.4·omega·dt < 1)
    roll_lift: float | None = ROLL_LIFT          # m: the edger's roll cap (WP13's roll_cap); None: uncapped


def start(task: PadEdge, seed: int) -> dict[str, Any]:
    """Their pad-edge start (2 mm), cut to the window plus the forms beside it."""
    return PR.window(TASK, seed, task.temperature, task.air, task.window)


def chart_fields(f: dict[str, Any]) -> dict[str, np.ndarray]:
    """base (fixed), film and paste mask on the chart [nx, ny]: the film is the paste above ``DEEP`` under the
    form top within ``DEEP_BAND`` of the forms, the surface layer elsewhere, the smear on the forms."""
    h, b, tb = f["h"], f["b"], f["tb"]
    inside, d = f["paste_mask"], f["d_edge"]
    base = np.where(inside, np.where(d < DEEP_BAND, np.minimum(FORM_H - DEEP, b + tb), b + tb), b)
    return {"base": base, "film": np.maximum(h - base, 0.0), "paste": inside}


def edger_body(task: PadEdge) -> PR.ToolBody:
    return PR.ToolBody("edger", EDGER["half"], EDGER["mass"], task.cart)


def build_scene(task: PadEdge, f: dict[str, Any]):
    return PR.build_scene(f, (edger_body(task),))


def edger_tool(profile: bool = True, roll_lift: float | None = None) -> Applicator:
    """The edger's sole on the film: GA's Sole "edger" (``paste.py:557-566``) when the surface-flow shape exists,
    else its flat alone (cost probes only)."""
    hx, hy, hz = EDGER["half"]
    if profile:
        # GA's Sole (hx, hy) includes the lip; d = hy - side·(e2·r) from the lip's face with e2 = normal x axis =
        # -y here, so side -1 puts the lip on the body's +y (the strokes turn +y toward the form)
        return Applicator(body="edger", shape="edger", size=(hx, hy + EDGER["lip_w"]), center=(0.0, 0.0, -hz),
                          normal=(0.0, 0.0, -1.0), axis=(1.0, 0.0, 0.0), stiffness=PAD_STIFFNESS, damping=0.0,
                          implicit=True, edge_radius=EDGER["radius"], edge_lip=EDGER["lip"],
                          edge_lip_w=EDGER["lip_w"], edge_side=-1,
                          **({} if roll_lift is None else {"roll_lift": roll_lift}))
    return Applicator(body="edger", shape="rect", size=(hx, hy), center=(0.0, 0.0, -hz), stiffness=PAD_STIFFNESS,
                      damping=0.0, implicit=True)


# --- the strokes -----------------------------------------------------------------------------------------------------

def sides(f: dict[str, Any]) -> list[tuple[str, tuple[float, float], tuple[float, float], float]]:
    """Each form the chart holds: (side, start, end, yaw) of the edger's centre for a run along it, its lip on the form
    (the sole's +y toward the form), its ends stopping at the perpendicular forms or the window's edge. The whole pad
    is edged counter-clockwise, each run ending in the corner the next one starts from; a window's west run goes
    north from the corner, so both runs leave the corner and end at the window's open edges. Each form's face is the
    paste's cell edge beside it (``pad_run.paste_faces``; the east one is half a cell inside ``INTERIOR / 2``)."""
    w, east, s, n = PR.paste_faces(f)
    e = EDGER["half"][1] + EDGER["lip_w"]                 # centre to the lip's outer face
    L = EDGER["half"][0]
    x0, x1 = max(w, f["x"][0]) + L, min(east, f["x"][-1]) - L     # a window's open edges: its last cell centres
    y0, y1 = max(s, f["y"][0]) + L, min(n, f["y"][-1]) - L
    whole = PR.whole(f)
    out = []
    if f["y"][0] < s:
        out.append(("s", (x0, s + e), (x1, s + e), math.pi))
    if f["x"][-1] > east:
        out.append(("e", (east - e, y0), (east - e, y1), -math.pi / 2))
    if f["y"][-1] > n:
        out.append(("n", (x1, n - e), (x0, n - e), 0.0))
    if f["x"][0] < w:
        out.append(("w", (w + e, y1), (w + e, y0), math.pi / 2) if whole else
                   ("w", (w + e, y0), (w + e, y1), math.pi / 2))
    return out


def attitude(p0: tuple[float, float], p1: tuple[float, float], yaw: float) -> tuple[float, float]:
    """Pitch and body height of a run: the leading end up by ``PITCH`` (their edger's 2°, report §3.6), so the
    trailing end, the sole's lowest point, rides at the form-top plane."""
    along = math.cos(yaw) * (p1[0] - p0[0]) + math.sin(yaw) * (p1[1] - p0[1])      # travel along the body's x
    pitch = PITCH if along < 0.0 else -PITCH                                         # +pitch lifts the body's -x end
    L, hz = EDGER["half"][0], EDGER["half"][2]
    return pitch, FORM_H + L * math.sin(PITCH) + hz * math.cos(PITCH)


def _toward(p0: tuple[float, float], p1: tuple[float, float], length: float) -> tuple[float, float]:
    """The point ``length`` from p1 back toward p0 (p0 when the run is shorter)."""
    run = math.dist(p0, p1)
    if run <= length:
        return p0
    k = length / run
    return (p1[0] + k * (p0[0] - p1[0]), p1[1] + k * (p0[1] - p1[1]))


def strokes(f: dict[str, Any]) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]]:
    """(start, far end, return end, yaw) per run: each form from one end to the other and back a blade length and
    ``RETURN_EXTRA``, lifting mid-side, away from the corners and the other forms' arcs. On the whole pad the loop
    closes with the first form's first ``CLOSE`` again: the last run's overflow lands on its arc there."""
    out = []
    for _, p0, p1, yaw in sides(f):
        out.append((p0, p1, _toward(p0, p1, 2 * EDGER["half"][0] + RETURN_EXTRA), yaw))
    if PR.whole(f) and out:
        p0, p1, _, yaw = out[0]
        end = _toward(p1, p0, CLOSE)
        out.append((p0, end, _toward(p0, end, 2 * EDGER["half"][0] + RETURN_EXTRA), yaw))
    return out


def program(task: PadEdge, f: dict[str, Any], t0: float = 0.0, passes: int = 1) -> tuple[ToolProgram, float]:
    """From the cart at ``t0``: per run (``strokes``), set down at the start and run to the far end pitched
    (``attitude``: the trailing end, the sole's lowest point, at the form-top plane, HEIGHT), rock to the other pitch
    and run back so the low end passes the last blade length too (one way, it stops a blade length short), lift;
    then back to the cart. Returns the program and its end time."""
    keys = [Stroke(t0, task.cart, mode=HEIGHT)]
    t, pos = t0, task.cart[:2]
    for _ in range(passes):
        for p0, p1, p2, yaw in strokes(f):
            pitch, z = attitude(p0, p1, yaw)
            up = z + 0.03
            t += math.dist(pos, p0) / task.air_speed + 0.3
            keys.append(Stroke(t, (*p0, up), yaw=yaw, pitch=pitch, mode=HEIGHT))
            t += 0.5
            keys.append(Stroke(t, (*p0, z), yaw=yaw, pitch=pitch, mode=HEIGHT))
            t += max(math.dist(p0, p1), 1e-3) / task.speed
            keys.append(Stroke(t, (*p1, z), yaw=yaw, pitch=pitch, mode=HEIGHT))
            t += ROCK_S
            keys.append(Stroke(t, (*p1, z), yaw=yaw, pitch=-pitch, mode=HEIGHT))
            t += max(math.dist(p1, p2), 1e-3) / task.speed
            keys.append(Stroke(t, (*p2, z), yaw=yaw, pitch=-pitch, mode=HEIGHT))
            t += 0.5
            keys.append(Stroke(t, (*p2, up), yaw=yaw, pitch=-pitch, mode=HEIGHT))
            pos = p2
    t += math.dist(pos, task.cart[:2]) / task.air_speed + 0.3
    keys.append(Stroke(t, task.cart, mode=HEIGHT))
    return ToolProgram("edger", tuple(keys)), t + 0.5


# --- plug-ins, the wait and the verdict's grids ------------------------------------------------------------------

def plugins(task: PadEdge, f: dict[str, Any], prog: ToolProgram, profile: bool = True):
    nx, ny = f["h"].shape
    flow = SurfaceFlow(charts=(PlaneChart("pad", cells=(nx, ny)),), material=PR.surface_film(f, task.temperature),
                       inertia=False, max_depth=0.025,
                       u_cap=U_CAP, tools=(edger_tool(profile, task.roll_lift),), tool_speed=max(task.speed, 0.1),
                       per_cell_rheology=True)
    carriage = ToolCarriage(programs=(prog,), omega=task.omega, omega_rot=task.omega)
    layer = ConcretePaste(material="concrete", temperature=task.temperature, air=task.air, film_yield="surface",
                          tools=(SoleWork("edger", *EDGER["work"], half=(EDGER["half"][0],
                                                                         EDGER["half"][1] + EDGER["lip_w"])),),
                          flow=flow, carriage=carriage)
    hx, hy, hz = EDGER["half"]
    lip = EdgerLip(half=(hx, hy + EDGER["lip_w"], hz), lip=EDGER["lip"], flow=flow, carriage=carriage, paste=layer)
    return carriage, flow, layer, lip


def set_start(flow: SurfaceFlow, layer: ConcretePaste, fs: list[dict[str, Any]]) -> None:
    PR.set_start(flow, layer, fs, chart_fields)


def edge_wet(layer: ConcretePaste, f: dict[str, Any]) -> np.ndarray:
    """Per world: the largest bleed film [m] on the edge band's paste (within 80 mm of a form)."""
    r = layer.read()
    band = (f["paste_mask"] & (f["d_edge"] < 0.080)).reshape(-1)
    return np.asarray(r["w"])[:, band].max(axis=1)


def grids(flow: SurfaceFlow, layer: ConcretePaste, lip: EdgerLip, fs: list[dict[str, Any]]) -> list[PasteGrid]:
    """The verdict's grids per world: heights relative to the form top, the lip's mark as ``C``."""
    return PR.grids(flow, layer, fs, C=lip.marks().reshape(-1, *fs[0]["h"].shape))


def start_grid(f: dict[str, Any]) -> PasteGrid:
    return PR.start_grid(f, C=f["compact"])


def window_differs(extent: tuple[float, float] | None) -> tuple[str, ...]:
    """A corner window's verdict says it is not the whole pad's."""
    if extent is None:
        return ()
    return (f"window: a {extent[0]:g} x {extent[1]:g} m south-west corner of the pad, not the whole pad: the shares "
            f"count the window's own lines and cells, and the run-end corners and the closing re-cut are not run",)


# --- the run ------------------------------------------------------------------------------------------------------

WAIT_AT = 10                               # periods on the cart before the wait (the world watched at rest)
WAIT_CHUNK = 60.0                          # s per wait command while the sheen stands
LIMIT_S = 110 * 60.0                       # s of material clock the task allows (T:49)


@dataclass
class EdgeResult:
    verdicts: list[dict]                   # per world, by the AB-gym rule (concrete_verdict): judged, summary, ...
    success: int                           # worlds whose judged Gym-Anything criteria all pass
    worlds: int
    waited_s: float
    material_s: float                      # material clock used: waits + stepped time (their limit 110 min)
    sim_s: float
    wall_s: float
    ledger_rel: float
    faces_at_cap: int
    cells_over_depth: int                  # cells deeper than max_depth (flagged, not refused), peak over the run
    cell_m: float
    not_measured: tuple[str, ...] = NOT_MEASURED


def run(task: PadEdge | None = None, *, worlds: int = 1, device: str = "cpu", profile: bool = True,
        npz: str | None = None) -> EdgeResult:
    """Their pad-edge, scripted: on the cart, wait while the edge band's sheen stands, edge every form the chart holds,
    back to the cart; the verdict per world (world w: seed + w)."""
    import time                                                                    # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, build_simulation            # noqa: PLC0415
    from alloy.tasks import pad_metrics as PM                                       # noqa: PLC0415

    task = task or PadEdge()
    fs = [start(task, task.seed + w) for w in range(worlds)]
    prog, duration = program(task, fs[0], t0=(WAIT_AT + 5) * 0.02)
    carriage, flow, layer, lip = plugins(task, fs[0], prog, profile)
    sim = build_simulation(TestScene(build=build_scene(task, fs[0]), ground=False, worlds=worlds, dt=task.exchange_dt),
                           [carriage, flow, layer, lip], device)
    set_start(flow, layer, fs)
    starts = [start_grid(f) for f in fs]
    v0 = flow.read()["film_volume"]
    t0 = time.perf_counter()
    for _ in range(WAIT_AT):
        sim.step_period()
    waited = 0.0
    while float(edge_wet(layer, fs[0]).max()) > task.sheen and waited < LIMIT_S:
        waited += material_wait(sim, flow, layer, carriage, WAIT_CHUNK)
    periods = int(math.ceil(duration / PR.PERIOD)) - WAIT_AT
    cap, over = PR.step_watch(sim, flow, periods)
    wall = time.perf_counter() - t0
    ends = grids(flow, layer, lip, fs)
    off = PR.tools_back(sim, carriage, (edger_body(task),))
    verdicts = [PM.pad_edge(g, s, fs[0]["x"], fs[0]["y"], PadScalars(tools_off=int(o), tools_total=1))
                for g, s, o in zip(ends, starts, off, strict=True)]
    if npz:
        np.savez_compressed(npz, h=np.stack([g.h for g in ends]), h0=np.stack([g.h for g in starts]),
                            lip=np.stack([g.C for g in ends]), wet=np.stack([g.wet_worked for g in ends]),
                            x=fs[0]["x"], y=fs[0]["y"], d=fs[0]["d_edge"], paste=fs[0]["paste_mask"])
    sim_s = (periods + WAIT_AT) * PR.PERIOD
    out = [PR.verdict_dict(v, differs=window_differs(task.window)) for v in verdicts]
    return EdgeResult(verdicts=out, success=sum(d["success"] for d in out), worlds=worlds, waited_s=waited,
                      material_s=waited + sim_s, sim_s=sim_s, wall_s=wall,
                      ledger_rel=PR.ledger_rel(flow, v0), faces_at_cap=cap, cells_over_depth=over,
                      cell_m=float(fs[0]["cell"]))


def main(argv: list[str] | None = None) -> int:
    import argparse                                                                # noqa: PLC0415
    import json                                                                    # noqa: PLC0415
    import logging                                                                 # noqa: PLC0415
    from dataclasses import asdict                                                 # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=float, nargs=2, default=None, help="m east and north of the SW inner corner")
    ap.add_argument("--flat", action="store_true", help="the edger's flat alone (no profile; cost probes)")
    ap.add_argument("--roll-lift", type=float, default=ROLL_LIFT, help="the edger's roll cap lift [m]")
    ap.add_argument("--json", default=None)
    ap.add_argument("--npz", default=None, help="end fields per world (h, lip, wet) and the start")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(PadEdge(seed=args.seed, window=tuple(args.window) if args.window else None, roll_lift=args.roll_lift),
              worlds=args.worlds,
              device=args.device, profile=not args.flat, npz=args.npz)
    out = json.dumps(asdict(res), indent=1, default=float)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(out)
    print(out)
    return 0 if res.success == res.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
